#!/usr/bin/env python3
"""
Benchmark two coupling reduction approaches:

A) Materialize concat along payload dim then reduce
B) Separable reduce (reduce each, then sum) without concat

Assumptions (defaults):
- block payload = 256 elements
- block grid = (4096, 1024)
- number of coupled tensors ("groups") = 2..4

Notes:
- The default problem is *huge* (4096*1024*256 ≈ 1.07e9 elems per tensor).
  With fp16 that's ~2.0 GiB per tensor. Concat can allocate a big temporary.
- If you OOM, rerun with smaller --grid0/--grid1.
"""

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass

import torch


@dataclass
class BenchResult:
    name: str
    ms: float
    ok: bool
    extra: str = ""


def fmt_bytes(n: float) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    u = 0
    while n >= 1024.0 and u < len(units) - 1:
        n /= 1024.0
        u += 1
    return f"{n:.2f} {units[u]}"


def cuda_time_ms(fn, iters: int, warmup: int) -> float:
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def cpu_time_ms(fn, iters: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0 / iters


def try_run(name: str, fn, timer, iters: int, warmup: int) -> BenchResult:
    try:
        ms = timer(fn, iters=iters, warmup=warmup)
        return BenchResult(name=name, ms=ms, ok=True)
    except RuntimeError as e:
        msg = str(e)
        if "out of memory" in msg.lower():
            return BenchResult(name=name, ms=float("nan"), ok=False, extra="OOM")
        return BenchResult(name=name, ms=float("nan"), ok=False, extra=msg[:200])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32", "bfloat16"])
    ap.add_argument("--grid0", type=int, default=2048)
    ap.add_argument("--grid1", type=int, default=1024)
    ap.add_argument("--payload", type=int, default=256)
    ap.add_argument("--kmin", type=int, default=2)
    ap.add_argument("--kmax", type=int, default=4)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--compile", action="store_true", help="torch.compile the benchmarked fns")
    ap.add_argument("--permute", action="store_true", help="apply a dummy permute of grid axes before reduce")
    args = ap.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; falling back to CPU.")
        device = torch.device("cpu")

    dtype = getattr(torch, args.dtype)
    grid = (args.grid0, args.grid1)
    payload = args.payload

    nblocks = grid[0] * grid[1]
    nelems_per_tensor = nblocks * payload
    bytes_per_tensor = nelems_per_tensor * torch.tensor([], dtype=dtype).element_size()

    print("=== Setup ===")
    print(f"device         : {device}")
    print(f"dtype          : {dtype}")
    print(f"grid           : {grid}  (#blocks={nblocks:,})")
    print(f"payload        : {payload}")
    print(f"elements/tensor: {nelems_per_tensor:,}")
    print(f"bytes/tensor   : {fmt_bytes(bytes_per_tensor)}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        print(f"gpu            : {props.name}  (VRAM={fmt_bytes(props.total_memory)})")
    print()

    timer = cuda_time_ms if device.type == "cuda" else cpu_time_ms

    # Pre-allocate a reference "orders" behavior: optional permute grid axes then keep payload last.
    # If permute=True, we simulate the coupling alignment step v = v.permute(1,0,2) (swap grid dims).
    def maybe_permute(v: torch.Tensor, explains="v shape [g0,g1,payload]") -> torch.Tensor:
        if not args.permute:
            return v
        # swap grid axes; payload stays last
        return v.permute(1, 0, 2).contiguous()  # contiguous to isolate permute cost if desired

    # Benchmark for K = 2..4
    for K in range(args.kmin, args.kmax + 1):
        print(f"=== K={K} coupled tensors ===")

        # Allocate K tensors shaped [g0, g1, payload]
        # (Keep as 3D so it mirrors block views.)
        tensors = []
        try:
            for i in range(K):
                t = torch.randn((grid[0], grid[1], payload), device=device, dtype=dtype)
                tensors.append(t)
        except RuntimeError as e:
            print("Allocation failed:", e)
            print("Try smaller --grid0/--grid1.")
            return

        # Approach A: concat then reduce L2 over payload
        def concat_then_reduce_l2():
            vs = [maybe_permute(t) for t in tensors]
            cat = torch.cat(vs, dim=-1)          # [grid..., K*payload]
            out = (cat * cat).sum(dim=-1).sqrt() # [grid...]
            # prevent DCE
            return out

        # Approach B: separable reduce, no concat
        def separable_reduce_l2():
            vs = [maybe_permute(t) for t in tensors]
            # stack then reduce over payload, sum across K
            # shape: [K, grid0, grid1]
            s = torch.stack([(v * v).sum(dim=-1) for v in vs], dim=0).sum(dim=0)
            out = s.sqrt()
            return out

        # Optional torch.compile
        fA = concat_then_reduce_l2
        fB = separable_reduce_l2
        if args.compile:
            # fullgraph=False improves robustness; set True if you want stricter graphs
            fA = torch.compile(fA, fullgraph=False)
            fB = torch.compile(fB, fullgraph=False)

        # Run once to ensure correctness + shape
        with torch.no_grad():
            a = fA()
            b = fB()
            # They should match closely (within fp16 noise); use fp32 accumulator for check on CPU if needed.
            max_abs = (a - b).abs().max().item()
            print(f"max|A-B|: {max_abs:.6g}")

        # Time them
        with torch.no_grad():
            rA = try_run("A concat->reduce L2", fA, timer, iters=args.iters, warmup=args.warmup)
            rB = try_run("B separable reduce L2", fB, timer, iters=args.iters, warmup=args.warmup)

        # Estimate *read* traffic (very rough):
        # Both read K tensors (K*bytes_per_tensor). A also writes + reads concat temp roughly K*bytes_per_tensor.
        read_bytes = K * bytes_per_tensor
        concat_temp_bytes = K * bytes_per_tensor  # cat output
        # In reality, reductions also write output [grid0,grid1], tiny compared to inputs.
        print(f"approx input read          : {fmt_bytes(read_bytes)} / iter")
        print(f"approx concat temp (A only): {fmt_bytes(concat_temp_bytes)} / iter")
        print()

        def show(res: BenchResult):
            if res.ok:
                print(f"{res.name:<22} : {res.ms:>8.3f} ms")
            else:
                print(f"{res.name:<22} :   (failed) {res.extra}")

        show(rA)
        show(rB)

        if rA.ok and rB.ok:
            speedup = rA.ms / rB.ms if rB.ms > 0 else float("inf")
            print(f"speedup (A/B): {speedup:.3f}x  (higher is better for separable)")
        print()

        # Cleanup between K runs to reduce fragmentation on smaller GPUs
        del tensors, a, b
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("Done.")


if __name__ == "__main__":
    main()
