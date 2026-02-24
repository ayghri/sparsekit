#!/usr/bin/env python3
import time
import statistics as stats

def bench(fn, iters=10):
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        ts.append(t1 - t0)
    return {
        "mean_ms": 1000 * stats.mean(ts),
        "min_ms": 1000 * min(ts),
        "median_ms": 1000 * stats.median(ts),
    }

def empty_loop(n):
    for _ in range(n):
        pass

def loop_add(n):
    x = 0
    for i in range(n):
        x += i
    return x

def loop_call(n):
    def f(x): return x + 1
    x = 0
    for _ in range(n):
        x = f(x)
    return x

def main():
    Ns = [10, 100, 1_000, 10_000, 100_000, 1_000_000]

    print("=== Python loop overhead (CPU) ===")
    for n in Ns:
        r = bench(lambda: empty_loop(n), iters=20)
        per_iter_ns = (r["median_ms"] * 1e6) / n
        print(f"empty loop n={n:>9}: median {r['median_ms']:.4f} ms  => {per_iter_ns:.1f} ns/iter")

    print("\n=== Loop doing simple work (CPU) ===")
    for n in [100_000, 1_000_000, 5_000_000]:
        r = bench(lambda: loop_add(n), iters=10)
        per_iter_ns = (r["median_ms"] * 1e6) / n
        print(f"add loop  n={n:>9}: median {r['median_ms']:.4f} ms  => {per_iter_ns:.1f} ns/iter")

    print("\n=== Loop + Python function call overhead (CPU) ===")
    for n in [100_000, 1_000_000]:
        r = bench(lambda: loop_call(n), iters=10)
        per_iter_ns = (r["median_ms"] * 1e6) / n
        print(f"call loop n={n:>9}: median {r['median_ms']:.4f} ms  => {per_iter_ns:.1f} ns/iter")

    # Optional: compare against CUDA launch + tiny kernels
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
            x = torch.randn(1024, device=device)

            def cuda_sync():
                torch.cuda.synchronize()

            # Warmup
            for _ in range(50):
                y = x * 1.0001
            cuda_sync()

            def launch_one():
                y = x * 1.0001  # tiny kernel
                return y

            def launch_k(k):
                y = x
                for _ in range(k):
                    y = y * 1.0001
                return y

            # Measure per-launch with sync included
            def timed_launch_one():
                y = launch_one()
                cuda_sync()
                return y

            r = bench(timed_launch_one, iters=100)
            print("\n=== CUDA tiny-kernel launch+exec (includes synchronize) ===")
            print(f"one tiny op: median {r['median_ms']:.4f} ms  (~{r['median_ms']*1000:.1f} µs)")

            for k in [2, 4, 8, 16]:
                def timed_k():
                    y = launch_k(k)
                    cuda_sync()
                    return y
                r = bench(timed_k, iters=50)
                print(f"k={k:>2} launches: median {r['median_ms']:.4f} ms  => {(r['median_ms']*1000)/k:.1f} µs/launch")
    except Exception as e:
        print("\n(CUDA part skipped)", e)

if __name__ == "__main__":
    main()
