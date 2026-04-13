#!/usr/bin/env python
"""Layer-by-layer pruning benchmark for Qwen3 models.

Supports SparseGPT and StructuredOBS (True OBS) with 2:4 and 4:8 patterns.
Records per-linear and per-decoder-layer statistics for paper figures.

Usage:
    python scripts/prune_gpt.py --method sparsegpt_24
    python scripts/prune_gpt.py --method true_obs_24 --ng 64
    python scripts/prune_gpt.py --method true_obs_48 --eval_every 4
"""

import argparse
import math
import os
import sys
import time

import pandas as pd
import torch
from torch import nn
from torch.nn.attention import sdpa_kernel, SDPBackend
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from lm_eval.evaluator import simple_evaluate
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import TaskManager

from sparsekit import BlockSpec, ScopeSpec, StructuredOBS
from sparsekit.pruners import compute_hessian, output_error
from sparsekit.pruners.obd import obd as obd_tensor
from sparsekit.training.data import get_c4
from sparsekit.training.hooks import ModuleInputCatcher, transfer_to_device
from sparsegpt import SparseGPT

os.environ["HF_HOME"] = "/buckets/datasets/huggingface"

# ── Method definitions ──────────────────────────────────────────────────

METHODS = {
    "sparsegpt_24": dict(pruner="sparsegpt", prune_n=2, prune_m=4, label="SparseGPT 2:4"),
    "sparsegpt_48": dict(pruner="sparsegpt", prune_n=4, prune_m=8, label="SparseGPT 4:8"),
    "true_obs_24": dict(pruner="true_obs", block_size=1, group_size=4, label="True OBS 2:4"),
    "true_obs_48": dict(pruner="true_obs", block_size=1, group_size=8, label="True OBS 4:8"),
    "obd_24": dict(pruner="obd", group_size=4, num_keep=2, label="S-OBD 2:4"),
    "obd_48": dict(pruner="obd", group_size=8, num_keep=4, label="S-OBD 4:8"),
}


# ── Hessian ─────────────────────────────────────────────────────────────


def compute_H(captured_inputs, K, device):
    """Compute H = X^T X / N from captured linear-layer inputs.

    Also handles dead columns (zero diagonal) by setting H[dead,dead]=1.
    Returns (H, dead_mask) where dead_mask is a boolean tensor of dead columns.
    """
    H = torch.zeros(K, K, device=device, dtype=torch.float32)
    N = 0
    for ins in captured_inputs:
        x = ins["args"][0].to(device=device, dtype=torch.float32)
        if x.ndim == 3:
            x = x.reshape(-1, K)
        n = x.shape[0]
        H.addmm_(x.T, x)
        N += n
    H /= N
    # Handle dead input features (zero or near-zero diagonal → singular H)
    d = torch.diag(H)
    dead = d < 1e-6 * d.max().clamp(min=1e-12)
    H[dead, dead] = 1.0
    return H, dead


# ── Error metrics ───────────────────────────────────────────────────────


def weight_error(W_pruned, W_orig):
    """Relative weight error: ||dW|| / ||W||."""
    dW = (W_pruned - W_orig).float()
    return (dW.norm() / W_orig.float().norm()).item()


def sparsity_pct(W):
    """Percentage of zero elements."""
    return 100.0 * (W.abs() < 1e-10).float().mean().item()


# ── Pruning ─────────────────────────────────────────────────────────────


def prune_sparsegpt(layer, captured_inputs, prune_n, prune_m, device):
    """Apply SparseGPT pruning to a single nn.Linear layer."""
    gpt = SparseGPT(layer)
    for ins in captured_inputs:
        x = ins["args"][0].to(device)
        if x.ndim == 3:
            for b in range(x.shape[0]):
                gpt.add_batch(x[b], None)
        else:
            gpt.add_batch(x, None)
    gpt.fasterprune(0.5, prune_n=prune_n, prune_m=prune_m)
    gpt.free()


def prune_obd(layer, H, scope_size, num_keep):
    """Apply OBD pruning to a single nn.Linear layer in-place."""
    layer.weight.data = obd_tensor(
        layer.weight.data, H, scope_size, num_keep
    )


def prune_true_obs(layer, H, C, block_size, group_size, ng, chunk_size):
    """Apply StructuredOBS True OBS pruning to a single nn.Linear layer."""
    bpg = group_size // block_size
    nnz = bpg // 2

    block = BlockSpec(layer.weight, shape=(1, block_size))
    scope = ScopeSpec(block, shape=(1, bpg))
    obs = StructuredOBS(scope, H, inv_h=C)
    obs.prune_true_obs(
        nnz=nnz,
        ng=ng,
        chunk_size=chunk_size,
        order="largest_first",
        scoring="independent",
        c_dtype=torch.float32,
    )


# ── Layer helpers ───────────────────────────────────────────────────────


def get_prunable_linears(decoder_layer):
    """Return dict of {name: nn.Linear} for all projection layers."""
    linears = {}
    for name, mod in decoder_layer.self_attn.named_children():
        if "_proj" in name and isinstance(mod, nn.Linear):
            linears[f"self_attn.{name}"] = mod
    for name, mod in decoder_layer.mlp.named_children():
        if "_proj" in name and isinstance(mod, nn.Linear):
            linears[f"mlp.{name}"] = mod
    return linears


# Pruning stages: linears in same stage share inputs; later stages
# depend on pruned outputs from earlier stages.
STAGES = [
    ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
    ["self_attn.o_proj"],
    ["mlp.gate_proj", "mlp.up_proj"],
    ["mlp.down_proj"],
]


def capture_first_layer_inputs(model, tokenized_inputs, device):
    """Capture inputs to the first decoder layer via full-model forward."""
    catcher = ModuleInputCatcher(device=torch.device("cpu"))
    first_layer = model.model.layers[0]
    catcher.attach(first_layer, "layer_0", raise_error=True)

    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        with torch.no_grad():
            for sample in tqdm(tokenized_inputs, desc="Capturing layer 0 inputs"):
                try:
                    model(
                        **transfer_to_device(sample, device),
                        labels=None,
                        use_cache=False,
                    )
                except Exception:
                    pass

    inputs = catcher.inputs["layer_0"]
    catcher.detach("layer_0")
    return inputs


def _get_hidden_states(out):
    """Extract hidden_states from decoder layer output (Tensor or tuple)."""
    if isinstance(out, torch.Tensor):
        return out
    return out[0]


def capture_all_layers_original(model, tokenized_inputs, device):
    """Pre-capture inputs for ALL decoder layers using the original model.

    Returns dict: {layer_idx: [{"args": ..., "kwargs": ...}, ...]}
    """
    num_layers = len(model.model.layers)
    catcher = ModuleInputCatcher(device=torch.device("cpu"))
    for li in range(num_layers):
        layer = model.model.layers[li]
        catcher.attach(layer, f"layer_{li}")

    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        with torch.no_grad():
            for sample in tqdm(tokenized_inputs, desc="Capturing all layer inputs"):
                model(
                    **transfer_to_device(sample, device),
                    labels=None,
                    use_cache=False,
                )

    all_inputs = {}
    for li in range(num_layers):
        name = f"layer_{li}"
        all_inputs[li] = catcher.inputs[name]
        catcher.detach(name)

    return all_inputs


def capture_linear_inputs_from_layer_inputs(decoder_layer, layer_inputs, linears, device):
    """Forward layer_inputs through decoder, capturing inputs for given linears.

    Returns {name: [{"args": ..., "kwargs": ...}, ...]}
    """
    catcher = ModuleInputCatcher(device=torch.device("cpu"))
    for name, mod in linears.items():
        catcher.attach(mod, name)

    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        with torch.no_grad():
            for inp in layer_inputs:
                inp_dev = transfer_to_device(inp, device)
                decoder_layer(*inp_dev["args"], **inp_dev["kwargs"])

    linear_inputs = {}
    for name in linears:
        linear_inputs[name] = catcher.inputs[name]
        catcher.detach(name)
    return linear_inputs


def capture_all_linear_inputs(decoder_layer, layer_inputs, linears, device):
    """Forward through decoder, capturing inputs to ALL linears + original outputs.

    Returns:
        linear_inputs: {name: [{"args": ..., "kwargs": ...}, ...]}
        orig_outputs: list of decoder output hidden states (on CPU)
    """
    catcher = ModuleInputCatcher(device=torch.device("cpu"))
    for name, mod in linears.items():
        catcher.attach(mod, name)

    orig_outputs = []
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        with torch.no_grad():
            for inp in layer_inputs:
                inp_dev = transfer_to_device(inp, device)
                out = decoder_layer(*inp_dev["args"], **inp_dev["kwargs"])
                orig_outputs.append(_get_hidden_states(out).cpu())

    linear_inputs = {}
    for name in linears:
        linear_inputs[name] = catcher.inputs[name]
        catcher.detach(name)

    return linear_inputs, orig_outputs


def propagate_through_pruned_layer(decoder_layer, layer_inputs, device):
    """Forward layer_inputs through the (pruned) decoder layer.

    Returns new layer_inputs list (on CPU) for the next decoder layer.
    """
    new_inputs = []
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        with torch.no_grad():
            for inp in layer_inputs:
                inp_dev = transfer_to_device(inp, device)
                out = decoder_layer(*inp_dev["args"], **inp_dev["kwargs"])
                # Package output as input for the next layer
                # Decoder layers return (hidden_states, ...) and next layer
                # expects (hidden_states,) as args + same kwargs
                new_args = (_get_hidden_states(out).cpu(),)
                # Keep kwargs (attention_mask, position_ids, etc.) from original
                new_inputs.append(
                    {"args": new_args, "kwargs": inp["kwargs"]}
                )
    return new_inputs


def compute_decoder_error(decoder_layer, layer_inputs, orig_outputs, device):
    """Compute relative decoder output error after pruning."""
    err_sq = 0.0
    ref_sq = 0.0
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        with torch.no_grad():
            for i, inp in enumerate(layer_inputs):
                inp_dev = transfer_to_device(inp, device)
                out = decoder_layer(*inp_dev["args"], **inp_dev["kwargs"])
                pruned = _get_hidden_states(out).float()
                orig = orig_outputs[i].to(device).float()
                err_sq += (pruned - orig).pow(2).sum().item()
                ref_sq += orig.pow(2).sum().item()
    return math.sqrt(err_sq / ref_sq) if ref_sq > 0 else float("inf")


# ── Perplexity evaluation ──────────────────────────────────────────────


def evaluate_perplexity(hf_model, task_manager):
    """Evaluate wikitext perplexity. Returns dict with metrics."""
    old_stdout, old_stderr = sys.stdout, sys.stderr
    devnull = open(os.devnull, "w")
    sys.stdout = devnull
    sys.stderr = devnull
    try:
        with torch.no_grad():
            results = simple_evaluate(
                model=hf_model,
                tasks=["wikitext"],
                num_fewshot=0,
                task_manager=task_manager,
                log_samples=False,
                batch_size=2,
                verbosity="ERROR",
            )
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        devnull.close()

    if results is None:
        return {}
    return results["results"].get("wikitext", {})


# ── Main ────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Layer-by-layer pruning benchmark")
    parser.add_argument(
        "--method",
        choices=list(METHODS.keys()),
        required=True,
        help="Pruning method",
    )
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B", help="Model name")
    parser.add_argument("--device", default="cuda", help="CUDA device")
    parser.add_argument("--num_samples", type=int, default=1024, help="Calibration samples")
    parser.add_argument("--seq_length", type=int, default=1024, help="Sequence length")
    parser.add_argument(
        "--eval_every",
        type=int,
        default=-1,
        help="Evaluate perplexity every N layers (-1=only final)",
    )
    parser.add_argument("--ng", type=int, default=64, help="Groups per batch (True OBS)")
    parser.add_argument("--chunk_size", type=int, default=16, help="Rows per chunk (True OBS)")
    parser.add_argument(
        "--original_h",
        action="store_true",
        help="Use original (unpruned) model activations for H computation",
    )
    parser.add_argument("--output", default="results", help="Output directory")
    parser.add_argument(
        "--save_model",
        default=None,
        help="Save pruned model to this directory (HuggingFace format)",
    )
    args = parser.parse_args()

    method_cfg = METHODS[args.method]
    device = torch.device(args.device)
    os.makedirs(args.output, exist_ok=True)

    print(f"Method: {method_cfg['label']}")
    print(f"Model:  {args.model}")
    print(f"Device: {device}")
    print(f"Samples: {args.num_samples}, Seq length: {args.seq_length}")
    print()

    # ── Load model ──
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype="auto", device_map=args.device
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    num_layers = len(model.model.layers)
    print(f"  {num_layers} decoder layers")
    print()

    # ── Load calibration data ──
    print("Loading C4 calibration data...")
    c4_data = get_c4(
        num_samples=args.num_samples,
        seq_len=args.seq_length,
        tokenizer=tokenizer,
        seed=42,
    )
    tokenized_inputs = [
        {"input_ids": d[0], "attention_mask": torch.ones_like(d[0])}
        for d in c4_data
    ]
    print(f"  {len(tokenized_inputs)} samples")
    print()

    # ── Setup evaluation ──
    hf_model = HFLM(model, tokenizer=tokenizer)
    task_manager = TaskManager()

    # ── Baseline perplexity ──
    print("Evaluating baseline perplexity...")
    baseline = evaluate_perplexity(hf_model, task_manager)
    print(f"  Baseline: word_ppl={baseline.get('word_perplexity,none', 'N/A'):.2f}")
    print()

    linear_results = []
    layer_results = [
        {
            "layer_idx": -1,
            "num_linears": 0,
            "layer_prune_time_s": 0.0,
            "decoder_error_pct": 0.0,
            "word_perplexity": baseline.get("word_perplexity,none"),
            "byte_perplexity": baseline.get("byte_perplexity,none"),
            "bits_per_byte": baseline.get("bits_per_byte,none"),
            "cumulative_time_s": 0.0,
        }
    ]
    cumulative_time = 0.0

    # ── Capture layer inputs ──
    if args.original_h:
        # Pre-capture inputs for ALL layers from the original (unpruned) model.
        # Each layer's H is computed from clean activations, not post-pruning.
        print("Capturing inputs for ALL layers from original model...")
        original_layer_inputs = capture_all_layers_original(
            model, tokenized_inputs, device
        )
        print(f"  {num_layers} layers x {len(original_layer_inputs[0])} samples")
        # Still need propagated inputs for decoder error + next-layer propagation
        layer_inputs = original_layer_inputs[0]
    else:
        print("Capturing first decoder layer inputs...")
        layer_inputs = capture_first_layer_inputs(model, tokenized_inputs, device)
    print(f"  {len(layer_inputs)} input samples captured")
    print()

    # ── Layer-by-layer pruning ──
    for layer_idx in range(num_layers):
        print(f"{'='*60}")
        print(f"Decoder layer {layer_idx}/{num_layers-1}")
        print(f"{'='*60}")

        decoder = model.model.layers[layer_idx]
        linears = get_prunable_linears(decoder)
        layer_prune_time = 0.0

        # Choose which inputs to use for H computation
        if args.original_h:
            h_layer_inputs = original_layer_inputs[layer_idx]
        else:
            h_layer_inputs = layer_inputs

        # Capture inputs to ALL linears + original decoder outputs
        print("  Capturing linear inputs + original outputs...")
        all_linear_inputs, orig_outputs = capture_all_linear_inputs(
            decoder, h_layer_inputs, linears, device
        )

        # Prune each linear
        for lin_name, linear in linears.items():
            M, K = linear.weight.shape

            t_h = time.time()
            H, dead = compute_H(all_linear_inputs[lin_name], K, device)
            if dead.any():
                linear.weight.data[:, dead] = 0
            W_orig = linear.weight.data.clone()
            torch.cuda.synchronize(device)
            h_time = time.time() - t_h

            c_time = 0.0

            if method_cfg["pruner"] == "sparsegpt":
                t_p = time.time()
                prune_sparsegpt(
                    linear,
                    all_linear_inputs[lin_name],
                    method_cfg["prune_n"],
                    method_cfg["prune_m"],
                    device,
                )
                torch.cuda.synchronize(device)
                prune_time = time.time() - t_p

            elif method_cfg["pruner"] == "obd":
                t_p = time.time()
                prune_obd(
                    linear, H, method_cfg["group_size"], method_cfg["num_keep"]
                )
                torch.cuda.synchronize(device)
                prune_time = time.time() - t_p

            elif method_cfg["pruner"] == "true_obs":
                t_c = time.time()
                C = StructuredOBS.compute_inverse(H, damp=0.01)
                torch.cuda.synchronize(device)
                c_time = time.time() - t_c

                t_p = time.time()
                prune_true_obs(
                    linear,
                    H,
                    C,
                    method_cfg["block_size"],
                    method_cfg["group_size"],
                    args.ng,
                    args.chunk_size,
                )
                torch.cuda.synchronize(device)
                prune_time = time.time() - t_p
                del C

            total_time = h_time + c_time + prune_time
            layer_prune_time += total_time

            out_err = output_error(linear.weight.data, W_orig, H)
            wt_err = weight_error(linear.weight.data, W_orig)
            sp = sparsity_pct(linear.weight.data)

            linear_results.append(
                {
                    "layer_idx": layer_idx,
                    "linear_name": lin_name,
                    "M": M,
                    "K": K,
                    "h_time_s": round(h_time, 3),
                    "c_time_s": round(c_time, 3),
                    "prune_time_s": round(prune_time, 3),
                    "total_time_s": round(total_time, 3),
                    "output_error_pct": round(out_err * 100, 4),
                    "weight_error_pct": round(wt_err * 100, 4),
                    "sparsity_pct": round(sp, 2),
                }
            )

            print(
                f"  {lin_name:25s} ({M:4d},{K:4d})  "
                f"out_err={out_err*100:6.2f}%  wt_err={wt_err*100:6.2f}%  "
                f"sp={sp:.0f}%  t={total_time:.1f}s"
            )

            del H, W_orig
            torch.cuda.empty_cache()

        del all_linear_inputs
        cumulative_time += layer_prune_time

        # Decoder output error (compare pruned vs original, same inputs)
        print("  Computing decoder output error...")
        dec_err = compute_decoder_error(
            decoder, h_layer_inputs, orig_outputs, device
        )
        print(f"  Decoder error: {dec_err*100:.2f}%")
        del orig_outputs

        # Propagate through pruned layer → next layer inputs
        # (always from propagated inputs for correct next-layer state)
        print("  Propagating inputs through pruned layer...")
        layer_inputs = propagate_through_pruned_layer(
            decoder, layer_inputs, device
        )

        torch.cuda.empty_cache()

        # Perplexity evaluation
        word_ppl = None
        byte_ppl = None
        bits = None

        should_eval = layer_idx == num_layers - 1 or (
            args.eval_every > 0 and (layer_idx + 1) % args.eval_every == 0
        )

        if should_eval:
            print(f"  Evaluating perplexity (after layer {layer_idx})...")
            ppl_result = evaluate_perplexity(hf_model, task_manager)
            word_ppl = ppl_result.get("word_perplexity,none")
            byte_ppl = ppl_result.get("byte_perplexity,none")
            bits = ppl_result.get("bits_per_byte,none")
            print(f"  Word perplexity: {word_ppl:.2f}" if word_ppl else "  PPL: N/A")

        layer_results.append(
            {
                "layer_idx": layer_idx,
                "num_linears": len(linears),
                "layer_prune_time_s": round(layer_prune_time, 3),
                "decoder_error_pct": round(dec_err * 100, 4),
                "word_perplexity": word_ppl,
                "byte_perplexity": byte_ppl,
                "bits_per_byte": bits,
                "cumulative_time_s": round(cumulative_time, 3),
            }
        )

        # Save incrementally
        _save_results(args, linear_results, layer_results)
        print()

    # ── Save pruned model ──
    if args.save_model:
        print(f"Saving pruned model to {args.save_model}...")
        model.save_pretrained(args.save_model)
        tokenizer.save_pretrained(args.save_model)
        print(f"  Saved.")

    # ── Summary ──
    print()
    print("=" * 60)
    print(f"PRUNING COMPLETE — {method_cfg['label']}")
    print("=" * 60)
    _print_summary(args, layer_results)


def _run_prefix(args):
    """Build unique filename prefix from run parameters."""
    # Extract short model name: "Qwen/Qwen3-0.6B" -> "qwen3_0.6b"
    model_short = args.model.split("/")[-1].lower().replace("-", "_")
    parts = [model_short, args.method, f"n{args.num_samples}"]
    if args.original_h:
        parts.append("origH")
    return os.path.join(args.output, "_".join(parts))


def _save_results(args, linear_results, layer_results):
    """Save CSVs incrementally."""
    prefix = _run_prefix(args)
    pd.DataFrame(linear_results).to_csv(f"{prefix}_linear.csv", index=False)
    pd.DataFrame(layer_results).to_csv(f"{prefix}_layer.csv", index=False)


def _print_summary(args, layer_results):
    """Print final summary table."""
    df = pd.DataFrame(layer_results)
    print(df.to_string(index=False))
    prefix = _run_prefix(args)
    print(f"\nResults saved to:")
    print(f"  {prefix}_linear.csv")
    print(f"  {prefix}_layer.csv")


if __name__ == "__main__":
    main()
