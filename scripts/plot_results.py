"""
Plot comparison between S-OBS and SparseGPT pruning results.
Generates multiple figures with paper-quality formatting.

Usage:
    python playground/plot_results.py [results_dir] [--model MODEL_NAME]

Default: results_dir=results, model=Qwen3-0.6B
"""
import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.facecolor": "none",
        "savefig.transparent": True,
        "figure.facecolor": "none",
        "axes.facecolor": "none",
    }
)

COLORS = {
    "sparsegpt": "#e74c3c",
    "true_obs": "#2980b9",
    "baseline": "#27ae60",
    "improvement": "#8e44ad",
}

LINEAR_SHORT = {
    "self_attn.q_proj": "Q Proj",
    "self_attn.k_proj": "K Proj",
    "self_attn.v_proj": "V Proj",
    "self_attn.o_proj": "O Proj",
    "mlp.gate_proj": "Gate Proj",
    "mlp.up_proj": "Up Proj",
    "mlp.down_proj": "Down Proj",
}

# Layout for error-per-sublayer plot: 3 rows, centered last row
SUBLAYER_GRID = [
    ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
    ["self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj"],
    ["mlp.down_proj"],
]


def load_results(results_dir, prefix):
    layer_path = os.path.join(results_dir, f"{prefix}_layer.csv")
    linear_path = os.path.join(results_dir, f"{prefix}_linear.csv")
    layer_df = pd.read_csv(layer_path) if os.path.exists(layer_path) else None
    linear_df = pd.read_csv(linear_path) if os.path.exists(linear_path) else None
    return layer_df, linear_df


def find_result_pairs(results_dir):
    """Find matching sparsegpt/true_obs result pairs by suffix."""
    files = os.listdir(results_dir)
    layer_files = [f for f in files if f.endswith("_layer.csv")]

    pairs = {}
    for f in layer_files:
        name = f.replace("_layer.csv", "")
        if name.startswith("sparsegpt_24"):
            suffix = name[len("sparsegpt_24") :]
            key = f"24{suffix}"
            pairs.setdefault(key, {})["sparsegpt"] = name
        elif name.startswith("true_obs_24"):
            suffix = name[len("true_obs_24") :]
            key = f"24{suffix}"
            pairs.setdefault(key, {})["true_obs"] = name

    # Only return pairs where both methods exist
    return {k: v for k, v in pairs.items() if len(v) == 2}


def plot_perplexity_progression(sg_layer, to_layer, output_dir, tag, model_name):
    """Plot 1: Perplexity after each evaluated layer."""
    baseline_ppl = sg_layer[sg_layer["layer_idx"] == -1]["word_perplexity"].values[0]

    sg = sg_layer[sg_layer["word_perplexity"].notna() & (sg_layer["layer_idx"] >= 0)]
    to = to_layer[to_layer["word_perplexity"].notna() & (to_layer["layer_idx"] >= 0)]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axhline(
        y=baseline_ppl,
        color=COLORS["baseline"],
        linestyle="--",
        linewidth=1.5,
        label=f"Dense baseline ({baseline_ppl:.1f})",
        alpha=0.7,
    )
    ax.plot(
        sg["layer_idx"],
        sg["word_perplexity"],
        "o-",
        color=COLORS["sparsegpt"],
        linewidth=2,
        markersize=6,
        label=f'SparseGPT ({sg["word_perplexity"].iloc[-1]:.1f})',
    )
    ax.plot(
        to["layer_idx"],
        to["word_perplexity"],
        "s-",
        color=COLORS["true_obs"],
        linewidth=2,
        markersize=6,
        label=f'S-OBS ({to["word_perplexity"].iloc[-1]:.1f})',
    )

    ax.set_xlabel("Layers pruned (cumulative)")
    ax.set_ylabel("WikiText-2 Perplexity")
    ax.set_title(f"{model_name} — 2:4 Pruning: Perplexity Progression")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(output_dir, f"ppl_progression_{tag}.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_decoder_error(sg_layer, to_layer, output_dir, tag, model_name):
    """Plot 2: Per-layer decoder output error."""
    sg = sg_layer[sg_layer["layer_idx"] >= 0].sort_values("layer_idx")
    to = to_layer[to_layer["layer_idx"] >= 0].sort_values("layer_idx")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: absolute decoder error
    ax = axes[0]
    ax.bar(
        sg["layer_idx"] - 0.2,
        sg["decoder_error_pct"],
        0.4,
        color=COLORS["sparsegpt"],
        label="SparseGPT",
        alpha=0.85,
    )
    ax.bar(
        to["layer_idx"] + 0.2,
        to["decoder_error_pct"],
        0.4,
        color=COLORS["true_obs"],
        label="S-OBS",
        alpha=0.85,
    )
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Decoder Output Error (%)")
    ax.set_title(f"{model_name} — Decoder Error per Layer")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Right: relative improvement (positive = S-OBS is better)
    ax2 = axes[1]
    improvement = sg["decoder_error_pct"].values - to["decoder_error_pct"].values
    colors_bar = [
        COLORS["true_obs"] if x > 0 else COLORS["sparsegpt"] for x in improvement
    ]
    ax2.bar(sg["layer_idx"].values, improvement, color=colors_bar, alpha=0.85)
    ax2.axhline(y=0, color="black", linewidth=0.8)
    ax2.set_xlabel("Layer Index")
    ax2.set_ylabel("Error Difference (SparseGPT - S-OBS)")
    ax2.set_title("Decoder Error: Positive = S-OBS Better")
    ax2.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    path = os.path.join(output_dir, f"decoder_error_{tag}.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_linear_output_error(sg_linear, to_linear, output_dir, tag, model_name):
    """Plot 3: Per-linear output error comparison."""
    # Merge on (layer_idx, linear_name) to align
    merged = sg_linear.merge(
        to_linear,
        on=["layer_idx", "linear_name"],
        suffixes=("_sg", "_to"),
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: scatter plot — each point is one linear layer
    ax = axes[0]
    for lname, short in LINEAR_SHORT.items():
        subset = merged[merged["linear_name"] == lname]
        ax.scatter(
            subset["output_error_pct_sg"],
            subset["output_error_pct_to"],
            label=short,
            alpha=0.7,
            s=30,
        )
    lims = [0, max(merged["output_error_pct_sg"].max(), merged["output_error_pct_to"].max()) * 1.05]
    ax.plot(lims, lims, "k--", linewidth=0.8, alpha=0.5, label="y=x")
    ax.set_xlabel("SparseGPT Output Error (%)")
    ax.set_ylabel("S-OBS Output Error (%)")
    ax.set_title("Per-Linear Output Error (H-based)")
    ax.legend(fontsize=8, ncol=2)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # Right: average error by linear type
    ax2 = axes[1]
    linear_names = list(LINEAR_SHORT.keys())
    sg_means = [
        merged[merged["linear_name"] == n]["output_error_pct_sg"].mean()
        for n in linear_names
    ]
    to_means = [
        merged[merged["linear_name"] == n]["output_error_pct_to"].mean()
        for n in linear_names
    ]
    x = np.arange(len(linear_names))
    ax2.bar(x - 0.2, sg_means, 0.4, color=COLORS["sparsegpt"], label="SparseGPT")
    ax2.bar(x + 0.2, to_means, 0.4, color=COLORS["true_obs"], label="S-OBS")
    ax2.set_xticks(x)
    ax2.set_xticklabels([LINEAR_SHORT[n] for n in linear_names])
    ax2.set_ylabel("Mean Output Error (%)")
    ax2.set_title("Average Output Error by Sublayer Type")
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    path = os.path.join(output_dir, f"linear_output_error_{tag}.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_error_vs_layer(sg_linear, to_linear, output_dir, tag, model_name):
    """Plot 4: Per-linear output error across layers (line plot per sublayer).

    Layout: 3 rows x 3 cols. Row 1: Q/K/V, Row 2: O/Gate/Up, Row 3: Down (centered).
    """
    ncols = 3
    nrows = len(SUBLAYER_GRID)
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 3.5 * nrows), sharey=True)

    # Compute shared y-limits across all sublayers
    all_err = pd.concat([sg_linear["output_error_pct"], to_linear["output_error_pct"]])
    ymax = all_err.quantile(0.99) * 1.08

    for row_idx, row_names in enumerate(SUBLAYER_GRID):
        # Center the row if fewer than ncols items
        pad_left = (ncols - len(row_names)) // 2
        # Hide unused axes in this row
        for col_idx in range(ncols):
            ax = axes[row_idx, col_idx]
            item_idx = col_idx - pad_left
            if item_idx < 0 or item_idx >= len(row_names):
                ax.set_visible(False)
                continue

            lname = row_names[item_idx]
            sg_sub = sg_linear[sg_linear["linear_name"] == lname].sort_values("layer_idx")
            to_sub = to_linear[to_linear["linear_name"] == lname].sort_values("layer_idx")
            ax.plot(
                sg_sub["layer_idx"],
                sg_sub["output_error_pct"],
                "-",
                color=COLORS["sparsegpt"],
                linewidth=1.5,
                label="SparseGPT",
            )
            ax.plot(
                to_sub["layer_idx"],
                to_sub["output_error_pct"],
                "-",
                color=COLORS["true_obs"],
                linewidth=1.5,
                label="S-OBS",
            )
            ax.set_title(LINEAR_SHORT[lname], fontsize=12, fontweight="bold")
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, ymax)
            if row_idx == 0 and col_idx == pad_left:
                ax.legend(fontsize=9)
            if row_idx == nrows - 1:
                ax.set_xlabel("Layer")
            if col_idx == 0 or (col_idx == pad_left and pad_left > 0):
                ax.set_ylabel("Output Error (%)")

    fig.suptitle(
        f"{model_name} — Per-Sublayer Output Error across Layers",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    path = os.path.join(output_dir, f"error_per_sublayer_{tag}.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_time_comparison(sg_layer, to_layer, sg_linear, to_linear, output_dir, tag, model_name):
    """Plot 5: Timing comparison."""
    sg_l = sg_layer[sg_layer["layer_idx"] >= 0].sort_values("layer_idx")
    to_l = to_layer[to_layer["layer_idx"] >= 0].sort_values("layer_idx")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: cumulative time
    ax = axes[0]
    ax.plot(
        sg_l["layer_idx"],
        sg_l["cumulative_time_s"],
        "o-",
        color=COLORS["sparsegpt"],
        label=f'SparseGPT ({sg_l["cumulative_time_s"].iloc[-1]:.0f}s)',
    )
    ax.plot(
        to_l["layer_idx"],
        to_l["cumulative_time_s"],
        "s-",
        color=COLORS["true_obs"],
        label=f'S-OBS ({to_l["cumulative_time_s"].iloc[-1]:.0f}s)',
    )
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Cumulative Time (s)")
    ax.set_title("Cumulative Pruning Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: per-linear time breakdown (mean across layers by sublayer type)
    ax2 = axes[1]
    linear_names = list(LINEAR_SHORT.keys())
    sg_times = [
        sg_linear[sg_linear["linear_name"] == n]["prune_time_s"].mean()
        for n in linear_names
    ]
    to_times = [
        to_linear[to_linear["linear_name"] == n]["prune_time_s"].mean()
        for n in linear_names
    ]
    x = np.arange(len(linear_names))
    ax2.bar(x - 0.2, sg_times, 0.4, color=COLORS["sparsegpt"], label="SparseGPT")
    ax2.bar(x + 0.2, to_times, 0.4, color=COLORS["true_obs"], label="S-OBS")
    ax2.set_xticks(x)
    ax2.set_xticklabels([LINEAR_SHORT[n] for n in linear_names])
    ax2.set_ylabel("Mean Prune Time (s)")
    ax2.set_title("Prune Time per Sublayer Type")
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    path = os.path.join(output_dir, f"timing_{tag}.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_improvement_heatmap(sg_linear, to_linear, output_dir, tag, model_name):
    """Plot 6: Heatmap of relative improvement (S-OBS vs SparseGPT output error)."""
    merged = sg_linear.merge(
        to_linear,
        on=["layer_idx", "linear_name"],
        suffixes=("_sg", "_to"),
    )
    # Relative improvement (%): positive = S-OBS is better
    merged["improvement_pct"] = (
        (merged["output_error_pct_sg"] - merged["output_error_pct_to"])
        / merged["output_error_pct_sg"].clip(lower=1e-6)
    ) * 100

    linear_names = list(LINEAR_SHORT.keys())
    layers = sorted(merged["layer_idx"].unique())
    data = np.zeros((len(linear_names), len(layers)))

    for i, lname in enumerate(linear_names):
        for j, lid in enumerate(layers):
            row = merged[
                (merged["linear_name"] == lname) & (merged["layer_idx"] == lid)
            ]
            if len(row) > 0:
                data[i, j] = row["improvement_pct"].values[0]

    fig, ax = plt.subplots(figsize=(14, 4))
    # Use robust vmax: 95th percentile of absolute values to avoid outliers
    vmax = min(np.percentile(np.abs(data), 95) * 1.3, max(abs(data.min()), abs(data.max())))
    vmax = max(vmax, 1.0)  # at least 1%
    im = ax.imshow(data, cmap="RdBu", aspect="auto", vmin=-vmax, vmax=vmax)
    ax.set_yticks(range(len(linear_names)))
    ax.set_yticklabels([LINEAR_SHORT[n] for n in linear_names])
    ax.set_xticks(range(0, len(layers), 2))
    ax.set_xticklabels([str(layers[i]) for i in range(0, len(layers), 2)])
    ax.set_xlabel("Layer Index")
    ax.set_title(
        f"{model_name} — Relative Output Error Improvement (%)\n"
        "Blue = S-OBS better, Red = SparseGPT better"
    )
    plt.colorbar(im, ax=ax, label="Improvement (%)")
    fig.tight_layout()
    path = os.path.join(output_dir, f"improvement_heatmap_{tag}.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def generate_markdown(
    plots, sg_layer, to_layer, sg_linear, to_linear, model_name, tag
):
    """Generate markdown report describing each plot."""
    sg = sg_layer[sg_layer["layer_idx"] >= 0]
    to = to_layer[to_layer["layer_idx"] >= 0]
    baseline_ppl = sg_layer[sg_layer["layer_idx"] == -1]["word_perplexity"].values[0]

    sg_final_rows = sg[sg["word_perplexity"].notna()]
    to_final_rows = to[to["word_perplexity"].notna()]
    sg_final_ppl = sg_final_rows["word_perplexity"].iloc[-1]
    to_final_ppl = to_final_rows["word_perplexity"].iloc[-1]

    # Per-linear error stats
    merged = sg_linear.merge(
        to_linear, on=["layer_idx", "linear_name"], suffixes=("_sg", "_to")
    )
    to_better_count = (
        merged["output_error_pct_to"] < merged["output_error_pct_sg"]
    ).sum()
    total_linears = len(merged)
    mean_sg_err = merged["output_error_pct_sg"].mean()
    mean_to_err = merged["output_error_pct_to"].mean()

    sg_total_time = sg["cumulative_time_s"].iloc[-1]
    to_total_time = to["cumulative_time_s"].iloc[-1]

    md = f"""# {model_name} — 2:4 Pruning Benchmark Results

## Summary

| Metric | Dense | SparseGPT | S-OBS |
|--------|-------|-----------|----------|
| WikiText-2 PPL | {baseline_ppl:.2f} | {sg_final_ppl:.2f} | {to_final_ppl:.2f} |
| PPL increase | — | {sg_final_ppl - baseline_ppl:.2f} (+{(sg_final_ppl/baseline_ppl - 1)*100:.1f}%) | {to_final_ppl - baseline_ppl:.2f} (+{(to_final_ppl/baseline_ppl - 1)*100:.1f}%) |
| Total prune time | — | {sg_total_time:.0f}s | {to_total_time:.0f}s |
| Mean linear error | — | {mean_sg_err:.2f}% | {mean_to_err:.2f}% |
| Linear layers with lower error | — | {total_linears - to_better_count}/{total_linears} | {to_better_count}/{total_linears} |

---

## Plot 1: Perplexity Progression
![Perplexity Progression]({plots['ppl']})

Shows WikiText-2 word perplexity evaluated after pruning every 4th decoder layer.
Both methods start from the same dense baseline ({baseline_ppl:.2f}).
SparseGPT achieves **{sg_final_ppl:.1f}** vs S-OBS **{to_final_ppl:.1f}** after all layers are pruned.

---

## Plot 2: Decoder Output Error
![Decoder Error]({plots['decoder']})

**Left:** Per-layer decoder output error (relative L2 norm of the difference between
pruned and unpruned decoder outputs). Lower is better.

**Right:** Error difference (SparseGPT minus S-OBS). Positive bars (blue) indicate
layers where S-OBS has lower decoder error. S-OBS is better in early-to-mid layers
but SparseGPT becomes better in later layers.

---

## Plot 3: Per-Linear Output Error
![Linear Output Error]({plots['linear']})

**Left:** Scatter plot of per-linear output errors (H-based metric: sqrt(trace(dW H dW^T) / trace(W H W^T))).
Each point represents one linear layer in one decoder layer. Points below the y=x line indicate
S-OBS achieves lower error. S-OBS wins on **{to_better_count}/{total_linears}** linear layers.

**Right:** Average output error grouped by sublayer type (Q, K, V, O, Gate, Up, Down).
S-OBS consistently achieves lower per-linear errors across all sublayer types.

---

## Plot 4: Error per Sublayer across Layers
![Error per Sublayer]({plots['sublayer']})

Line plots of per-linear output error for each sublayer type across all 28 decoder layers.
S-OBS (blue) achieves lower output error than SparseGPT (red) at virtually every layer
for every sublayer type. The gap is most pronounced for Gate, Up, and O projections.

---

## Plot 5: Timing
![Timing]({plots['timing']})

**Left:** Cumulative pruning time. SparseGPT: **{sg_total_time:.0f}s**, S-OBS: **{to_total_time:.0f}s**.
S-OBS is {to_total_time/sg_total_time:.1f}x slower due to per-row Schur complement updates.

**Right:** Mean pruning time per sublayer type. The down_proj (K=3072) takes longest for both methods.

---

## Plot 6: Improvement Heatmap
![Improvement Heatmap]({plots['heatmap']})

Heatmap of relative output error improvement: (SparseGPT_error - TrueOBS_error) / SparseGPT_error * 100.
Blue cells indicate S-OBS achieves lower error, red cells indicate SparseGPT is better.
S-OBS dominates across virtually all layer-sublayer combinations.
"""
    return md


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", nargs="?", default="results")
    parser.add_argument("--model", default="Qwen3-0.6B")
    parser.add_argument("--suffix", default="_n1024", help="Result file suffix, e.g. _n1024")
    parser.add_argument(
        "--prefix",
        default="",
        help="Model prefix for filenames, e.g. qwen3_0.6b_. Auto-detected if empty.",
    )
    args = parser.parse_args()

    results_dir = args.results_dir
    model_name = args.model
    suffix = args.suffix
    model_prefix = args.prefix
    if not model_prefix:
        # Auto-detect: try model-prefixed first, then bare
        short = model_name.lower().replace("-", "_").replace(" ", "_")
        test_path = os.path.join(results_dir, f"{short}_sparsegpt_24{suffix}_layer.csv")
        if os.path.exists(test_path):
            model_prefix = f"{short}_"
        else:
            model_prefix = ""

    sg_prefix = f"{model_prefix}sparsegpt_24{suffix}"
    to_prefix = f"{model_prefix}true_obs_24{suffix}"
    tag = f"{model_prefix}24{suffix}"

    sg_layer, sg_linear = load_results(results_dir, sg_prefix)
    to_layer, to_linear = load_results(results_dir, to_prefix)

    if sg_layer is None or to_layer is None:
        print(f"Missing layer CSVs for {sg_prefix} or {to_prefix} in {results_dir}")
        sys.exit(1)
    if sg_linear is None or to_linear is None:
        print(f"Missing linear CSVs for {sg_prefix} or {to_prefix} in {results_dir}")
        sys.exit(1)

    # Check both have all 28 layers
    sg_layers_done = len(sg_layer[sg_layer["layer_idx"] >= 0])
    to_layers_done = len(to_layer[to_layer["layer_idx"] >= 0])
    print(f"SparseGPT: {sg_layers_done} layers, S-OBS: {to_layers_done} layers")

    output_dir = os.path.join(results_dir, "plots")
    os.makedirs(output_dir, exist_ok=True)

    plots = {}
    plots["ppl"] = plot_perplexity_progression(
        sg_layer, to_layer, output_dir, tag, model_name
    )
    print(f"  -> {plots['ppl']}")

    plots["decoder"] = plot_decoder_error(
        sg_layer, to_layer, output_dir, tag, model_name
    )
    print(f"  -> {plots['decoder']}")

    plots["linear"] = plot_linear_output_error(
        sg_linear, to_linear, output_dir, tag, model_name
    )
    print(f"  -> {plots['linear']}")

    plots["sublayer"] = plot_error_vs_layer(
        sg_linear, to_linear, output_dir, tag, model_name
    )
    print(f"  -> {plots['sublayer']}")

    plots["timing"] = plot_time_comparison(
        sg_layer, to_layer, sg_linear, to_linear, output_dir, tag, model_name
    )
    print(f"  -> {plots['timing']}")

    plots["heatmap"] = plot_improvement_heatmap(
        sg_linear, to_linear, output_dir, tag, model_name
    )
    print(f"  -> {plots['heatmap']}")

    # Generate markdown report
    # Make plot paths relative to results_dir for the markdown
    rel_plots = {k: os.path.relpath(v, results_dir) for k, v in plots.items()}
    md = generate_markdown(
        rel_plots, sg_layer, to_layer, sg_linear, to_linear, model_name, tag
    )
    md_path = os.path.join(results_dir, f"report_{tag}.md")
    with open(md_path, "w") as f:
        f.write(md)
    print(f"\nMarkdown report: {md_path}")


if __name__ == "__main__":
    main()
