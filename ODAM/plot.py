#!/usr/bin/env python3
"""
plot.py
=======

Tạo các hình phục vụ luận điểm:
    "ODAM scalarization có gradient conflict / dominance;
     DPGA giảm ảnh hưởng xấu bằng projection + norm cap + gate."

Input:
    --baseline-dir runs/baseline        (optional)
    --odam-dir     runs/odam
    --dpga-dir     runs/dpga
    --output       figures

Mỗi run directory có:
    metrics.csv
    gradient_diagnostics_rank0.csv
    gradient_diagnostics_rank1.csv
    ...

Output:
    gradient_summary.csv
    01_unsafe_descent_rate.png
    02_final_gradient_alignment.png
    03_effective_aux_to_det_ratio.png
    04_odam_module_conflict_heatmap.png
    05_dpga_projection_rate.png
    06_dpga_cap_activation_rate.png
    07_ap_curve.png
    08_mr2_curve.png       (nếu có)

Dependencies:
    pip install pandas matplotlib numpy
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


METHOD_LABELS = {
    "baseline": "Faster R-CNN",
    "odam": "Faster R-CNN + ODAM",
    "dpga": "Faster R-CNN + DPGA-ODAM",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline-dir", type=str, default=None)
    p.add_argument("--odam-dir", type=str, required=True)
    p.add_argument("--dpga-dir", type=str, required=True)
    p.add_argument("--output", type=str, required=True)
    return p.parse_args()


def load_gradient_run(run_dir: Path) -> pd.DataFrame:
    files = sorted(
        run_dir.glob("gradient_diagnostics_rank*.csv")
    )

    if not files:
        raise FileNotFoundError(
            f"No gradient_diagnostics_rank*.csv in {run_dir}"
        )

    dfs = [
        pd.read_csv(path)
        for path in files
    ]
    df = pd.concat(
        dfs,
        ignore_index=True,
    )

    numeric_cols = [
        "epoch",
        "step",
        "rank",
        "loss_det",
        "loss_odam",
        "cosine_raw",
        "det_norm",
        "odam_norm_raw",
        "odam_norm_safe",
        "aux_to_det_raw",
        "aux_to_det_effective",
        "directional_margin",
        "final_cosine_to_det",
        "final_angle_deg",
        "conflict_raw",
        "dominance_raw",
        "dominance_effective",
        "unsafe_descent",
        "projected",
        "cap_active",
        "norm_scale",
        "gate",
        "alpha",
        "effective_weight",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce",
            )

    return df


def load_metrics(run_dir: Optional[Path], method: str) -> Optional[pd.DataFrame]:
    if run_dir is None:
        return None

    path = run_dir / "metrics.csv"
    if not path.exists():
        return None

    df = pd.read_csv(path)
    df["method"] = method
    return df


def save_gradient_summary(
    frames: Dict[str, pd.DataFrame],
    output: Path,
):
    rows = []

    for method, df in frames.items():
        grouped = df.groupby(
            "module",
            dropna=False,
        )

        for module, g in grouped:
            rows.append(
                {
                    "method": method,
                    "module": module,
                    "samples": len(g),
                    "raw_conflict_rate": g["conflict_raw"].mean(),
                    "raw_dominance_rate": g["dominance_raw"].mean(),
                    "effective_dominance_rate": g[
                        "dominance_effective"
                    ].mean(),
                    "unsafe_descent_rate": g["unsafe_descent"].mean(),
                    "mean_cosine_raw": g["cosine_raw"].mean(),
                    "median_cosine_raw": g["cosine_raw"].median(),
                    "mean_aux_to_det_raw": g["aux_to_det_raw"].mean(),
                    "mean_aux_to_det_effective": g[
                        "aux_to_det_effective"
                    ].mean(),
                    "mean_directional_margin": g[
                        "directional_margin"
                    ].mean(),
                    "mean_final_cosine_to_det": g[
                        "final_cosine_to_det"
                    ].mean(),
                    "mean_final_angle_deg": g[
                        "final_angle_deg"
                    ].mean(),
                    "projection_rate": g["projected"].mean(),
                    "cap_activation_rate": g["cap_active"].mean(),
                    "mean_gate": g["gate"].mean(),
                    "mean_alpha": g["alpha"].mean(),
                }
            )

    pd.DataFrame(rows).to_csv(
        output / "gradient_summary.csv",
        index=False,
    )


def line_by_epoch(
    frames: Dict[str, pd.DataFrame],
    column: str,
    ylabel: str,
    title: str,
    output_path: Path,
    ylim=None,
):
    fig, ax = plt.subplots()

    for method, df in frames.items():
        series = (
            df.groupby("epoch")[column]
            .mean()
            .sort_index()
        )
        ax.plot(
            series.index,
            series.values,
            marker="o",
            label=METHOD_LABELS[method],
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend()

    if ylim is not None:
        ax.set_ylim(*ylim)

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)


def odam_conflict_heatmap(
    odam_df: pd.DataFrame,
    output_path: Path,
):
    pivot = (
        odam_df
        .pivot_table(
            index="module",
            columns="epoch",
            values="conflict_raw",
            aggfunc="mean",
        )
        .sort_index()
    )

    fig, ax = plt.subplots()
    im = ax.imshow(
        pivot.values,
        aspect="auto",
        vmin=0.0,
        vmax=1.0,
    )

    ax.set_yticks(
        np.arange(len(pivot.index))
    )
    ax.set_yticklabels(
        list(pivot.index)
    )

    ax.set_xticks(
        np.arange(len(pivot.columns))
    )
    ax.set_xticklabels(
        [str(int(x)) for x in pivot.columns]
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Module")
    ax.set_title(
        "ODAM raw gradient conflict rate by module"
    )

    cbar = fig.colorbar(
        im,
        ax=ax,
    )
    cbar.set_label(
        "P(cos(g_det, g_odam) < 0)"
    )

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)


def dpga_module_bar(
    dpga_df: pd.DataFrame,
    column: str,
    ylabel: str,
    title: str,
    output_path: Path,
):
    values = (
        dpga_df.groupby("module")[column]
        .mean()
        .sort_values(ascending=False)
    )

    fig, ax = plt.subplots()

    ax.bar(
        values.index,
        values.values,
    )

    ax.set_xlabel("Module")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.tick_params(
        axis="x",
        rotation=30,
    )
    ax.grid(
        True,
        axis="y",
        alpha=0.25,
    )

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_metric_curve(
    metrics: List[pd.DataFrame],
    column: str,
    ylabel: str,
    title: str,
    output_path: Path,
):
    usable = [
        df
        for df in metrics
        if df is not None and column in df.columns
    ]

    if not usable:
        return

    fig, ax = plt.subplots()
    plotted = False

    for df in usable:
        method = str(df["method"].iloc[0])

        y = pd.to_numeric(
            df[column],
            errors="coerce",
        )
        x = pd.to_numeric(
            df["epoch"],
            errors="coerce",
        )

        valid = y.notna() & x.notna()
        if not valid.any():
            continue

        ax.plot(
            x[valid],
            y[valid],
            marker="o",
            label=METHOD_LABELS.get(method, method),
        )
        plotted = True

    if not plotted:
        plt.close(fig)
        return

    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)


def main():
    args = parse_args()

    output = Path(args.output)
    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    odam_dir = Path(args.odam_dir)
    dpga_dir = Path(args.dpga_dir)

    grad_frames = {
        "odam": load_gradient_run(odam_dir),
        "dpga": load_gradient_run(dpga_dir),
    }

    save_gradient_summary(
        grad_frames,
        output,
    )

    # ------------------------------------------------------------------
    # Figure 1:
    # How often the final update is no longer a first-order descent
    # direction for L_det.
    # ------------------------------------------------------------------
    line_by_epoch(
        grad_frames,
        column="unsafe_descent",
        ylabel="Unsafe update rate",
        title=(
            "First-order detector-descent violations"
        ),
        output_path=output / "01_unsafe_descent_rate.png",
        ylim=(0, 1),
    )

    # ------------------------------------------------------------------
    # Figure 2:
    # Alignment of final gradient with detection gradient.
    # Higher is safer / more detection-priority.
    # ------------------------------------------------------------------
    line_by_epoch(
        grad_frames,
        column="final_cosine_to_det",
        ylabel="cos(g_final, g_det)",
        title=(
            "Alignment of the final update with detection gradient"
        ),
        output_path=output / "02_final_gradient_alignment.png",
        ylim=(-1, 1),
    )

    # ------------------------------------------------------------------
    # Figure 3:
    # Effective auxiliary magnitude relative to detector gradient.
    # DPGA should be bounded by module rho/gate/alpha.
    # ------------------------------------------------------------------
    line_by_epoch(
        grad_frames,
        column="aux_to_det_effective",
        ylabel="Effective ||g_aux|| / ||g_det||",
        title=(
            "Effective auxiliary-to-detection gradient ratio"
        ),
        output_path=output / "03_effective_aux_to_det_ratio.png",
    )

    # ------------------------------------------------------------------
    # Figure 4:
    # Proves one global lambda is coarse because conflicts differ by module.
    # ------------------------------------------------------------------
    odam_conflict_heatmap(
        grad_frames["odam"],
        output / "04_odam_module_conflict_heatmap.png",
    )

    # ------------------------------------------------------------------
    # Figure 5-6:
    # Shows DPGA actually intervenes where needed.
    # ------------------------------------------------------------------
    dpga_module_bar(
        grad_frames["dpga"],
        column="projected",
        ylabel="Projection activation rate",
        title="DPGA conflict-projection activation by module",
        output_path=output / "05_dpga_projection_rate.png",
    )

    dpga_module_bar(
        grad_frames["dpga"],
        column="cap_active",
        ylabel="Norm-cap activation rate",
        title="DPGA norm-cap activation by module",
        output_path=output / "06_dpga_cap_activation_rate.png",
    )

    # ------------------------------------------------------------------
    # Detection performance curves.
    # ------------------------------------------------------------------
    baseline_dir = (
        Path(args.baseline_dir)
        if args.baseline_dir
        else None
    )

    metrics = [
        load_metrics(
            baseline_dir,
            "baseline",
        ),
        load_metrics(
            odam_dir,
            "odam",
        ),
        load_metrics(
            dpga_dir,
            "dpga",
        ),
    ]

    plot_metric_curve(
        metrics,
        column="AP",
        ylabel="COCO AP",
        title="Detection performance across training",
        output_path=output / "07_ap_curve.png",
    )

    plot_metric_curve(
        metrics,
        column="MR-2_generic",
        ylabel="MR-2 (lower is better)",
        title="Log-average miss rate across training",
        output_path=output / "08_mr2_curve.png",
    )

    print(f"Saved figures to: {output}")
    print(f"Summary: {output / 'gradient_summary.csv'}")


if __name__ == "__main__":
    main()
