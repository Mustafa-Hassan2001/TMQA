"""Benchmarking tone-mapping operators with TMQI and validating TMQI
against subjective rankings."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

from .dataset import Scene
from .hdr_io import read_hdr, read_ldr, write_ldr
from .tmqi import tmqi
from .tonemap import OPERATORS, tonemap

SCORE_COLUMNS = ["Q", "S", "N"]


# --------------------------------------------------------------------------
# Benchmark: apply our eight TMOs to HDR scenes and score them
# --------------------------------------------------------------------------
def benchmark_operators(
    hdr_images: Mapping[str, np.ndarray] | Iterable[Scene],
    operators: list[str] | None = None,
    save_dir: str | Path | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """Tone-map every HDR image with every operator and compute TMQI.

    ``hdr_images`` is either a ``{name: hdr_array}`` mapping or an iterable of
    :class:`~tmqa.dataset.Scene` objects (HDR files are read lazily).
    Returns one row per (scene, operator) with Q, S, N, per-scale S and the
    within-scene rank of the operator (1 = highest Q).
    """
    operators = operators or list(OPERATORS)
    if isinstance(hdr_images, Mapping):
        items = list(hdr_images.items())
        loader = lambda v: v  # noqa: E731
    else:
        items = [(s.name, s.hdr_path) for s in hdr_images]
        loader = read_hdr
    if save_dir is not None:
        save_dir = Path(save_dir)

    rows = []
    for name, src in items:
        hdr = loader(src)
        for op in operators:
            ldr = tonemap(hdr, op)
            res = tmqi(hdr, ldr)
            rows.append({"scene": name, "operator": op, **res.as_dict()})
            if save_dir is not None:
                (save_dir / name).mkdir(parents=True, exist_ok=True)
                write_ldr(save_dir / name / f"{op}.png", ldr)
            if verbose:
                print(f"  {name:<24} {op:<16} Q={res.Q:.4f} S={res.S:.4f} N={res.N:.4f}")
    df = pd.DataFrame(rows)
    df["rank"] = df.groupby("scene")["Q"].rank(ascending=False, method="min").astype(int)
    return df


def summarize_operators(df: pd.DataFrame) -> pd.DataFrame:
    """Per-operator mean/std of scores, mean rank and number of first places."""
    g = df.groupby("operator")
    summary = pd.DataFrame(
        {
            "Q_mean": g["Q"].mean(),
            "Q_std": g["Q"].std(ddof=0),
            "S_mean": g["S"].mean(),
            "N_mean": g["N"].mean(),
            "mean_rank": g["rank"].mean(),
            "wins": g["rank"].apply(lambda r: int((r == 1).sum())),
        }
    )
    return summary.sort_values("Q_mean", ascending=False)


# --------------------------------------------------------------------------
# Validation: correlation of TMQI with subjective rankings
# --------------------------------------------------------------------------
def score_dataset(scenes: Iterable[Scene], verbose: bool = False) -> pd.DataFrame:
    """Compute TMQI for every existing tone-mapped image of every scene."""
    rows = []
    for sc in scenes:
        if not sc.ldr_paths:
            continue
        hdr = read_hdr(sc.hdr_path)
        for p in sc.ldr_paths:
            ldr = read_ldr(p)
            if ldr.shape[:2] != hdr.shape[:2]:
                raise ValueError(f"{p}: size {ldr.shape[:2]} != HDR {hdr.shape[:2]}")
            res = tmqi(hdr, ldr)
            rows.append(
                {
                    "scene": sc.name,
                    "image": p.stem,
                    "subjective": sc.subjective.get(p.stem, np.nan),
                    **res.as_dict(),
                }
            )
            if verbose:
                print(f"  {sc.name:<24} {p.stem:<16} Q={res.Q:.4f} S={res.S:.4f} N={res.N:.4f}")
    return pd.DataFrame(rows)


def correlation_with_subjective(
    df: pd.DataFrame, higher_is_better: bool = False, columns: list[str] = SCORE_COLUMNS
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-scene Kendall (KRCC) and Spearman (SRCC) rank correlations.

    Subjective ranks (lower = better) are negated so that a positive
    correlation always means agreement with human judgement, as reported in
    the TMQI paper. Returns ``(per_scene, mean_over_scenes)``.
    """
    df = df.dropna(subset=["subjective"])
    if df.empty:
        raise ValueError("no subjective scores available")
    sign = 1.0 if higher_is_better else -1.0
    rows = []
    for scene, g in df.groupby("scene"):
        if len(g) < 3:
            continue
        subj = sign * g["subjective"].to_numpy()
        row = {"scene": scene, "n": len(g)}
        for c in columns:
            row[f"KRCC_{c}"] = kendalltau(g[c], subj).statistic
            row[f"SRCC_{c}"] = spearmanr(g[c], subj).statistic
        rows.append(row)
    per_scene = pd.DataFrame(rows)
    mean = per_scene.drop(columns=["scene", "n"]).agg(["mean", "std"])
    return per_scene, mean


# --------------------------------------------------------------------------
# Reports
# --------------------------------------------------------------------------
def _md_table(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    cols = [df.index.name or ""] + list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for idx, row in df.iterrows():
        cells = [str(idx)]
        for v in row:
            cells.append(format(v, floatfmt) if isinstance(v, (float, np.floating)) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_benchmark_report(df: pd.DataFrame, out_dir: str | Path, title: str) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "scores.csv", index=False)
    summary = summarize_operators(df)
    summary.to_csv(out_dir / "summary.csv")
    pivot = df.pivot(index="scene", columns="operator", values="Q")[summary.index]

    md = [
        f"# {title}",
        "",
        f"{df['scene'].nunique()} HDR scenes x {df['operator'].nunique()} tone-mapping operators.",
        "",
        "## Operator ranking (by mean TMQI)",
        "",
        _md_table(summary),
        "",
        "## TMQI Q per scene",
        "",
        _md_table(pivot),
        "",
    ]
    if _plot_benchmark(df, summary, out_dir):
        md += ["## Figures", "", "![TMQI by operator](tmqi_by_operator.png)", ""]
    report = out_dir / "report.md"
    report.write_text("\n".join(md))
    return report


def write_validation_report(
    scores: pd.DataFrame, per_scene: pd.DataFrame, mean: pd.DataFrame, out_dir: str | Path
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scores.to_csv(out_dir / "scores.csv", index=False)
    per_scene.to_csv(out_dir / "correlation_per_scene.csv", index=False)
    mean.to_csv(out_dir / "correlation_mean.csv")
    md = [
        "# TMQI validation against subjective rankings",
        "",
        f"{scores['scene'].nunique()} scenes, {len(scores)} tone-mapped images.",
        "",
        "## Mean rank correlation over scenes",
        "",
        _md_table(mean.T),
        "",
        "## Per scene",
        "",
        _md_table(per_scene.set_index("scene")),
        "",
    ]
    report = out_dir / "report.md"
    report.write_text("\n".join(md))
    return report


def _plot_benchmark(df: pd.DataFrame, summary: pd.DataFrame, out_dir: Path) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    order = list(summary.index)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=False)
    for ax, col, label in zip(
        axes, ["Q", "S", "N"], ["TMQI (Q)", "Structural fidelity (S)", "Naturalness (N)"]
    ):
        data = [df.loc[df["operator"] == op, col].to_numpy() for op in order]
        ax.boxplot(data, showmeans=True)
        ax.set_xticks(range(1, len(order) + 1), order, rotation=40, ha="right")
        ax.set_title(label)
        ax.set_ylim(0, 1.02)
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "tmqi_by_operator.png", dpi=120)
    plt.close(fig)
    return True
