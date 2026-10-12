"""Command-line interface: ``python -m tmqa <command> ...``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import evaluation
from .dataset import load_dataset
from .hdr_io import read_hdr, read_ldr, write_hdr, write_ldr
from .synthetic import make_scenes
from .tmqi import tmqi
from .tonemap import OPERATORS, tonemap


def _cmd_score(args: argparse.Namespace) -> int:
    hdr = read_hdr(args.hdr)
    results = {}
    for p in args.ldr:
        res = tmqi(hdr, read_ldr(p))
        results[str(p)] = res.as_dict()
        if not args.json:
            print(f"{p}: Q={res.Q:.4f}  S={res.S:.4f}  N={res.N:.4f}")
    if args.json:
        print(json.dumps(results, indent=2))
    return 0


def _cmd_tonemap(args: argparse.Namespace) -> int:
    hdr = read_hdr(args.hdr)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for op in args.operators or list(OPERATORS):
        ldr = tonemap(hdr, op)
        path = out / f"{Path(args.hdr).stem}_{op}.png"
        write_ldr(path, ldr)
        res = tmqi(hdr, ldr) if args.score else None
        msg = f"wrote {path}"
        if res:
            msg += f"  Q={res.Q:.4f} S={res.S:.4f} N={res.N:.4f}"
        print(msg)
    return 0


def _cmd_benchmark(args: argparse.Namespace) -> int:
    if args.dataset:
        scenes = load_dataset(args.dataset)
        if args.limit:
            scenes = scenes[: args.limit]
        source, title = scenes, f"TMO benchmark on {Path(args.dataset).name}"
    else:
        source = make_scenes(args.synthetic, seed=args.seed)
        title = "TMO benchmark on synthetic HDR scenes"
    print(f"Benchmarking {len(source)} scenes x {len(args.operators or OPERATORS)} operators")
    df = evaluation.benchmark_operators(
        source,
        args.operators,
        save_dir=Path(args.out) / "images" if args.save_images else None,
        verbose=True,
    )
    report = evaluation.write_benchmark_report(df, args.out, title)
    print()
    print(evaluation.summarize_operators(df).to_string(float_format=lambda v: f"{v:.4f}"))
    print(f"\nreport: {report}")
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    scenes = load_dataset(args.dataset, args.subjective)
    scores = evaluation.score_dataset(scenes, verbose=True)
    if scores.empty:
        print("no tone-mapped images found", file=sys.stderr)
        return 1
    per_scene, mean = evaluation.correlation_with_subjective(scores, args.higher_is_better)
    report = evaluation.write_validation_report(scores, per_scene, mean, args.out)
    print()
    print(mean.to_string(float_format=lambda v: f"{v:.4f}"))
    print(f"\nreport: {report}")
    return 0


def _cmd_synth(args: argparse.Namespace) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, hdr in make_scenes(args.n, args.height, args.width, args.seed).items():
        write_hdr(out / f"{name}.hdr", hdr)
        print(f"wrote {out / name}.hdr  (dynamic range {hdr.max() / hdr.min():.3g}:1)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tmqa", description="Quality assessment of tone-mapped HDR images (TMQI)."
    )
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("score", help="TMQI of tone-mapped image(s) against an HDR reference")
    s.add_argument("hdr")
    s.add_argument("ldr", nargs="+")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=_cmd_score)

    s = sub.add_parser("tonemap", help="apply tone-mapping operators to an HDR image")
    s.add_argument("hdr")
    s.add_argument("-o", "--out", default="tonemapped")
    s.add_argument("--operators", nargs="+", choices=list(OPERATORS))
    s.add_argument("--score", action="store_true", help="also print TMQI")
    s.set_defaults(func=_cmd_tonemap)

    s = sub.add_parser("benchmark", help="rank the eight TMOs with TMQI")
    g = s.add_mutually_exclusive_group()
    g.add_argument("--dataset", help="dataset directory or manifest CSV (e.g. TMQI database)")
    g.add_argument("--synthetic", type=int, default=4, help="number of synthetic scenes")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--limit", type=int, help="use only the first N dataset scenes")
    s.add_argument("--operators", nargs="+", choices=list(OPERATORS))
    s.add_argument("-o", "--out", default="results/benchmark")
    s.add_argument("--save-images", action="store_true")
    s.set_defaults(func=_cmd_benchmark)

    s = sub.add_parser("validate", help="correlate TMQI with subjective rankings")
    s.add_argument("dataset", help="dataset directory or manifest CSV")
    s.add_argument("--subjective", help="CSV with scene,image,subjective columns")
    s.add_argument(
        "--higher-is-better", action="store_true", help="subjective scores are MOS, not ranks"
    )
    s.add_argument("-o", "--out", default="results/validation")
    s.set_defaults(func=_cmd_validate)

    s = sub.add_parser("synth", help="write synthetic HDR scenes to disk")
    s.add_argument("-n", type=int, default=4)
    s.add_argument("--height", type=int, default=384)
    s.add_argument("--width", type=int, default=512)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("-o", "--out", default="data/synthetic")
    s.set_defaults(func=_cmd_synth)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
