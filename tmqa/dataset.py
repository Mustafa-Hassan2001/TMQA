"""Loading the TMQI database (or any HDR / tone-mapped collection).

The TMQI database (Yeganeh & Wang, 2013) contains 15 HDR scenes, each
tone-mapped by 8 operators, with subjective rankings (1 = best ... 8 = worst)
from 20 observers. It is distributed by the authors and is **not** included
in this repository. Two layouts are supported:

1. **Folder layout** (auto-discovered)::

       root/
         <scene>/
           <scene>.hdr          # exactly one HDR reference (.hdr/.pfm/.exr)
           <name>.png ...       # tone-mapped images (any LDR format)
         ...

   HDR files placed directly in ``root`` are also accepted (``root/<scene>.hdr``
   with LDR images in ``root/<scene>/``).

2. **Manifest CSV** with columns ``scene,hdr,ldr[,subjective]`` (paths
   relative to the CSV file).

Subjective scores can be supplied in a CSV (``scene,image,subjective``)
either as the ``subjective`` column of the manifest or via a separate file
(auto-detected names: ``subjective.csv``, ``ranks.csv``, ``scores.csv``).
By default lower values are better (ranks); pass ``higher_is_better=True``
for MOS-style scores.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

from .hdr_io import is_hdr_file, is_ldr_file

SUBJECTIVE_FILENAMES = ("subjective.csv", "ranks.csv", "scores.csv")


@dataclass
class Scene:
    name: str
    hdr_path: Path
    ldr_paths: list[Path] = field(default_factory=list)
    subjective: dict[str, float] = field(default_factory=dict)  # keyed by LDR stem


def _read_subjective(path: Path) -> dict[tuple[str, str], float]:
    out: dict[tuple[str, str], float] = {}
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        cols = {c.lower(): c for c in reader.fieldnames or []}
        score_col = cols.get("subjective") or cols.get("rank") or cols.get("score") or cols.get("mos")
        if not score_col or "scene" not in cols or "image" not in cols:
            raise ValueError(f"{path}: need columns scene,image,subjective|rank|score|mos")
        for row in reader:
            out[(row[cols["scene"]], Path(row[cols["image"]]).stem)] = float(row[score_col])
    return out


def _from_manifest(path: Path) -> list[Scene]:
    scenes: dict[str, Scene] = {}
    base = path.parent
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            name = row["scene"]
            sc = scenes.setdefault(name, Scene(name, base / row["hdr"]))
            if row.get("ldr"):
                ldr = base / row["ldr"]
                sc.ldr_paths.append(ldr)
                if row.get("subjective"):
                    sc.subjective[ldr.stem] = float(row["subjective"])
    return list(scenes.values())


def _discover(root: Path) -> list[Scene]:
    scenes: list[Scene] = []
    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        hdrs = sorted(p for p in sub.iterdir() if is_hdr_file(p))
        if not hdrs:
            hdrs = sorted(p for p in root.iterdir() if is_hdr_file(p) and p.stem == sub.name)
        if not hdrs:
            continue
        ldrs = sorted(p for p in sub.iterdir() if is_ldr_file(p))
        scenes.append(Scene(sub.name, hdrs[0], ldrs))
    # Lone HDR files at root level (no tone-mapped set): useful for benchmarking.
    known = {s.name for s in scenes}
    for p in sorted(root.iterdir()):
        if is_hdr_file(p) and p.stem not in known:
            scenes.append(Scene(p.stem, p, []))
    return scenes


def load_dataset(
    root: str | Path, subjective_csv: str | Path | None = None
) -> list[Scene]:
    """Load a dataset from a directory or a manifest CSV."""
    root = Path(root)
    if root.is_file() and root.suffix.lower() == ".csv":
        scenes = _from_manifest(root)
        search_dir = root.parent
    elif root.is_dir():
        scenes = _discover(root)
        search_dir = root
    else:
        raise FileNotFoundError(root)

    if subjective_csv is None:
        for name in SUBJECTIVE_FILENAMES:
            if (search_dir / name).exists():
                subjective_csv = search_dir / name
                break
    if subjective_csv is not None:
        table = _read_subjective(Path(subjective_csv))
        for sc in scenes:
            for ldr in sc.ldr_paths:
                if (sc.name, ldr.stem) in table:
                    sc.subjective[ldr.stem] = table[(sc.name, ldr.stem)]
    if not scenes:
        raise ValueError(f"no HDR scenes found under {root}")
    return scenes
