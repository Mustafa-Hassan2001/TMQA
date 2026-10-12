import csv

import numpy as np
import pytest

from tmqa import evaluation
from tmqa.cli import main
from tmqa.dataset import load_dataset
from tmqa.hdr_io import read_hdr, read_ldr, write_hdr, write_ldr
from tmqa.synthetic import make_scene
from tmqa.tonemap import OPERATORS, bilateral_filter, tonemap


@pytest.fixture(scope="module")
def scene():
    return make_scene(192, 224, seed=5)


def test_eight_operators():
    assert len(OPERATORS) == 8


@pytest.mark.parametrize("op", list(OPERATORS))
def test_operator_output_valid(scene, op):
    out = tonemap(scene, op)
    assert out.shape == scene.shape
    assert np.isfinite(out).all()
    assert out.min() >= 0 and out.max() <= 255
    assert out.std() > 1  # not a constant image


def test_operators_are_monotone_in_luminance():
    ramp = np.tile(np.logspace(-2, 4, 256), (8, 1))
    for op in ["linear", "logarithmic", "reinhard_global", "drago", "ward"]:
        out = tonemap(ramp, op)
        assert (np.diff(out[0]) >= -1e-9).all(), op


def test_unknown_operator(scene):
    with pytest.raises(KeyError):
        tonemap(scene, "nope")


def test_bilateral_preserves_edges():
    img = np.zeros((40, 40))
    img[:, 20:] = 5.0
    out = bilateral_filter(img, sigma_s=4, sigma_r=0.4)
    assert out[:, :18].max() < 0.1 and out[:, 22:].min() > 4.9


def test_rgbe_roundtrip(tmp_path, scene):
    write_hdr(tmp_path / "x.hdr", scene)
    back = read_hdr(tmp_path / "x.hdr")
    assert back.shape == scene.shape
    assert np.allclose(back, scene, rtol=1.5e-2, atol=1e-3)


def test_rgbe_rle_decoding(tmp_path):
    # Hand-build a 1x8 RLE scanline: run of 8 for R,G,B and E.
    header = b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 1 +X 8\n"
    line = bytes([2, 2, 0, 8]) + bytes([136, 128]) * 3 + bytes([136, 129])
    (tmp_path / "r.hdr").write_bytes(header + line)
    img = read_hdr(tmp_path / "r.hdr")
    assert img.shape == (1, 8, 3)
    assert np.allclose(img, (128.5 / 256) * 2.0)


def test_pfm_and_ldr_roundtrip(tmp_path, scene):
    write_hdr(tmp_path / "x.pfm", scene)
    assert np.allclose(read_hdr(tmp_path / "x.pfm"), scene, rtol=1e-6)
    ldr = tonemap(scene, "drago")
    write_ldr(tmp_path / "x.png", ldr)
    assert np.abs(read_ldr(tmp_path / "x.png") - ldr).max() <= 0.5 + 1e-9


def _make_dataset(root, scene, operators):
    """Folder-layout dataset with ranks = order of TMQI (perfect agreement)."""
    from tmqa.tmqi import tmqi

    sc = root / "scene_a"
    sc.mkdir(parents=True)
    write_hdr(sc / "scene_a.hdr", scene)
    hdr = read_hdr(sc / "scene_a.hdr")
    qs = {}
    for op in operators:
        write_ldr(sc / f"{op}.png", tonemap(hdr, op))
        qs[op] = tmqi(hdr, read_ldr(sc / f"{op}.png")).Q
    ranked = sorted(qs, key=qs.get, reverse=True)
    with open(root / "ranks.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["scene", "image", "rank"])
        for r, op in enumerate(ranked, 1):
            w.writerow(["scene_a", f"{op}.png", r])


def test_dataset_validation_pipeline(tmp_path, scene):
    ops = ["linear", "drago", "reinhard_global", "fattal"]
    _make_dataset(tmp_path / "db", scene, ops)
    scenes = load_dataset(tmp_path / "db")
    assert len(scenes) == 1 and len(scenes[0].ldr_paths) == 4
    assert len(scenes[0].subjective) == 4
    scores = evaluation.score_dataset(scenes)
    per_scene, mean = evaluation.correlation_with_subjective(scores)
    assert np.isclose(per_scene["KRCC_Q"].iloc[0], 1.0)
    assert np.isclose(mean.loc["mean", "SRCC_Q"], 1.0)


def test_benchmark_and_report(tmp_path, scene):
    df = evaluation.benchmark_operators({"s": scene}, ["linear", "drago", "fattal"])
    assert len(df) == 3 and sorted(df["rank"]) == [1, 2, 3]
    report = evaluation.write_benchmark_report(df, tmp_path, "t")
    assert report.exists() and (tmp_path / "summary.csv").exists()


def test_cli_end_to_end(tmp_path, scene, capsys):
    _make_dataset(tmp_path / "db", scene, ["linear", "drago", "ward"])
    assert main(["validate", str(tmp_path / "db"), "-o", str(tmp_path / "val")]) == 0
    assert (tmp_path / "val" / "report.md").exists()
    hdr = tmp_path / "db" / "scene_a" / "scene_a.hdr"
    assert main(["score", str(hdr), str(tmp_path / "db" / "scene_a" / "drago.png")]) == 0
    assert "Q=" in capsys.readouterr().out
    assert main(["tonemap", str(hdr), "-o", str(tmp_path / "tm"), "--operators", "drago"]) == 0
    assert (tmp_path / "tm" / "scene_a_drago.png").exists()
