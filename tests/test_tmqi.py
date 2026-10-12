import numpy as np
import pytest
from scipy import ndimage

from tmqa.synthetic import make_scene
from tmqa.tmqi import (
    MEAN_MU,
    _downsample,
    _filter_valid,
    block_std_mean,
    gaussian_window,
    statistical_naturalness,
    structural_fidelity,
    tmqi,
)
from tmqa.tonemap import tonemap


@pytest.fixture(scope="module")
def scene():
    return make_scene(256, 320, seed=3)


def test_gaussian_window_matches_fspecial():
    w = gaussian_window(11, 1.5)
    assert w.shape == (11, 11)
    assert np.isclose(w.sum(), 1.0)
    assert np.allclose(w, w.T)
    assert w[5, 5] == w.max()


def test_filter_valid_matches_direct_sum():
    rng = np.random.default_rng(0)
    img = rng.random((20, 25))
    w = gaussian_window(5, 1.0)
    out = _filter_valid(img, w)
    assert out.shape == (16, 21)
    expected = sum(w[i, j] * img[i : i + 16, j : j + 21] for i in range(5) for j in range(5))
    assert np.allclose(out, expected)


def test_downsample_shape_and_mean():
    img = np.arange(35, dtype=float).reshape(5, 7)
    d = _downsample(img)
    assert d.shape == (3, 4)
    assert np.isclose(d[0, 0], img[:2, :2].mean())


def test_block_std_uses_sample_std_and_zero_padding():
    img = np.zeros((11, 22))
    img[:, 11:] = np.arange(121).reshape(11, 11)
    assert np.isclose(block_std_mean(img), np.arange(121).std(ddof=1) / 2)


def test_naturalness_peaks_at_natural_statistics():
    rng = np.random.default_rng(1)
    # Mean near 116 and moderate contrast: high naturalness.
    good = np.clip(MEAN_MU + 18 * rng.standard_normal((220, 220)), 0, 255)
    dark = np.clip(good - 90, 0, 255)
    flat = np.full((220, 220), MEAN_MU)
    n_good = statistical_naturalness(good)
    assert 0 < n_good <= 1
    assert statistical_naturalness(dark) < n_good
    assert statistical_naturalness(flat) < 1e-6


def test_identity_like_mapping_has_high_structural_fidelity():
    rng = np.random.default_rng(2)
    base = ndimage.gaussian_filter(rng.random((256, 256)), 2)
    hdr = 0.05 + base / base.max()  # low dynamic range "HDR"
    ldr = 255 * (hdr - hdr.min()) / (hdr.max() - hdr.min())
    S, s_local, maps = structural_fidelity(hdr * 2**32 / hdr.max(), ldr)
    assert len(s_local) == 5 and len(maps) == 5
    assert S > 0.95


def test_structural_fidelity_drops_with_blur(scene):
    ldr = tonemap(scene, "reinhard_global")
    blurred = ndimage.gaussian_filter(ldr, sigma=(3, 3, 0))
    assert tmqi(scene, blurred).S < tmqi(scene, ldr).S


def test_tmqi_range_and_formula(scene):
    res = tmqi(scene, tonemap(scene, "drago"))
    assert 0 <= res.S <= 1 and 0 <= res.N <= 1 and 0 <= res.Q <= 1
    assert np.isclose(res.Q, 0.8012 * res.S**0.3046 + 0.1988 * res.N**0.7088)
    assert set(res.as_dict()) == {"Q", "S", "N", "S1", "S2", "S3", "S4", "S5"}


def test_tmqi_accepts_grayscale(scene):
    ldr = tonemap(scene, "drago")
    gray_hdr = scene @ np.array([0.2126, 0.7152, 0.0722])
    gray_ldr = ldr @ np.array([0.2126, 0.7152, 0.0722])
    assert np.isclose(tmqi(gray_hdr, gray_ldr).Q, tmqi(scene, ldr).Q)


def test_tmqi_is_invariant_to_hdr_scaling(scene):
    ldr = tonemap(scene, "reinhard_global")
    assert np.isclose(tmqi(scene, ldr).Q, tmqi(scene * 37.5, ldr).Q, atol=1e-6)


def test_size_checks(scene):
    with pytest.raises(ValueError):
        tmqi(scene, np.zeros((10, 10, 3)))
    with pytest.raises(ValueError):
        tmqi(np.ones((100, 100)), np.ones((100, 100)))
