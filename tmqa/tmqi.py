"""Tone-Mapped image Quality Index (TMQI).

Implementation of

    H. Yeganeh and Z. Wang, "Objective Quality Assessment of Tone-Mapped
    Images," IEEE Transactions on Image Processing, 22(2):657-667, 2013.

TMQI compares an HDR reference with a tone-mapped LDR image using two terms:

* **Structural fidelity** ``S`` -- a multi-scale, SSIM-like comparison of local
  structure. Because HDR and LDR signals live on different intensity scales,
  the luminance term of SSIM is dropped and local standard deviations are
  passed through a psychometric (Gaussian CDF) non-linearity whose threshold
  is derived from a contrast sensitivity function (CSF) at each scale.
* **Statistical naturalness** ``N`` -- the likelihood of the LDR image's global
  brightness (mean) and contrast (mean of block standard deviations) under
  Gaussian / Beta densities fitted to ~3000 natural images.

The two are combined as ``Q = a * S**alpha + (1 - a) * N**beta``.

The numerical details (filters, constants, block processing, HDR luminance
rescaling) follow the authors' reference MATLAB release so that scores are
comparable with published TMQI values.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage
from scipy.stats import beta as beta_dist
from scipy.stats import norm

# Parameters fitted in the paper.
A = 0.8012
ALPHA = 0.3046
BETA = 0.7088
LEVELS = 5
SCALE_WEIGHTS = np.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])

# Statistical naturalness model.
MEAN_MU = 115.94
MEAN_SIGMA = 27.99
STD_BETA_A = 4.4
STD_BETA_B = 10.1
STD_NORMALIZER = 64.29
NATURALNESS_BLOCK = 11

# Structural fidelity stabilising constants.
C1 = 0.01
C2 = 10.0

_Y_COEFFS = np.array([0.2126, 0.7152, 0.0722])  # Y row of the sRGB->XYZ matrix


@dataclass
class TMQIResult:
    """Full output of :func:`tmqi`."""

    Q: float
    S: float
    N: float
    s_local: np.ndarray = field(repr=False)
    s_maps: list[np.ndarray] = field(repr=False)

    def as_dict(self) -> dict[str, float]:
        d = {"Q": self.Q, "S": self.S, "N": self.N}
        d.update({f"S{i + 1}": float(v) for i, v in enumerate(self.s_local)})
        return d


def gaussian_window(size: int = 11, sigma: float = 1.5) -> np.ndarray:
    """Equivalent of MATLAB ``fspecial('gaussian', size, sigma)``."""
    ax = np.arange(size) - (size - 1) / 2.0
    g = np.exp(-(ax**2) / (2 * sigma**2))
    w = np.outer(g, g)
    return w / w.sum()


def luminance(img: np.ndarray) -> np.ndarray:
    """Y channel (CIE XYZ) of an RGB image; grayscale input is returned as-is."""
    img = np.asarray(img, dtype=np.float64)
    if img.ndim == 2:
        return img
    if img.ndim == 3 and img.shape[2] == 1:
        return img[..., 0]
    return img[..., :3] @ _Y_COEFFS


def hdr_luminance(hdr: np.ndarray) -> np.ndarray:
    """HDR luminance stretched to ``[0, 2**32 - 1]`` as in the reference code."""
    L = luminance(hdr)
    lmin, lmax = L.min(), L.max()
    if lmax <= lmin:
        return np.zeros_like(L)
    return np.round((2.0**32 - 1) / (lmax - lmin)) * (L - lmin)


# --------------------------------------------------------------------------
# Structural fidelity
# --------------------------------------------------------------------------
def _filter_valid(img: np.ndarray, window: np.ndarray) -> np.ndarray:
    """``filter2(window, img, 'valid')``, separable fast path when possible."""
    kh, kw = window.shape
    u, s, vt = np.linalg.svd(window)
    if s[1:].sum() <= 1e-12 * s[0]:
        col = u[:, 0] * np.sqrt(s[0])
        row = vt[0] * np.sqrt(s[0])
        out = ndimage.correlate1d(img, col, axis=0, mode="constant")
        out = ndimage.correlate1d(out, row, axis=1, mode="constant")
    else:
        out = ndimage.correlate(img, window, mode="constant")
    # ndimage centres the kernel at size//2; crop to the 'valid' region.
    oy, ox = kh // 2, kw // 2
    return out[oy : oy + img.shape[0] - kh + 1, ox : ox + img.shape[1] - kw + 1]


def csf_threshold(sf: float) -> float:
    """Mean of the psychometric function at spatial frequency ``sf`` (cycles/degree).

    Uses the Mannos-Sakrison style CSF from the paper; the visibility threshold
    is the inverse of contrast sensitivity, scaled to the 8-bit range.
    """
    csf = 100.0 * 2.6 * (0.0192 + 0.114 * sf) * np.exp(-((0.114 * sf) ** 1.1))
    return 128.0 / (1.4 * csf)


def local_structural_fidelity(
    img_hdr: np.ndarray, img_ldr: np.ndarray, window: np.ndarray, sf: float
) -> tuple[float, np.ndarray]:
    """Single-scale structural fidelity map and its mean."""
    window = window / window.sum()
    mu1 = _filter_valid(img_hdr, window)
    mu2 = _filter_valid(img_ldr, window)
    sigma1_sq = _filter_valid(img_hdr * img_hdr, window) - mu1 * mu1
    sigma2_sq = _filter_valid(img_ldr * img_ldr, window) - mu2 * mu2
    sigma1 = np.sqrt(np.maximum(sigma1_sq, 0.0))
    sigma2 = np.sqrt(np.maximum(sigma2_sq, 0.0))
    sigma12 = _filter_valid(img_hdr * img_ldr, window) - mu1 * mu2

    u = csf_threshold(sf)
    sig = u / 3.0
    sigma1p = norm.cdf(sigma1, loc=u, scale=sig)
    sigma2p = norm.cdf(sigma2, loc=u, scale=sig)

    s_map = ((2 * sigma1p * sigma2p + C1) / (sigma1p**2 + sigma2p**2 + C1)) * (
        (sigma12 + C2) / (sigma1 * sigma2 + C2)
    )
    return float(s_map.mean()), s_map


def _downsample(img: np.ndarray) -> np.ndarray:
    """2x2 mean filter (symmetric border) followed by dyadic decimation."""
    p = np.pad(img, ((0, 1), (0, 1)), mode="symmetric")
    f = (p[:-1, :-1] + p[1:, :-1] + p[:-1, 1:] + p[1:, 1:]) / 4.0
    return f[::2, ::2]


def structural_fidelity(
    L_hdr: np.ndarray,
    L_ldr: np.ndarray,
    levels: int = LEVELS,
    weights: np.ndarray = SCALE_WEIGHTS,
    window: np.ndarray | None = None,
) -> tuple[float, np.ndarray, list[np.ndarray]]:
    """Multi-scale structural fidelity ``S`` with per-scale scores and maps."""
    if window is None:
        window = gaussian_window()
    weights = np.asarray(weights, dtype=np.float64)
    if len(weights) != levels:
        raise ValueError("number of weights must equal number of levels")
    min_side = window.shape[0] * 2 ** (levels - 1)
    if min(L_hdr.shape) < min_side:
        raise ValueError(
            f"image too small for {levels}-scale TMQI: need min side >= {min_side}, "
            f"got {L_hdr.shape}"
        )

    s_local = np.empty(levels)
    s_maps: list[np.ndarray] = []
    f = 32.0
    for lvl in range(levels):
        f /= 2.0
        s_local[lvl], smap = local_structural_fidelity(L_hdr, L_ldr, window, f)
        s_maps.append(smap)
        if lvl < levels - 1:
            L_hdr = _downsample(L_hdr)
            L_ldr = _downsample(L_ldr)
    # Guard against tiny negative values that would produce NaNs.
    S = float(np.prod(np.power(np.maximum(s_local, 0.0), weights)))
    return S, s_local, s_maps


# --------------------------------------------------------------------------
# Statistical naturalness
# --------------------------------------------------------------------------
def block_std_mean(L: np.ndarray, block: int = NATURALNESS_BLOCK) -> float:
    """Mean of non-overlapping block standard deviations (MATLAB ``blkproc``).

    Partial border blocks are zero padded, matching ``blkproc``'s behaviour,
    and every pixel of a block carries that block's standard deviation, so the
    mean is taken over the padded image.
    """
    h, w = L.shape
    H = -(-h // block) * block
    W = -(-w // block) * block
    p = np.zeros((H, W))
    p[:h, :w] = L
    blocks = p.reshape(H // block, block, W // block, block).swapaxes(1, 2)
    stds = blocks.reshape(H // block, W // block, -1).std(axis=2, ddof=1)
    return float(stds.mean())


def statistical_naturalness(L_ldr: np.ndarray) -> float:
    """Statistical naturalness ``N`` of an LDR luminance image in ``[0, 255]``."""
    u = float(L_ldr.mean())
    sig = block_std_mean(L_ldr)

    mode = (STD_BETA_A - 1) / (STD_BETA_A + STD_BETA_B - 2)
    pc = beta_dist.pdf(sig / STD_NORMALIZER, STD_BETA_A, STD_BETA_B) / beta_dist.pdf(
        mode, STD_BETA_A, STD_BETA_B
    )
    pb = norm.pdf(u, MEAN_MU, MEAN_SIGMA) / norm.pdf(MEAN_MU, MEAN_MU, MEAN_SIGMA)
    return float(pb * pc)


# --------------------------------------------------------------------------
# TMQI
# --------------------------------------------------------------------------
def tmqi(
    hdr: np.ndarray,
    ldr: np.ndarray,
    window: np.ndarray | None = None,
    levels: int = LEVELS,
    weights: np.ndarray = SCALE_WEIGHTS,
) -> TMQIResult:
    """Compute TMQI between an HDR image and its tone-mapped LDR version.

    Parameters
    ----------
    hdr:
        Linear HDR image, ``(H, W, 3)`` RGB or ``(H, W)`` luminance.
    ldr:
        Tone-mapped image in ``[0, 255]``, ``(H, W, 3)`` or ``(H, W)``.
    """
    hdr = np.asarray(hdr, dtype=np.float64)
    ldr = np.asarray(ldr, dtype=np.float64)
    if hdr.shape[:2] != ldr.shape[:2]:
        raise ValueError(f"size mismatch: HDR {hdr.shape[:2]} vs LDR {ldr.shape[:2]}")

    L_hdr = hdr_luminance(hdr)
    L_ldr = luminance(ldr)

    S, s_local, s_maps = structural_fidelity(L_hdr, L_ldr, levels, weights, window)
    N = statistical_naturalness(L_ldr)
    Q = A * S**ALPHA + (1 - A) * N**BETA
    return TMQIResult(Q=float(Q), S=S, N=N, s_local=s_local, s_maps=s_maps)
