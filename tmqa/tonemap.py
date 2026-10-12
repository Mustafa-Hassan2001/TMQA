"""Eight classic tone-mapping operators (TMOs).

Every operator takes a linear HDR RGB image and returns an 8-bit-range RGB
image (float, ``[0, 255]``). Operators compute a display luminance ``Ld``
from the world luminance ``Lw``; colour is restored with Schlick's
saturation-controlled ratio ``C_d = (C_w / L_w) ** s * L_d`` and the result is
gamma encoded for display.

Operators
---------
``linear``           Exposure scaling + clipping (baseline).
``logarithmic``      Global logarithmic mapping.
``reinhard_global``  Reinhard et al. 2002, photographic operator (global).
``reinhard_local``   Reinhard et al. 2002, dodging-and-burning (local).
``drago``            Drago et al. 2003, adaptive logarithmic mapping.
``durand``           Durand & Dorsey 2002, bilateral-filter base/detail split.
``ward``             Ward Larson et al. 1997, histogram adjustment.
``fattal``           Fattal et al. 2002, gradient-domain compression.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy import ndimage
from scipy.fft import dctn, idctn

from .tmqi import luminance

EPS = 1e-6
GAMMA = 2.2


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _log_average(L: np.ndarray, delta: float = EPS) -> float:
    return float(np.exp(np.mean(np.log(delta + L))))


def _normalize_percentile(L: np.ndarray, lo: float = 0.1, hi: float = 99.9) -> np.ndarray:
    a, b = np.percentile(L, [lo, hi])
    if b <= a:
        return np.clip(L - a, 0, 1)
    return np.clip((L - a) / (b - a), 0.0, 1.0)


def _apply_color(
    hdr: np.ndarray, Lw: np.ndarray, Ld: np.ndarray, saturation: float = 0.6
) -> np.ndarray:
    """Restore colour and gamma-encode to ``[0, 255]``."""
    if hdr.ndim == 2:
        out = Ld
    else:
        ratio = hdr[..., :3] / np.maximum(Lw, EPS)[..., None]
        out = np.power(np.maximum(ratio, 0.0), saturation) * Ld[..., None]
    out = np.clip(out, 0.0, 1.0)
    return np.power(out, 1.0 / GAMMA) * 255.0


# --------------------------------------------------------------------------
# operators: each returns display luminance in [0, 1]
# --------------------------------------------------------------------------
def linear_luminance(Lw: np.ndarray, percentile: float = 99.0) -> np.ndarray:
    """Map the given percentile of luminance to display white and clip."""
    white = np.percentile(Lw, percentile)
    return np.clip(Lw / max(white, EPS), 0.0, 1.0)


def logarithmic_luminance(Lw: np.ndarray, q: float = 1.0, k: float = 1.0) -> np.ndarray:
    """``Ld = log(1 + q Lw) / log(1 + k Lmax)`` on log-average-normalised input."""
    L = Lw / _log_average(Lw)
    return np.log1p(q * L) / np.log1p(k * L.max())


def reinhard_global_luminance(Lw: np.ndarray, key: float = 0.18) -> np.ndarray:
    """Reinhard photographic operator with burn-out (Eq. 4 of the paper)."""
    Lm = key / _log_average(Lw) * Lw
    Lwhite = Lm.max()
    return Lm * (1 + Lm / Lwhite**2) / (1 + Lm)


def reinhard_local_luminance(
    Lw: np.ndarray,
    key: float = 0.18,
    phi: float = 8.0,
    eps: float = 0.05,
    num_scales: int = 8,
) -> np.ndarray:
    """Reinhard dodging-and-burning with a Gaussian centre-surround scale space."""
    Lm = key / _log_average(Lw) * Lw
    alpha1 = 1.0 / (2.0 * np.sqrt(2.0))
    ratio = 1.6
    scales = [ratio**i for i in range(num_scales + 1)]
    blurs = [ndimage.gaussian_filter(Lm, sigma=alpha1 * s, mode="reflect") for s in scales]

    V1_sel = blurs[-2].copy()
    chosen = np.zeros(Lm.shape, dtype=bool)
    for i in range(num_scales):
        s = scales[i]
        V1, V2 = blurs[i], blurs[i + 1]
        V = (V1 - V2) / (2.0**phi * key / s**2 + V1)
        stop = (np.abs(V) >= eps) & ~chosen
        # The largest scale before the threshold is crossed is the previous one.
        prev = blurs[i - 1] if i > 0 else V1
        V1_sel[stop] = prev[stop]
        chosen |= stop
    Ld = Lm / (1.0 + V1_sel)
    return np.clip(Ld, 0.0, 1.0)


def drago_luminance(Lw: np.ndarray, bias: float = 0.85, Ld_max: float = 100.0) -> np.ndarray:
    """Drago adaptive logarithmic mapping."""
    L = Lw / _log_average(Lw)
    Lmax = L.max()
    exponent = np.log(bias) / np.log(0.5)
    Ld = (Ld_max * 0.01 / np.log10(Lmax + 1)) * np.log(L + 1) / np.log(
        2 + 8 * np.power(L / Lmax, exponent)
    )
    return np.clip(Ld, 0.0, 1.0)


def bilateral_filter(
    img: np.ndarray, sigma_s: float, sigma_r: float, n_segments: int | None = None
) -> np.ndarray:
    """Fast bilateral filter using Durand & Dorsey's piecewise-linear scheme.

    The range axis is sampled at ``n_segments`` intensity levels; for each one a
    normalised Gaussian-blurred response is computed and the output is linearly
    interpolated between the two nearest levels.
    """
    lo, hi = float(img.min()), float(img.max())
    if hi - lo < 1e-12:
        return img.copy()
    if n_segments is None:
        n_segments = int(np.ceil((hi - lo) / sigma_r)) + 1
        n_segments = max(n_segments, 2)
    levels = np.linspace(lo, hi, n_segments)
    responses = np.empty((n_segments,) + img.shape)
    for j, ij in enumerate(levels):
        G = np.exp(-((img - ij) ** 2) / (2 * sigma_r**2))
        K = ndimage.gaussian_filter(G, sigma_s, mode="reflect")
        H = ndimage.gaussian_filter(G * img, sigma_s, mode="reflect")
        responses[j] = H / np.maximum(K, 1e-12)
    pos = (img - lo) / (hi - lo) * (n_segments - 1)
    i0 = np.clip(np.floor(pos).astype(int), 0, n_segments - 2)
    t = pos - i0
    rows, cols = np.indices(img.shape)
    return (1 - t) * responses[i0, rows, cols] + t * responses[i0 + 1, rows, cols]


def durand_luminance(
    Lw: np.ndarray, contrast: float = 50.0, sigma_r: float = 0.4, sigma_s_frac: float = 0.02
) -> np.ndarray:
    """Durand & Dorsey fast bilateral filtering for HDR display."""
    logL = np.log10(np.maximum(Lw, EPS))
    sigma_s = max(sigma_s_frac * max(Lw.shape), 1.0)
    base = bilateral_filter(logL, sigma_s, sigma_r)
    detail = logL - base
    span = base.max() - base.min()
    cf = np.log10(contrast) / span if span > 0 else 1.0
    log_out = base * cf + detail - base.max() * cf
    return np.clip(np.power(10.0, log_out), 0.0, 1.0)


def ward_histogram_luminance(
    Lw: np.ndarray, Ld_min: float = 1.0, Ld_max: float = 100.0, n_bins: int = 100
) -> np.ndarray:
    """Ward Larson histogram adjustment with linear-contrast ceiling."""
    # Foveal adaptation image: roughly 1 degree samples -> downsample.
    factor = max(1, int(round(max(Lw.shape) / 128)))
    fovea = Lw[::factor, ::factor] if factor > 1 else Lw
    logL = np.log(np.maximum(fovea, EPS))
    bmin, bmax = logL.min(), logL.max()
    if bmax - bmin < 1e-9:
        return np.full(Lw.shape, 0.5)
    hist, edges = np.histogram(logL, bins=n_bins, range=(bmin, bmax))
    hist = hist.astype(np.float64)
    db = (bmax - bmin) / n_bins
    log_dr = np.log(Ld_max) - np.log(Ld_min)

    # Iteratively trim bins exceeding the linear ceiling.
    tolerance = 0.025 * hist.sum()
    for _ in range(1000):
        total = hist.sum()
        if total < tolerance:  # degenerate: fall back to linear in log
            hist = np.ones(n_bins)
            break
        ceiling = total * db / log_dr
        excess = np.maximum(hist - ceiling, 0.0)
        trimmings = excess.sum()
        if trimmings <= tolerance:
            break
        hist = np.minimum(hist, ceiling)
    cdf = np.concatenate([[0.0], np.cumsum(hist)]) / hist.sum()

    logLw = np.log(np.maximum(Lw, EPS))
    P = np.interp(logLw, edges, cdf)
    Ld = np.exp(np.log(Ld_min) + log_dr * P)
    return np.clip((Ld - Ld_min) / (Ld_max - Ld_min), 0.0, 1.0)


def _solve_poisson_neumann(div: np.ndarray) -> np.ndarray:
    """Solve ``laplacian(I) = div`` with Neumann boundaries using the DCT."""
    h, w = div.shape
    f = dctn(div, type=2, norm="ortho")
    yy = 2 * np.cos(np.pi * np.arange(h) / h) - 2
    xx = 2 * np.cos(np.pi * np.arange(w) / w) - 2
    denom = yy[:, None] + xx[None, :]
    denom[0, 0] = 1.0
    f /= denom
    f[0, 0] = 0.0
    return idctn(f, type=2, norm="ortho")


def fattal_luminance(
    Lw: np.ndarray, alpha_frac: float = 0.1, beta: float = 0.85, min_size: int = 32
) -> np.ndarray:
    """Fattal, Lischinski & Werman gradient-domain HDR compression."""
    H = np.log(np.maximum(Lw, EPS))

    # Gaussian pyramid and per-level attenuation factors.
    pyramid = [H]
    while min(pyramid[-1].shape) // 2 >= min_size:
        blurred = ndimage.gaussian_filter(pyramid[-1], 1.0, mode="reflect")
        pyramid.append(blurred[::2, ::2])

    Phi = None
    for k in range(len(pyramid) - 1, -1, -1):
        Hk = pyramid[k]
        gy, gx = np.gradient(Hk)
        mag = np.hypot(gx, gy) / (2.0**k)
        alpha = alpha_frac * max(mag.mean(), EPS)
        phi = (alpha / np.maximum(mag, EPS)) * np.power(np.maximum(mag, EPS) / alpha, beta)
        if Phi is None:
            Phi = phi
        else:
            zoom = (Hk.shape[0] / Phi.shape[0], Hk.shape[1] / Phi.shape[1])
            Phi = ndimage.zoom(Phi, zoom, order=1, mode="nearest")[: Hk.shape[0], : Hk.shape[1]]
            if Phi.shape != Hk.shape:  # zoom rounding
                Phi = np.pad(
                    Phi,
                    ((0, Hk.shape[0] - Phi.shape[0]), (0, Hk.shape[1] - Phi.shape[1])),
                    mode="edge",
                )
            Phi = Phi * phi

    # Attenuated forward-difference gradients and their divergence.
    gx = np.zeros_like(H)
    gy = np.zeros_like(H)
    gx[:, :-1] = H[:, 1:] - H[:, :-1]
    gy[:-1, :] = H[1:, :] - H[:-1, :]
    Phi_x = np.zeros_like(H)
    Phi_y = np.zeros_like(H)
    Phi_x[:, :-1] = 0.5 * (Phi[:, 1:] + Phi[:, :-1])
    Phi_y[:-1, :] = 0.5 * (Phi[1:, :] + Phi[:-1, :])
    Gx, Gy = gx * Phi_x, gy * Phi_y
    div = Gx.copy()
    div[:, 1:] -= Gx[:, :-1]
    div += Gy
    div[1:, :] -= Gy[:-1, :]

    I = _solve_poisson_neumann(div)
    return _normalize_percentile(np.exp(I - I.max()), 0.5, 99.5)


# --------------------------------------------------------------------------
# registry and public API
# --------------------------------------------------------------------------
LuminanceOperator = Callable[[np.ndarray], np.ndarray]

OPERATORS: dict[str, LuminanceOperator] = {
    "linear": linear_luminance,
    "logarithmic": logarithmic_luminance,
    "reinhard_global": reinhard_global_luminance,
    "reinhard_local": reinhard_local_luminance,
    "drago": drago_luminance,
    "durand": durand_luminance,
    "ward": ward_histogram_luminance,
    "fattal": fattal_luminance,
}

# Colour saturation exponent per operator (operators that compress strongly
# desaturate colours, so they get a lower exponent).
_SATURATION = {"linear": 1.0, "fattal": 0.6, "durand": 0.8}


def tonemap(hdr: np.ndarray, method: str, saturation: float | None = None, **params) -> np.ndarray:
    """Tone-map an HDR image with the named operator.

    Returns an RGB (or gray) float image in ``[0, 255]``.
    """
    if method not in OPERATORS:
        raise KeyError(f"unknown operator {method!r}; choose from {sorted(OPERATORS)}")
    hdr = np.maximum(np.asarray(hdr, dtype=np.float64), 0.0)
    Lw = np.maximum(luminance(hdr), 0.0)
    Ld = OPERATORS[method](Lw, **params)
    s = saturation if saturation is not None else _SATURATION.get(method, 0.7)
    return _apply_color(hdr, Lw, Ld, s)


def tonemap_all(hdr: np.ndarray, methods: list[str] | None = None) -> dict[str, np.ndarray]:
    """Apply several operators (all eight by default)."""
    return {m: tonemap(hdr, m) for m in (methods or list(OPERATORS))}
