"""Procedural HDR scenes for demos and tests.

The TMQI database is not redistributable, so these scenes let the full
pipeline run out-of-the-box. Each scene combines a very bright sky with a sun,
a dim textured foreground in shadow and a lit interior "window", giving a
dynamic range of roughly 10^5 - 10^6 : 1 with texture at all scales.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def _fractal_noise(shape: tuple[int, int], rng: np.random.Generator, octaves: int = 6) -> np.ndarray:
    h, w = shape
    out = np.zeros(shape)
    amp, total = 1.0, 0.0
    for o in range(octaves):
        cell = max(2, 2 ** (octaves - o + 1))
        small = rng.standard_normal((h // cell + 2, w // cell + 2))
        up = ndimage.zoom(small, cell, order=3)[:h, :w]
        out += amp * up
        total += amp
        amp *= 0.55
    out /= total
    return (out - out.min()) / (out.max() - out.min() + 1e-12)


def make_scene(
    height: int = 384, width: int = 512, seed: int = 0, dynamic_range: float = 1e5
) -> np.ndarray:
    """Generate a linear-radiance HDR RGB scene."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width] / np.array([height, width])[:, None, None]
    horizon = 0.45 + 0.1 * rng.random()

    # Sky: bright gradient with soft clouds.
    clouds = _fractal_noise((height, width), rng, octaves=5)
    sky_L = (2000 + 3000 * (1 - yy / horizon)) * (0.7 + 0.6 * clouds)
    sky_rgb = np.stack([0.75 * sky_L, 0.9 * sky_L, 1.2 * sky_L], axis=-1)

    # Sun near the top.
    sx, sy = 0.2 + 0.6 * rng.random(), 0.12 + 0.15 * rng.random()
    r2 = ((xx - sx) * width / height) ** 2 + (yy - sy) ** 2
    sun = dynamic_range * 5 * np.exp(-r2 / (2 * 0.012**2)) + 20000 * np.exp(-r2 / (2 * 0.06**2))
    sky_rgb += np.stack([sun, 0.95 * sun, 0.8 * sun], axis=-1)

    # Ground in shadow: dark texture with fine detail.
    tex = _fractal_noise((height, width), rng, octaves=7)
    fine = _fractal_noise((height, width), rng, octaves=3)
    ground_L = 0.5 + 8.0 * tex**2 + 1.5 * fine
    ground_rgb = np.stack([0.9 * ground_L, 0.8 * ground_L, 0.6 * ground_L], axis=-1)

    is_sky = (yy < horizon + 0.03 * np.sin(xx * 12 + rng.random() * 6))[..., None]
    img = np.where(is_sky, sky_rgb, ground_rgb)

    # A building with a lit window (mid-range luminance with sharp edges).
    bx0, bx1 = sorted(rng.uniform(0.05, 0.95, 2))
    if bx1 - bx0 < 0.2:
        bx1 = min(bx0 + 0.3, 0.98)
    building = (xx > bx0) & (xx < bx1) & (yy > horizon - 0.25)
    wall = 3.0 + 4.0 * _fractal_noise((height, width), rng, octaves=4)
    img[building] = np.stack([wall, 0.95 * wall, 0.9 * wall], axis=-1)[building]
    wx0, wx1 = bx0 + 0.3 * (bx1 - bx0), bx0 + 0.6 * (bx1 - bx0)
    window = (xx > wx0) & (xx < wx1) & (yy > horizon - 0.18) & (yy < horizon - 0.05)
    lamp = 150 + 250 * _fractal_noise((height, width), rng, octaves=3)
    img[window] = np.stack([1.1 * lamp, 0.9 * lamp, 0.6 * lamp], axis=-1)[window]

    return np.maximum(img, 1e-4)


def make_scenes(n: int = 4, height: int = 384, width: int = 512, seed: int = 0) -> dict[str, np.ndarray]:
    return {f"synthetic_{i:02d}": make_scene(height, width, seed + i) for i in range(n)}
