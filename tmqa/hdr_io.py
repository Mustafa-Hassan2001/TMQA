"""Reading and writing HDR (Radiance .hdr, .pfm, optionally .exr) and LDR images.

The Radiance RGBE codec is implemented here so that the project has no hard
dependency on FreeImage / OpenEXR bindings. All HDR images are returned as
``float64`` arrays of shape ``(H, W, 3)`` holding linear radiance values. LDR
images are returned as ``float64`` arrays in the range ``[0, 255]``.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from PIL import Image

HDR_EXTENSIONS = {".hdr", ".pic", ".rgbe", ".pfm", ".exr"}
LDR_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".ppm"}


# --------------------------------------------------------------------------
# Radiance RGBE
# --------------------------------------------------------------------------
def _read_rle_scanline(data: bytes, pos: int, width: int) -> tuple[np.ndarray, int]:
    """Decode one new-style RLE scanline starting at ``pos`` (after the 4-byte header)."""
    line = np.empty((4, width), dtype=np.uint8)
    for ch in range(4):
        x = 0
        while x < width:
            count = data[pos]
            pos += 1
            if count > 128:  # run
                count -= 128
                line[ch, x : x + count] = data[pos]
                pos += 1
            else:  # literal
                line[ch, x : x + count] = np.frombuffer(data, np.uint8, count, pos)
                pos += count
            x += count
    return line.T, pos


def read_rgbe(path: str | Path) -> np.ndarray:
    """Read a Radiance RGBE (``.hdr``) file into linear float RGB."""
    data = Path(path).read_bytes()
    pos = 0
    header_lines = []
    while True:
        end = data.index(b"\n", pos)
        line = data[pos:end].decode("ascii", errors="replace")
        pos = end + 1
        if line == "":
            break
        header_lines.append(line)
    if not header_lines or not header_lines[0].startswith("#?"):
        raise ValueError(f"{path}: not a Radiance HDR file")
    exposure = 1.0
    for line in header_lines:
        if line.startswith("FORMAT=") and "32-bit_rle_rgbe" not in line:
            raise ValueError(f"{path}: unsupported format {line}")
        if line.startswith("EXPOSURE="):
            exposure *= float(line.split("=", 1)[1])
    end = data.index(b"\n", pos)
    res = data[pos:end].decode("ascii")
    pos = end + 1
    m = re.match(r"([-+])Y\s+(\d+)\s+([-+])X\s+(\d+)", res.strip())
    if not m:
        raise ValueError(f"{path}: unsupported resolution string {res!r}")
    height, width = int(m.group(2)), int(m.group(4))

    rgbe = np.empty((height, width, 4), dtype=np.uint8)
    for y in range(height):
        if (
            8 <= width < 32768
            and data[pos] == 2
            and data[pos + 1] == 2
            and ((data[pos + 2] << 8) | data[pos + 3]) == width
        ):
            rgbe[y], pos = _read_rle_scanline(data, pos + 4, width)
        else:  # flat (uncompressed) scanline
            n = width * 4
            rgbe[y] = np.frombuffer(data, np.uint8, n, pos).reshape(width, 4)
            pos += n

    if m.group(1) == "+":
        rgbe = rgbe[::-1]
    if m.group(3) == "-":
        rgbe = rgbe[:, ::-1]

    e = rgbe[..., 3].astype(np.int32)
    scale = np.where(e > 0, np.ldexp(1.0, e - (128 + 8)), 0.0)
    rgb = (rgbe[..., :3].astype(np.float64) + 0.5) * scale[..., None]
    rgb[e == 0] = 0.0
    return rgb / exposure


def write_rgbe(path: str | Path, img: np.ndarray) -> None:
    """Write linear float RGB to an (uncompressed) Radiance RGBE file."""
    img = np.asarray(img, dtype=np.float64)
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=2)
    img = np.maximum(img, 0.0)
    h, w, _ = img.shape
    vmax = img.max(axis=2)
    mant, expo = np.frexp(vmax)
    valid = vmax >= 1e-32
    scale = np.where(valid, mant * 256.0 / np.where(valid, vmax, 1.0), 0.0)
    rgbe = np.zeros((h, w, 4), dtype=np.uint8)
    rgbe[..., :3] = np.clip(np.floor(img * scale[..., None]), 0, 255).astype(np.uint8)
    rgbe[..., 3] = np.where(valid, expo + 128, 0).astype(np.uint8)
    header = f"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y {h} +X {w}\n".encode("ascii")
    Path(path).write_bytes(header + rgbe.tobytes())


# --------------------------------------------------------------------------
# Portable Float Map
# --------------------------------------------------------------------------
def read_pfm(path: str | Path) -> np.ndarray:
    data = Path(path).read_bytes()
    parts = data.split(b"\n", 3)
    kind = parts[0].strip()
    w, h = (int(v) for v in parts[1].split())
    scale = float(parts[2])
    channels = 3 if kind == b"PF" else 1
    dtype = "<f4" if scale < 0 else ">f4"
    arr = np.frombuffer(parts[3], dtype=dtype, count=w * h * channels)
    arr = arr.reshape(h, w, channels)[::-1].astype(np.float64)
    if channels == 1:
        arr = np.repeat(arr, 3, axis=2)
    return arr


def write_pfm(path: str | Path, img: np.ndarray) -> None:
    img = np.asarray(img, dtype="<f4")
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=2)
    h, w, _ = img.shape
    header = f"PF\n{w} {h}\n-1.0\n".encode("ascii")
    Path(path).write_bytes(header + np.ascontiguousarray(img[::-1]).tobytes())


# --------------------------------------------------------------------------
# Generic entry points
# --------------------------------------------------------------------------
def read_hdr(path: str | Path) -> np.ndarray:
    """Read an HDR image (``.hdr``/``.pic``/``.pfm``; ``.exr`` needs imageio)."""
    ext = Path(path).suffix.lower()
    if ext in {".hdr", ".pic", ".rgbe"}:
        return read_rgbe(path)
    if ext == ".pfm":
        return read_pfm(path)
    if ext == ".exr":
        try:
            import imageio.v3 as iio
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("Reading .exr requires the 'imageio' package") from exc
        img = np.asarray(iio.imread(path), dtype=np.float64)
        return img[..., :3] if img.ndim == 3 else np.repeat(img[..., None], 3, axis=2)
    raise ValueError(f"Unsupported HDR format: {path}")


def write_hdr(path: str | Path, img: np.ndarray) -> None:
    ext = Path(path).suffix.lower()
    if ext == ".pfm":
        write_pfm(path, img)
    else:
        write_rgbe(path, img)


def read_ldr(path: str | Path) -> np.ndarray:
    """Read an 8-bit image as float RGB in ``[0, 255]``."""
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.float64)


def write_ldr(path: str | Path, img: np.ndarray) -> None:
    """Write an image in ``[0, 255]`` (float or uint8) as an 8-bit file."""
    arr = np.clip(np.round(np.asarray(img, dtype=np.float64)), 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def is_hdr_file(path: str | Path) -> bool:
    return Path(path).suffix.lower() in HDR_EXTENSIONS


def is_ldr_file(path: str | Path) -> bool:
    return Path(path).suffix.lower() in LDR_EXTENSIONS
