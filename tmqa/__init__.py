"""TMQA: objective quality assessment of tone-mapped HDR images.

Implements the Tone-Mapped image Quality Index (TMQI; Yeganeh & Wang, 2013),
which combines multi-scale structural fidelity with statistical naturalness,
together with eight classic tone-mapping operators and tooling to benchmark
them on the TMQI database.
"""

from .hdr_io import read_hdr, read_ldr, write_hdr, write_ldr
from .tmqi import TMQIResult, statistical_naturalness, structural_fidelity, tmqi
from .tonemap import OPERATORS, tonemap, tonemap_all

__all__ = [
    "OPERATORS",
    "TMQIResult",
    "read_hdr",
    "read_ldr",
    "statistical_naturalness",
    "structural_fidelity",
    "tmqi",
    "tonemap",
    "tonemap_all",
    "write_hdr",
    "write_ldr",
]
__version__ = "0.1.0"
