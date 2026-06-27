"""Render all-sky HEALPix maps to image files (Mollweide projection)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from . import config


def _save(fig_or_none, outfile: Path) -> Path:
    import matplotlib.pyplot as plt

    outfile = Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close("all")
    return outfile


def plot_allsky(
    m: np.ndarray,
    outfile,
    title: str = "",
    unit: str = r"$\mu$K",
    coord: str = "G",
    cmap: str = "RdBu_r",
    vrange: Optional[Tuple[float, float]] = None,
    mask: Optional[np.ndarray] = None,
    graticule: bool = True,
) -> Path:
    """Mollweide all-sky plot of a HEALPix map saved to *outfile*.

    Parameters
    ----------
    m : ndarray
        HEALPix map (RING ordering).
    vrange : (min, max), optional
        Colour-scale limits. If None, a symmetric range at the 2nd/98th
        percentile of the unmasked pixels is used.
    mask : ndarray, optional
        Pixels with mask < 0.5 are greyed out (set to UNSEEN for display).
    """
    import healpy as hp

    disp = m.astype(np.float64).copy()
    if mask is not None:
        disp[mask < 0.5] = hp.UNSEEN

    if vrange is None:
        finite = disp[np.isfinite(disp) & (disp != hp.UNSEEN)]
        if finite.size:
            hi = np.percentile(np.abs(finite - np.median(finite)), 98)
            med = np.median(finite)
            vmin, vmax = med - hi, med + hi
        else:  # pragma: no cover
            vmin, vmax = -1.0, 1.0
    else:
        vmin, vmax = vrange

    hp.mollview(disp, title=title, unit=unit, coord=coord, cmap=cmap,
                min=vmin, max=vmax, badcolor="0.85")
    if graticule:
        hp.graticule(dpar=30, dmer=30, color="0.5", alpha=0.5)
    return _save(None, config.FIGURE_DIR / outfile if not str(outfile).startswith("/") else outfile)


def plot_gnomonic(
    m: np.ndarray,
    lon: float,
    lat: float,
    outfile,
    reso_arcmin: float = 1.5,
    xsize: int = 200,
    title: str = "",
    unit: str = "",
    coord: str = "G",
    cmap: str = "RdBu_r",
    vrange: Optional[Tuple[float, float]] = None,
) -> Path:
    """Gnomonic (tangent-plane) cutout centred on (lon, lat) in degrees."""
    import healpy as hp

    kw = {}
    if vrange is not None:
        kw["min"], kw["max"] = vrange
    hp.gnomview(m, rot=(lon, lat), reso=reso_arcmin, xsize=xsize,
                title=title, unit=unit, coord=coord, cmap=cmap, **kw)
    hp.graticule(dpar=5, dmer=5, color="0.5", alpha=0.5)
    return _save(None, config.FIGURE_DIR / outfile if not str(outfile).startswith("/") else outfile)
