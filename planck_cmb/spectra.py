"""Angular power-spectrum estimation and plotting.

The CMB anisotropy field T(n) is expanded in spherical harmonics,
    T(n) = sum_{l,m} a_{lm} Y_{lm}(n),
and its statistics are captured by the angular power spectrum
    C_l = <|a_{lm}|^2>.
The conventional plotting quantity is D_l = l(l+1) C_l / (2 pi).

When a sky cut is applied the raw (pseudo-)spectrum is biased low by the sky
fraction f_sky; :func:`estimate_cl` applies the simple ``1/f_sky`` correction,
which is adequate for visualisation (a full MASTER/NaMaster deconvolution is out
of scope here).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from . import config


def estimate_cl(
    m: np.ndarray,
    mask: Optional[np.ndarray] = None,
    lmax: Optional[int] = None,
    apply_fsky_correction: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate C_l from a temperature map.

    Returns ``(ell, cl)`` where ``cl`` is in (map unit)^2, e.g. uK^2.
    """
    import healpy as hp

    nside = hp.get_nside(m)
    lmax = lmax or (3 * nside - 1)

    work = m.astype(np.float64).copy()
    fsky = 1.0
    if mask is not None:
        work = work * mask
        fsky = float(np.mean(mask ** 2))

    cl = hp.anafast(work, lmax=lmax)
    if mask is not None and apply_fsky_correction and fsky > 0:
        cl = cl / fsky
    ell = np.arange(cl.size)
    return ell, cl


def cl_to_dl(ell: np.ndarray, cl: np.ndarray) -> np.ndarray:
    """Convert C_l to D_l = l(l+1)C_l / (2 pi)."""
    ell = np.asarray(ell, dtype=np.float64)
    return ell * (ell + 1.0) * cl / (2.0 * np.pi)


def bin_spectrum(
    ell: np.ndarray,
    cl: np.ndarray,
    delta_ell: int = 30,
    lmin: int = 2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bin a spectrum into bandpowers of width *delta_ell*.

    Returns ``(ell_centre, value, error)`` where the error is the scatter within
    each band divided by sqrt(N) (a crude but useful uncertainty estimate).
    """
    ell = np.asarray(ell)
    cl = np.asarray(cl)
    edges = np.arange(lmin, ell.max() + delta_ell, delta_ell)
    centres, vals, errs = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (ell >= lo) & (ell < hi)
        if not np.any(sel):
            continue
        centres.append(0.5 * (lo + hi))
        vals.append(np.mean(cl[sel]))
        n = np.count_nonzero(sel)
        errs.append(np.std(cl[sel]) / np.sqrt(max(n, 1)))
    return np.array(centres), np.array(vals), np.array(errs)


def plot_spectrum(
    ell: np.ndarray,
    cl: np.ndarray,
    outfile,
    title: str = "CMB temperature power spectrum",
    delta_ell: int = 30,
    lmin: int = 2,
    lmax: Optional[int] = None,
    theory: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> Path:
    """Plot the binned D_l spectrum (and optional theory curve) to *outfile*."""
    import matplotlib.pyplot as plt

    lmax = lmax or int(ell.max())
    sel = (ell >= lmin) & (ell <= lmax)
    ell_s, cl_s = ell[sel], cl[sel]
    dl = cl_to_dl(ell_s, cl_s)

    bc, bcl, berr = bin_spectrum(ell_s, cl_s, delta_ell=delta_ell, lmin=lmin)
    bdl = cl_to_dl(bc, bcl)
    bderr = cl_to_dl(bc, berr)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ell_s, dl, color="0.7", lw=0.6, label="unbinned")
    ax.errorbar(bc, bdl, yerr=bderr, fmt="o", ms=4, color="C0",
                capsize=2, label=f"binned (Δℓ={delta_ell})")
    if theory is not None:
        t_ell, t_dl = theory
        ax.plot(t_ell, t_dl, color="C3", lw=1.5, label="theory")
    ax.set_xlabel(r"Multipole moment $\ell$")
    ax.set_ylabel(r"$D_\ell = \ell(\ell+1)C_\ell/2\pi\ \ [\mu\mathrm{K}^2]$")
    ax.set_title(title)
    ax.set_xlim(lmin, lmax)
    ax.grid(alpha=0.3)
    ax.legend()

    outfile = Path(outfile)
    if not str(outfile).startswith("/"):
        outfile = config.FIGURE_DIR / outfile
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return outfile
