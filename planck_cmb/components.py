"""Decompose the temperature sky into monopole, dipole and higher multipoles.

The observed CMB temperature field is dominated by the l=0 monopole (the mean
2.7255 K) and the l=1 kinematic dipole (~3.36 mK, from the Sun's motion through
the CMB rest frame). To study the primordial anisotropies these must be removed.

This module wraps the HEALPix spherical-harmonic machinery to:
  * fit and report the monopole and dipole (amplitude + direction),
  * subtract any range of low multipoles, and
  * isolate an individual multipole band as its own map.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class DipoleFit:
    """Result of a dipole fit."""

    monopole: float                 # l=0 amplitude (same unit as the map)
    amplitude: float                # |d|, the dipole amplitude
    direction_vec: np.ndarray       # unit vector (x, y, z) in map coordinates
    l_gal: float                    # Galactic longitude of the dipole [deg]
    b_gal: float                    # Galactic latitude of the dipole [deg]

    def __str__(self) -> str:
        return (
            f"monopole = {self.monopole:.6g}\n"
            f"dipole   = {self.amplitude:.6g} "
            f"toward (l, b) = ({self.l_gal:.3f} deg, {self.b_gal:.3f} deg)"
        )


def fit_monopole(m: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """Return the mean (l=0 monopole) over the unmasked sky."""

    good = _good_pixels(m, mask)
    return float(np.mean(m[good]))


def fit_dipole(m: np.ndarray, mask: Optional[np.ndarray] = None) -> DipoleFit:
    """Fit the monopole and dipole; report direction in Galactic coordinates."""
    import healpy as hp

    work = m.copy()
    if mask is not None:
        work[mask < 0.5] = hp.UNSEEN
    mono, vec = hp.fit_dipole(work, bad=hp.UNSEEN)
    vec = np.asarray(vec, dtype=np.float64)
    amp = float(np.linalg.norm(vec))
    unit = vec / amp if amp > 0 else vec
    # HEALPix maps are in Galactic coordinates -> convert direction to (l, b).
    theta, phi = hp.vec2ang(unit)
    l_gal = float(np.degrees(np.atleast_1d(phi)[0])) % 360.0
    b_gal = float(90.0 - np.degrees(np.atleast_1d(theta)[0]))
    return DipoleFit(monopole=float(mono), amplitude=amp,
                     direction_vec=unit, l_gal=l_gal, b_gal=b_gal)


def remove_monopole(m: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Return a copy of *m* with the best-fit monopole subtracted."""
    import healpy as hp

    work = m.copy()
    if mask is not None:
        work[mask < 0.5] = hp.UNSEEN
    return hp.remove_monopole(work, bad=hp.UNSEEN, copy=True)


def remove_monopole_dipole(m: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Return a copy of *m* with the monopole and dipole subtracted.

    Pixels outside the mask are restored to their (corrected) values rather than
    left as UNSEEN, so the result is a full-sky anisotropy map. The fit itself
    only uses unmasked pixels.
    """
    import healpy as hp

    fit = fit_dipole(m, mask)
    npix = m.size
    nside = hp.npix2nside(npix)
    x, y, z = hp.pix2vec(nside, np.arange(npix))
    dipole = (fit.direction_vec[0] * x
              + fit.direction_vec[1] * y
              + fit.direction_vec[2] * z) * fit.amplitude
    return m - fit.monopole - dipole


def isolate_multipoles(
    m: np.ndarray,
    lmin: int,
    lmax: int,
    nside_out: Optional[int] = None,
) -> np.ndarray:
    """Return the map reconstructed from multipoles in ``[lmin, lmax]`` only.

    Useful for visualising "just the dipole" (lmin=lmax=1), "just the
    quadrupole" (l=2), the large-scale anomalies, etc.
    """
    import healpy as hp

    nside = hp.get_nside(m)
    nside_out = nside_out or nside
    lmax_t = 3 * nside - 1
    alm = hp.map2alm(m, lmax=lmax_t)
    fl = np.zeros(lmax_t + 1)
    fl[lmin:lmax + 1] = 1.0
    alm = hp.almxfl(alm, fl)
    return hp.alm2map(alm, nside=nside_out, lmax=lmax_t)


def decompose(
    m: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> dict:
    """Full low-multipole decomposition of a temperature map.

    Returns a dict with the monopole value, the :class:`DipoleFit`, and separate
    HEALPix maps for the monopole, dipole, quadrupole, octupole and the
    anisotropy residual (monopole+dipole removed).
    """
    import healpy as hp

    nside = hp.get_nside(m)
    npix = m.size

    fit = fit_dipole(m, mask)

    monopole_map = np.full(npix, fit.monopole, dtype=np.float64)
    x, y, z = hp.pix2vec(nside, np.arange(npix))
    dipole_map = (fit.direction_vec[0] * x
                  + fit.direction_vec[1] * y
                  + fit.direction_vec[2] * z) * fit.amplitude

    return {
        "monopole_value": fit.monopole,
        "dipole_fit": fit,
        "monopole_map": monopole_map,
        "dipole_map": dipole_map,
        "quadrupole_map": isolate_multipoles(m, 2, 2),
        "octupole_map": isolate_multipoles(m, 3, 3),
        "anisotropy_map": remove_monopole_dipole(m, mask),
    }


def _good_pixels(m: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
    import healpy as hp

    good = np.isfinite(m) & (m != hp.UNSEEN)
    if mask is not None:
        good &= mask >= 0.5
    return good
