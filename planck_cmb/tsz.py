"""Thermal Sunyaev-Zeldovich (tSZ) extraction from the Compton-*y* map.

The tSZ effect is the inverse-Compton scattering of CMB photons off hot
electrons in the intracluster medium, parameterised by the Compton-y parameter
    y = (sigma_T / m_e c^2) * integral P_e dl.
Planck's component-separation pipelines (MILCA / NILC) deliver an all-sky y map
in which galaxy clusters appear as compact positive sources.

This module provides:
  * coordinate conversion (equatorial -> Galactic, the y-map's frame),
  * per-object gnomonic cutouts,
  * azimuthally-averaged radial y-profiles,
  * aperture-photometry y values, and
  * catalog stacking to beat down the noise on individual faint objects.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def radec_to_galactic(ra_deg, dec_deg) -> Tuple[np.ndarray, np.ndarray]:
    """Convert equatorial (J2000) coordinates to Galactic (l, b) in degrees."""
    from astropy import coordinates as coords
    import astropy.units as u

    c = coords.SkyCoord(np.atleast_1d(ra_deg) * u.deg,
                        np.atleast_1d(dec_deg) * u.deg, frame="icrs").galactic
    return np.asarray(c.l.deg), np.asarray(c.b.deg)


def cutout(y_map: np.ndarray, l_gal: float, b_gal: float,
           size_deg: float = 1.0, reso_arcmin: float = 1.5) -> np.ndarray:
    """Return a square gnomonic cutout array of the y-map around (l, b)."""
    import healpy as hp

    xsize = int(np.ceil(size_deg * 60.0 / reso_arcmin))
    proj = hp.projector.GnomonicProj(rot=(l_gal, b_gal),
                                     reso=reso_arcmin, xsize=xsize, ysize=xsize)
    nside = hp.get_nside(y_map)
    return proj.projmap(y_map, lambda x, y, z: hp.vec2pix(nside, x, y, z))


def radial_profile(
    y_map: np.ndarray,
    l_gal: float,
    b_gal: float,
    rbins_arcmin: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Azimuthally-averaged y as a function of angular radius from (l, b).

    Returns ``(r_centre_arcmin, y_mean, y_err)``.
    """
    import healpy as hp

    if rbins_arcmin is None:
        rbins_arcmin = np.linspace(0, 60, 13)  # 0..60', 5' bins
    nside = hp.get_nside(y_map)
    centre = hp.ang2vec(np.radians(90.0 - b_gal), np.radians(l_gal))
    rmax = np.radians(rbins_arcmin[-1] / 60.0)
    ipix = hp.query_disc(nside, centre, rmax)
    vecs = np.array(hp.pix2vec(nside, ipix))
    ang_arcmin = np.degrees(hp.rotator.angdist(vecs, centre[:, None])) * 60.0

    # Fixed-length output: empty bins (common at coarse Nside) become NaN so
    # downstream stacking can use nanmean instead of dropping whole objects.
    r_c = 0.5 * (rbins_arcmin[:-1] + rbins_arcmin[1:])
    y_m = np.full(r_c.size, np.nan)
    y_e = np.full(r_c.size, np.nan)
    for i, (lo, hi) in enumerate(zip(rbins_arcmin[:-1], rbins_arcmin[1:])):
        sel = (ang_arcmin >= lo) & (ang_arcmin < hi)
        n = np.count_nonzero(sel)
        if n == 0:
            continue
        vals = y_map[ipix[sel]]
        y_m[i] = np.mean(vals)
        y_e[i] = np.std(vals) / np.sqrt(n)
    return r_c, y_m, y_e


def aperture_y(
    y_map: np.ndarray,
    l_gal: float,
    b_gal: float,
    aperture_arcmin: float = 5.0,
    bg_inner_arcmin: float = 10.0,
    bg_outer_arcmin: float = 20.0,
) -> float:
    """Background-subtracted mean y in a disc aperture (AP photometry).

    The mean y inside *aperture_arcmin* minus the mean y in the surrounding
    annulus ``[bg_inner, bg_outer]`` removes the local large-scale background.
    """
    import healpy as hp

    nside = hp.get_nside(y_map)
    centre = hp.ang2vec(np.radians(90.0 - b_gal), np.radians(l_gal))
    ipix = hp.query_disc(nside, centre, np.radians(bg_outer_arcmin / 60.0))
    vecs = np.array(hp.pix2vec(nside, ipix))
    ang = np.degrees(hp.rotator.angdist(vecs, centre[:, None])) * 60.0

    in_ap = ang <= aperture_arcmin
    in_bg = (ang >= bg_inner_arcmin) & (ang <= bg_outer_arcmin)
    if not np.any(in_ap):
        return np.nan
    bg = np.mean(y_map[ipix[in_bg]]) if np.any(in_bg) else 0.0
    return float(np.mean(y_map[ipix[in_ap]]) - bg)


def stack(
    y_map: np.ndarray,
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    size_deg: float = 1.0,
    reso_arcmin: float = 2.0,
    mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, int]:
    """Stack square cutouts of the y-map at the given catalog positions.

    Returns ``(mean_stack_2d, n_used)``. Objects whose centre falls in a masked
    region (mask < 0.5) are skipped.
    """
    import healpy as hp

    l_gal, b_gal = radec_to_galactic(ra_deg, dec_deg)
    xsize = int(np.ceil(size_deg * 60.0 / reso_arcmin))
    acc = np.zeros((xsize, xsize), dtype=np.float64)
    n_used = 0
    nside = hp.get_nside(y_map)

    for l, b in zip(l_gal, b_gal):
        if mask is not None:
            ipix = hp.ang2pix(nside, np.radians(90.0 - b), np.radians(l))
            if mask[ipix] < 0.5:
                continue
        c = cutout(y_map, l, b, size_deg=size_deg, reso_arcmin=reso_arcmin)
        if c.shape != (xsize, xsize) or not np.all(np.isfinite(c)):
            continue
        acc += c
        n_used += 1
    if n_used == 0:
        return acc, 0
    return acc / n_used, n_used
