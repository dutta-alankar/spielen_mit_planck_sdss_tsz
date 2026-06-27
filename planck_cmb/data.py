"""Download and cache Planck data products, with a simulation fallback.

The real Planck maps are large (the SMICA map is ~600 MB, the y-map ~200 MB).
:func:`download` streams them to ``data/raw`` with a resumable-friendly progress
bar and verifies the cache on subsequent calls.

For environments without network access (CI, laptops offline, quick demos) the
:func:`load_cmb_map` and :func:`load_y_map` helpers accept ``simulate=True`` and
synthesise a statistically reasonable sky from a theory power spectrum so the
rest of the pipeline can run end-to-end. Simulated data is clearly flagged in
the returned metadata.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np

from . import config

try:  # optional, only needed for real downloads
    import requests
except ImportError:  # pragma: no cover
    requests = None

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


# --------------------------------------------------------------------------- #
# Downloading
# --------------------------------------------------------------------------- #
def download(key: str, force: bool = False, timeout: int = 60) -> Path:
    """Download Planck product *key* into the cache and return its path.

    Parameters
    ----------
    key : str
        A key from :data:`planck_cmb.config.DATA_PRODUCTS`.
    force : bool
        Re-download even if a cached copy exists.
    timeout : int
        Per-request connection timeout in seconds.
    """
    dest = config.local_path(key)
    if dest.exists() and not force and dest.stat().st_size > 0:
        return dest
    if requests is None:
        raise RuntimeError("The 'requests' package is required to download data.")

    url = config.url_for(key)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"Downloading {config.describe(key)}\n  {url}")

    with requests.get(url, stream=True, timeout=timeout) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("Content-Length", 0))
        bar = tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) if tqdm else None
        with open(tmp, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                if chunk:
                    fh.write(chunk)
                    if bar:
                        bar.update(len(chunk))
        if bar:
            bar.close()
    os.replace(tmp, dest)
    return dest


# --------------------------------------------------------------------------- #
# Loading maps (real or simulated)
# --------------------------------------------------------------------------- #
def load_cmb_map(
    field: int = 0,
    simulate: bool = False,
    nside_sim: int = 256,
    download_if_missing: bool = True,
):
    """Return the Planck CMB temperature map in microkelvin, plus metadata.

    Parameters
    ----------
    field : int
        FITS field to read (0 = intensity / temperature for the SMICA product).
    simulate : bool
        If True (or if the file is missing and downloads are disabled) return a
        Gaussian realisation of the LCDM CMB instead of the real map.
    nside_sim : int
        HEALPix Nside for the simulated map.
    download_if_missing : bool
        Fetch the real product if it is not already cached.

    Returns
    -------
    (numpy.ndarray, dict)
        The map in microkelvin and a metadata dict with at least ``simulated``,
        ``nside`` and ``source`` keys.
    """
    import healpy as hp

    path = config.local_path("cmb_smica")
    if not simulate and (path.exists() or download_if_missing):
        try:
            if not path.exists():
                download("cmb_smica")
            m = hp.read_map(path, field=field)
            # SMICA intensity is stored in K_CMB; convert to microkelvin.
            m = np.asarray(m, dtype=np.float64) * 1e6
            return m, {
                "simulated": False,
                "nside": hp.get_nside(m),
                "source": "Planck SMICA (COM_CMB_IQU-smica_2048_R3.00)",
                "unit": "uK_CMB",
            }
        except Exception as exc:  # fall back to simulation on any failure
            print(f"[data] real CMB map unavailable ({exc}); simulating instead.")

    m = simulate_cmb_map(nside=nside_sim)
    return m, {
        "simulated": True,
        "nside": nside_sim,
        "source": "Gaussian LCDM realisation (synthetic)",
        "unit": "uK_CMB",
    }


def load_y_map(
    field: int = 0,
    simulate: bool = False,
    nside_sim: int = 256,
    download_if_missing: bool = True,
):
    """Return the Planck thermal-SZ Compton-*y* map plus metadata.

    Field 0 of the MILCA product is the full-sky y map; field 1 is NILC.
    """
    import healpy as hp

    path = config.local_path("ymap_milca")
    if not simulate and (path.exists() or download_if_missing):
        try:
            if not path.exists():
                download("ymap_milca")
            y = hp.read_map(path, field=field)
            return np.asarray(y, dtype=np.float64), {
                "simulated": False,
                "nside": hp.get_nside(y),
                "source": "Planck MILCA tSZ y-map (COM_CompMap_YSZ_R2.00)",
                "unit": "dimensionless (Compton y)",
            }
        except Exception as exc:
            print(f"[data] real y-map unavailable ({exc}); simulating instead.")

    y = simulate_y_map(nside=nside_sim)
    return y, {
        "simulated": True,
        "nside": nside_sim,
        "source": "Synthetic tSZ y-map (random cluster injection)",
        "unit": "dimensionless (Compton y)",
    }


def load_mask(nside: Optional[int] = None, simulate: bool = False,
              download_if_missing: bool = True):
    """Return the common analysis mask (1 = keep, 0 = reject).

    If *nside* is given the mask is up/downgraded to that resolution. When the
    real mask is unavailable a simple Galactic-plane cut (|b| > 20 deg) is used.
    """
    import healpy as hp

    path = config.local_path("cmb_mask")
    mask = None
    if not simulate and (path.exists() or download_if_missing):
        try:
            if not path.exists():
                download("cmb_mask")
            mask = hp.read_map(path, field=0)
        except Exception as exc:
            print(f"[data] real mask unavailable ({exc}); using |b|>20deg cut.")

    if mask is None:
        target = nside or 256
        mask = _galactic_plane_mask(target, b_cut_deg=20.0)

    if nside is not None and hp.get_nside(mask) != nside:
        mask = hp.ud_grade(mask, nside)
        mask = (mask > 0.5).astype(np.float64)
    return mask


# --------------------------------------------------------------------------- #
# Simulation helpers
# --------------------------------------------------------------------------- #
def _theory_dl(lmax: int) -> np.ndarray:
    """A smooth toy LCDM temperature D_ell = l(l+1)C_l/2pi in uK^2.

    This is *not* a fit to data; it is a compact analytic stand-in with a
    Sachs-Wolfe plateau and damped acoustic peaks, good enough to generate a
    realistic-looking sky for testing the pipeline.
    """
    ell = np.arange(lmax + 1)
    dl = np.zeros_like(ell, dtype=np.float64)
    l = ell[2:].astype(np.float64)
    plateau = 1000.0
    # Acoustic peaks via a damped cosine envelope.
    peaks = 1.0 + 0.9 * np.cos(np.pi * (l - 220.0) / 300.0) * np.exp(-((l - 220.0) ** 2) / (2 * 350.0 ** 2))
    damping = np.exp(-(l / 1600.0) ** 2)
    sw = (l / 10.0) ** -0.6  # rise toward low-l Sachs-Wolfe plateau
    dl[2:] = plateau * (0.4 + sw) * np.clip(peaks, 0.05, None) * damping
    return dl


def simulate_cmb_map(nside: int = 256, lmax: Optional[int] = None, seed: int = 42) -> np.ndarray:
    """Gaussian realisation of the CMB temperature sky in microkelvin."""
    import healpy as hp

    lmax = lmax or 3 * nside - 1
    ell = np.arange(lmax + 1)
    dl = _theory_dl(lmax)
    cl = np.zeros_like(dl)
    cl[2:] = dl[2:] * 2.0 * np.pi / (ell[2:] * (ell[2:] + 1.0))
    # healpy's synfast uses the global RNG; seed it for reproducibility.
    np.random.seed(seed)
    m = hp.synfast(cl, nside=nside, lmax=lmax)
    # Inject a realistic dipole (~3.36 mK) along the known CMB dipole direction.
    m = m + _dipole_template(nside, amp_uk=3362.0, l_gal=264.021, b_gal=48.253)
    return m.astype(np.float64)


def simulate_y_map(nside: int = 256, n_clusters: int = 400, seed: int = 7) -> np.ndarray:
    """Synthetic tSZ Compton-y map: smooth background + injected clusters."""
    import healpy as hp

    rng = np.random.default_rng(seed)
    npix = hp.nside2npix(nside)
    y = np.abs(rng.normal(0.0, 1e-7, npix))  # faint positive background
    # Inject Gaussian-profile clusters at random sky positions.
    theta = np.arccos(rng.uniform(-1, 1, n_clusters))
    phi = rng.uniform(0, 2 * np.pi, n_clusters)
    vecs = hp.ang2vec(theta, phi)
    amps = rng.uniform(1e-6, 2e-5, n_clusters)
    sigma = np.radians(rng.uniform(0.05, 0.2, n_clusters))
    for v, a, s in zip(vecs, amps, sigma):
        ipix = hp.query_disc(nside, v, 4 * s)
        ang = hp.rotator.angdist(np.array(hp.pix2vec(nside, ipix)),
                                 np.asarray(v)[:, None])
        y[ipix] += a * np.exp(-0.5 * (ang / s) ** 2)
    return y.astype(np.float64)


def _dipole_template(nside: int, amp_uk: float, l_gal: float, b_gal: float) -> np.ndarray:
    import healpy as hp

    npix = hp.nside2npix(nside)
    vec = hp.ang2vec(np.radians(90.0 - b_gal), np.radians(l_gal))
    x, y, z = hp.pix2vec(nside, np.arange(npix))
    proj = vec[0] * x + vec[1] * y + vec[2] * z
    return amp_uk * proj


def _galactic_plane_mask(nside: int, b_cut_deg: float = 20.0) -> np.ndarray:
    import healpy as hp

    npix = hp.nside2npix(nside)
    theta, _ = hp.pix2ang(nside, np.arange(npix))
    b = 90.0 - np.degrees(theta)
    return (np.abs(b) > b_cut_deg).astype(np.float64)
