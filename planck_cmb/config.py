"""Central configuration: filesystem layout and Planck data-product URLs.

All large binary products are downloaded on demand into ``data/raw`` and cached;
they are excluded from version control via ``.gitignore``.

The download mirrors point at the public NASA LAMBDA / IRSA Planck release-3
archive. These URLs occasionally change between releases; override them with the
``PLANCK_CMB_<KEY>_URL`` environment variable if a mirror moves.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

# --------------------------------------------------------------------------- #
# Filesystem layout
# --------------------------------------------------------------------------- #
# Resolve relative to the repository root (two levels up from this file) so the
# package behaves the same regardless of the current working directory.
PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent

DATA_DIR = Path(os.environ.get("PLANCK_CMB_DATA_DIR", REPO_ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FIGURE_DIR = Path(os.environ.get("PLANCK_CMB_FIGURE_DIR", REPO_ROOT / "figures"))

for _d in (RAW_DIR, PROCESSED_DIR, FIGURE_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Remote data products
# --------------------------------------------------------------------------- #
# Each entry: key -> (default_url, local_filename, human description, archive_member).
# ``archive_member`` is None for plain files; for compressed tarballs it is the
# basename of the member that should be extracted into the cache.
_IRSA = "https://irsa.ipac.caltech.edu/data/Planck/release_3/all-sky-maps/maps"
_IRSA_R3 = "https://irsa.ipac.caltech.edu/data/Planck/release_3"
_IRSA_R2 = "https://irsa.ipac.caltech.edu/data/Planck/release_2"

DATA_PRODUCTS = {
    # Component-separated CMB temperature+polarization map (SMICA, Nside=2048).
    "cmb_smica": (
        f"{_IRSA}/component-maps/cmb/COM_CMB_IQU-smica_2048_R3.00_full.fits",
        "COM_CMB_IQU-smica_2048_R3.00_full.fits",
        "SMICA component-separated CMB map (I,Q,U; Nside=2048)",
        None,
    ),
    # Common confidence mask for the CMB analysis (lives under ancillary-data).
    "cmb_mask": (
        f"{_IRSA_R3}/ancillary-data/masks/COM_Mask_CMB-common-Mask-Int_2048_R3.00.fits",
        "COM_Mask_CMB-common-Mask-Int_2048_R3.00.fits",
        "Common intensity analysis mask (Nside=2048)",
        None,
    ),
    # MILCA thermal SZ Compton-y map (Nside=2048). IRSA distributes the y-maps
    # only inside a gzipped tarball; we extract the MILCA full-sky map from it.
    "ymap_milca": (
        f"{_IRSA_R2}/all-sky-maps/maps/component-maps/foregrounds/COM_CompMap_YSZ_R2.00.fits.tgz",
        "milca_ymaps.fits",
        "Planck tSZ Compton-y maps (MILCA & NILC; Nside=2048)",
        "milca_ymaps.fits",
    ),
}


def url_for(key: str) -> str:
    """Return the (possibly env-overridden) download URL for *key*."""
    env = os.environ.get(f"PLANCK_CMB_{key.upper()}_URL")
    if env:
        return env
    return DATA_PRODUCTS[key][0]


def local_path(key: str) -> Path:
    """Return the cache path for product *key* under ``data/raw``."""
    return RAW_DIR / DATA_PRODUCTS[key][1]


def describe(key: str) -> str:
    """Human-readable description of product *key*."""
    return DATA_PRODUCTS[key][2]


def archive_member(key: str) -> Optional[str]:
    """Return the tarball member to extract for *key*, or None for plain files."""
    entry = DATA_PRODUCTS[key]
    return entry[3] if len(entry) > 3 else None


# Physical / instrument constants used across the pipeline.
T_CMB_K = 2.7255          # CMB monopole temperature [K] (Fixsen 2009)
T_CMB_UK = T_CMB_K * 1e6  # ... in microkelvin
