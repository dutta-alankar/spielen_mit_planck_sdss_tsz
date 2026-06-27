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
# Each entry: key -> (default_url, local_filename, human description).
_IRSA = "https://irsa.ipac.caltech.edu/data/Planck/release_3/all-sky-maps/maps"

DATA_PRODUCTS = {
    # Component-separated CMB temperature+polarization map (SMICA, Nside=2048).
    "cmb_smica": (
        f"{_IRSA}/component-maps/cmb/COM_CMB_IQU-smica_2048_R3.00_full.fits",
        "COM_CMB_IQU-smica_2048_R3.00_full.fits",
        "SMICA component-separated CMB map (I,Q,U; Nside=2048)",
    ),
    # Common confidence mask for the CMB analysis.
    "cmb_mask": (
        f"{_IRSA}/component-maps/cmb/COM_CMB_IQU-common-Inpainting-Mask-Int_2048_R3.00.fits",
        "COM_CMB_IQU-common-Mask-Int_2048_R3.00.fits",
        "Common intensity analysis mask (Nside=2048)",
    ),
    # MILCA thermal SZ Compton-y map (Nside=2048).
    "ymap_milca": (
        "https://irsa.ipac.caltech.edu/data/Planck/release_2/all-sky-maps/maps/"
        "component-maps/foregrounds/COM_CompMap_YSZ_R2.00.fits",
        "COM_CompMap_YSZ_R2.00.fits",
        "Planck tSZ Compton-y maps (MILCA & NILC; Nside=2048)",
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


# Physical / instrument constants used across the pipeline.
T_CMB_K = 2.7255          # CMB monopole temperature [K] (Fixsen 2009)
T_CMB_UK = T_CMB_K * 1e6  # ... in microkelvin
