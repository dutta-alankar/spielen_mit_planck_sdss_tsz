"""planck_cmb: a pipeline for Planck CMB analysis and tSZ x SDSS cross-correlation.

Modules
-------
config      : paths and remote data-product URLs.
data        : download / cache Planck maps and catalogs (with simulation fallback).
components  : monopole / dipole / multipole decomposition of the temperature sky.
maps        : all-sky CMB anisotropy map rendering.
spectra     : angular power spectrum (C_ell / D_ell) estimation and plotting.
sdss        : query galaxies and clusters from the Sloan Digital Sky Survey.
tsz         : thermal Sunyaev-Zeldovich extraction, cutouts and stacking.
crosscorr   : cross-correlation of the tSZ y-map with SDSS objects.
"""

from . import config, data, components, maps, spectra, sdss, tsz, crosscorr

__version__ = "0.1.0"

__all__ = [
    "config",
    "data",
    "components",
    "maps",
    "spectra",
    "sdss",
    "tsz",
    "crosscorr",
    "__version__",
]
