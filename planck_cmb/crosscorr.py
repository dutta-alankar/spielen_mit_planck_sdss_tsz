"""Cross-correlate the Planck tSZ y-map with an SDSS object catalog.

Two complementary statistics are implemented:

1. **Stacked profile** -- the mean y-profile centred on the catalog objects. A
   detection appears as a central excess above the surrounding background. A
   random-position stack provides the null reference.

2. **Real-space angular cross-correlation** w(theta) -- the excess y measured in
   angular-separation bins around catalog objects relative to random points,
   estimated as <y>_data(theta) - <y>_random(theta).

Both come with a simple significance estimate from bootstrap/random trials.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from . import config, tsz


def _random_positions(n: int, seed: int = 0,
                      b_cut_deg: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """Uniform random (ra, dec) on the sphere, optionally avoiding |b|<cut."""
    rng = np.random.default_rng(seed)
    ra = rng.uniform(0, 360, n)
    dec = np.degrees(np.arcsin(rng.uniform(-1, 1, n)))
    if b_cut_deg > 0:
        l, b = tsz.radec_to_galactic(ra, dec)
        keep = np.abs(b) > b_cut_deg
        ra, dec = ra[keep], dec[keep]
    return ra, dec


def stacked_profile(
    y_map: np.ndarray,
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    rbins_arcmin: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray] = None,
    n_random: Optional[int] = None,
    seed: int = 0,
) -> dict:
    """Mean stacked y radial profile around the catalog, with a random null.

    Returns a dict with ``r``, ``y`` (mean profile), ``y_err``, and
    ``y_random`` / ``y_random_err`` for the random-position reference.
    """
    if rbins_arcmin is None:
        rbins_arcmin = np.linspace(0, 60, 13)
    l_gal, b_gal = tsz.radec_to_galactic(ra_deg, dec_deg)

    r_c = 0.5 * (rbins_arcmin[:-1] + rbins_arcmin[1:])

    def _mean_profile(lons, lats):
        rows = [tsz.radial_profile(y_map, l, b, rbins_arcmin)[1]
                for l, b in zip(lons, lats)]
        arr = np.vstack(rows) if rows else np.full((1, r_c.size), np.nan)
        # Per-bin number of objects that actually had pixels in that bin.
        n_eff = np.maximum(np.sum(np.isfinite(arr), axis=0), 1)
        # Bins with no finite samples yield all-NaN slices; nanmean/nanstd warn
        # on those, so silence the (expected) empty-slice warnings and let the
        # nan_to_num below turn the resulting NaNs into zeros.
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Mean of empty slice",
                                    category=RuntimeWarning)
            warnings.filterwarnings("ignore", message="Degrees of freedom <= 0",
                                    category=RuntimeWarning)
            mean = np.nanmean(arr, axis=0)
            err = np.nanstd(arr, axis=0) / np.sqrt(n_eff)
        return np.nan_to_num(mean), np.nan_to_num(err)

    y_mean, y_err = _mean_profile(l_gal, b_gal)

    n_random = n_random or len(ra_deg)
    rra, rdec = _random_positions(n_random, seed=seed)
    rl, rb = tsz.radec_to_galactic(rra, rdec)
    yr_mean, yr_err = _mean_profile(rl, rb)

    return {
        "r": r_c,
        "y": y_mean,
        "y_err": y_err,
        "y_random": yr_mean,
        "y_random_err": yr_err,
        "n_objects": int(np.size(ra_deg)),
        "n_random": int(np.size(rra)),
    }


def cross_correlation(
    y_map: np.ndarray,
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    theta_bins_arcmin: Optional[np.ndarray] = None,
    n_random: Optional[int] = None,
    seed: int = 0,
) -> dict:
    """Angular cross-correlation w(theta) = <y>_data - <y>_random.

    A positive central w(theta) indicates the catalog objects trace hot,
    pressure-rich gas (the tSZ signal of clusters/groups).
    """
    if theta_bins_arcmin is None:
        theta_bins_arcmin = np.linspace(0, 120, 13)
    data = stacked_profile(y_map, ra_deg, dec_deg,
                           rbins_arcmin=theta_bins_arcmin,
                           n_random=n_random, seed=seed)
    w = data["y"] - data["y_random"]
    w_err = np.sqrt(data["y_err"] ** 2 + data["y_random_err"] ** 2)
    # Central-bin significance.
    snr = float(w[0] / w_err[0]) if w_err[0] > 0 else np.nan
    return {
        "theta": data["r"],
        "w": w,
        "w_err": w_err,
        "central_snr": snr,
        "n_objects": data["n_objects"],
    }


def plot_stack(stack2d: np.ndarray, outfile, reso_arcmin: float = 2.0,
               title: str = "Stacked tSZ signal", n_used: Optional[int] = None) -> Path:
    """Image a 2-D stacked cutout (from :func:`planck_cmb.tsz.stack`)."""
    import matplotlib.pyplot as plt

    half = stack2d.shape[0] * reso_arcmin / 2.0
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(stack2d, origin="lower", cmap="inferno",
                   extent=[-half, half, -half, half])
    ax.set_xlabel("arcmin")
    ax.set_ylabel("arcmin")
    t = title + (f"  (N={n_used})" if n_used is not None else "")
    ax.set_title(t)
    fig.colorbar(im, ax=ax, label="Compton y")

    outfile = Path(outfile)
    if not str(outfile).startswith("/"):
        outfile = config.FIGURE_DIR / outfile
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return outfile


def plot_cross_correlation(result: dict, outfile,
                           title: str = "tSZ x SDSS cross-correlation") -> Path:
    """Plot w(theta) with error bars and the random null line."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(result["theta"], result["w"], yerr=result["w_err"],
                fmt="o-", capsize=3, color="C1", label="data - random")
    ax.axhline(0, color="0.5", lw=1, ls="--")
    ax.set_xlabel(r"Angular separation $\theta$ [arcmin]")
    ax.set_ylabel(r"$w(\theta) = \langle y\rangle_{\rm data} - \langle y\rangle_{\rm rand}$")
    snr = result.get("central_snr", float("nan"))
    ax.set_title(f"{title}  (central S/N = {snr:.1f})")
    ax.grid(alpha=0.3)
    ax.legend()

    outfile = Path(outfile)
    if not str(outfile).startswith("/"):
        outfile = config.FIGURE_DIR / outfile
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return outfile
