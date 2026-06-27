"""Command-line pipeline for the planck_cmb package.

Run ``planck-cmb --help`` (or ``python -m planck_cmb --help``) to see the
available stages. Every stage accepts ``--simulate`` so the whole pipeline can
be exercised without downloading the multi-gigabyte Planck products.

Stages
------
download    fetch (and cache) the Planck data products
components  fit/remove monopole & dipole, render low-multipole maps
map         all-sky CMB anisotropy map
spectra     angular power spectrum D_ell
tsz         thermal-SZ cutouts/stacks for a catalog
crosscorr   tSZ x SDSS cross-correlation
all         run the full pipeline end-to-end
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from . import config, data, components, maps, spectra, sdss, tsz, crosscorr


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--simulate", action="store_true",
                   help="use synthetic data instead of real Planck/SDSS products")
    p.add_argument("--nside-sim", type=int, default=256,
                   help="HEALPix Nside for simulated maps (default 256)")
    p.add_argument("--no-download", action="store_true",
                   help="never hit the network; require cached files or --simulate")


def cmd_download(args) -> int:
    keys = args.products or ["cmb_smica", "cmb_mask", "ymap_milca"]
    for k in keys:
        path = data.download(k)
        print(f"  cached: {path}")
    return 0


def cmd_components(args) -> int:
    m, meta = data.load_cmb_map(simulate=args.simulate, nside_sim=args.nside_sim,
                                download_if_missing=not args.no_download)
    mask = data.load_mask(nside=meta["nside"], simulate=args.simulate,
                          download_if_missing=not args.no_download)
    print(f"[components] source: {meta['source']}  (simulated={meta['simulated']})")

    dec = components.decompose(m, mask)
    print("\n" + str(dec["dipole_fit"]) + "\n")

    maps.plot_allsky(dec["monopole_map"], "01_monopole.png",
                     title="Monopole (l=0)", unit=meta["unit"])
    maps.plot_allsky(dec["dipole_map"], "02_dipole.png",
                     title="Dipole (l=1)", unit=meta["unit"])
    maps.plot_allsky(dec["quadrupole_map"], "03_quadrupole.png",
                     title="Quadrupole (l=2)", unit=meta["unit"])
    maps.plot_allsky(dec["octupole_map"], "04_octupole.png",
                     title="Octupole (l=3)", unit=meta["unit"])
    out = maps.plot_allsky(dec["anisotropy_map"], "05_anisotropy.png",
                           title="CMB anisotropy (monopole+dipole removed)",
                           unit=meta["unit"], mask=mask)
    print(f"[components] figures written under {config.FIGURE_DIR}")
    return 0


def cmd_map(args) -> int:
    m, meta = data.load_cmb_map(simulate=args.simulate, nside_sim=args.nside_sim,
                                download_if_missing=not args.no_download)
    mask = data.load_mask(nside=meta["nside"], simulate=args.simulate,
                          download_if_missing=not args.no_download)
    aniso = components.remove_monopole_dipole(m, mask)
    out = maps.plot_allsky(aniso, "05_anisotropy.png",
                           title=f"CMB anisotropy map ({meta['source']})",
                           unit=meta["unit"], mask=mask, vrange=(-300, 300))
    print(f"[map] wrote {out}")
    return 0


def cmd_spectra(args) -> int:
    m, meta = data.load_cmb_map(simulate=args.simulate, nside_sim=args.nside_sim,
                                download_if_missing=not args.no_download)
    mask = data.load_mask(nside=meta["nside"], simulate=args.simulate,
                          download_if_missing=not args.no_download)
    aniso = components.remove_monopole_dipole(m, mask)
    ell, cl = spectra.estimate_cl(aniso, mask=mask)
    theory_ell = np.arange(ell.size)
    theory_dl = data._theory_dl(ell.size - 1)
    out = spectra.plot_spectrum(ell, cl, "06_power_spectrum.png",
                                lmax=min(int(ell.max()), 2 * meta["nside"]),
                                theory=(theory_ell, theory_dl))
    print(f"[spectra] wrote {out}")
    return 0


def _get_catalog(args):
    """Return (ra, dec, label) for tSZ stages, real SDSS or mock."""
    if args.simulate or args.no_download:
        cat = sdss.mock_catalog(n=args.n_objects, kind="cluster")
        return np.asarray(cat["ra"]), np.asarray(cat["dec"]), "synthetic clusters"
    try:
        cat = sdss.query_clusters(limit=args.n_objects)
        return np.asarray(cat["ra"]), np.asarray(cat["dec"]), "SDSS redMaPPer clusters"
    except Exception as exc:
        print(f"[catalog] SDSS query failed ({exc}); using synthetic catalog.")
        cat = sdss.mock_catalog(n=args.n_objects, kind="cluster")
        return np.asarray(cat["ra"]), np.asarray(cat["dec"]), "synthetic clusters"


def cmd_tsz(args) -> int:
    y, meta = data.load_y_map(simulate=args.simulate, nside_sim=args.nside_sim,
                              download_if_missing=not args.no_download)
    print(f"[tsz] y-map: {meta['source']} (simulated={meta['simulated']})")
    ra, dec, label = _get_catalog(args)
    print(f"[tsz] catalog: {label} ({len(ra)} objects)")

    stack2d, n_used = tsz.stack(y, ra, dec, size_deg=1.0, reso_arcmin=2.0)
    out = crosscorr.plot_stack(stack2d, "07_tsz_stack.png", reso_arcmin=2.0,
                               title=f"Stacked tSZ ({label})", n_used=n_used)
    print(f"[tsz] stacked {n_used} objects -> {out}")
    return 0


def cmd_crosscorr(args) -> int:
    y, meta = data.load_y_map(simulate=args.simulate, nside_sim=args.nside_sim,
                              download_if_missing=not args.no_download)
    ra, dec, label = _get_catalog(args)
    print(f"[crosscorr] {label}: {len(ra)} objects vs {meta['source']}")

    res = crosscorr.cross_correlation(y, ra, dec, n_random=len(ra))
    out = crosscorr.plot_cross_correlation(
        res, "08_tsz_sdss_crosscorr.png",
        title=f"tSZ x {label}")
    print(f"[crosscorr] central S/N = {res['central_snr']:.2f} -> {out}")
    return 0


def cmd_all(args) -> int:
    print("=== Stage 1/5: low-multipole components ===")
    cmd_components(args)
    print("\n=== Stage 2/5: anisotropy all-sky map ===")
    cmd_map(args)
    print("\n=== Stage 3/5: CMB power spectrum ===")
    cmd_spectra(args)
    print("\n=== Stage 4/5: tSZ stacking ===")
    cmd_tsz(args)
    print("\n=== Stage 5/5: tSZ x SDSS cross-correlation ===")
    cmd_crosscorr(args)
    print(f"\nDone. Figures in {config.FIGURE_DIR}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="planck-cmb", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    pd = sub.add_parser("download", help="download Planck products")
    pd.add_argument("products", nargs="*", help="product keys (default: all)")
    pd.set_defaults(func=cmd_download)

    for name, func, helptext in [
        ("components", cmd_components, "monopole/dipole/multipole decomposition"),
        ("map", cmd_map, "all-sky CMB anisotropy map"),
        ("spectra", cmd_spectra, "angular power spectrum"),
        ("all", cmd_all, "run the full pipeline"),
    ]:
        sp = sub.add_parser(name, help=helptext)
        _add_common(sp)
        sp.set_defaults(func=func)

    for name, func, helptext in [
        ("tsz", cmd_tsz, "thermal-SZ cutouts/stacking"),
        ("crosscorr", cmd_crosscorr, "tSZ x SDSS cross-correlation"),
    ]:
        sp = sub.add_parser(name, help=helptext)
        _add_common(sp)
        sp.add_argument("--n-objects", type=int, default=500,
                        help="number of catalog objects to use")
        sp.set_defaults(func=func)

    # `all` also needs --n-objects for its tSZ stages.
    for sp in sub.choices.values():
        if not any(a.dest == "n_objects" for a in sp._actions):
            sp.add_argument("--n-objects", type=int, default=500,
                            help=argparse.SUPPRESS)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
