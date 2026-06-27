"""Fast, fully-offline tests exercising the pipeline on simulated data."""

import numpy as np
import pytest

from planck_cmb import data, components, spectra, tsz, crosscorr, sdss

NSIDE = 32


@pytest.fixture(scope="module")
def sim_cmb():
    return data.simulate_cmb_map(nside=NSIDE, seed=1)


@pytest.fixture(scope="module")
def sim_y():
    return data.simulate_y_map(nside=NSIDE, n_clusters=50, seed=2)


def test_simulated_cmb_shape(sim_cmb):
    import healpy as hp
    assert sim_cmb.shape[0] == hp.nside2npix(NSIDE)
    assert np.all(np.isfinite(sim_cmb))


def test_dipole_recovery():
    # Inject a known dipole and check the fitted direction.
    m = data.simulate_cmb_map(nside=64, seed=3)
    fit = components.fit_dipole(m)
    # The injected dipole points toward (l, b) = (264.02, 48.25).
    assert abs(fit.l_gal - 264.021) < 5.0
    assert abs(fit.b_gal - 48.253) < 5.0
    assert fit.amplitude > 3000.0  # ~3.36 mK injected


def test_monopole_dipole_removal(sim_cmb):
    aniso = components.remove_monopole_dipole(sim_cmb)
    fit = components.fit_dipole(aniso)
    # Residual monopole and dipole should be tiny compared to the originals.
    assert abs(fit.monopole) < 1.0
    assert fit.amplitude < 50.0


def test_isolate_quadrupole(sim_cmb):
    import healpy as hp
    quad = components.isolate_multipoles(sim_cmb, 2, 2)
    cl = hp.anafast(quad, lmax=8)
    # Power should be concentrated at l=2.
    assert np.argmax(cl) == 2


def test_power_spectrum_positive(sim_cmb):
    ell, cl = spectra.estimate_cl(sim_cmb, lmax=3 * NSIDE - 1)
    dl = spectra.cl_to_dl(ell, cl)
    assert np.all(cl[2:] >= 0)
    assert np.isfinite(dl).all()


def test_mask_fsky_correction(sim_cmb):
    # Remove monopole+dipole first (as the pipeline does); otherwise the 3 mK
    # dipole leaks through the mask edge and swamps the low-l spectrum.
    aniso = components.remove_monopole_dipole(sim_cmb)
    mask = data._galactic_plane_mask(NSIDE, b_cut_deg=20.0)
    _, cl_cut = spectra.estimate_cl(aniso, mask=mask)
    _, cl_full = spectra.estimate_cl(aniso)
    # fsky correction should bring the masked spectrum into the right ballpark.
    band = slice(5, 20)
    ratio = np.mean(cl_cut[band]) / np.mean(cl_full[band])
    # Crude 1/fsky correction: should recover the full-sky power to a factor ~few.
    assert 0.2 < ratio < 5.0


def test_tsz_aperture_detects_injected_cluster():
    import healpy as hp
    # Build a y-map with a single bright cluster at a known location. Nside=256
    # gives ~13.7' pixels, so aperture/annulus scales must be several pixels.
    nside = 256
    npix = hp.nside2npix(nside)
    y = np.zeros(npix)
    l0, b0 = 120.0, 40.0
    vec = hp.ang2vec(np.radians(90 - b0), np.radians(l0))
    ipix = hp.query_disc(nside, vec, np.radians(0.5))  # 30' cluster
    y[ipix] = 1e-4
    val = tsz.aperture_y(y, l0, b0, aperture_arcmin=30,
                         bg_inner_arcmin=60, bg_outer_arcmin=90)
    assert val > 1e-5


def test_stack_runs():
    y = data.simulate_y_map(nside=64, n_clusters=30, seed=4)
    cat = sdss.mock_catalog(n=20, kind="cluster")
    stack2d, n = tsz.stack(y, np.asarray(cat["ra"]), np.asarray(cat["dec"]),
                           size_deg=1.0, reso_arcmin=4.0)
    assert n > 0
    assert np.all(np.isfinite(stack2d))


def test_crosscorr_structure(sim_y):
    cat = sdss.mock_catalog(n=30, kind="cluster")
    res = crosscorr.cross_correlation(sim_y, np.asarray(cat["ra"]),
                                      np.asarray(cat["dec"]), n_random=30)
    for key in ("theta", "w", "w_err", "central_snr"):
        assert key in res
    assert res["theta"].shape == res["w"].shape


def test_mock_catalog_deterministic():
    a = sdss.mock_catalog(n=10, seed=5)
    b = sdss.mock_catalog(n=10, seed=5)
    assert np.allclose(a["ra"], b["ra"])
