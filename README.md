# planck-cmb

A reproducible Python pipeline for **Planck** Cosmic Microwave Background (CMB)
analysis and its cross-correlation with **Sloan Digital Sky Survey (SDSS)**
galaxies and clusters.

The pipeline takes you from the raw Planck sky maps all the way to a thermal
Sunyaev–Zeldovich (tSZ) detection around SDSS objects:

1. **Download** the raw Planck component-separated maps (SMICA CMB map, the
   common analysis mask, and the MILCA/NILC Compton-*y* tSZ map).
2. **Decompose** the temperature sky into its **monopole** (`ℓ=0`), **dipole**
   (`ℓ=1`), **quadrupole** (`ℓ=2`), **octupole** (`ℓ=3`) and higher multipoles.
3. **Render an all-sky map** of the CMB **anisotropy** (monopole + dipole
   removed) in Mollweide projection.
4. **Estimate the CMB power spectrum** `D_ℓ = ℓ(ℓ+1)C_ℓ/2π`.
5. **Map the thermal-SZ signal** of individual galaxies/clusters and **stack**
   them.
6. **Cross-correlate** the tSZ *y*-map with an **SDSS** galaxy/cluster catalog
   and quantify the detection significance.

Every stage runs on real Planck/SDSS data **or** on a built-in synthetic sky
(`--simulate`), so the full pipeline can be exercised offline in seconds.

---

## Installation

This project uses [**uv**](https://docs.astral.sh/uv/) for environment and
dependency management.

```bash
# clone, then from the repo root:
uv venv                       # create .venv
uv pip install -e .           # install planck-cmb + dependencies (editable)

# optional: dev tools (pytest)
uv pip install -e ".[dev]"
```

> **Note on active conda environments.** If you have a conda environment active,
> call the interpreter explicitly (`.venv/bin/python ...`) or `source
> .venv/bin/activate` first, so commands resolve to the project venv rather than
> conda.

To regenerate a fully pinned lockfile:

```bash
uv lock                       # writes uv.lock
uv sync                       # install exactly from the lockfile
```

---

## Quick start

Run the **entire pipeline** on synthetic data (no network, a few seconds):

```bash
uv run python -m planck_cmb all --simulate
```

Run it on the **real Planck data** (downloads several GB into `data/raw/` on
first use, then caches):

```bash
uv run python -m planck_cmb all
```

Outputs are written to `figures/`:

| File | Contents |
|------|----------|
| `01_monopole.png`            | `ℓ=0` monopole map |
| `02_dipole.png`              | `ℓ=1` kinematic dipole map |
| `03_quadrupole.png`          | `ℓ=2` quadrupole map |
| `04_octupole.png`            | `ℓ=3` octupole map |
| `05_anisotropy.png`          | all-sky CMB anisotropy (monopole+dipole removed) |
| `06_power_spectrum.png`      | angular power spectrum `D_ℓ` |
| `07_tsz_stack.png`           | stacked tSZ cutout around catalog objects |
| `08_tsz_sdss_crosscorr.png`  | tSZ × SDSS cross-correlation `w(θ)` |

---

## Usage by stage

Each stage is a subcommand (`--help` on any of them for options). All accept
`--simulate`, `--nside-sim N` and `--no-download`.

```bash
# 1. fetch and cache the raw Planck products
uv run python -m planck_cmb download                  # all products
uv run python -m planck_cmb download cmb_smica         # just one

# 2. monopole / dipole / multipole decomposition (prints the dipole direction)
uv run python -m planck_cmb components

# 3. all-sky CMB anisotropy map
uv run python -m planck_cmb map

# 4. angular power spectrum
uv run python -m planck_cmb spectra

# 5. thermal-SZ stacking on a catalog
uv run python -m planck_cmb tsz --n-objects 1000

# 6. tSZ × SDSS cross-correlation
uv run python -m planck_cmb crosscorr --n-objects 1000
```

The convenience wrapper `scripts/run_pipeline.py` runs everything:

```bash
uv run python scripts/run_pipeline.py --simulate
```

---

## Using the library

```python
from planck_cmb import data, components, spectra, tsz, crosscorr, sdss

# Load the CMB temperature map (real or simulated) in microkelvin.
cmb, meta = data.load_cmb_map(simulate=True)
mask = data.load_mask(nside=meta["nside"], simulate=True)

# Fit and report the monopole + dipole.
fit = components.fit_dipole(cmb, mask)
print(fit)            # amplitude and (l, b) direction in Galactic coords

# Anisotropy map and power spectrum.
aniso = components.remove_monopole_dipole(cmb, mask)
ell, cl = spectra.estimate_cl(aniso, mask=mask)

# tSZ × SDSS cross-correlation.
y, _ = data.load_y_map(simulate=True)
cat = sdss.mock_catalog(n=500, kind="cluster")
res = crosscorr.cross_correlation(y, cat["ra"], cat["dec"])
print("central S/N =", res["central_snr"])
```

---

## Data products

URLs live in [`planck_cmb/config.py`](planck_cmb/config.py) and point at the
public NASA LAMBDA / IRSA Planck release archive. They can be overridden per
product with an environment variable, e.g.:

```bash
export PLANCK_CMB_CMB_SMICA_URL="https://my-mirror/COM_CMB_IQU-smica_...fits"
```

| Key | Product |
|-----|---------|
| `cmb_smica` | SMICA component-separated CMB map (`I,Q,U`; Nside=2048) |
| `cmb_mask`  | Common intensity analysis mask |
| `ymap_milca`| Planck tSZ Compton-*y* maps (MILCA & NILC) |

SDSS objects are queried live through `astroquery.sdss` (DR17). When the live
service is unavailable the pipeline falls back to a deterministic synthetic
catalog so it always completes.

Large binary products are cached in `data/raw/` and are **git-ignored**.

---

## The science, briefly

- **Monopole & dipole.** The measured sky is dominated by the `2.7255 K`
  monopole and the `~3.36 mK` kinematic dipole from the Solar System's motion
  (`v ≈ 370 km/s`) relative to the CMB rest frame, toward Galactic
  `(l, b) ≈ (264°, 48°)`. Both are fit and removed before any anisotropy
  analysis. (The synthetic sky injects exactly this dipole, and the pipeline
  recovers it — a built-in sanity check.)
- **Anisotropy & power spectrum.** The residual `ΔT/T ~ 10⁻⁵` fluctuations
  carry the cosmological information; their statistics are summarised by the
  angular power spectrum `C_ℓ`, with the acoustic peaks appearing in
  `D_ℓ = ℓ(ℓ+1)C_ℓ/2π`. A `1/f_sky` correction accounts for the sky cut.
- **Thermal SZ.** Hot intracluster electrons inverse-Compton scatter CMB
  photons, imprinting a spectral distortion measured as the Compton-*y*
  parameter. Clusters appear as compact positive sources in the *y*-map.
- **Cross-correlation.** Stacking the *y*-map on SDSS cluster/galaxy positions
  and differencing against random positions yields `w(θ)`, a direct measure of
  the hot-gas pressure traced by the galaxy distribution.

---

## Testing

```bash
.venv/bin/python -m pytest        # or: uv run pytest  (if no conda env is active)
```

The suite is fully offline (synthetic sky) and verifies dipole recovery,
monopole/dipole removal, multipole isolation, power-spectrum positivity, tSZ
aperture photometry, and the cross-correlation machinery.

---

## Project layout

```
planck_cmb/
  config.py      # paths and Planck product URLs
  data.py        # download/cache + simulation fallback
  components.py  # monopole/dipole/multipole decomposition
  maps.py        # all-sky & gnomonic rendering
  spectra.py     # C_ell / D_ell estimation and plotting
  sdss.py        # SDSS galaxy/cluster queries (+ mock)
  tsz.py         # tSZ cutouts, profiles, aperture photometry, stacking
  crosscorr.py   # tSZ x SDSS cross-correlation
  cli.py         # command-line pipeline
scripts/run_pipeline.py
tests/test_pipeline.py
```

---

## License

[MIT](LICENSE) © 2026 Alankar Dutta.

This software is independent of and not endorsed by the Planck Collaboration,
ESA, NASA, or the SDSS Collaboration. Please cite the original Planck and SDSS
data papers when using their data products.
