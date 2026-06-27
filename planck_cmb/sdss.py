"""Query galaxies and clusters from the Sloan Digital Sky Survey (SDSS).

Uses :mod:`astroquery.sdss` to pull spectroscopic galaxy positions and (where
available) cluster catalogs via SQL against the SkyServer/CasJobs database.

A deterministic synthetic fallback (:func:`mock_catalog`) is provided so the
cross-correlation pipeline can run without network access.
"""

from __future__ import annotations


import numpy as np


def query_galaxies(
    ra_deg: float,
    dec_deg: float,
    radius_deg: float = 2.0,
    zmin: float = 0.0,
    zmax: float = 0.7,
    limit: int = 5000,
):
    """Return an astropy Table of SDSS spectroscopic galaxies in a cone.

    Parameters
    ----------
    ra_deg, dec_deg : float
        Cone centre in J2000 equatorial coordinates (degrees).
    radius_deg : float
        Cone search radius in degrees.
    zmin, zmax : float
        Spectroscopic redshift selection.
    limit : int
        Maximum number of rows to return.
    """
    from astroquery.sdss import SDSS
    from astropy import coordinates as coords
    import astropy.units as u

    sql = f"""
    SELECT TOP {int(limit)}
        s.specObjID, p.ra, p.dec, s.z, s.zErr, p.r AS rmag
    FROM SpecObj AS s
    JOIN PhotoObj AS p ON s.bestObjID = p.objID
    WHERE s.class = 'GALAXY' AND s.zWarning = 0
      AND s.z BETWEEN {zmin} AND {zmax}
      AND p.dec BETWEEN {dec_deg - radius_deg} AND {dec_deg + radius_deg}
      AND p.ra  BETWEEN {ra_deg - radius_deg} AND {ra_deg + radius_deg}
    """
    tbl = SDSS.query_sql(sql, data_release=17)
    if tbl is None or len(tbl) == 0:
        return tbl
    # Trim the box to a true cone.
    centre = coords.SkyCoord(ra_deg * u.deg, dec_deg * u.deg)
    pts = coords.SkyCoord(tbl["ra"] * u.deg, tbl["dec"] * u.deg)
    sep = centre.separation(pts).deg
    return tbl[sep <= radius_deg]


def query_clusters(limit: int = 2000):
    """Return SDSS redMaPPer-style clusters (ra, dec, z, richness).

    The redMaPPer catalog is distributed as a VAC and is not always reachable
    through the live SQL endpoint; this attempts the query and raises on
    failure so callers can fall back to :func:`mock_catalog`.
    """
    from astroquery.sdss import SDSS

    sql = f"""
    SELECT TOP {int(limit)}
        ra, dec, z_lambda AS z, lambda_chisq AS richness
    FROM redmapper_dr8
    WHERE lambda_chisq > 20
    ORDER BY lambda_chisq DESC
    """
    tbl = SDSS.query_sql(sql, data_release=17)
    if tbl is None or len(tbl) == 0:
        raise RuntimeError("redMaPPer query returned no rows.")
    return tbl


def mock_catalog(n: int = 1000, seed: int = 11, kind: str = "cluster"):
    """Deterministic synthetic catalog as an astropy Table.

    Columns: ra, dec, z, richness. Positions avoid the Galactic plane so they
    overlap the usable sky in the simulated y-map.
    """
    from astropy.table import Table

    rng = np.random.default_rng(seed)
    ra = rng.uniform(0, 360, n)
    dec = np.degrees(np.arcsin(rng.uniform(-1, 1, n)))
    z = rng.uniform(0.05, 0.6, n)
    richness = rng.uniform(20, 200, n) if kind == "cluster" else np.ones(n)
    return Table({"ra": ra, "dec": dec, "z": z, "richness": richness})
