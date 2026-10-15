"""
Helpers shared by the tests: a small CME ensemble builder, a cheap analytic observation operator that
stands in for SURF in the fast regression test, and baseline (golden file) handling.
"""
import datetime
import json
import pathlib

import numpy as np
import pytest

DATA_DIR = pathlib.Path(__file__).resolve().parent / "data"
T0 = datetime.datetime(2012, 1, 1, 0, 0, 0)


def make_ensemble(n: int = 6, seed: int = 0, surf_init_time: datetime.datetime = T0) -> dict:
    """A CmeParEns-style dictionary with random (but reproducible) members, lon spanning 0-360 deg."""
    import astropy.units as u

    rng = np.random.default_rng(seed)
    return {
        "surf_init_time": surf_init_time,
        "n_members": n,
        "t_init": [surf_init_time + datetime.timedelta(seconds=float(s)) for s in rng.uniform(1800, 7200, n)],
        "v": list(rng.uniform(400, 900, n)) * u.km / u.s,
        "width": list(rng.uniform(20, 60, n)) * u.deg,
        "lon": list(rng.uniform(0, 360, n)) * u.deg,
        "lat": list(rng.uniform(-10, 10, n)) * u.deg,
        "thick": list(rng.uniform(0, 2, n)) * u.solRad,
        "v_transit": [0] * n,
        "v_hit": [0] * n,
        "likelihood": [0] * n,
        "weight": [1.0 / n] * n,
        "log_weight": [-np.log(n)] * n,
    }


def analytic_elongation(cme_par_dict: dict, obs_time: datetime.datetime, obs_lon_deg: float,
                        cme_init_rad_rs: float = 12.0) -> np.ndarray:
    """
    Cheap, smooth stand-in for the SURF observation operator: elongation (deg) of the CME flank seen by an
    observer at 1 AU, for a CME moving radially at constant speed from cme_init_rad at t_init. The flank
    is the edge of the cone nearest the observer. NaN before launch (exercises the NaN handling).
    """
    import astropy.units as u

    solrad_au = 695700.0 / 1.495978707e8

    def as_list(x):
        # one value per member, or a single value (e.g. the truth CME)
        if isinstance(x, datetime.datetime) or (hasattr(x, "ndim") and x.ndim == 0):
            return [x]
        return list(x)

    t_init = as_list(cme_par_dict["t_init"])
    n = len(t_init)
    t = np.array([(obs_time - ti).total_seconds() for ti in t_init])
    v = np.array([float(x.to(u.km / u.s).value) for x in as_list(cme_par_dict["v"])])
    lon = np.array([float(x.to(u.deg).value) for x in as_list(cme_par_dict["lon"])])
    width = np.array([float(x.to(u.deg).value) for x in as_list(cme_par_dict["width"])])
    r_au = cme_init_rad_rs * solrad_au + v * t / 1.495978707e8
    # Angle between the observer-Sun line and the near flank of the cone (deg)
    dlon = np.abs(((lon - obs_lon_deg + 180.0) % 360.0) - 180.0)
    theta = np.deg2rad(np.clip(dlon - width / 2.0, 0.0, 180.0))
    el = np.rad2deg(np.arctan2(r_au * np.sin(theta + 0.3), 1.0 - r_au * np.cos(theta + 0.3)))
    el[t <= 0] = np.nan
    assert el.shape == (n,) and n == cme_par_dict["n_members"]
    return el


class AnalyticObservationOperator:
    """
    Drop-in replacement for sir_observation_operator.ObservationOperator (same constructor keywords,
    same make_obs_op() output shape) that uses analytic_elongation instead of running SURF.
    """
    calls = 0

    def __init__(self, use_model, cme_par_dict, obs_time_in_datetime, obs_cov, craft=None, obs_lon=None,
                 cme_init_rad=None, **kwargs):
        self.cme_par_dict = cme_par_dict
        self.obs_times = (
            [obs_time_in_datetime] if isinstance(obs_time_in_datetime, datetime.datetime)
            else list(obs_time_in_datetime)
        )
        lon = obs_lon if obs_lon is not None else 300.0
        self.obs_lon_deg = float(np.ravel(getattr(lon, "value", lon))[0])
        import astropy.units as u

        self.cme_init_rad_rs = 12.0 if cme_init_rad is None else float(cme_init_rad.to(u.solRad).value)

    def make_obs_op(self) -> np.ndarray:
        type(self).calls += 1
        return np.stack([
            analytic_elongation(self.cme_par_dict, t, self.obs_lon_deg, self.cme_init_rad_rs)
            for t in self.obs_times
        ], axis=1)


# ---------------------------------------------------------------------------------------------
# Regression baselines
# ---------------------------------------------------------------------------------------------
def check_against_golden(name: str, results: dict, update: bool, rtol: float = 1e-9, atol: float = 1e-9) -> None:
    """
    Compare ``results`` (name -> array-like) with the baseline tests/data/<name>.json.
    If the baseline does not exist yet, or ``update`` is True, write it and skip the test, so the first
    run records the current behaviour and later runs check that it is reproduced.
    """
    path = DATA_DIR / f"{name}.json"
    serialisable = {k: np.asarray(v, dtype=float).tolist() for k, v in results.items()}
    if update or not path.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(serialisable, indent=1))
        pytest.skip(f"regression baseline written to {path}; re-run to compare against it")

    golden = json.loads(path.read_text())
    assert set(golden) == set(serialisable), f"baseline keys differ: {sorted(golden)} vs {sorted(serialisable)}"
    for key in golden:
        np.testing.assert_allclose(
            np.asarray(serialisable[key], dtype=float), np.asarray(golden[key], dtype=float),
            rtol=rtol, atol=atol, equal_nan=True,
            err_msg=f"'{key}' differs from the regression baseline {path.name} "
                    f"(if the change is intended, re-run with --update-golden)",
        )
