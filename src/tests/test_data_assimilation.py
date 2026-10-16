"""
Tests for data_assimilation_routine.RunDataAssimilationRoutine, plus the REGRESSION tests that check
future versions of the data assimilation reproduce today's results.

The routine is built without running __init__ (which reads da_surf_setup.Setup, downloading WSA maps and
reading .env paths); instead a small synthetic OSSE configuration is set directly, and the real
run_data_assimilation() is called, writing cme_pars.nc files to a temporary directory.

Regression tests
----------------
* test_regression_fast_pipeline: the whole pipeline (synthetic observations, prior draw, Aux-PF steps,
  saving) with the cheap analytic observation operator instead of SURF. Runs in seconds, always.
* test_regression_full_surf_osse (slow, --run-slow): the same OSSE with the real compressible SURF
  model and synthetic heliospheric imager.

The first run writes the baselines to tests/data/*.json (the test is then reported as skipped); later runs
must reproduce them. After an intended change in results, re-run with --update-golden and commit the
new baseline files.
"""
import datetime
import glob
import os
import types

import numpy as np
import pytest

u = pytest.importorskip("astropy.units")
xr = pytest.importorskip("xarray")
for _mod in ("surf", "huxt", "sunpy", "dotenv", "stereo_spice", "tables"):
    pytest.importorskip(_mod)

import aux_pf
import data_assimilation_routine as dar
import sir_observations
from da_test_helpers import AnalyticArrivalTimes, AnalyticObservationOperator, check_against_golden
from init_sir import initialise_cme_parameter_ensemble_dict

T0 = datetime.datetime(2012, 1, 1, 0, 0, 0)
TRUTH_T_INIT = T0 + datetime.timedelta(hours=1)
SAVED_VARS = ["t_init", "v", "width", "lon", "lat", "thick", "log_weight"]


def make_routine(out_dir, monkeypatch, use_model="compress_surf", n_members=10, n_obs=4, n_runs=2,
                 elong_range=(4.0, 30.0)) -> "dar.RunDataAssimilationRoutine":
    """A RunDataAssimilationRoutine configured as a small synthetic OSSE (truth: 495 km/s, 37.4 deg, lon 0)."""
    monkeypatch.setenv("OUT_BASE_DIR", str(out_dir))
    r = object.__new__(dar.RunDataAssimilationRoutine)
    r.setup_class = types.SimpleNamespace(
        get_use_synthetic_obs=lambda: True,
        allowed_cov_types=lambda: {"uncorr", "donki", "mo_cone"},
    )
    # Model
    r.use_model = use_model
    r.surf_init_time = T0
    r.vr_in = np.ones(128) * 400 * u.km / u.s
    r.vr_longs, r.vr_lats = np.nan * u.rad, np.nan * u.rad
    r.br_in, r.br_longs, r.br_lats = np.nan * u.nT, np.nan * u.rad, np.nan * u.rad
    r.lon_start, r.lon_stop = -70 * u.deg, 70 * u.deg
    r.latitude_min, r.latitude_max, r.n_lat_surf = -30 * u.deg, 30 * u.deg, 1
    r.sim_time, r.dt_scale = 2 * u.day, 1
    r.r_min, r.cme_init_rad = 21.5 * u.solRad, 12 * u.solRad
    r.cme_fixed_duration, r.fixed_duration = True, 12 * 3600 * u.s
    r.n_lon_in, r.n_lat_in = 128, 1
    # Observations
    r.n_obs = n_obs
    r.obs_radius, r.obs_lon, r.obs_lat = 1.0 * u.AU, 300 * u.deg, 0 * u.deg
    r.obs_cov = [0.5]
    r.obs_times = [TRUTH_T_INIT + datetime.timedelta(hours=10 + 3 * i) for i in range(1, 1 + n_obs)]
    r.elong_min, r.elong_max = elong_range
    r.use_synthetic_obs, r.obs_filenames, r.obs_rng_seed = True, None, 4096
    r.ssw_event, r.craft, r.img = None, "twin", None
    r.bias_term_bool, r.bias_term_5rs, r.bias_term_21rs = False, None, None
    r.use_parallel = False
    # Truth CME
    truth = initialise_cme_parameter_ensemble_dict(1, T0)
    truth.update(t_init=TRUTH_T_INIT, v=495 * u.km / u.s, width=37.4 * u.deg, lon=0 * u.deg,
                 lat=0 * u.deg, thick=0 * u.solRad)
    r.true_cme_par_dict = truth
    # First guess and prior
    r.fg_mean_cme_t_init, r.fg_mean_cme_speed, r.fg_mean_cme_width = TRUTH_T_INIT, 495, 37.4
    r.fg_mean_cme_lon, r.fg_mean_cme_lat, r.fg_mean_cme_thick = 0.0, 0.0, 0.0
    r.fg_sd_cme_t_init, r.fg_sd_cme_speed, r.fg_sd_cme_width = 0, 50, 5
    r.fg_sd_cme_lon, r.fg_sd_cme_lat, r.fg_sd_cme_thick = 5, 0, 0
    r.cme_cov_type, r.scale_corr, r.use_log_v = "uncorr", True, True
    r.prior_sd_t_init, r.prior_sd_speed, r.prior_sd_width = 0, 0.1, 5
    r.prior_sd_lon, r.prior_sd_lat, r.prior_sd_thick = 5, 0, 0
    # Data assimilation
    r.n_members, r.n_runs, r.delta_aux_pf = n_members, n_runs, 0.98
    r.pars_in_state, r.time_tolerance = ["v", "width", "lon"], 0.5 * u.day
    # Synthetic observations, then one observer position per observation (as __init__ does)
    r.observations = r.get_observations()
    r.obs_radius = [r.obs_radius] * n_obs
    r.obs_lon = [r.obs_lon] * n_obs
    r.obs_lat = [r.obs_lat] * n_obs
    return r


@pytest.fixture
def analytic_model(monkeypatch):
    """Replace SURF with the analytic observation operator everywhere the DA uses it."""
    monkeypatch.setattr(aux_pf, "ObservationOperator", AnalyticObservationOperator)
    monkeypatch.setattr(sir_observations, "ObservationOperator", AnalyticObservationOperator)
    monkeypatch.setattr(dar, "EnsembleArrivalTimes", AnalyticArrivalTimes)


def read_runs(out_dir) -> list[dict]:
    """Read every run_NNN/cme_pars.nc under out_dir, in run order."""
    files = sorted(glob.glob(os.path.join(str(out_dir), "**", "run_*", "cme_pars.nc"), recursive=True))
    runs = []
    for f in files:
        with xr.open_dataset(f) as ds:
            runs.append({k: np.asarray(ds[k].values, dtype=float) for k in SAVED_VARS})
    return runs


# ---------------------------------------------------------------------------------------------
# Helper methods
# ---------------------------------------------------------------------------------------------
def test_random_seeds_are_distinct_and_reproducible():
    r = object.__new__(dar.RunDataAssimilationRoutine)
    seeds = [r.generate_random_seed(42, 10000, n) for n in range(-5, 6)]
    assert len(set(seeds)) == len(seeds)
    assert r.generate_random_seed(42, 10000, 3) == 30042
    assert r.generate_random_seed() == 42


@pytest.mark.parametrize("val, lower, upper, expected", [
    (6.0, True, True, True), (6.0, False, True, False), (10.0, True, False, False),
    (8.0, False, False, True), (5.9, True, True, False), (np.nan, True, True, False),
])
def test_obs_in_elong_range(val, lower, upper, expected):
    r = object.__new__(dar.RunDataAssimilationRoutine)
    r.elong_min, r.elong_max = 6.0, 10.0
    assert r.obs_in_elong_range(val, inclusive_lower=lower, inclusive_upper=upper) == expected


def test_save_ensemble_state_units_and_longitude_wrap():
    from da_test_helpers import make_ensemble
    r = object.__new__(dar.RunDataAssimilationRoutine)
    r.n_members = 6
    d = make_ensemble(n=6, seed=9)
    saved = np.zeros((2, 6, 7))
    r.save_ensemble_state(saved, 1, d)
    np.testing.assert_allclose(saved[1, :, 1], d["v"].to(u.km / u.s).value)
    lon = saved[1, :, 3]
    assert np.all(lon <= 180) and np.all(lon > -180)
    np.testing.assert_allclose(np.mod(lon, 360), np.mod(d["lon"].to(u.deg).value, 360))
    np.testing.assert_allclose(saved[1, :, 6], d["log_weight"])
    np.testing.assert_array_equal(saved[0], 0)


def test_unsupported_model_cannot_be_saved(tmp_path, monkeypatch, analytic_model):
    r = make_routine(tmp_path, monkeypatch, n_runs=1)
    r.use_model = "not_a_model"
    with pytest.raises(SystemExit):
        r.make_output_dataset(np.zeros((r.n_obs + 1, r.n_members, 7)))


# ---------------------------------------------------------------------------------------------
# Whole pipeline with the analytic model
# ---------------------------------------------------------------------------------------------
def test_synthetic_observations_are_reproducible(tmp_path, monkeypatch, analytic_model):
    a = make_routine(tmp_path / "a", monkeypatch).observations
    b = make_routine(tmp_path / "b", monkeypatch).observations
    assert np.all(np.isfinite(a)) and len(a) == 4
    np.testing.assert_array_equal(a, b)


def test_pipeline_output_structure(tmp_path, monkeypatch, analytic_model):
    r = make_routine(tmp_path, monkeypatch, n_members=10, n_obs=4, n_runs=2)
    r.run_data_assimilation(run_start=0)
    runs = read_runs(tmp_path)
    assert len(runs) == 2
    for run in runs:
        for k in SAVED_VARS:
            assert run[k].shape == (5, 10), k
        np.testing.assert_allclose(run["log_weight"][0], -np.log(10))           # prior: uniform
        np.testing.assert_allclose(np.exp(run["log_weight"]).sum(axis=1), 1.0)  # normalised every step
        assert np.all(run["lon"] <= 180) and np.all(run["lon"] > -180)
        np.testing.assert_array_equal(run["lat"], 0)                              # not estimated, no spread
    # different runs use different random seeds
    assert not np.allclose(runs[0]["v"][0], runs[1]["v"][0])


def test_pipeline_is_deterministic(tmp_path, monkeypatch, analytic_model):
    for name in ("a", "b"):
        make_routine(tmp_path / name, monkeypatch, n_runs=1).run_data_assimilation(run_start=0)
    a, b = read_runs(tmp_path / "a"), read_runs(tmp_path / "b")
    for k in SAVED_VARS:
        np.testing.assert_array_equal(a[0][k], b[0][k])


def test_observations_outside_the_elongation_range_are_skipped(tmp_path, monkeypatch, analytic_model):
    r = make_routine(tmp_path, monkeypatch, n_runs=1, elong_range=(1000.0, 2000.0))
    r.run_data_assimilation(run_start=0)
    run = read_runs(tmp_path)[0]
    for k in SAVED_VARS:  # nothing assimilated: every step equals the prior
        np.testing.assert_array_equal(run[k], np.broadcast_to(run[k][0], run[k].shape))


def test_negative_runs_use_the_unperturbed_first_guess(tmp_path, monkeypatch, analytic_model):
    r = make_routine(tmp_path, monkeypatch, n_runs=1)
    r.run_data_assimilation(run_start=-1)
    assert glob.glob(os.path.join(str(tmp_path), "**", "run_-01", "cme_pars.nc"), recursive=True)
    assert r.fg_sd_cme_speed == 0 and r.fg_sd_cme_lon == 0


# ---------------------------------------------------------------------------------------------
# Regression tests
# ---------------------------------------------------------------------------------------------
def _results(routine, runs) -> dict:
    out = {"observations": routine.observations}
    for i, run in enumerate(runs):
        for k in SAVED_VARS:
            out[f"run{i}_{k}"] = run[k]
    return out


@pytest.mark.regression
def test_regression_fast_pipeline(tmp_path, monkeypatch, analytic_model, update_golden):
    """Full DA pipeline with the analytic observation operator must reproduce the stored baseline."""
    r = make_routine(tmp_path, monkeypatch, n_members=10, n_obs=4, n_runs=2)
    r.run_data_assimilation(run_start=0)
    check_against_golden("regression_fast_pipeline", _results(r, read_runs(tmp_path)), update_golden)


@pytest.mark.slow
@pytest.mark.regression
def test_regression_full_surf_osse(tmp_path, monkeypatch, update_golden):
    """
    Small OSSE with the real compressible SURF model and synthetic HI observations (4 members, 2 obs,
    1 run): about 20 SURF runs. Must reproduce the stored baseline.
    """
    r = make_routine(tmp_path, monkeypatch, use_model="compress_surf", n_members=4, n_obs=2, n_runs=1)
    r.run_data_assimilation(run_start=0)
    runs = read_runs(tmp_path)
    assert len(runs) == 1
    check_against_golden("regression_full_surf_osse", _results(r, runs), update_golden, rtol=1e-7, atol=1e-7)
