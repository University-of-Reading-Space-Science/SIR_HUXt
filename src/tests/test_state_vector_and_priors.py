"""Tests for to_state_vector / from_state_vector and the prior samplers (need astropy)."""
import datetime
import os

import numpy as np
import pytest

u = pytest.importorskip("astropy.units")

from da_test_helpers import make_ensemble
from from_state_vector import FromStateVector
from to_state_vector import ToStateVector


# ---------------------------------------------------------------------------------------------
# State vector conversions
# ---------------------------------------------------------------------------------------------
def test_cme_par_array_columns_and_units():
    d = make_ensemble(n=5, seed=1)
    arr = ToStateVector(d).cme_par_dict_to_cme_par_array()
    assert arr.shape == (5, 6)
    np.testing.assert_allclose(arr[:, 0], [(t - d["surf_init_time"]).total_seconds() for t in d["t_init"]])
    np.testing.assert_allclose(arr[:, 1], d["v"].to(u.km / u.s).value)
    np.testing.assert_allclose(arr[:, 2], d["width"].to(u.deg).value)
    np.testing.assert_allclose(arr[:, 4], d["lat"].to(u.deg).value)
    np.testing.assert_allclose(arr[:, 5], d["thick"].to(u.solRad).value)


def test_longitude_is_wrapped_to_plus_minus_180_in_the_array():
    d = make_ensemble(n=50, seed=2)
    arr = ToStateVector(d).cme_par_dict_to_cme_par_array()
    lon_in = d["lon"].to(u.deg).value
    assert np.all(arr[:, 3] > -180) and np.all(arr[:, 3] <= 180)
    np.testing.assert_allclose(np.mod(arr[:, 3], 360), np.mod(lon_in, 360))


def test_state_vector_holds_the_requested_parameters_in_order():
    d = make_ensemble(n=4, seed=3)
    tsv = ToStateVector(d, pars_in_state_vector=["v", "lon", "width"])
    arr = tsv.cme_par_dict_to_cme_par_array()
    sv = tsv.cme_par_dict_to_state_vector()
    np.testing.assert_allclose(sv, arr[:, [1, 3, 2]])


def test_round_trip_dict_array_dict():
    d = make_ensemble(n=6, seed=4)
    arr = ToStateVector(d).cme_par_dict_to_cme_par_array()
    import copy
    out = FromStateVector(np.copy(arr), copy.deepcopy(d), state_ens=arr.copy()).cme_par_array_to_cme_par_dict()
    for i in range(6):
        assert abs((out["t_init"][i] - d["t_init"][i]).total_seconds()) < 1e-3
    for key, unit in [("v", u.km / u.s), ("width", u.deg), ("lon", u.deg), ("lat", u.deg), ("thick", u.solRad)]:
        np.testing.assert_allclose(u.Quantity(out[key]).to(unit).value, d[key].to(unit).value, atol=1e-9)


def test_dictionary_longitudes_are_back_in_0_360():
    d = make_ensemble(n=30, seed=5)
    arr = ToStateVector(d).cme_par_dict_to_cme_par_array()
    out = FromStateVector(np.copy(arr), dict(d), state_ens=arr.copy()).cme_par_array_to_cme_par_dict()
    lon = u.Quantity(out["lon"]).to(u.deg).value
    assert np.all(lon >= 0) and np.all(lon < 360)


def test_state_vector_to_array_only_changes_state_columns():
    d = make_ensemble(n=5, seed=6)
    arr = ToStateVector(d).cme_par_dict_to_cme_par_array()
    new_state = np.arange(10.0).reshape(5, 2)
    out = FromStateVector(np.copy(arr), d, new_state, pars_in_state_vector=["v", "lat"]).state_vector_to_cme_par_array()
    np.testing.assert_allclose(out[:, [1, 4]], new_state)
    np.testing.assert_allclose(out[:, [0, 2, 3, 5]], arr[:, [0, 2, 3, 5]])


def test_lon_conversion_does_not_modify_the_array_in_place():
    d = make_ensemble(n=5, seed=7)
    arr = ToStateVector(d).cme_par_dict_to_cme_par_array()
    arr[:, 3] = [-170, -10, 0, 10, 170]
    before = arr.copy()
    FromStateVector(arr, d, state_ens=arr.copy()).convert_lon_array_to_dict()
    np.testing.assert_array_equal(arr, before)


def test_z_score_round_trip():
    d = make_ensemble(n=8, seed=8)
    tsv = ToStateVector(d, pars_in_state_vector=["v", "width"])
    arr = tsv.cme_par_dict_to_cme_par_array()
    z, mean, std = tsv.cme_par_to_z_state_vector()
    np.testing.assert_allclose(z.mean(axis=0), 0, atol=1e-12)
    np.testing.assert_allclose(z.std(axis=0, ddof=1), 1)
    back = FromStateVector(np.zeros_like(arr), d, z, pars_in_state_vector=["v", "width"],
                           x_mean=mean, x_std=std).z_state_vector_to_cme_par_array()
    np.testing.assert_allclose(back[:, [1, 2]], arr[:, [1, 2]])


def test_from_state_vector_checks_the_state_shape():
    d = make_ensemble(n=4)
    arr = ToStateVector(d).cme_par_dict_to_cme_par_array()
    with pytest.raises(AssertionError):
        FromStateVector(arr, d, np.zeros((4, 2)), pars_in_state_vector=["v"])


# ---------------------------------------------------------------------------------------------
# Priors
# ---------------------------------------------------------------------------------------------
@pytest.fixture
def priors():
    pytest.importorskip("seaborn")
    import prior_uncorrelated
    return prior_uncorrelated


def test_uncorrelated_cov_is_diagonal_variances(priors):
    cov = priors.uncorrelated_cov_mat(sd_t_init=10, sd_v=2, sd_width=3, sd_lon=4, sd_lat=5, sd_thick=6)
    np.testing.assert_allclose(cov, np.diag([100, 4, 9, 16, 25, 36]))


def test_uncorrelated_gaussian_samples_have_the_requested_moments(priors):
    mean = [100.0, 500.0, 40.0, 10.0, 0.0, 1.0]
    s = priors.make_uncorrelated_samples(4000, mean, np.random.default_rng(0), sd_v=50, sd_width=5, sd_lon=3)
    assert s.shape == (6, 4000)
    np.testing.assert_allclose(s.mean(axis=1), mean, atol=3)
    np.testing.assert_allclose(s.std(axis=1), [0, 50, 5, 3, 0, 0], atol=2)


def test_uncorrelated_log_speed_samples_are_positive_and_do_not_change_the_mean_list(priors):
    mean = [0.0, 500.0, 40.0, 0.0, 0.0, 0.0]
    s = priors.make_uncorrelated_samples(2000, mean, np.random.default_rng(1), sd_v=0.1, use_log_v=True)
    assert np.all(s[1] > 0)
    assert np.median(s[1]) == pytest.approx(500, rel=0.02)
    assert mean[1] == 500.0  # caller's list untouched


def test_uniform_samples_stay_within_their_bounds(priors):
    mean = [100.0, 500.0, 40.0, 10.0, 2.0, 1.0]
    s = priors.make_uncorr_uniform_samples(3000, mean, np.random.default_rng(2), sd_t_init=60, sd_v=0.1,
                                           sd_width=5, sd_lon=5, sd_lat=1, sd_thick=0.5)
    lo = [40, 450, 35, 5, 1, 0.5]
    hi = [160, 550, 45, 15, 3, 1.5]
    assert np.all(s.min(axis=1) >= lo) and np.all(s.max(axis=1) <= hi)
    # sd_v is a FRACTIONAL half-width for the uniform sampler
    assert s[1].min() < 455 and s[1].max() > 545


def test_samplers_are_reproducible(priors):
    mean = [0.0, 500.0, 40.0, 0.0, 0.0, 0.0]
    a = priors.make_uncorr_uniform_samples(10, mean, np.random.default_rng(3), sd_v=0.1, sd_lon=5)
    b = priors.make_uncorr_uniform_samples(10, mean, np.random.default_rng(3), sd_v=0.1, sd_lon=5)
    np.testing.assert_array_equal(a, b)


def test_blair_covariance_from_a_csv(tmp_path):
    pytest.importorskip("seaborn")
    import pandas as pd
    import prior_blair
    rng = np.random.default_rng(4)
    v = rng.normal(800, 100, 200)
    df = pd.DataFrame({"V": v, "Ang_rad": 0.05 * v + rng.normal(0, 3, 200),
                       "lon": rng.normal(0, 20, 200), "lat": rng.normal(0, 10, 200), "other": 1.0})
    path = tmp_path / "blair.csv"
    df.to_csv(path, index=False)
    cov, corr = prior_blair.make_cov_blair(str(path), vars_req=np.array(["v", "width", "lon"]),
                                           sd_t_init=60, sd_thick=0.5)
    assert cov.shape == corr.shape == (6, 6)
    assert cov[1, 1] == pytest.approx(np.var(v, ddof=1))
    assert corr[1, 2] > 0.5                       # speed and width were made correlated
    np.testing.assert_array_equal(cov[4], 0)      # lat not requested -> no (co)variance
    np.testing.assert_array_equal(cov[0], 0)      # t_init not requested either
    np.testing.assert_allclose(cov, cov.T)


def _make_cone_tree(root, stamps):
    for st in stamps:
        d = root / st.strftime("%Y") / st.strftime("%m")
        d.mkdir(parents=True, exist_ok=True)
        (d / st.strftime("cone_cme_%Y%m%d%H.in")).write_text("")


@pytest.mark.parametrize("start, end, expected", [
    # single month
    (datetime.datetime(2020, 1, 5), datetime.datetime(2020, 1, 20), ["2020010600", "2020011512"]),
    # several months in one year
    (datetime.datetime(2020, 1, 10), datetime.datetime(2020, 3, 1),
     ["2020011512", "2020012500", "2020020100", "2020022800"]),
    # across a year boundary
    (datetime.datetime(2019, 12, 31), datetime.datetime(2020, 1, 6), ["2019123106", "2020010100", "2020010600"]),
])
def test_mo_cone_file_list_respects_the_date_range(tmp_path, start, end, expected):
    pytest.importorskip("xarray")
    pytest.importorskip("surf")
    import prior_mo_cone
    stamps = [datetime.datetime.strptime(s, "%Y%m%d%H") for s in
              ["2019123006", "2019123106", "2020010100", "2020010600", "2020011512", "2020012500",
               "2020020100", "2020022800", "2020030112", "2020040100"]]
    _make_cone_tree(tmp_path, stamps)
    files = prior_mo_cone.make_mo_cone_file_list(str(tmp_path), start, end)
    got = [os.path.basename(f)[len("cone_cme_"):-len(".in")] for f in files]
    assert got == expected
