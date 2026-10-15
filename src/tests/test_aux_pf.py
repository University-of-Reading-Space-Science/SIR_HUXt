"""
Tests for aux_pf.AuxPF (Liu & West auxiliary particle filter). The forward model is replaced by cheap
stand-ins, so these check the filter itself: shrinkage, weights, resampling and bookkeeping.
"""
import copy
import datetime

import numpy as np
import pytest

u = pytest.importorskip("astropy.units")
pytest.importorskip("surf")
pytest.importorskip("huxt")
pytest.importorskip("sunpy")

import aux_pf
from da_test_helpers import AnalyticObservationOperator, analytic_elongation, make_ensemble

OBS_TIME = datetime.datetime(2012, 1, 1, 14)


class ConstantOperator:
    """Observation operator returning a fixed (n_members, 1) array."""
    values = None

    def __init__(self, cme_par_dict, **kwargs):
        self.n = cme_par_dict["n_members"]

    def make_obs_op(self):
        return np.asarray(type(self).values, dtype=float).reshape(self.n, 1)


class SpeedOperator:
    """Observation operator hx = v / 100 (so the likelihood favours a particular speed)."""

    def __init__(self, cme_par_dict, **kwargs):
        self.v = np.array([float(x.to(u.km / u.s).value) for x in cme_par_dict["v"]])

    def make_obs_op(self):
        return (self.v / 100.0).reshape(-1, 1)


def make_filter(monkeypatch, operator, n=8, seed=0, obs=6.0, delta=0.98, pars=("v", "lon", "width"),
                ensemble=None, rng_seed=1):
    monkeypatch.setattr(aux_pf, "ObservationOperator", operator)
    d = ensemble if ensemble is not None else make_ensemble(n=n, seed=seed)
    return aux_pf.AuxPF(
        use_model="compress_surf", cme_par_dict=d, obs=obs, obs_cov=[0.5],
        obs_radius=1 * u.AU, obs_lon=300 * u.deg, obs_lat=0 * u.deg, obs_time=OBS_TIME,
        pars_in_state=list(pars), delta_aux_pf=delta, r_min=21.5 * u.solRad, cme_init_rad=12 * u.solRad,
        rng=np.random.default_rng(rng_seed),
    )


@pytest.mark.parametrize("delta", [0.9, 0.95, 0.98, 0.999])
def test_liu_west_shrinkage_factors(monkeypatch, delta):
    f = make_filter(monkeypatch, ConstantOperator, delta=delta)
    a, h2 = f.get_shrink_fact()
    assert a == pytest.approx((3 * delta - 1) / (2 * delta))
    assert a ** 2 + h2 == pytest.approx(1.0)
    assert 0 < a <= 1


def test_shrinkage_keeps_the_weighted_mean_and_reduces_the_spread(monkeypatch):
    d = make_ensemble(n=10, seed=3)
    lw = np.random.default_rng(0).normal(size=10)
    d["log_weight"] = list(lw - np.log(np.sum(np.exp(lw))))
    f = make_filter(monkeypatch, ConstantOperator, ensemble=d, delta=0.9)
    w = np.exp(f.log_weights) / np.sum(np.exp(f.log_weights))
    np.testing.assert_allclose(np.average(f.shrunk_pars, axis=0, weights=w),
                               np.average(f.state_vector, axis=0, weights=w))
    a, _ = f.get_shrink_fact()
    np.testing.assert_allclose(np.std(f.shrunk_pars, axis=0), a * np.std(f.state_vector, axis=0))


def test_kernel_covariance_is_h2_times_weighted_covariance(monkeypatch):
    f = make_filter(monkeypatch, ConstantOperator, n=12)
    np.testing.assert_allclose(f.scaled_cov_state, f.cov_shrink_fact2 * f.cov_state)
    np.testing.assert_allclose(f.cov_state, np.cov(f.state_vector, rowvar=False))  # uniform weights


def test_initialisation_does_not_modify_the_input_ensemble(monkeypatch):
    d = make_ensemble(n=6, seed=4)
    before = copy.deepcopy(d)
    make_filter(monkeypatch, ConstantOperator, ensemble=d)
    for k in ["v", "lon", "width", "lat", "thick"]:
        np.testing.assert_array_equal(d[k].value, before[k].value)
    assert d["t_init"] == before["t_init"]


@pytest.mark.parametrize("log_w, expected", [
    (np.log(np.full(10, 0.1)), 10.0),
    (np.log([1.0, 1e-300, 1e-300, 1e-300]), 1.0),
    (np.log([0.5, 0.5, 0.0 + 1e-300, 1e-300]), 2.0),
])
def test_effective_sample_size(monkeypatch, log_w, expected):
    f = make_filter(monkeypatch, ConstantOperator)
    assert f.calc_ess_log_weights(log_w) == pytest.approx(expected, rel=1e-6)
    assert f.calc_ess_lin_weights(np.exp(log_w)) == pytest.approx(expected, rel=1e-6)


def test_posterior_weights_are_normalised(monkeypatch):
    f = make_filter(monkeypatch, AnalyticObservationOperator, n=10)
    out = f.aux_pf()
    assert np.sum(out["weight"]) == pytest.approx(1.0)
    np.testing.assert_allclose(np.exp(out["log_weight"]), out["weight"])
    assert len(out["v"]) == 10


def test_assimilation_moves_the_ensemble_towards_the_observation(monkeypatch):
    # hx = v/100 and obs = 6 -> speeds near 600 km/s are favoured
    d = make_ensemble(n=200, seed=5)
    f = make_filter(monkeypatch, SpeedOperator, ensemble=d, obs=6.0, delta=0.95)
    prior_v = np.array(d["v"].to(u.km / u.s).value)  # copy: aux_pf updates the input ensemble in place
    prior_err = np.abs(np.mean(prior_v) - 600)
    out = f.aux_pf()
    v = np.array([float(x.to(u.km / u.s).value) for x in out["v"]])
    w = np.asarray(out["weight"])
    assert np.abs(np.average(v, weights=w) - 600) < prior_err
    assert np.std(v) < np.std(prior_v)


def test_all_nan_observation_operator_keeps_uniform_weights(monkeypatch):
    ConstantOperator.values = [np.nan] * 6
    f = make_filter(monkeypatch, ConstantOperator, n=6)
    out = f.aux_pf()
    np.testing.assert_allclose(out["weight"], 1 / 6)
    np.testing.assert_allclose(out["log_weight"], -np.log(6))


def test_identical_likelihoods_keep_uniform_weights(monkeypatch):
    ConstantOperator.values = [6.0] * 6
    out = make_filter(monkeypatch, ConstantOperator, n=6).aux_pf()
    np.testing.assert_allclose(out["weight"], 1 / 6)


def test_non_state_parameters_are_inherited_from_the_ancestor(monkeypatch):
    """Parameters not in the state vector (t_init, lat, thick here) must stay together as one member's set."""
    d = make_ensemble(n=12, seed=6)
    t0 = d["surf_init_time"]

    def triples(ens):
        return [
            (round((t - t0).total_seconds(), 3),
             round(float(u.Quantity(la).to(u.deg).value), 6),
             round(float(u.Quantity(th).to(u.solRad).value), 6))
            for t, la, th in zip(ens["t_init"], ens["lat"], ens["thick"])
        ]

    before = set(triples(d))
    out = make_filter(monkeypatch, SpeedOperator, ensemble=d, obs=6.0).aux_pf()
    assert set(triples(out)) <= before


def test_same_seed_gives_the_same_result(monkeypatch):
    outs = []
    for _ in range(2):
        outs.append(make_filter(monkeypatch, AnalyticObservationOperator, n=8, rng_seed=7).aux_pf())
    for k in ["v", "lon", "width"]:
        np.testing.assert_array_equal(u.Quantity(outs[0][k]).value, u.Quantity(outs[1][k]).value)
    np.testing.assert_array_equal(outs[0]["log_weight"], outs[1]["log_weight"])


def test_sim_time_reaches_the_observation(monkeypatch):
    f = make_filter(monkeypatch, ConstantOperator)
    expected = (OBS_TIME - f.surf_init_time).total_seconds() / 86400 + 0.5
    assert f.sim_time.to(u.day).value == pytest.approx(expected)


def test_analytic_operator_sanity():
    d = make_ensemble(n=5, seed=8)
    el = analytic_elongation(d, OBS_TIME, 300.0)
    assert el.shape == (5,) and np.all(np.isfinite(el))
    assert np.all(np.isnan(analytic_elongation(d, datetime.datetime(2012, 1, 1, 0, 1), 300.0)))
