"""Tests for sir_likelihood, sir_resampling_log_weights and sir_resampling_linear_weights (numpy only)."""
import numpy as np
import pytest

from sir_likelihood import LikelihoodFunction
from sir_resampling_linear_weights import ResampleParsLinearWeights
from sir_resampling_log_weights import ResampleParsLogWeights, jacobian_log


# ---------------------------------------------------------------------------------------------
# Likelihood
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("obs, cov, hx, expected", [
    (1.0, 0.5, 2.0, -1.0),            # -0.5 * 1^2 / 0.5
    (1.0, [0.5], [2.0], -1.0),        # length-1 lists behave like scalars
    (3.0, 2.0, 3.0, 0.0),             # perfect match
    (0.0, 4.0, 2.0, -0.5),
])
def test_scalar_log_likelihood(obs, cov, hx, expected):
    assert LikelihoodFunction(obs=obs, obs_cov=cov, hx=hx).make_log_likelihood() == pytest.approx(expected)


def test_vector_log_likelihood_matches_quadratic_form():
    obs = np.array([1.0, 2.0, 3.0])
    hx = np.array([1.5, 1.0, 3.0])
    R = np.array([[1.0, 0.2, 0.0], [0.2, 2.0, 0.0], [0.0, 0.0, 0.5]])
    d = obs - hx
    expected = -0.5 * d @ np.linalg.inv(R) @ d
    assert LikelihoodFunction(obs, R, hx).make_log_likelihood() == pytest.approx(expected)


def test_vector_covariance_given_as_diagonal_or_scalar():
    obs, hx = np.array([1.0, 2.0]), np.array([2.0, 2.0])
    a = LikelihoodFunction(obs, [0.5, 0.5], hx).make_log_likelihood()
    b = LikelihoodFunction(obs, 0.5, hx).make_log_likelihood()
    c = LikelihoodFunction(obs, 0.5 * np.eye(2), hx).make_log_likelihood()
    assert a == pytest.approx(b) == pytest.approx(c) == pytest.approx(-1.0)


def test_likelihood_is_exp_of_log_likelihood():
    lf = LikelihoodFunction(1.0, 0.3, 1.4)
    assert lf.make_likelihood() == pytest.approx(np.exp(lf.make_log_likelihood()))


def test_nan_observation_operator_gives_nan():
    assert np.isnan(LikelihoodFunction(np.array([1.0]), 0.5, np.array([np.nan])).make_log_likelihood())


@pytest.mark.parametrize("obs, cov, hx", [
    (1.0, 0.0, 1.0),                                  # zero variance
    (1.0, -1.0, 1.0),                                 # negative variance
    (np.array([1.0, 2.0]), 0.5, np.array([1.0])),     # shape mismatch
    (np.array([1.0, 2.0]), np.array([[1.0, 0.5], [0.0, 1.0]]), np.array([1.0, 2.0])),  # not symmetric
])
def test_invalid_inputs_are_rejected(obs, cov, hx):
    with pytest.raises(AssertionError):
        LikelihoodFunction(obs, cov, hx)


# ---------------------------------------------------------------------------------------------
# Log-sum-exp
# ---------------------------------------------------------------------------------------------
def test_jacobian_log_is_log_sum_exp():
    x = np.random.default_rng(1).normal(size=20) * 10
    assert jacobian_log(list(x)) == pytest.approx(np.log(np.sum(np.exp(x))))


def test_jacobian_log_is_stable_for_large_magnitudes():
    assert jacobian_log([-1000.0, -1000.0]) == pytest.approx(-1000.0 + np.log(2))
    assert jacobian_log([1000.0, 1000.0]) == pytest.approx(1000.0 + np.log(2))


def test_jacobian_log_ignores_nan_and_handles_inf():
    assert jacobian_log([np.nan, 0.0, 0.0]) == pytest.approx(np.log(2))
    assert jacobian_log([-np.inf, 0.0]) == pytest.approx(0.0)
    assert jacobian_log([-np.inf, -np.inf]) == -np.inf
    assert np.isnan(jacobian_log([np.nan, np.nan]))


# ---------------------------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------------------------
def test_normalised_log_weights_sum_to_one():
    lw = np.random.default_rng(2).normal(size=15) * 5
    norm = ResampleParsLogWeights(list(lw)).norm_log_weights()
    assert np.sum(np.exp(norm)) == pytest.approx(1.0)


def test_log_cdf_is_monotonic_and_ends_at_zero():
    lw = list(np.random.default_rng(3).normal(size=12))
    cdf = ResampleParsLogWeights(lw).make_cdf_log_weights()
    assert np.all(np.diff(cdf) >= 0)
    assert cdf[-1] == pytest.approx(0.0, abs=1e-8)


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("init_position", [0.0, 0.003, 0.02])
def test_log_and_linear_resampling_agree(seed, init_position):
    lw = np.random.default_rng(seed).normal(size=40) * 4
    a = ResampleParsLogWeights(list(lw)).systematic_resampling_log_weights(init_position)
    b = ResampleParsLinearWeights(list(np.exp(lw))).systematic_resampling(init_position)
    assert a == b


def test_systematic_resampling_known_answer():
    # weights 0.1, 0.2, 0.3, 0.4 -> CDF 0.1, 0.3, 0.6, 1.0; positions u0 + k/4 with u0 = 0.1 are
    #  0.1, 0.35, 0.6, 0.85, each selecting the first particle whose CDF is >= the position
    w = [0.1, 0.2, 0.3, 0.4]
    assert ResampleParsLinearWeights(w).systematic_resampling(0.1) == [0, 2, 2, 3]
    assert ResampleParsLogWeights(list(np.log(w))).systematic_resampling_log_weights(0.1) == [0, 2, 2, 3]


def test_resampling_selects_the_only_non_zero_weight():
    w = [0.0, 0.0, 1.0, 0.0]
    assert ResampleParsLinearWeights(w).systematic_resampling(0.1) == [2, 2, 2, 2]


def test_resampled_counts_follow_the_weights():
    n = 1000
    rng = np.random.default_rng(4)
    w = rng.uniform(size=n)
    idx = ResampleParsLogWeights(list(np.log(w)), rng=rng).systematic_resampling_log_weights()
    counts = np.bincount(idx, minlength=n)
    expected = n * w / w.sum()
    # systematic resampling: each count is floor or ceil of its expectation
    assert np.all(np.abs(counts - expected) < 1.0 + 1e-9)


def test_indices_are_valid_and_sorted():
    lw = list(np.random.default_rng(5).normal(size=30))
    idx = ResampleParsLogWeights(lw, rng=np.random.default_rng(0)).systematic_resampling_log_weights()
    assert len(idx) == 30 and min(idx) >= 0 and max(idx) <= 29 and idx == sorted(idx)


def test_invalid_log_weights_get_zero_weight():
    idx = ResampleParsLogWeights([np.nan, 0.0, -np.inf, 1.0, np.inf]).systematic_resampling_log_weights(0.1)
    assert set(idx) <= {1, 3}


def test_all_invalid_log_weights_fall_back_to_uniform():
    for bad in ([np.nan] * 5, [-1e31] * 5, [np.inf] * 5):
        r = ResampleParsLogWeights(bad)
        assert np.allclose(r.norm_log_weights(), -np.log(5))
        assert r.systematic_resampling_log_weights(0.1) == [0, 1, 2, 3, 4]


def test_resampling_is_reproducible_with_a_seeded_rng():
    lw = list(np.random.default_rng(6).normal(size=25))
    a = ResampleParsLogWeights(lw, rng=np.random.default_rng(42)).systematic_resampling_log_weights()
    b = ResampleParsLogWeights(lw, rng=np.random.default_rng(42)).systematic_resampling_log_weights()
    assert a == b


def test_n_ensemble_must_match_number_of_weights():
    with pytest.raises(AssertionError):
        ResampleParsLogWeights([0.0, 0.0], n_ensemble=3)
    with pytest.raises(AssertionError):
        ResampleParsLinearWeights([1.0, 1.0], n_ensemble=3)
