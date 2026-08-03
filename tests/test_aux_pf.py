import pytest
import numpy as np
import datetime
import astropy.units as u

from aux_pf import AuxPF

from SIR_HUXt.src.cme_par_ens import CmeParEns


# ============================================================
# MOCK ObservationOperator (ONLY physics layer)
# ============================================================

class MockObservationOperator:
    def __init__(self, cme_par_dict, *args, **kwargs):
        v = np.array([v.value for v in cme_par_dict["v"]])
        self.hx = 0.02 * v

    def make_obs_op(self):
        return self.hx.reshape(1, len(self.hx))


@pytest.fixture(autouse=True)
def patch_obs(monkeypatch):
    monkeypatch.setattr("aux_pf.ObservationOperator", MockObservationOperator)


class TestAuxPF:
    # ============================================================
    # __INIT__ TEST
    # ============================================================
    def assert_cme_par_dict(self, cme_par_dict: CmeParEns) -> None:
        print(f"cme_par_dict: {cme_par_dict}")
        assert (
                cme_par_dict["surf_init_time"] == datetime.datetime(2008, 1, 1, 0, 0, 0)
        )
        assert cme_par_dict["t_init"] == [datetime.datetime(2008, 1, 1)] * 5

        exp_v = np.linspace(400, 600, 5)
        print(f"exp_v = {exp_v}")
        assert (
                (cme_par_dict["v"].value == pytest.approx(exp_v))
                and (cme_par_dict["v"].unit == u.km / u.s)
        )
        assert (
                (cme_par_dict["width"].value == np.ones(5) * 30)
                and (cme_par_dict["width"].unit == u.deg)
        )
        assert (
                (cme_par_dict["lon"].value == np.linspace(-20, 20, 5))
                and (cme_par_dict["lon"].unit == u.deg)
        )
        assert (
                (cme_par_dict["lat"].value == np.zeros(5))
                and (cme_par_dict["lat"].unit == u.deg)
        )
        assert (
                (cme_par_dict["thick"].value == np.zeros(5))
                and (cme_par_dict["thick"].unit == u.solRad)
        )
        assert (
                cme_par_dict["weight"] == pytest.approx(list(np.ones(5) / 5))
        )
        assert (
                cme_par_dict["log_weight"] == pytest.approx(list(np.log(np.ones(5) / 5)))
        )
        assert (
                cme_par_dict["likelihood"] == list(np.zeros(5))
        )
        assert cme_par_dict["n_members"] == 5

        return None


    def assert_true_cme_par_dict(self, true_cme_par_dict: CmeParEns) -> None:
        assert (true_cme_par_dict["n_members"] == 1)
        assert (true_cme_par_dict["surf_init_time"] == datetime.datetime(2008, 1, 1, 0, 0, 0))
        assert (true_cme_par_dict["t_init"] == datetime.datetime(2008, 1, 1, 1, 0, 0))
        assert (
            (true_cme_par_dict["v"].value == pytest.approx(495))
            and (true_cme_par_dict["v"].unit == u.km / u.s)
        )
        assert (
            (true_cme_par_dict["width"].value == pytest.approx(37.4))
            and (true_cme_par_dict["width"].unit == u.deg)
        )
        assert (
            (true_cme_par_dict["lon"].value == pytest.approx(0))
            and (true_cme_par_dict["lon"].unit == u.deg)
        )
        assert (
            (true_cme_par_dict["lat"].value == pytest.approx(0))
            and (true_cme_par_dict["lat"].unit == u.deg)
        )
        assert (
            (true_cme_par_dict["thick"].value == pytest.approx(0))
            and (true_cme_par_dict["thick"].unit == u.solRad)
        )

        return None

    def assert_aux_pf_init_inputs(
            self,
            get_aux_pf: AuxPF,
    ):
        pass

    def test_aux_pf_init(
            self,
            get_aux_pf:AuxPF,
    ):
        # Check cme_par_dict is initialised correctly
        self.assert_cme_par_dict(get_aux_pf.cme_par_dict)

        # Check true_cme_par_dict is initialised correctly
        self.assert_true_cme_par_dict(get_aux_pf.true_cme_par_dict)

        assert (get_aux_pf.obs == pytest.approx([10.0, 12.0]))
        assert (get_aux_pf.obs_cov == pytest.approx(0.1))
        assert (
            get_aux_pf.obs_lon.value == pytest.approx(0)
            and (get_aux_pf.obs_lon.unit == u.deg)
        )
        assert (get_aux_pf.obs_lat == [
            datetime.datetime(2008, 1, 1, 9, 0, 0),
            datetime.datetime(2008, 1, 1, 10, 0, 0)
        ])
        assert (get_aux_pf.pars_in_state == ["v", "width"])

        assert (
            (get_aux_pf.vr_in.value == pytest.approx(400 * np.ones(128)))
            and get_aux_pf.vr_out.unit == u.km / u.s
        )
        assert (
            (get_aux_pf.lon_start.value == pytest.approx(290))
            and (get_aux_pf.lon_start.unit == u.deg)
        )
        assert (
            (get_aux_pf.lon_stop.value == pytest.approx(380))
            and (get_aux_pf.lon_stop.unit == u.deg)
        )
        assert (
            (get_aux_pf.time_tolerance.value == pytest.approx(0.5))
            and (get_aux_pf.time_tolerance.unit == u.day)
        )
        assert (get_aux_pf.dt_scale == 20)
        assert (get_aux_pf.delta_aux_pf == pytest.approx(0.98))
        assert (
            get_aux_pf.r_min.value == pytest.approx(30)
            and (get_aux_pf.r_min.unit == u.solRad)
        )
        assert (get_aux_pf.cme_fixed_duration)
        assert (
            (get_aux_pf.fixed_duration.value == pytest.approx(12 * 60 *60))
            and (get_aux_pf.fixed_duration.unit == u.s)
        )
        assert (
                get_aux_pf.rng == pytest.approx(np.random_default_rng(42))
        )

        assert (
            get_aux_pf.weights == pytest.approx(list(np.ones(5) / 5))
        )
        assert (
            get_aux_pf.log_weights == pytest.approx(list(np.log(np.ones(5) / 5.0)))
        )

        assert (
            get_aux_pf.state_shrink_fact == pytest.approx(194.0/196.0)
        )
        return None


    # ============================================================
    # STATE COVARIANCE
    # ============================================================
    def test_get_state_cov(self, get_aux_pf):
        cov, scaled = get_aux_pf.get_state_cov()

        assert cov.shape[0] == get_aux_pf.state_vector.shape[1]
        assert np.allclose(scaled, cov * get_aux_pf.cov_shrink_fact2)

        return None


    # ============================================================
    # SHRINK FACTORS
    # ============================================================
    def test_get_shrink_fact(self, get_aux_pf):
        s, c = get_aux_pf.get_shrink_fact()

        assert 0 < s < 1
        assert np.isclose(c, 1 - s**2)

        return None


    # ============================================================
    # SHRINK SINGLE PARTICLE
    # ============================================================
    def test_shrink_par_single(self, get_aux_pf):
        x = get_aux_pf.state_vector[0]
        mean = np.mean(get_aux_pf.state_vector, axis=0)

        shrunk = get_aux_pf.shrink_par_single_ens(x, mean)

        expected = get_aux_pf.state_shrink_fact * x + (1 - get_aux_pf.state_shrink_fact) * mean

        assert np.allclose(shrunk, expected)

        return None


    # ============================================================
    # SHRINK FULL ENSEMBLE
    # ============================================================
    def test_shrink_par(self, get_aux_pf):
        shrunk = get_aux_pf.shrink_par()

        assert shrunk.shape == get_aux_pf.state_vector.shape

        # values move toward mean
        mean = np.mean(get_aux_pf.state_vector, axis=0)
        d_before = np.linalg.norm(get_aux_pf.state_vector - mean)
        d_after = np.linalg.norm(shrunk - mean)

        assert d_after <= d_before

        return None


    # ============================================================
    # PUT SHRUNK PARS
    # ============================================================
    def test_put_shrunk_pars(self, get_aux_pf):
        arr = get_aux_pf.put_shrunk_pars_in_cme_par_array()

        assert arr.shape[0] == get_aux_pf.n_members

        return None


    # ============================================================
    # DICTIONARY CONVERSION
    # ============================================================
    def test_make_state_vector_dictionary(self, get_aux_pf):
        d = get_aux_pf.make_state_vector_dictionary()

        assert isinstance(d, dict)
        assert "v" in d

        return None


    # ============================================================
    # SIM TIME
    # ============================================================
    def test_get_sim_time(self, get_aux_pf):
        sim_time = get_aux_pf.get_sim_time()

        assert sim_time.unit == u.day
        assert (sim_time.value > 0).all()

        return None


    # ============================================================
    # PARTICLE WEIGHTS
    # ============================================================
    def test_get_particle_weights_handles_inf(self, get_cme_par_dict):
        get_cme_par_dict["weight"][0] = np.inf

        pf = AuxPF(
            get_cme_par_dict,
            10.0, 1.0,
            0 * u.deg,
            datetime.datetime(2008, 1, 2)
        )

        assert pf.weights[0] == 0

        return None


    def test_get_particle_log_weights_handles_inf(self, get_cme_par_dict):
        get_cme_par_dict["log_weight"][0] = np.inf

        pf = AuxPF(
            get_cme_par_dict,
            10.0, 1.0,
            0 * u.deg,
            datetime.datetime(2008, 1, 2)
        )

        assert pf.log_weights[0] < -200  # ~log(1e-100)

        return None


    # ============================================================
    # OBSERVATION OPERATOR
    # ============================================================
    def test_get_observation_operator(self, get_aux_pf):
        hx = get_aux_pf.get_observation_operator(get_aux_pf.state_vector_shrunk_dict)

        assert hx.shape == (1, get_aux_pf.n_members)

        return None


    # ============================================================
    # LIKELIHOODS
    # ============================================================

    def test_likelihood_functions(self, get_aux_pf):
        L = get_aux_pf.calculate_likelihood_single_ens(10.0)
        logL = get_aux_pf.calculate_log_likelihood_single_ens(10.0)

        assert (L > 0).all()
        assert np.isfinite(logL).all()

        return None


    # ============================================================
    # AUX PROBABILITIES
    # ============================================================
    def test_get_aux_prob(self, get_aux_pf):
        likelihoods = list(range(1, get_aux_pf.n_members + 1))
        aux = get_aux_pf.get_aux_prob(likelihoods)

        assert len(aux) == get_aux_pf.n_members
        assert aux[-1] > aux[0]

        return None


    def test_get_aux_prob_log_weights(self, get_aux_pf):
        logL = np.zeros(get_aux_pf.n_members)

        logp = get_aux_pf.get_aux_prob_log_weights(logL)

        assert np.isclose(np.sum(np.exp(logp)), 1.0)

        return None


    # ============================================================
    # RESAMPLING
    # ============================================================
    def test_resample_pars(self, get_aux_pf):
        sample = get_aux_pf.resample_pars(0)

        assert len(sample) == get_aux_pf.n_pars_in_state

        return None


    # ============================================================
    # FULL AUXPF ALGORITHM
    # ============================================================
    def test_aux_pf_full(self, get_aux_pf):
        updated = get_aux_pf.aux_pf()

        weights = np.array(updated["weight"])

        # weights must be valid probabilities
        assert np.isfinite(weights).all()
        assert np.isclose(np.sum(weights), 1.0)

        # ensure variance exists (non-degenerate)
        assert np.std(weights) > 0

        return None


    # ============================================================
    # POSTERIOR IMPROVEMENT TEST
    # ============================================================
    def test_aux_pf_improves_estimate(self, get_aux_pf):
        def weighted_mean(cme_dict):
            v = np.array([v.value for v in cme_dict["v"]])
            w = np.array(cme_dict["weight"])
            return np.sum(v * w)

        prior_mean = weighted_mean(get_aux_pf.cme_par_dict)
        post = get_aux_pf.aux_pf()
        post_mean = weighted_mean(post)

        # posterior should shift (non-trivial update)
        assert prior_mean != post_mean

        return None

