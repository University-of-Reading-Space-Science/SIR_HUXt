"""
Auxiliary particle filter (Liu & West 2001) for one observation: takes the CME parameter ensemble,
runs it through the observation operator (SURF) and returns the updated particles and weights.
"""

import copy
import datetime
import warnings
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import astropy.units as u
from astropy.units import Quantity
from astropy.time import Time

from sir_resampling_log_weights import jacobian_log, ResampleParsLogWeights
from to_state_vector import ToStateVector
from from_state_vector import FromStateVector
from sir_observation_operator import ObservationOperator, ObsOpSettings, WorkerPool
from sir_likelihood import LikelihoodFunction
from cme_par_ens import CmeParEns
from state_transform import StateTransform

warnings.filterwarnings("ignore", message="invalid value encountered in divide")


@dataclass
class PFSettings:
    """
    Settings of the auxiliary particle filter (the defaults are those used by the DA routine unless
    Setup.get_pf_settings says otherwise).
    skip_update_if_any_member_nan: If True, an observation is not assimilated (the ensemble and its
        weights are left unchanged) when the CME of any ensemble member does not produce an
        elongation, either for the shrunk particles used to choose the resampling (first stage) or for
        the resampled particles (second stage). If False, only an observation where no member
        produces an elongation is skipped, and members without an elongation are given a negligible
        weight.
    adaptive_resampling: Adaptive resampling (Liu & Chen 1995; Kong, Liu & Wong 1994). If True, each
        observation first updates the weights of the current particles, w_i <- w_i * p(y | theta_i).
        Only if the effective sample size of these updated weights falls below
        resample_ess_fraction * N is the full auxiliary-PF step (Liu-West shrinkage, resampling and
        jitter) carried out. Resampling at every observation repeatedly removes particles, so the
        ensemble's family tree (and its spread) collapses even when the weights are fairly even. If
        False, the auxiliary-PF step is done at every observation.
        Cost: a step without resampling needs one ensemble of model runs (the current particles); a
        step with resampling needs three (current, shrunk and resampled particles).
    resample_ess_fraction: ESS threshold for resampling, as a fraction of the ensemble size
    """

    skip_update_if_any_member_nan: bool = True
    adaptive_resampling: bool = True
    resample_ess_fraction: float = 0.5


def n_members_without_elongation(obs_op: npt.NDArray[np.float64]) -> int:
    """Number of members (rows of the observation operator) with a non-finite elongation"""
    return int(np.sum(~np.all(np.isfinite(np.asarray(obs_op, dtype=float)), axis=1)))


class AuxPF:
    """
    Class to perform Auxiliary PF for a single observation
    """

    def __init__(
        self,
        use_model: str,
        cme_par_dict: CmeParEns,
        obs: list[float] | float,
        obs_cov: list[float] | float,
        obs_radius: Quantity[u.AU] | list[Quantity[u.AU]],
        obs_lon: Quantity[u.deg] | list[Quantity[u.deg]],
        obs_lat: Quantity[u.deg] | list[Quantity[u.deg]],
        obs_time: datetime.datetime,
        craft: str = None,
        pars_in_state: list[str] | None = None,
        vr_in: Quantity[u.km / u.s] = np.zeros(128) + 400 * u.km / u.s,
        vr_longs: Quantity[u.rad] = np.nan * u.rad,
        vr_lats: Quantity[u.rad] = np.nan * u.rad,
        n_lat_surf: int = 1,
        lon_start: Quantity[u.deg] = 290 * u.deg,
        lon_stop: Quantity[u.deg] = 380 * u.deg,
        time_tolerance: Quantity[u.day] = 0.5 * u.day,
        dt_scale: int | float = 20,
        delta_aux_pf: float = 0.98,
        r_min: Quantity[u.solRad] = 30 * u.solRad,
        cme_init_rad: Quantity[u.solRad] = 12 * u.solRad,
        cme_fixed_duration: bool = True,
        fixed_duration: Quantity[u.s] = 12 * 60 * 60 * u.s,
        plot_surf_output: bool = False,
        bias_term_bool: bool = False,
        bias_term_5rs: float = None,
        bias_term_21rs: float = None,
        use_parallel: bool = False,
        rng: np.random.Generator | None = None,
        state_transform: StateTransform | None = None,
        pf_settings: PFSettings | None = None,
        obs_op_settings: ObsOpSettings | None = None,
        worker_pool: WorkerPool | None = None,
    ) -> None:
        """
        Set up the auxiliary PF for one observation (the other parameters are described where they are used)
        :param state_transform: Optional StateTransform (state_transform.py) giving the coordinates in
            which the Liu-West shrinkage and jitter are done, and how the jitter covariance is
            regularised (correlation handling). None (default) = the natural state vector.
            Its parameter list must equal pars_in_state.
        :param pf_settings: Filter settings (PFSettings; None = the defaults)
        :param obs_op_settings: Observation-operator settings (ObsOpSettings; None = the defaults)
        :param worker_pool: Worker pool shared by the run for parallel model runs (None = a pool is
            started and closed for each ensemble of model runs)
        """
        self.pf_settings: PFSettings = (
            PFSettings() if pf_settings is None else pf_settings
        )
        self.obs_op_settings: ObsOpSettings | None = obs_op_settings
        self.worker_pool: WorkerPool | None = worker_pool
        self.use_model: str = use_model
        self.craft: str = craft
        self.cme_par_dict: CmeParEns = cme_par_dict

        self.obs: list[float] | float = obs
        self.obs_cov: list[float] | float = obs_cov
        self.obs_radius: Quantity[u.AU] | list[Quantity[u.AU]] = obs_radius
        self.obs_lon: Quantity[u.deg] | list[Quantity[u.deg]] = obs_lon
        self.obs_lat: Quantity[u.deg] | list[Quantity[u.deg]] = obs_lat
        self.obs_time: datetime.datetime = obs_time
        self.pars_in_state: list[str] = (
            ["v", "lon", "width"] if pars_in_state is None else list(pars_in_state)
        )
        self.vr_in: Quantity[u.km / u.s] = vr_in
        self.vr_longs: Quantity[u.rad] = vr_longs
        self.vr_lats: Quantity[u.rad] = vr_lats
        self.n_lat_surf: int = n_lat_surf
        self.lon_start: Quantity[u.deg] = lon_start
        self.lon_stop: Quantity[u.deg] = lon_stop
        self.time_tolerance: Quantity[u.day] = time_tolerance
        self.dt_scale: int | float = dt_scale
        self.delta_aux_pf: float = delta_aux_pf
        self.r_min: Quantity[u.solRad] = r_min
        self.cme_init_rad: Quantity[u.solRad] = cme_init_rad
        self.cme_fixed_duration: bool = cme_fixed_duration
        self.fixed_duration: Quantity[u.s] = fixed_duration
        self.plot_surf_output: bool = plot_surf_output
        self.use_parallel: bool = use_parallel
        self.rng: np.random.Generator = np.random.default_rng() if rng is None else rng

        self.surf_init_time: datetime.datetime = cme_par_dict["surf_init_time"]
        self.n_members: int = cme_par_dict["n_members"]

        state_vector_class: ToStateVector = ToStateVector(
            self.cme_par_dict, pars_in_state_vector=self.pars_in_state
        )
        self.cme_par_array: npt.NDArray[np.float64] = (
            state_vector_class.cme_par_dict_to_cme_par_array()
        )
        self.state_vector: npt.NDArray[np.float64] = (
            state_vector_class.cme_par_dict_to_state_vector()
        )

        # Working coordinates of the Liu-West step (the natural state vector unless a StateTransform
        #  with correlation handling is given; see state_transform.py)
        self.state_transform: StateTransform | None = state_transform
        if self.state_transform is not None:
            assert list(self.state_transform.pars) == list(
                self.pars_in_state
            ), f"StateTransform parameters {self.state_transform.pars} != pars_in_state {self.pars_in_state}"
            self.state_work: npt.NDArray[np.float64] = self.state_transform.forward(
                self.state_vector
            )
        else:
            self.state_work: npt.NDArray[np.float64] = self.state_vector

        self.weights: npt.NDArray[np.float64] = self.get_particle_weights()
        self.log_weights: npt.NDArray[np.float64] = self.get_particle_log_weights()

        # Liu-West shrinkage of the particles towards their weighted mean, and the jitter covariance
        self.cov_shrink_fact2: float
        self.state_shrink_fact: float
        self.state_shrink_fact, self.cov_shrink_fact2 = self.get_shrink_fact()
        self.cov_state: npt.NDArray[np.float64]
        self.scaled_cov_state: npt.NDArray[np.float64]
        self.cov_state, self.scaled_cov_state = self.get_state_cov()
        self.shrunk_pars: npt.NDArray[np.float64] = self.shrink_par()
        self.cme_par_array = self.put_shrunk_pars_in_cme_par_array()
        self.state_vector_shrunk_dict: CmeParEns = self.make_state_vector_dictionary()

        self.sim_time: Quantity = self.get_sim_time()

        # Linear elongation-bias correction of the observation operator (SURF/HUXt solvers only)
        self.bias_term_bool: bool = (
            bias_term_bool
            and (bias_term_5rs is not None)
            and (bias_term_21rs is not None)
        )
        self.bias_term_5rs: float = bias_term_5rs
        self.bias_term_21rs: float = bias_term_21rs

    def unnormalised_weights(self) -> npt.NDArray[np.float64]:
        """
        exp(log_weight - max log_weight): the particle weights up to a constant (subtracting the maximum
        first so that very negative log-weights cannot all underflow to zero)
        """
        log_w: npt.NDArray[np.float64] = np.asarray(self.log_weights, dtype=float)
        return np.exp(log_w - np.max(log_w))

    def get_state_cov(self) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Weighted covariance of the state vector (in the working coordinates), consistent with the
        weighted mean used in shrink_par (Liu & West, 2001)
        :return: cov_state: Covariance of the state vector
        :return: scaled_cov_state: Covariance of the state vector scaled by cov_shrink_fact2 (the jitter)
        """
        weights: npt.NDArray[np.float64] | None = self.unnormalised_weights()
        if (not np.all(np.isfinite(weights))) or (np.sum(weights) <= 0):
            weights = None
        cov_state: npt.NDArray[np.float64] = np.cov(
            self.state_work, rowvar=False, aweights=weights
        )
        if cov_state.ndim == 0:
            cov_state = np.array([[cov_state]])

        # Correlation handling: shrink the noisy ensemble covariance towards its diagonal
        if self.state_transform is not None:
            cov_state = self.state_transform.regularise_cov(cov_state)

        scaled_cov_state: npt.NDArray[np.float64] = cov_state * self.cov_shrink_fact2

        # Correlation handling: minimum jitter spread on every working axis
        if self.state_transform is not None:
            scaled_cov_state = self.state_transform.floor_jitter_cov(scaled_cov_state)

        return cov_state, scaled_cov_state

    def get_shrink_fact(self) -> tuple[float, float]:
        """
        Calculate the shrinkage factors for the state and covariance respectively
        :return: state_shrink_fact, cov_shrink_fact2: Shrinkage factors for the state and covariance respectively
        """
        state_shrink_fact: float = ((3 * self.delta_aux_pf) - 1) / (
            2 * self.delta_aux_pf
        )
        cov_shrink_fact: float = 1 - state_shrink_fact**2

        return state_shrink_fact, cov_shrink_fact

    def shrink_par_single_ens(
        self, state_1ens: npt.NDArray[np.float64], mean_state: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """
        Function to shrink a single ensemble member's state vector to the mean
        :param state_1ens: State vector for the ensemble member required
        :param mean_state: Mean of the state vector
        :return: shrunk_state: State vector shrunk to the mean
        """
        return (
            self.state_shrink_fact * state_1ens
            + (1 - self.state_shrink_fact) * mean_state
        )

    def shrink_par(self) -> npt.NDArray[np.float64]:
        """
        Liu-West shrinkage of the parameters towards the weighted mean, in the working coordinates
        (self.state_work; the natural state vector unless a StateTransform is used)
        :return: shrunk_pars (nEns, nPar) array containing parameters shrunk to mean (working coordinates)
        """
        weights: npt.NDArray[np.float64] = self.unnormalised_weights()
        mean_par: npt.NDArray[np.float64] = np.average(
            self.state_work, axis=0, weights=weights / weights.sum()
        )

        return np.array(
            [
                self.shrink_par_single_ens(self.state_work[m, :], mean_par)
                for m in range(self.n_members)
            ]
        )

    def put_shrunk_pars_in_cme_par_array(self) -> npt.NDArray[np.float64]:
        """
        Utilise the state_vector_to_cme_par_array function in the FromStateVector class
         to add the shrunken parameters to self.cme_par_array
        :return: updated_cme_par_array: self.cme_par_array updated with shrunken parameters
        """
        from_state_class: FromStateVector = FromStateVector(
            cme_par_array=self.cme_par_array,
            cme_par_dict=self.cme_par_dict,
            state_ens=self.to_natural(self.shrunk_pars),
            pars_in_state_vector=self.pars_in_state,
        )
        return from_state_class.state_vector_to_cme_par_array()

    def to_natural(
        self, state_work: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """
        Convert an ensemble from the working coordinates of the Liu-West step to the natural state vector
        :param state_work: (nEns, nPar) array in working coordinates
        :return: (nEns, nPar) array in natural units (as ToStateVector)
        """
        if self.state_transform is None:
            return state_work
        return self.state_transform.inverse(state_work)

    def make_state_vector_dictionary(self) -> CmeParEns:
        """
        Utilise the cme_par_array_to_cme_par_dict function in the FromStateVector class
         to create a dictionary containing the shrunken parameters for data assimilation using the
         Auxillary Particle Filter
        :return: state_vector_dict: State vector dictionary with shrunken parameters replacing original parameters
        """
        # cme_par_array_to_cme_par_dict fills the dictionary it is given in place, so pass copies to
        #  stop the shrunk parameters overwriting self.cme_par_dict
        from_state_class: FromStateVector = FromStateVector(
            cme_par_array=np.copy(self.cme_par_array),
            cme_par_dict=copy.deepcopy(self.cme_par_dict),
            state_ens=self.state_vector,
            pars_in_state_vector=self.pars_in_state,
        )
        return from_state_class.cme_par_array_to_cme_par_dict()

    def get_sim_time(self) -> Quantity[u.day]:
        """
        Get SURF simulation time for current observation time
        :return: sim_time: SURF simulation time
        """
        init_astro_time: Time = Time(
            self.surf_init_time, format="datetime", scale="utc"
        )
        obs_astro_time: Time = Time(self.obs_time, format="datetime", scale="utc")

        return (obs_astro_time - init_astro_time).to(u.day) + self.time_tolerance

    def get_particle_weights(self) -> npt.NDArray[np.float64]:
        """
        The particle weights, with non-finite weights set to zero (i.e. the particle is discarded)
        :return: weights: Particle filter weights
        """
        weights: npt.NDArray[np.float64] = np.array(
            self.cme_par_dict["weight"], dtype=float
        )
        weights[~np.isfinite(weights)] = 0
        return weights

    def get_particle_log_weights(self) -> npt.NDArray[np.float64]:
        """
        The logarithms of the particle weights, with non-finite values set to log(1e-100)
        (i.e. the particle is discarded)
        :return: log_weights: Logarithm of particle filter weights
        """
        log_weights: npt.NDArray[np.float64] = np.array(
            self.cme_par_dict["log_weight"], dtype=float
        )
        log_weights[~np.isfinite(log_weights)] = np.log(1e-100)
        return log_weights

    def get_observation_operator(
        self, state_dict_in: CmeParEns
    ) -> npt.NDArray[np.float64]:
        """
        Function to get the observation operator
        :return: hx: Observation operator for self.obs_time
        """
        obs_op_class: ObservationOperator = ObservationOperator(
            use_model=self.use_model,
            cme_par_dict=state_dict_in,
            obs=self.obs,
            obs_radius=self.obs_radius,
            obs_lon=self.obs_lon,
            obs_lat=self.obs_lat,
            obs_time_in_datetime=self.obs_time,
            obs_cov=self.obs_cov,
            craft=self.craft,
            surf_init_time=self.surf_init_time,
            vr_in=self.vr_in,
            vr_longs=self.vr_longs,
            vr_lats=self.vr_lats,
            n_lat_surf=self.n_lat_surf,
            lon_start=self.lon_start,
            lon_stop=self.lon_stop,
            sim_time=self.sim_time,
            dt_scale=self.dt_scale,
            r_min=self.r_min,
            cme_init_rad=self.cme_init_rad,
            cme_fixed_duration=self.cme_fixed_duration,
            fixed_duration=self.fixed_duration,
            plot_surf_output=self.plot_surf_output,
            bias_term_bool=self.bias_term_bool,
            bias_term_5rs=self.bias_term_5rs,
            bias_term_21rs=self.bias_term_21rs,
            use_parallel=self.use_parallel,
            settings=self.obs_op_settings,
            worker_pool=self.worker_pool,
        )
        return obs_op_class.make_obs_op()

    def calculate_likelihood_single_ens(self, hx: npt.NDArray[np.float64]) -> float:
        """
        Function to calculate the likelihood for a single ensemble member
        :param hx: Observation operator for a single ensemble member
        :return: likelihood: Likelihood value for a single ensemble member
        """
        return LikelihoodFunction(
            obs=self.obs, obs_cov=self.obs_cov, hx=hx
        ).make_likelihood()

    def calculate_log_likelihood_single_ens(self, hx: npt.NDArray[np.float64]) -> float:
        """
        Function to calculate the log likelihood for a single ensemble member
        :param hx: Observation operator for a single ensemble member
        :return: log_likelihood: Log likelihood value for a single ensemble member
        """
        return LikelihoodFunction(
            obs=self.obs, obs_cov=self.obs_cov, hx=hx
        ).make_log_likelihood()

    def get_aux_prob(
        self, likelihoods: list[float] | npt.NDArray[np.float64]
    ) -> list[float]:
        """
        Function to get the auxillary probabilities from the weights and the likelihoods
        :param likelihoods: Likelihood function
        :return: aux_prob: Auxillary probabilities
        """
        return [self.weights[i] * likelihoods[i] for i in range(self.n_members)]

    def get_aux_prob_log_weights(
        self, log_likelihood: list[float] | npt.NDArray[np.float64]
    ) -> list[float]:
        """
        Function to get the auxillary probabilities from the log weights
         and the log likelihoods
        :param log_likelihood: Log likelihoods
        :return: log_aux_prob_norm: Normalised logarithm of auxillary probabilities
        """
        log_aux_prob_unnorm: list[float] = [
            self.log_weights[i] + log_likelihood[i] for i in range(self.n_members)
        ]
        jacob_log: float = jacobian_log(log_aux_prob_unnorm)

        return [log_aux_prob_unnorm[i] - jacob_log for i in range(self.n_members)]

    def resample_pars(self, ind_req: int) -> npt.NDArray[np.float64]:
        """Jittered draw N(shrunk_pars[ind_req], h^2 cov_state) for one resampled particle"""
        return self.rng.multivariate_normal(
            mean=self.shrunk_pars[ind_req, :], cov=self.scaled_cov_state
        )

    def log_ess_log_weights(
        self, log_weight_in: list[float] | npt.NDArray[np.float64]
    ) -> float:
        """
        Function to calculate the Effective Sample Size of Aux-PF with logarithmic weights
        :return: log_ess: Logarithm of Effective Sample Size
        """
        # np.asarray: with a python list, `2 * log_weight_in` would repeat the list rather than
        #  double the log-weights
        log_weight_in = np.asarray(log_weight_in, dtype=float)

        return 2 * jacobian_log(log_weight_in) - jacobian_log(2 * log_weight_in)

    def calc_ess_log_weights(
        self, log_weight_in: list[float] | npt.NDArray[np.float64]
    ) -> float:
        """
        Function to calculate the Effective Sample Size with logarithmic weights
        :return: ess: Effective Sample Size
        """
        log_weight_in = np.asarray(log_weight_in, dtype=float)
        log_weight_in = log_weight_in - jacobian_log(log_weight_in)

        return np.exp(self.log_ess_log_weights(log_weight_in=log_weight_in))

    def skip_update(self, reason: str, note: str = "") -> CmeParEns:
        """
        Leave this observation unassimilated: record why, warn, and return the ensemble unchanged
        :param reason: Why the observation was not assimilated (stored in self.skip_reason)
        :param note: Optional text added to the warning
        :return: self.cme_par_dict
        """
        self.skip_reason = reason
        warnings.warn(
            f"Observation at {self.obs_time} not assimilated: {reason}{note}."
        )
        self.elon_posterior = np.full(self.n_members, np.nan)
        self.parent_index = np.arange(self.n_members, dtype=int)
        self.update_skipped: bool = True
        self.resampled = False
        return self.cme_par_dict

    def aux_pf(self) -> CmeParEns:
        """
        Function to perform the auxillary particle filter routine
        :return: cme_par_dict: Updated cme_par_dict with parameters changed by Auxillary PF
        """
        # Effective sample size of the prior weights (diagnostic only). The weights are carried through
        #  unchanged: resetting them without resampling would throw away the information they carry.
        ess_prior: float = self.calc_ess_log_weights(self.log_weights)
        print(f"ess_prior = {ess_prior}")

        # Keep the ensemble as it is now, so the update can be discarded if a resampled member's CME
        #  does not produce an elongation (see PFSettings.skip_update_if_any_member_nan)
        cme_par_dict_in: CmeParEns = copy.deepcopy(self.cme_par_dict)
        cme_par_array_in: npt.NDArray[np.float64] = np.copy(self.cme_par_array)
        self.skip_reason: str | None = None
        self.resampled: bool = False
        self.ess_updated: float = np.nan

        if self.pf_settings.adaptive_resampling:
            # Elongations and likelihoods of the current (unshrunk, unjittered) particles
            obs_op_current: npt.NDArray[np.float64] = self.get_observation_operator(
                self.cme_par_dict
            )
            elon_current: npt.NDArray[np.float64] = np.asarray(
                obs_op_current, dtype=float
            )[:, 0].copy()
            log_likelihood_current: npt.NDArray[np.float64] = np.array(
                [
                    self.calculate_log_likelihood_single_ens(obs_op_current[i, :])
                    for i in range(self.n_members)
                ],
                dtype=float,
            )

            # Skip this observation (as below) if a member produced no elongation, or none did
            n_no_elon: int = n_members_without_elongation(obs_op_current)
            if (
                self.pf_settings.skip_update_if_any_member_nan and n_no_elon > 0
            ) or not np.any(np.isfinite(log_likelihood_current)):
                self.elon_first_stage = elon_current
                return self.skip_update(
                    f"{n_no_elon}/{self.n_members} members produced no elongation"
                )

            # Updated weights w_i * p(y | theta_i) and their effective sample size
            log_weights_upd: npt.NDArray[np.float64] = (
                np.asarray(self.log_weights, dtype=float) + log_likelihood_current
            )
            log_weights_upd = np.where(
                np.isfinite(log_weights_upd), log_weights_upd, -1e31
            )
            log_weights_upd = log_weights_upd - jacobian_log(log_weights_upd)
            self.ess_updated = float(self.calc_ess_log_weights(log_weights_upd))
            print(
                f"ess_updated = {self.ess_updated:.2f} (resampling threshold "
                f"{self.pf_settings.resample_ess_fraction * self.n_members:.1f})"
            )

            if (
                self.ess_updated
                >= self.pf_settings.resample_ess_fraction * self.n_members
            ):
                # Enough effective particles: keep the particles, carry the updated weights forward
                self.cme_par_dict["log_weight"] = list(log_weights_upd)
                self.cme_par_dict["weight"] = list(np.exp(log_weights_upd))
                self.elon_first_stage = elon_current
                self.elon_posterior = elon_current.copy()
                self.parent_index = np.arange(self.n_members, dtype=int)
                self.update_skipped = False
                print("ESS above threshold: weights updated, no resampling")
                return self.cme_par_dict

            print("ESS below threshold: auxiliary-PF step with resampling")
        self.resampled = True

        # First stage: likelihoods of the shrunk particles. Their modelled elongations (indexed by the
        #  pre-resampling particle) are kept for the routine to write out.
        obs_op_shrunk: npt.NDArray[np.float64] = self.get_observation_operator(
            self.state_vector_shrunk_dict
        )
        self.elon_first_stage: npt.NDArray[np.float64] = np.asarray(
            obs_op_shrunk, dtype=float
        )[:, 0].copy()
        log_likelihood_shrunk_ens: list[float] = [
            self.calculate_log_likelihood_single_ens(obs_op_shrunk[i, :])
            for i in range(self.n_members)
        ]

        # Skip the update, returning the ensemble and its weights unchanged, if
        #  - any member's CME produced no elongation and PFSettings.skip_update_if_any_member_nan is set, or
        #  - no member produced a finite likelihood (e.g. every modelled CME is outside the imager's
        #    field of view): the observation then carries no information, and resampling with the
        #    resulting uniform weights would only destroy ensemble diversity.
        n_no_elon = n_members_without_elongation(obs_op_shrunk)
        no_finite_lik: bool = not np.any(
            np.isfinite(np.asarray(log_likelihood_shrunk_ens, dtype=float))
        )
        if (
            self.pf_settings.skip_update_if_any_member_nan and n_no_elon > 0
        ) or no_finite_lik:
            return self.skip_update(
                f"{n_no_elon}/{self.n_members} (shrunk) members produced no elongation"
            )
        self.update_skipped = False

        # Choose the particles to resample from the auxiliary probabilities
        log_aux_prob: list[float] = [
            x if np.isfinite(x) else -1e31
            for x in self.get_aux_prob_log_weights(log_likelihood_shrunk_ens)
        ]
        resample_ind: list[int] = ResampleParsLogWeights(
            log_weights=log_aux_prob, n_ensemble=self.n_members, rng=self.rng
        ).systematic_resampling_log_weights()

        # Jittered draws N(shrunk_par, h^2 * cov_state) in the working coordinates, converted to the
        #  natural state vector
        resample_pars: npt.NDArray[np.float64] = self.to_natural(
            np.array([self.resample_pars(ind_req) for ind_req in resample_ind])
        )

        # Each new particle inherits ALL of its parent's parameters. Only the parameters in the state
        #  vector are overwritten with the jittered draws below; the others (e.g. t_init, lat, thick when
        #  they are not in the state) come from the parent particle resample_ind[i].
        from_state_class: FromStateVector = FromStateVector(
            cme_par_array=np.copy(self.cme_par_array[resample_ind, :]),
            cme_par_dict=self.cme_par_dict,
            state_ens=resample_pars,
            pars_in_state_vector=self.pars_in_state,
        )
        self.cme_par_array = from_state_class.state_vector_to_cme_par_array()
        self.cme_par_dict = from_state_class.cme_par_array_to_cme_par_dict()

        # Second stage: elongations of the resampled (posterior) particles, and which prior particle
        #  each came from
        obs_op_post: npt.NDArray[np.float64] = self.get_observation_operator(
            self.cme_par_dict
        )
        self.elon_posterior: npt.NDArray[np.float64] = np.asarray(
            obs_op_post, dtype=float
        )[:, 0].copy()
        self.parent_index: npt.NDArray[np.int_] = np.asarray(resample_ind, dtype=int)

        # If a resampled member's CME produced no elongation, discard the update: restore the
        #  ensemble and weights from before this observation (see PFSettings.skip_update_if_any_member_nan)
        n_no_elon_post: int = n_members_without_elongation(obs_op_post)
        if self.pf_settings.skip_update_if_any_member_nan and n_no_elon_post > 0:
            self.cme_par_dict = cme_par_dict_in
            self.cme_par_array = cme_par_array_in
            return self.skip_update(
                f"{n_no_elon_post}/{self.n_members} resampled members produced no elongation",
                note="; the update has been discarded",
            )

        # Second-stage weights p(y | theta_i) / p(y | mu_parent(i)). A member whose posterior likelihood
        #  is NaN (model CME not visible) is given a negligible weight, so that no NaN weight reaches the
        #  output or the ESS/weighted statistics.
        log_likelihood_post: list[float] = [
            self.calculate_log_likelihood_single_ens(obs_op_post[i, :])
            for i in range(self.n_members)
        ]
        log_weights_post: list[float] = [
            log_likelihood_post[i] - log_likelihood_shrunk_ens[resample_ind[i]]
            for i in range(len(resample_ind))
        ]
        log_weights_post = [x if np.isfinite(x) else -1e31 for x in log_weights_post]

        log_weights_post = ResampleParsLogWeights(
            log_weights=log_weights_post, n_ensemble=self.n_members
        ).norm_log_weights()
        self.cme_par_dict["weight"] = [np.exp(w) for w in log_weights_post]
        self.cme_par_dict["log_weight"] = log_weights_post

        print(f"ess_post_log = {self.calc_ess_log_weights(log_weights_post)}")

        return self.cme_par_dict
