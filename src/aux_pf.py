# @package sir
# This module will take inputs from SURF and perform an SIR to
#  generate an updated set of weights and particles
import copy
import numpy as np
import numpy.typing as npt
import datetime

import astropy.units as u
from astropy.units import Quantity
from astropy.time import Time

from sir_resampling_log_weights import jacobian_log, ResampleParsLogWeights
from to_state_vector import ToStateVector
from from_state_vector import FromStateVector

from sir_observation_operator import ObservationOperator
from sir_likelihood import LikelihoodFunction
import matplotlib.pyplot as plt
from cme_par_ens import CmeParEns
import seaborn as sns
import colorcet as cc
import pytest
from cme_par_dict_structure import required_dict_keys
from state_transform import StateTransform

import warnings
warnings.filterwarnings(
    "ignore", message="invalid value encountered in divide"
)

# If True, an observation is not assimilated (the ensemble and its weights are left unchanged)
#  when the CME of any ensemble member does not produce an elongation, either for the shrunk
#  particles used to choose the resampling (first stage) or for the resampled particles (second
#  stage). If False, only an observation where no member produces an elongation is skipped, and
#  members without an elongation are given a negligible weight.
SKIP_UPDATE_IF_ANY_MEMBER_NAN: bool = True

# Adaptive resampling (Liu & Chen 1995; Kong, Liu & Wong 1994). If True, each observation first
#  updates the weights of the current particles, w_i <- w_i * p(y | theta_i). Only if the effective
#  sample size of these updated weights falls below RESAMPLE_ESS_FRACTION * N is the full
#  auxiliary-PF step (Liu-West shrinkage, resampling and jitter) carried out. Resampling at every
#  observation repeatedly removes particles, so the ensemble's family tree (and its spread) collapses
#  even when the weights are fairly even. If False, the auxiliary-PF step is done at every observation.
#  Cost: a step without resampling needs one ensemble of model runs (the current particles); a step
#  with resampling needs three (current, shrunk and resampled particles).
ADAPTIVE_RESAMPLING: bool = True
RESAMPLE_ESS_FRACTION: float = 0.5

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
            true_cme_par_dict: CmeParEns = None,
            infl_fact: float = 1,
            craft: str=None,
            pars_in_state: list[str] | None = None,
            vr_in: npt.NDArray[Quantity[u.km/u.s]]=np.zeros(128) + 400 * u.km / u.s,
            vr_longs: npt.NDArray[Quantity[u.rad]]=np.nan * u.rad,
            vr_lats: npt.NDArray[Quantity[u.rad]]=np.nan * u.rad,
            n_lat_surf: int = 1,
            lon_start: Quantity[u.deg]=290 * u.deg,
            lon_stop: Quantity[u.deg]=380 * u.deg,
            time_tolerance: Quantity[u.day]=0.5 * u.day,
            dt_scale: int | float=20,
            delta_aux_pf: float=0.98,
            r_min: Quantity[u.solRad]=30 * u.solRad,
            cme_init_rad: Quantity[u.solRad]=12 * u.solRad,
            cme_fixed_duration: bool = True,
            fixed_duration: Quantity[u.s] = 12 * 60 * 60 * u.s,
            plot_surf_output: bool = False,
            bias_term_bool: bool = False,
            bias_term_5rs: float = None,
            bias_term_21rs: float = None,
            use_parallel: bool = False,
            rng=None,
            state_transform: StateTransform | None = None
    ):
        """
        Set up the auxiliary PF for one observation (the other parameters are described where they are used)
        :param state_transform: Optional StateTransform (state_transform.py) giving the coordinates in
            which the Liu-West shrinkage and jitter are done, and how the jitter covariance is
            regularised (correlation handling). None (default) = the natural state vector, as before.
            Its parameter list must equal pars_in_state.
        """
        self.use_model = use_model
        self.craft = craft
        self.cme_par_dict:CmeParEns = cme_par_dict

        self.obs = obs
        self.obs_cov = obs_cov
        self.obs_radius = obs_radius
        self.obs_lon = obs_lon
        self.obs_lat = obs_lat
        self.obs_time = obs_time
        self.true_cme_par_dict: CmeParEns = true_cme_par_dict
        self.infl_fact = infl_fact
        # (avoid a mutable default argument)
        self.pars_in_state = ["v", "lon", "width"] if pars_in_state is None else list(pars_in_state)
        self.vr_in = vr_in
        self.vr_longs = vr_longs
        self.vr_lats = vr_lats
        self.n_lat_surf = n_lat_surf
        self.lon_start = lon_start
        self.lon_stop = lon_stop
        self.time_tolerance = time_tolerance
        self.dt_scale = dt_scale
        self.delta_aux_pf = delta_aux_pf
        self.r_min = r_min
        self.cme_init_rad = cme_init_rad
        self.cme_fixed_duration = cme_fixed_duration
        self.fixed_duration = fixed_duration
        self.plot_surf_output = plot_surf_output
        self.use_parallel = use_parallel

        if rng is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = rng

        self.n_pars_in_state = len(self.pars_in_state)
        self.surf_init_time = cme_par_dict["surf_init_time"]
        self.n_members = cme_par_dict["n_members"]

        state_vector_class = ToStateVector(
            self.cme_par_dict,
            pars_in_state_vector=self.pars_in_state
        )
        self.cme_par_array = state_vector_class.cme_par_dict_to_cme_par_array()
        self.state_vector = state_vector_class.cme_par_dict_to_state_vector()

        # Working coordinates of the Liu-West step (the natural state vector unless a StateTransform
        #  with correlation handling is given; see state_transform.py)
        self.state_transform: StateTransform | None = state_transform
        if self.state_transform is not None:
            assert list(self.state_transform.pars) == list(self.pars_in_state), (
                f"StateTransform parameters {self.state_transform.pars} != pars_in_state {self.pars_in_state}"
            )
            self.state_work: npt.NDArray[float] = self.state_transform.forward(self.state_vector)
        else:
            self.state_work: npt.NDArray[float] = self.state_vector

        # Extract weights of particles
        self.weights: list[float] = self.get_particle_weights()
        self.log_weights: list[float] = self.get_particle_log_weights()

        # Calculate shrinkage parameter for state and covariance matrix from delta_aux_pf
        self.state_shrink_fact, self.cov_shrink_fact2 = self.get_shrink_fact()

        # Calculate the covariance of the ensemble of parameters
        self.cov_state, self.scaled_cov_state = self.get_state_cov()

        # Calculate shrunk_parameters
        # Shrink the parameters to the mean
        self.shrunk_pars = self.shrink_par()

        # Put shrunk_parameters back into cme_par_array
        self.cme_par_array = self.put_shrunk_pars_in_cme_par_array()

        # Create a state_vector dictionary containing all shrunken parameters
        self.state_vector_shrunk_dict = self.make_state_vector_dictionary()

        # Get SURF simulation time for current observation time
        self.sim_time = self.get_sim_time()

        # Get bias_term_bool to determine whether to include a bias in the observation operator
        self.bias_term_bool = bias_term_bool
        self.bias_term_5rs = bias_term_5rs
        self.bias_term_21rs = bias_term_21rs

        # Ensure all bias terms exist, if not, set bias_term_bool to False
        if (self.bias_term_5rs is None) or (self.bias_term_21rs is None):
            self.bias_term_bool = False


    def get_state_cov(self) -> tuple[npt.NDArray[float], npt.NDArray[float]]:
        """
        Function to calculate the covariance of the state vector
          (i.e. the covariance in the parameters obtained from the ensemble)
        :return: cov_state: Covariance of the state vector
        :return: scaled_cov_state: Covariance of the state vector scaled by cov_shrink_fact2
        """
        # Weighted covariance, consistent with the weighted mean used in shrink_par (Liu & West, 2001).
        #  (Previously the unweighted covariance was used.)
        # Subtract the maximum before exponentiating so the weights cannot all underflow to zero
        #  (which previously made this silently fall back to the unweighted covariance)
        log_w = np.asarray(self.log_weights, dtype=float)
        weights = np.exp(log_w - np.max(log_w))
        if (not np.all(np.isfinite(weights))) or (np.sum(weights) <= 0):
            weights = None
        # (in the working coordinates: the natural state vector unless a StateTransform is used)
        cov_state: float | npt.NDArray = np.cov(self.state_work, rowvar=False, aweights=weights)
        if cov_state.ndim == 0:
            cov_state: npt.NDArray[float] = np.array([[cov_state]])

        # Correlation handling: shrink the noisy ensemble covariance towards its diagonal
        if self.state_transform is not None:
            cov_state = self.state_transform.regularise_cov(cov_state)

        scaled_cov_state = cov_state * self.cov_shrink_fact2

        # Correlation handling: minimum jitter spread on every working axis
        if self.state_transform is not None:
            scaled_cov_state = self.state_transform.floor_jitter_cov(scaled_cov_state)

        return cov_state, scaled_cov_state


    def get_shrink_fact(self) -> tuple[float, float]:
        """
        Calculate the shrinkage factors for the state and covariance respectively
        :return: state_shrink_fact, cov_shrink_fact2: Shrinkage factors for the state and covariance respectively
        """
        state_shrink_fact: float = ((3 * self.delta_aux_pf) - 1) / (2 * self.delta_aux_pf)
        cov_shrink_fact: float = 1 - state_shrink_fact ** 2

        return state_shrink_fact, cov_shrink_fact


    def shrink_par_single_ens(self, state_1ens, mean_state) -> list[float]:
        """
        Function to shrink a single ensemble member's state vector to the mean
        :param state_1ens: State vector for the ensemble member required
        :param mean_state: Mean of the state vector
        :return: shrunk_state: State vector shrunk to the mean
        """
        term1 = self.state_shrink_fact * state_1ens
        term2 = (1 - self.state_shrink_fact) * mean_state

        shrunk_state = term1 + term2

        return shrunk_state


    def shrink_par(self) -> npt.NDArray[float]:
        """
        Liu-West shrinkage of the parameters towards the weighted mean, in the working coordinates
        (self.state_work; the natural state vector unless a StateTransform is used)
        :return: shrunk_pars (nEns, nPar) array containing parameters shrunk to mean (working coordinates)
        """
        if self.log_weights is None:
            # If no weights are provided, calculate an arithmetic mean
            mean_par: npt.NDArray[float] = np.mean(self.state_work, axis=0)
        else:
            # If weights are provided, calculate weighted mean
            # Transform from log-space and normalise weights
            # (subtract the maximum first so that very negative log-weights cannot underflow to 0/0)
            log_w = np.asarray(self.log_weights, dtype=float)
            weights: npt.NDArray[float] = np.exp(log_w - np.max(log_w))
            weights: npt.NDArray[float] = weights / weights.sum()

            # Compute weighted mean parameter
            mean_par: npt.NDArray[float] = np.average(self.state_work, axis=0, weights=weights)

        # Shrink towards the mean
        shrunk_pars = np.array([
            self.shrink_par_single_ens(self.state_work[m, :], mean_par)
            for m in range(self.n_members)
        ])

        return shrunk_pars


    def put_shrunk_pars_in_cme_par_array(self):
        """
        Utilise the state_vector_to_cme_par_array function in the FromStateVector class
         to add the shrunken parameters to self.cme_par_array
        :return: updated_cme_par_array: self.cme_par_array updated with shrunken parameters
        """
        from_state_class = FromStateVector(
            cme_par_array=self.cme_par_array,
            cme_par_dict=self.cme_par_dict,
            state_ens=self.to_natural(self.shrunk_pars),
            pars_in_state_vector=self.pars_in_state
        )
        updated_cme_par_array = from_state_class.state_vector_to_cme_par_array()

        return updated_cme_par_array


    def to_natural(self, state_work: npt.NDArray[float]) -> npt.NDArray[float]:
        """
        Convert an ensemble from the working coordinates of the Liu-West step to the natural state vector
        :param state_work: (nEns, nPar) array in working coordinates
        :return: (nEns, nPar) array in natural units (as ToStateVector)
        """
        if self.state_transform is None:
            return state_work
        return self.state_transform.inverse(state_work)


    def make_state_vector_dictionary(self):
        """
        Utilise the cme_par_array_to_cme_par_dict function in the FromStateVector class
         to create a dictionary containing the shrunken parameters for data assimilation using the
         Auxillary Particle Filter
        :return: state_vector_dict: State vector dictionary with shrunken parameters replacing original parameters
        """
        from_state_class = FromStateVector(
            cme_par_array=np.copy(self.cme_par_array),
            cme_par_dict=copy.deepcopy(self.cme_par_dict),
            state_ens=self.state_vector,
            pars_in_state_vector=self.pars_in_state
        )
        state_vector_dict = from_state_class.cme_par_array_to_cme_par_dict()

        # NB: cme_par_array_to_cme_par_dict fills the dictionary it was given in place,
        #  so the copies above stop the shrunk parameters overwriting self.cme_par_dict

        return state_vector_dict


    def get_sim_time(self) -> Quantity[u.day]:
        """
        Get SURF simulation time for current observation time
        :return: sim_time: SURF simulation time
        """
        init_astro_time = Time(self.surf_init_time, format="datetime", scale="utc")
        obs_astro_time = Time(self.obs_time, format="datetime", scale="utc")

        sim_time = (obs_astro_time - init_astro_time).to(u.day) + self.time_tolerance

        return sim_time


    def get_particle_weights(self) -> list[float]:
        """
        Function to get the particle weights and remove any infinite weights and set them equal to zero
        so that they're ignored by the particle filter
        :return: weights: Particle filter weights
        """
        ## Function to convert the CME parameters into a state vector
        # Extract weights of particles
        # np.array so boolean indexing works (cme_par_dict stores a python list)
        weights: npt.NDArray[float] = np.array(self.cme_par_dict["weight"], dtype=float)

        # If there are non-finite weights, set weight to zero (i.e. discard particle)
        weight_cond = ~np.isfinite(weights)
        if np.sum(weight_cond) > 0:
            weights[weight_cond] = 0

        return weights


    def get_particle_log_weights(self) -> list[float]:
        """
        Function to get the logarithm of the particle weights and remove any infinite weights
        and set them equal to a value close to zero (log(1e-100))
        so that they're ignored by the particle filter
        :return: log_weights: Logarithm of particle filter weights
        """
        # Extract weights of particles
        # np.array so boolean indexing works (cme_par_dict stores a python list)
        log_weights: npt.NDArray[float] = np.array(self.cme_par_dict["log_weight"], dtype=float)

        # If there are non-finite log_weights, set log_weight to log(1e-100)
        #  (i.e. discard particle)
        log_weight_cond = ~np.isfinite(log_weights)
        if np.sum(log_weight_cond) > 0:
            log_weights[log_weight_cond] = np.log(1e-100)

        return log_weights


    def get_observation_operator(self, state_dict_in):
        """
        Function to get the observation operator
        :return: hx: Observation operator for self.obs_time
        """
        obs_op_class = ObservationOperator(
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
        )

        hx = obs_op_class.make_obs_op()

        return hx


    def calculate_diff_between_obs_and_obs_op(self, hx):
        """
        Function to calculate the difference between the observation operator and the observation
        :param hx: Observation operator
        :return: obs_diff: difference between the observation operator and the observation
        """
        obs_diff = self.obs - hx

        return obs_diff


    def calculate_likelihood_single_ens(self, hx):
        """
        Function to calculate the likelihood for a single ensemble member
        :param hx: Observation operator for a single ensemble member
        :return: likelihood: Likelihood valuefor a single ensemble member
        """
        likelihood_class = LikelihoodFunction(
            obs=self.obs,
            obs_cov=self.obs_cov,
            hx=hx
        )
        likelihood = likelihood_class.make_likelihood()

        return likelihood


    def calculate_log_likelihood_single_ens(self, hx):
        """
        Function to calculate the likelihood for a single ensemble member
        :param hx: Observation operator for a single ensemble member
        :return: likelihood: Likelihood valuefor a single ensemble member
        """
        likelihood_class = LikelihoodFunction(
            obs=self.obs,
            obs_cov=self.obs_cov,
            hx=hx
        )
        log_likelihood = likelihood_class.make_log_likelihood()

        return log_likelihood


    def get_aux_prob(self, likelihoods):
        """
        Function to get the auxillary probabilities from the weights and the likelihoods
        :param likelihoods: Likelihood function
        :return: aux_prob: Auxillary probabilities
        """
        aux_prob = [
            self.weights[i] * likelihoods[i] for i in range(self.n_members)
        ]
        return aux_prob


    def get_aux_prob_log_weights(self, log_likelihood):
        """
        Function to get the auxillary probabilities from the log weights
         and the log likelihoods
        :param log_likelihood: Log likelihoods
        :return: log_aux_prob_norm: Normalised logarithm of auxillary probabilities
        """
        # Calculate logarithm of auxillary probability
        log_aux_prob_unnorm = [
            self.log_weights[i] + log_likelihood[i] for i in range(self.n_members)
        ]
        jacob_log = jacobian_log(log_aux_prob_unnorm)

        # Normalised logarithm of auxillary proabilities
        log_aux_prob_norm = [
            log_aux_prob_unnorm[i] - jacob_log
            for i in range(self.n_members)
        ]

        return log_aux_prob_norm


    def resample_pars(self, ind_req: int) -> list[float]:
        resampled_par = self.rng.multivariate_normal(
            mean=self.shrunk_pars[ind_req, :], cov=self.scaled_cov_state
        )

        return resampled_par


    def calc_ess_lin_weights(self, weight_in):
        """
        Function to calculate the Effective Sample Size with linear weights
        :return: ess: Effective Sample Size
        """
        numer = np.sum(weight_in) ** 2
        denom = np.sum([w * w for w in weight_in])

        ess = numer / denom

        return ess


    def log_ess_log_weights(self, log_weight_in):
        """
        Function to calculate the Effective Sample Size of Aux-PF with logarithmic weights
        :return: log_ess: Logarithm of Effective Sample Size
        """
        # np.asarray: with a python list, `2 * log_weight_in` would repeat the list rather than
        #  double the log-weights, giving a wrong ESS
        log_weight_in = np.asarray(log_weight_in, dtype=float)

        # Calculate the logarithm of the Effective Sample Size
        term1 = 2 * jacobian_log(log_weight_in)
        term2 = jacobian_log(2 * log_weight_in)

        log_ess = term1 - term2

        return log_ess


    def calc_ess_log_weights(self, log_weight_in):
        """
        Function to calculate the Effective Sample Size with logarithmic weights
        :return: ess: Effective Sample Size
        """
        # Normalise weights (np.asarray: `2 * list` would repeat the list rather than double it)
        log_weight_in = np.asarray(log_weight_in, dtype=float)
        log_weight_in = log_weight_in - jacobian_log(log_weight_in)

        # Calculate the effective sample size
        log_ess = self.log_ess_log_weights(log_weight_in=log_weight_in)
        ess = np.exp(log_ess)

        return ess


    def aux_pf(self):
        """
        Function to perform the auxillary particle filter routine
        :return: cme_par_dict: Updated cme_par_dict with parameters changed by Auxillary PF
        """
        # Calculate the effective sample size from the prior weights (diagnostic only).
        #  NB: the weights were previously reset to 1/N here whenever ESS < N/2. Resetting weights
        #  *without* resampling throws away the information they carry (low-weight particles become
        #  as likely as high-weight ones). The auxiliary PF already resamples every step using the
        #  first-stage weights w_{t-1} * p(y | mu), which is what controls degeneracy, so the prior
        #  weights must be carried through unchanged. (The reset also happened after the shrinkage
        #  mean and covariance had been computed with the old weights, so the two were inconsistent.)
        ess_prior = self.calc_ess_log_weights(self.log_weights)
        print(f"ess_prior = {ess_prior}")

        # Keep the ensemble as it is now, so the update can be discarded if a resampled member's CME
        #  does not produce an elongation (see SKIP_UPDATE_IF_ANY_MEMBER_NAN)
        cme_par_dict_in = copy.deepcopy(self.cme_par_dict)
        cme_par_array_in = np.copy(self.cme_par_array)
        self.skip_reason: str | None = None
        self.resampled: bool = False
        self.ess_updated: float = np.nan

        if ADAPTIVE_RESAMPLING:
            # Elongations and likelihoods of the current (unshrunk, unjittered) particles
            obs_op_current = self.get_observation_operator(self.cme_par_dict)
            elon_current = np.asarray(obs_op_current, dtype=float)[:, 0].copy()
            log_likelihood_current = np.array([
                self.calculate_log_likelihood_single_ens(obs_op_current[i, :])
                for i in range(self.n_members)
            ], dtype=float)

            # Skip this observation (as below) if a member produced no elongation, or none did
            n_no_elon = int(np.sum(~np.all(np.isfinite(np.asarray(obs_op_current, dtype=float)), axis=1)))
            no_finite_lik = not np.any(np.isfinite(log_likelihood_current))
            if (SKIP_UPDATE_IF_ANY_MEMBER_NAN and n_no_elon > 0) or no_finite_lik:
                self.skip_reason = f"{n_no_elon}/{self.n_members} members produced no elongation"
                warnings.warn(f"Observation at {self.obs_time} not assimilated: {self.skip_reason}.")
                self.elon_first_stage = elon_current
                self.elon_posterior = np.full(self.n_members, np.nan)
                self.parent_index = np.arange(self.n_members, dtype=int)
                self.update_skipped = True
                return self.cme_par_dict

            # Updated weights w_i * p(y | theta_i) and their effective sample size
            log_weights_upd = np.asarray(self.log_weights, dtype=float) + log_likelihood_current
            log_weights_upd = np.where(np.isfinite(log_weights_upd), log_weights_upd, -1e31)
            log_weights_upd = log_weights_upd - jacobian_log(log_weights_upd)
            self.ess_updated = float(self.calc_ess_log_weights(log_weights_upd))
            print(f"ess_updated = {self.ess_updated:.2f} (resampling threshold "
                  f"{RESAMPLE_ESS_FRACTION * self.n_members:.1f})")

            if self.ess_updated >= RESAMPLE_ESS_FRACTION * self.n_members:
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

        # Calculate the observation operator, hx, to calculate the likelihoods for each ensemble member
        obs_op_shrunk = self.get_observation_operator(self.state_vector_shrunk_dict)
        # Keep the modelled elongations (one value per member at this observation time) so the
        #  routine can write them out. Indexed by the pre-resampling particle.
        self.elon_first_stage: npt.NDArray[float] = np.asarray(obs_op_shrunk, dtype=float)[:, 0].copy()

        # Calculate likelihoods for all ensemble members
        log_likelihood_shrunk_ens = [
            self.calculate_log_likelihood_single_ens(obs_op_shrunk[i, :])
            for i in range(self.n_members)
        ]

        # Skip the update, returning the ensemble and its weights unchanged, if
        #  - any member's CME produced no elongation and SKIP_UPDATE_IF_ANY_MEMBER_NAN is set, or
        #  - no member produced a finite likelihood (e.g. every modelled CME is outside the imager's
        #    field of view): the observation then carries no information, and resampling with the
        #    resulting uniform weights would only destroy ensemble diversity.
        n_no_elon = int(np.sum(~np.all(np.isfinite(np.asarray(obs_op_shrunk, dtype=float)), axis=1)))
        no_finite_lik = not np.any(np.isfinite(np.asarray(log_likelihood_shrunk_ens, dtype=float)))
        if (SKIP_UPDATE_IF_ANY_MEMBER_NAN and n_no_elon > 0) or no_finite_lik:
            self.skip_reason = (
                f"{n_no_elon}/{self.n_members} (shrunk) members produced no elongation"
            )
            warnings.warn(
                f"Observation at {self.obs_time} not assimilated: {self.skip_reason}."
            )
            self.elon_posterior = np.full(self.n_members, np.nan)
            self.parent_index = np.arange(self.n_members, dtype=int)
            self.update_skipped = True
            self.resampled = False
            return self.cme_par_dict
        self.update_skipped = False

        # Calculate the auxillary probabilities for selecting the new particles
        log_aux_prob = self.get_aux_prob_log_weights(log_likelihood_shrunk_ens)
        log_aux_prob = [
            x if np.isfinite(x) else -1e31 for x in log_aux_prob
        ]

        # Get indices to resample
        resample_class = ResampleParsLogWeights(
            log_weights=log_aux_prob,
            n_ensemble=self.n_members,
            rng=self.rng
        )
        resample_ind: list[int] = resample_class.systematic_resampling_log_weights()

        # Draw random samples from normal(shrunk_par, h^2 * cov_state) (in the working coordinates),
        #  then convert them to the natural state vector
        resample_pars: npt.NDArray[float] = self.to_natural(np.array(
            [self.resample_pars(ind_req) for ind_req in resample_ind]
        ))

        # Each new particle inherits ALL of its parent's parameters. Only the parameters in the state
        #  vector are overwritten with the jittered draws below; the others (e.g. t_init, lat, thick when
        #  they are not in the state) must come from the parent particle resample_ind[i], not from row i.
        #  (Previously row i kept its own non-state parameters, producing mismatched hybrid particles.)
        parent_cme_par_array: npt.NDArray[float] = np.copy(self.cme_par_array[resample_ind, :])

        # Place new resampled pars into the cme_par_dict
        from_state_class: FromStateVector = FromStateVector(
            cme_par_array=parent_cme_par_array,
            cme_par_dict=self.cme_par_dict,
            state_ens=resample_pars,
            pars_in_state_vector=self.pars_in_state
        )
        self.cme_par_array = from_state_class.state_vector_to_cme_par_array()
        self.cme_par_dict = from_state_class.cme_par_array_to_cme_par_dict()

        # Calculate the observation operator, hx, to calculate the posterior log likelihoods for each ensemble member
        obs_op_post = self.get_observation_operator(self.cme_par_dict)
        # Elongations of the resampled (posterior) particles, and which prior particle each came from
        self.elon_posterior: npt.NDArray[float] = np.asarray(obs_op_post, dtype=float)[:, 0].copy()
        self.parent_index: npt.NDArray[int] = np.asarray(resample_ind, dtype=int)

        # If a resampled member's CME produced no elongation, discard the update: restore the
        #  ensemble and weights from before this observation (see SKIP_UPDATE_IF_ANY_MEMBER_NAN)
        n_no_elon_post = int(np.sum(~np.all(np.isfinite(np.asarray(obs_op_post, dtype=float)), axis=1)))
        if SKIP_UPDATE_IF_ANY_MEMBER_NAN and n_no_elon_post > 0:
            self.skip_reason = (
                f"{n_no_elon_post}/{self.n_members} resampled members produced no elongation"
            )
            warnings.warn(
                f"Observation at {self.obs_time} not assimilated: {self.skip_reason}; "
                f"the update has been discarded."
            )
            self.cme_par_dict = cme_par_dict_in
            self.cme_par_array = cme_par_array_in
            self.elon_posterior = np.full(self.n_members, np.nan)
            self.parent_index = np.arange(self.n_members, dtype=int)
            self.update_skipped = True
            self.resampled = False
            return self.cme_par_dict

        # Calculate likelihoods for all ensemble members
        log_likelihood_post = [
            self.calculate_log_likelihood_single_ens(obs_op_post[i, :])
            for i in range(self.n_members)
        ]
        log_weights_post = [
            log_likelihood_post[i] - log_likelihood_shrunk_ens[resample_ind[i]]
            for i in range(len(resample_ind))
        ]

        # A member whose posterior likelihood is NaN (model CME not visible) would otherwise keep a
        #  NaN weight, which is written to the output and breaks the ESS/weighted statistics.
        #  Give such members a negligible weight explicitly.
        log_weights_post = [
            x if np.isfinite(x) else -1e31 for x in log_weights_post
        ]

        # Normalise the log_weights
        resamp_par_class = ResampleParsLogWeights(log_weights=log_weights_post, n_ensemble=self.n_members)
        log_weights_post = resamp_par_class.norm_log_weights()
        weights_post = [np.exp(w) for w in log_weights_post]

        # Update weights in cme_par_dict
        self.cme_par_dict["weight"] = weights_post
        self.cme_par_dict["log_weight"] = log_weights_post

        # Calculate the effective sample size from the posterior weights
        ess_post_log = self.calc_ess_log_weights(log_weights_post)
        print(f"ess_post_log = {ess_post_log}")

        return self.cme_par_dict
