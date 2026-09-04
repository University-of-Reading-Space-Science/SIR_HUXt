# @package sir
# This module will take inputs from SURF and perform an SIR to
#  generate an updated set of weights and particles
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

import warnings
warnings.filterwarnings(
    "ignore", message="invalid value encountered in divide"
)

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
            pars_in_state: list[str] = ["v", "lon", "width"],
            vr_in: npt.NDArray[Quantity[u.km/u.s]]=np.zeros(128) + 400 * u.km / u.s,
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
            rng=None
    ):
        self.use_model = use_model
        self.cme_par_dict:CmeParEns = cme_par_dict

        self.obs = obs
        self.obs_cov = obs_cov
        self.obs_radius = obs_radius
        self.obs_lon = obs_lon
        self.obs_lat = obs_lat
        self.obs_time = obs_time
        self.true_cme_par_dict: CmeParEns = true_cme_par_dict
        self.infl_fact = infl_fact
        self.pars_in_state = pars_in_state
        self.vr_in = vr_in
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
        cov_state: float | npt.NDArray = np.cov(self.state_vector, rowvar=False)
        if cov_state.ndim == 0:
            cov_state: npt.NDArray[float] = np.array([[cov_state]])

        scaled_cov_state = cov_state * self.cov_shrink_fact2

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
        Liu-West shrinkage of the parameters towards the weighted mean
        :return: shrunk_pars (nEns, nPar) array containing parameters shrunk to mean
        """
        if self.log_weights is None:
            # If no weights are provided, calculate an arithmetic mean
            mean_par: npt.NDArray[float] = np.mean(self.state_vector, axis=0)
        else:
            # If weights are provided, calculate weighted mean
            # Transform from log-space and normalise weights
            weights: npt.NDArray[float] = np.asarray(
                np.exp(self.log_weights)
            )
            weights: npt.NDArray[float] = weights / weights.sum()

            # Compute weighted mean parameter
            mean_par: npt.NDArray[float] = np.average(self.state_vector, axis=0, weights=weights)

        # Shrink towards the mean
        shrunk_pars = np.array([
            self.shrink_par_single_ens(self.state_vector[m, :], mean_par)
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
            state_ens=self.shrunk_pars,
            pars_in_state_vector=self.pars_in_state
        )
        updated_cme_par_array = from_state_class.state_vector_to_cme_par_array()

        return updated_cme_par_array


    def make_state_vector_dictionary(self):
        """
        Utilise the cme_par_array_to_cme_par_dict function in the FromStateVector class
         to create a dictionary containing the shrunken parameters for data assimilation using the
         Auxillary Particle Filter
        :return: state_vector_dict: State vector dictionary with shrunken parameters replacing original parameters
        """
        from_state_class = FromStateVector(
            cme_par_array=self.cme_par_array,
            cme_par_dict=self.cme_par_dict,
            state_ens=self.state_vector,
            pars_in_state_vector=self.pars_in_state
        )
        state_vector_dict = from_state_class.cme_par_array_to_cme_par_dict()

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
        weights: list[float] = self.cme_par_dict["weight"]

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
        log_weights: list[float] = self.cme_par_dict["log_weight"]

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
            obs_radius=self.obs_radius,
            obs_lon=self.obs_lon,
            obs_lat=self.obs_lat,
            obs_time_in_datetime=self.obs_time,
            obs_cov=self.obs_cov,
            surf_init_time=self.surf_init_time,
            vr_in=self.vr_in,
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
        # Normalise weights
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
        # Calculate the effective sample size from the prior weights
        ess_prior = self.calc_ess_log_weights(self.log_weights)
        if ess_prior < (self.n_members / 2.0):
            self.cme_par_dict["weight"] = [1.0 / self.n_members for _ in range(self.n_members)]
            self.weights = [1.0 / self.n_members for _ in range(self.n_members)]

            self.cme_par_dict["log_weight"] = [-np.log(self.n_members) for _ in range(self.n_members)]
            self.log_weights = [-np.log(self.n_members) for _ in range(self.n_members)]

        ess_prior2 = self.calc_ess_log_weights(self.log_weights)
        #print(f"ess_prior: {ess_prior}")
        #print(f"ess_prior2 = {ess_prior2}")

        # Calculate the observation operator, hx, to calculate the likelihoods for each ensemble member
        obs_op_shrunk = self.get_observation_operator(self.state_vector_shrunk_dict)

        # Calculate likelihoods for all ensemble members
        log_likelihood_shrunk_ens = [
            self.calculate_log_likelihood_single_ens(obs_op_shrunk[i, :])
            for i in range(self.n_members)
        ]

        # Calculate the auxillary probabilities for selecting the new particles
        log_aux_prob = self.get_aux_prob_log_weights(log_likelihood_shrunk_ens)
        log_aux_prob = [
            x if ~np.isnan(x) else -1e31 for x in log_aux_prob
        ]

        # Get indices to resample
        resample_class = ResampleParsLogWeights(
            log_weights=log_aux_prob,
            n_ensemble=self.n_members,
            rng=self.rng
        )
        resample_ind: list[int] = resample_class.systematic_resampling_log_weights()

        # Draw random samples from normal(shrunk_par, h^2 * cov_state)
        resample_pars: npt.NDArray[float] = np.array(
            [self.resample_pars(ind_req) for ind_req in resample_ind]
        )

        # Place new resampled pars into the cme_par_dict
        from_state_class: FromStateVector = FromStateVector(
            cme_par_array=self.cme_par_array,
            cme_par_dict=self.cme_par_dict,
            state_ens=resample_pars,
            pars_in_state_vector=self.pars_in_state
        )
        self.cme_par_array = from_state_class.state_vector_to_cme_par_array()
        self.cme_par_dict = from_state_class.cme_par_array_to_cme_par_dict()

        # Calculate the observation operator, hx, to calculate the posterior log likelihoods for each ensemble member
        obs_op_post = self.get_observation_operator(self.cme_par_dict)

        # Calculate likelihoods for all ensemble members
        log_likelihood_post = [
            self.calculate_log_likelihood_single_ens(obs_op_post[i, :])
            for i in range(self.n_members)
        ]
        log_weights_post = [
            log_likelihood_post[i] - log_likelihood_shrunk_ens[resample_ind[i]]
            for i in range(len(resample_ind))
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
