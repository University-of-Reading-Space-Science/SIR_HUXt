"""
aux_pf.py

One assimilation step of the auxiliary particle filter (Aux-PF) with Liu & West (2001) kernel
shrinkage, used to estimate static CME parameters from heliospheric-imager elongation observations.

Algorithm for one observation y (``AuxPF.aux_pf``), with particles x_i and log-weights log w_i:

    1. Shrink each particle towards the weighted ensemble mean:
           m_i = a x_i + (1 - a) xbar,      a = (3 delta - 1) / (2 delta)
    2. Run the forward model for each m_i and compute the first-stage (auxiliary) probabilities
           log g_i = log w_i + log p(y | m_i)
    3. Systematically resample ancestor indices k_i from g.
    4. Jitter: draw new particles x_i' ~ N(m_{k_i}, h^2 V),  h^2 = 1 - a^2, V = weighted ensemble covariance.
    5. Run the forward model for each x_i' and set the second-stage weights
           log w_i' = log p(y | x_i') - log p(y | m_{k_i}),   then normalise.

``delta_aux_pf`` (typically 0.95-0.99) is Liu & West's discount factor: values close to one give little
shrinkage and little jitter.

Optional look-ahead reuse (``lookahead``, see data_assimilation_routine ``reuse_lookahead_tracks``): the
first-stage weights may use any positive g_i, provided the second stage divides by the same g. Instead of running
the model at the shrunk points m_i (step 2), g_i = p(y | x_i) can be taken from the elongation tracks that the
previous step's second-stage runs already produced for the current particles x_i (their runs cover all later
observation times). Steps 1 (shrink) and 4 (jitter) are unchanged, so the kernel still preserves the ensemble mean
and covariance; only N rather than 2N model runs are needed per observation. Members without a usable stored track
fall back to a run at m_i (standard Liu-West), and the reuse is skipped if the observer position differs from the
one the tracks were computed for.
"""
import copy
import datetime
import warnings

import numpy as np
import numpy.typing as npt
from numpy.random import Generator

import astropy.units as u
from astropy.units import Quantity
from astropy.time import Time

from cme_par_ens import CmeParEns
from from_state_vector import FromStateVector
from sir_likelihood import LikelihoodFunction
from sir_observation_operator import ObservationOperator
from sir_resampling_log_weights import jacobian_log, ResampleParsLogWeights
from to_state_vector import ToStateVector
from timing import timed

# Members whose CME is never tracked give NaN observation operators; silence the resulting warnings
warnings.filterwarnings(
    "ignore", message="invalid value encountered in divide"
)


def same_observer(a, b) -> bool:
    """Whether two observer ids (AuxPF.observer_of) describe the same observer."""
    if a is None or b is None:
        return False
    a, b = tuple(a), tuple(b)
    if a and isinstance(a[0], str):
        if not (b and isinstance(b[0], str)) or a[0] != b[0] or len(a) != len(b):
            return False
        if a[0] == "craft":
            return a[1] == b[1]
        return bool(np.allclose(np.asarray(a[1:], dtype=float), np.asarray(b[1:], dtype=float), rtol=0, atol=1e-6))
    # legacy (r, lon, lat) tuples
    return len(a) == len(b) and bool(np.allclose(np.asarray(a, dtype=float), np.asarray(b, dtype=float),
                                                 rtol=0, atol=1e-6))


def subset_cme_par_dict(cme_par_dict: CmeParEns, indices: npt.ArrayLike) -> CmeParEns:
    """A copy of cme_par_dict containing only the members ``indices`` (per-member entries are indexed)."""
    idx = np.asarray(indices, dtype=int)
    n = cme_par_dict["n_members"]
    out = {}
    for key, value in cme_par_dict.items():
        if isinstance(value, Quantity) and value.ndim == 1 and len(value) == n:
            out[key] = value[idx]
        elif isinstance(value, (list, tuple, np.ndarray)) and len(value) == n:
            out[key] = [value[i] for i in idx]
        else:
            out[key] = copy.deepcopy(value)
    out["n_members"] = int(idx.size)
    return out


class AuxPF:
    """
    Auxiliary particle filter (Liu & West kernel shrinkage) for a single observation.
    See the module docstring for the algorithm.

    Example::

        cme_par_dict = AuxPF(use_model, cme_par_dict, obs, obs_cov, ...).aux_pf()

    NB: the input ``cme_par_dict`` is updated in place and also returned.
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
            true_cme_par_dict: CmeParEns | None = None,
            infl_fact: float = 1,
            craft: str | None = None,
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
            bias_term_5rs: float | None = None,
            bias_term_21rs: float | None = None,
            use_parallel: bool = False,
            fixed_compressible_sim_time: bool = True,
            profile_times: list[datetime.datetime] | None = None,
            lookahead: dict | None = None,
            sim_time_end: datetime.datetime | None = None,
            check_lookahead: bool = False,
            moving_craft_observer: bool = False,
            arrival_body: str | None = None,
            arrival_sim_time: Quantity[u.day] | None = None,
            n_workers: int | None = None,
            rng: Generator | None = None
    ) -> None:
        """
        Set up the filter and perform the Liu-West shrinkage of the current ensemble (step 1).

        :param use_model: Forward model: "surf", "compress_surf", "compress_surf3d" or "huxt"
        :param cme_par_dict: Current CME parameter ensemble (particles and weights)
        :param obs: Observed elongation (deg)
        :param obs_cov: Observation error variance (deg^2)
        :param obs_radius, obs_lon, obs_lat: Observer position
        :param obs_time: Time of the observation
        :param true_cme_par_dict: Truth CME parameters (OSSEs only; stored but not used here)
        :param infl_fact: Covariance inflation factor (stored but not currently used)
        :param craft: Observing spacecraft, e.g. "sta", or "twin"/None for a synthetic observer
        :param pars_in_state: CME parameters being estimated (default ["v", "lon", "width"])
        :param vr_in, vr_longs, vr_lats: Ambient solar wind speed at the inner boundary and its coordinates
        :param n_lat_surf: Number of latitudes for SURF3d
        :param lon_start, lon_stop: Longitude range of the model domain
        :param time_tolerance: Extra run time beyond the observation time
        :param dt_scale: Model output cadence
        :param delta_aux_pf: Liu-West discount factor (see module docstring)
        :param r_min: Inner boundary radius of the model
        :param cme_init_rad: Radius at which the CME parameters (t_init) are defined
        :param cme_fixed_duration, fixed_duration: Cone-CME duration settings
        :param plot_surf_output: Plot the model at the observation time for every member (slow)
        :param bias_term_bool, bias_term_5rs, bias_term_21rs: Optional linear elongation bias correction
        :param use_parallel: Run the ensemble members in parallel (multiprocessing)
        :param fixed_compressible_sim_time: see ObservationOperator (False = shorter, faster compressible runs)
        :param profile_times: Optional times (e.g. all observation times) at which to also record each member's
            modelled elongation profile, for output only (see the diagnostics set by aux_pf())
        :param lookahead: Optional stored tracks of the CURRENT particles for the first stage (see the module
            docstring): {"profiles": (n_members, n_profile_times) elongations from the previous step's
            second-stage runs, "index": column of profiles at obs_time, "observer": (r AU, lon deg, lat deg) the
            tracks were computed for}
        :param sim_time_end: Optional time the model runs must reach (surf / huxt only; e.g. the last observation
            time, so that the second-stage tracks can be reused at later observations)
        :param check_lookahead: Diagnostic only: when stored tracks are used, ALSO run the shrunk points and keep
            their elongations in self.hx_at_shrunk_points, to measure the look-ahead approximation (extra runs,
            not counted in n_model_runs; the filter still uses the stored tracks)
        :param moving_craft_observer: see ObservationOperator (real craft: image from the craft's trajectory, so
            stored tracks can be reused at later observation times)
        :param arrival_body, arrival_sim_time: If arrival_body is set, the second-stage runs also give every new
            particle's arrival at that body (self.post_arrivals), see ObservationOperator
        :param n_workers: Members to run at the same time (default: limited by CPUs and memory, memory_budget.py)
        :param rng: Random number generator (resampling and jitter)
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
        self.fixed_compressible_sim_time = fixed_compressible_sim_time
        self.profile_times = None if profile_times is None else list(profile_times)
        self.lookahead = lookahead
        self.sim_time_end = sim_time_end
        self.check_lookahead = check_lookahead
        self.moving_craft_observer = moving_craft_observer
        self.arrival_body = arrival_body
        self.arrival_sim_time = arrival_sim_time
        self.n_workers = n_workers
        #  post_arrivals: [(arrival datetime or None, speed km/s, hit)] of the new particles (arrival_body only)
        self.post_arrivals: list | None = None
        self.last_member_arrivals: list | None = None
        self.hx_at_shrunk_points: npt.NDArray[float] | None = None

        # Diagnostics filled in by aux_pf() (for output; see data_assimilation_routine / da_output):
        #  hx_shrunk / hx_post: modelled elongation at obs_time of the shrunk (first-stage) and the new
        #  (posterior) particles; profile_shrunk / profile_post: their profiles at profile_times;
        #  ancestors: resampled indices; ess_prior / ess_post: effective sample sizes before / after.
        self.hx_shrunk: npt.NDArray[float] | None = None
        self.hx_post: npt.NDArray[float] | None = None
        self.profile_shrunk: npt.NDArray[float] | None = None
        self.profile_post: npt.NDArray[float] | None = None
        self.ancestors: npt.NDArray[int] | None = None
        self.ess_prior: float = np.nan
        self.ess_post: float = np.nan
        self.last_model_profiles: npt.NDArray[float] | None = None
        #  lookahead_reused: True for members whose first-stage value came from a stored track;
        #  n_model_runs: members run through the forward model in this step (first + second stage);
        #  post_observer: observer position (r AU, lon deg, lat deg) of the second-stage runs
        self.lookahead_reused: npt.NDArray[bool] = np.zeros(cme_par_dict["n_members"], dtype=bool)
        self.n_model_runs: int = 0
        self.post_observer: tuple | None = None
        self.last_observer: tuple | None = None

        if rng is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = rng

        # Convert the ensemble into unit-free arrays
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

        # Step 1: shrink the parameters towards the weighted mean
        self.shrunk_pars = self.shrink_par()

        # Put shrunk_parameters back into cme_par_array
        self.cme_par_array = self.put_shrunk_pars_in_cme_par_array()

        # Create a state_vector dictionary containing all shrunken parameters
        self.state_vector_shrunk_dict = self.make_state_vector_dictionary()

        # Model run length needed to reach the current observation time
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
        weights = np.exp(np.asarray(self.log_weights, dtype=float))
        if (not np.all(np.isfinite(weights))) or (np.sum(weights) <= 0):
            weights = None
        cov_state: float | npt.NDArray = np.cov(self.state_vector, rowvar=False, aweights=weights)
        if cov_state.ndim == 0:
            cov_state: npt.NDArray[float] = np.array([[cov_state]])

        scaled_cov_state = cov_state * self.cov_shrink_fact2

        return cov_state, scaled_cov_state


    def get_shrink_fact(self) -> tuple[float, float]:
        """
        Liu & West shrinkage factors from the discount factor delta:
            a   = (3 delta - 1) / (2 delta)   (state shrinkage)
            h^2 = 1 - a^2                     (kernel covariance scaling)
        :return: state_shrink_fact (a), cov_shrink_fact (h^2)
        """
        state_shrink_fact: float = ((3 * self.delta_aux_pf) - 1) / (2 * self.delta_aux_pf)
        cov_shrink_fact: float = 1 - state_shrink_fact ** 2

        return state_shrink_fact, cov_shrink_fact


    def shrink_par_single_ens(
            self,
            state_1ens: npt.NDArray[float],
            mean_state: npt.NDArray[float]
    ) -> npt.NDArray[float]:
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


    def put_shrunk_pars_in_cme_par_array(self) -> npt.NDArray[float]:
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


    def make_state_vector_dictionary(self) -> CmeParEns:
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

        # surf / huxt run only as long as needed; extend them so their tracks cover later observations
        #  (the compressible models already run for a fixed length, see ObservationOperator)
        end = getattr(self, "sim_time_end", None)
        if end is not None and self.use_model in ["surf", "huxt"] and end > self.obs_time:
            end_astro_time = Time(end, format="datetime", scale="utc")
            sim_time = (end_astro_time - init_astro_time).to(u.day) + self.time_tolerance

        return sim_time


    def get_particle_weights(self) -> npt.NDArray[float]:
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


    def get_particle_log_weights(self) -> npt.NDArray[float]:
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


    def make_observation_operator(self, state_dict_in: CmeParEns, with_arrival: bool = False):
        """
        The observation operator for an ensemble (constructing it runs no model).
        :param with_arrival: also compute the members' arrival at self.arrival_body from the same runs
        """
        extra = {}
        if getattr(self, "n_workers", None) is not None:
            extra["n_workers"] = self.n_workers
        if self.moving_craft_observer:
            extra["moving_craft_observer"] = True
        if with_arrival and self.arrival_body is not None:
            extra["arrival_body"] = self.arrival_body
            extra["arrival_sim_time"] = self.arrival_sim_time
        return ObservationOperator(
            use_model=self.use_model,
            cme_par_dict=state_dict_in,
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
            fixed_compressible_sim_time=self.fixed_compressible_sim_time,
            profile_times=self.profile_times,
            **extra,
        )


    def observer_of(self, obs_op_class) -> tuple:
        """
        What the operator's modelled tracks depend on, observer-wise (ObservationOperator.observer_id):
        ("craft", name) when the imager follows a real craft's trajectory, else ("position", r AU, lon deg,
        lat deg) of the fixed observer (for a real craft without moving_craft_observer: its position at the
        observation time; operators without observer_id: the position given to AuxPF).
        """
        oid = getattr(obs_op_class, "observer_id", None)
        if isinstance(oid, tuple):
            return oid

        def value(x, unit, default):
            x = getattr(obs_op_class, x, None)
            x = default if x is None else x
            q = u.Quantity(x)
            return float(np.ravel(q.to(unit).value)[0]) if q.unit != u.dimensionless_unscaled \
                else float(np.ravel(q.value)[0])
        return ("position", round(value("obs_radius", u.AU, self.obs_radius), 9),
                round(value("obs_lon", u.deg, self.obs_lon), 9), round(value("obs_lat", u.deg, self.obs_lat), 9))


    def get_observation_operator(self, state_dict_in: CmeParEns, with_arrival: bool = False
                                 ) -> npt.NDArray[float]:
        """
        Run the forward model for every ensemble member in state_dict_in and return the predicted
        observations.
        :param state_dict_in: CME parameter ensemble to run
        :param with_arrival: also compute the arrivals (self.last_member_arrivals), see make_observation_operator
        :return: hx: (n_members, n_obs_times) array of model elongations at self.obs_time
        """
        obs_op_class = self.make_observation_operator(state_dict_in, with_arrival=with_arrival)
        hx = obs_op_class.make_obs_op()
        self.last_member_arrivals = getattr(obs_op_class, "member_arrivals", None) if with_arrival else None
        self.n_model_runs += int(state_dict_in["n_members"])
        self.last_observer = self.observer_of(obs_op_class)

        # Diagnostic profiles of this call (None if not requested or not provided by the operator)
        self.last_model_profiles = getattr(obs_op_class, "model_profiles", None)

        return hx


    def lookahead_usable(self) -> bool:
        """True if stored tracks were given and were computed for the observer this step uses."""
        la = self.lookahead
        if la is None or la.get("profiles") is None or la.get("observer") is None:
            return False
        profiles = np.asarray(la["profiles"], dtype=float)
        if profiles.ndim != 2 or profiles.shape[0] != self.n_members or not (0 <= la["index"] < profiles.shape[1]):
            return False
        expected = self.observer_of(self.make_observation_operator(self.cme_par_dict))
        return same_observer(expected, la["observer"])


    def get_first_stage_observations(self) -> npt.NDArray[float]:
        """
        Modelled observations for the first-stage weights: the stored tracks of the current particles x_i where
        available (see the module docstring), otherwise a model run at the shrunk point m_i (standard Liu-West).
        Sets self.lookahead_reused and self.last_model_profiles (the profiles behind the values used).
        :return: (n_members, 1) array
        """
        if not self.lookahead_usable():
            self.lookahead_reused = np.zeros(self.n_members, dtype=bool)
            return self.get_observation_operator(self.state_vector_shrunk_dict)

        profiles = np.array(self.lookahead["profiles"], dtype=float)
        hx = profiles[:, int(self.lookahead["index"])].reshape(self.n_members, 1)
        reused = np.isfinite(hx[:, 0])
        missing = np.flatnonzero(~reused)
        if missing.size:
            # Members without a stored value: run them at their shrunk points
            sub = subset_cme_par_dict(self.state_vector_shrunk_dict, missing)
            hx_sub = self.get_observation_operator(sub)
            hx[missing, 0] = np.asarray(hx_sub, dtype=float)[:, 0]
            if self.last_model_profiles is not None and np.shape(self.last_model_profiles)[1] == profiles.shape[1]:
                profiles[missing] = self.last_model_profiles
            else:
                profiles[missing] = np.nan
        if self.check_lookahead:
            saved = (self.n_model_runs, self.last_observer)
            self.hx_at_shrunk_points = np.asarray(
                self.get_observation_operator(self.state_vector_shrunk_dict), dtype=float)[:, 0]
            self.n_model_runs, self.last_observer = saved
        self.lookahead_reused = reused
        self.last_model_profiles = profiles
        print(f"First stage: {int(reused.sum())}/{self.n_members} members from stored tracks, "
              f"{missing.size} run at their shrunk points")
        return hx


    def calculate_diff_between_obs_and_obs_op(
            self,
            hx: npt.NDArray[float] | float
    ) -> npt.NDArray[float] | float:
        """
        Function to calculate the difference between the observation operator and the observation
        :param hx: Observation operator
        :return: obs_diff: difference between the observation operator and the observation
        """
        obs_diff = self.obs - hx

        return obs_diff


    def calculate_likelihood_single_ens(self, hx: npt.NDArray[float] | float) -> float:
        """
        Likelihood p(y | x) for a single ensemble member
        :param hx: Observation operator for a single ensemble member
        :return: likelihood: Likelihood value for a single ensemble member
        """
        likelihood_class = LikelihoodFunction(
            obs=self.obs,
            obs_cov=self.obs_cov,
            hx=hx
        )
        likelihood = likelihood_class.make_likelihood()

        return likelihood


    def calculate_log_likelihood_single_ens(self, hx: npt.NDArray[float] | float) -> float:
        """
        Log-likelihood log p(y | x) for a single ensemble member
        :param hx: Observation operator for a single ensemble member
        :return: log_likelihood: Log-likelihood value for a single ensemble member (nan if hx is nan)
        """
        likelihood_class = LikelihoodFunction(
            obs=self.obs,
            obs_cov=self.obs_cov,
            hx=hx
        )
        log_likelihood = likelihood_class.make_log_likelihood()

        return log_likelihood


    def get_aux_prob(self, likelihoods: list[float]) -> list[float]:
        """
        Function to get the auxillary probabilities from the weights and the likelihoods
        :param likelihoods: Likelihood function
        :return: aux_prob: Auxillary probabilities
        """
        aux_prob = [
            self.weights[i] * likelihoods[i] for i in range(self.n_members)
        ]
        return aux_prob


    def get_aux_prob_log_weights(self, log_likelihood: list[float]) -> list[float]:
        """
        First-stage (auxiliary) probabilities in log-space, normalised:
            log g_i = log w_i + log p(y | m_i) - log( sum_j w_j p(y | m_j) )
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


    def resample_pars(self, ind_req: int) -> npt.NDArray[float]:
        """
        Draw a jittered particle from N(m_k, h^2 V) around shrunk particle k (step 4).
        :param ind_req: Index k of the (resampled) ancestor
        :return: New state vector
        """
        resampled_par = self.rng.multivariate_normal(
            mean=self.shrunk_pars[ind_req, :], cov=self.scaled_cov_state
        )

        return resampled_par


    def calc_ess_lin_weights(self, weight_in: list[float] | npt.NDArray[float]) -> float:
        """
        Effective Sample Size from linear weights: ESS = (sum w)^2 / sum w^2
        :return: ess: Effective Sample Size
        """
        numer = np.sum(weight_in) ** 2
        denom = np.sum([w * w for w in weight_in])

        ess = numer / denom

        return ess


    def log_ess_log_weights(self, log_weight_in: npt.NDArray[float]) -> float:
        """
        Logarithm of the Effective Sample Size from log-weights:
            log ESS = 2 log(sum w) - log(sum w^2)
        :return: log_ess: Logarithm of Effective Sample Size
        """
        # Calculate the logarithm of the Effective Sample Size
        term1 = 2 * jacobian_log(log_weight_in)
        term2 = jacobian_log(2 * log_weight_in)

        log_ess = term1 - term2

        return log_ess


    def calc_ess_log_weights(self, log_weight_in: list[float] | npt.NDArray[float]) -> float:
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


    @timed("aux_pf.step_total")
    def aux_pf(self) -> CmeParEns:
        """
        Perform steps 2-5 of the auxiliary particle filter (step 1 is done in __init__).
        :return: cme_par_dict: Updated cme_par_dict with parameters changed by Auxillary PF
        """
        # If the prior weights are degenerate (ESS < N/2), reset them to uniform.
        #  NB: no resampling is done here, and the shrinkage in __init__ used the old weights.
        ess_prior = self.calc_ess_log_weights(self.log_weights)
        self.ess_prior = float(ess_prior)
        if ess_prior < (self.n_members / 2.0):
            self.cme_par_dict["weight"] = [1.0 / self.n_members for _ in range(self.n_members)]
            self.weights = [1.0 / self.n_members for _ in range(self.n_members)]

            self.cme_par_dict["log_weight"] = [-np.log(self.n_members) for _ in range(self.n_members)]
            self.log_weights = [-np.log(self.n_members) for _ in range(self.n_members)]

        ess_prior2 = self.calc_ess_log_weights(self.log_weights)

        # Step 2: run the model for the shrunk particles m_i and compute the first-stage probabilities
        with timed("aux_pf.obs_op_shrunk_ensemble"):
            obs_op_shrunk = self.get_first_stage_observations()
        self.hx_shrunk = np.asarray(obs_op_shrunk, dtype=float)[:, 0]
        self.profile_shrunk = self.last_model_profiles

        # Calculate likelihoods for all ensemble members
        log_likelihood_shrunk_ens = [
            self.calculate_log_likelihood_single_ens(obs_op_shrunk[i, :])
            for i in range(self.n_members)
        ]

        # Calculate the auxillary probabilities for selecting the new particles
        log_aux_prob = self.get_aux_prob_log_weights(log_likelihood_shrunk_ens)
        log_aux_prob = [
            x if np.isfinite(x) else -1e31 for x in log_aux_prob
        ]

        # Step 3: systematically resample ancestor indices from the auxiliary probabilities
        resample_class = ResampleParsLogWeights(
            log_weights=log_aux_prob,
            n_ensemble=self.n_members,
            rng=self.rng
        )
        with timed("aux_pf.resample"):
            resample_ind: list[int] = resample_class.systematic_resampling_log_weights()
        self.ancestors = np.asarray(resample_ind, dtype=int)

        # Step 4: draw jittered particles from normal(shrunk_par, h^2 * cov_state)
        resample_pars: npt.NDArray[float] = np.array(
            [self.resample_pars(ind_req) for ind_req in resample_ind]
        )

        # Place new resampled pars into the cme_par_dict.
        #  Reorder the rows by ancestor first, so that parameters NOT in the state vector (e.g. t_init, lat,
        #  thick for the OSSE) are inherited from the resampled particle rather than staying with member i.
        #  (No effect while those parameters have zero prior spread, as in the current setup.)
        self.cme_par_array = np.copy(self.cme_par_array[np.asarray(resample_ind, dtype=int), :])
        from_state_class: FromStateVector = FromStateVector(
            cme_par_array=self.cme_par_array,
            cme_par_dict=self.cme_par_dict,
            state_ens=resample_pars,
            pars_in_state_vector=self.pars_in_state
        )
        self.cme_par_array = from_state_class.state_vector_to_cme_par_array()
        self.cme_par_dict = from_state_class.cme_par_array_to_cme_par_dict()

        # Step 5: run the model for the new particles and compute the second-stage weights
        with timed("aux_pf.obs_op_posterior_ensemble"):
            obs_op_post = self.get_observation_operator(self.cme_par_dict, with_arrival=True)
        self.post_arrivals = self.last_member_arrivals
        self.hx_post = np.asarray(obs_op_post, dtype=float)[:, 0]
        self.profile_post = self.last_model_profiles
        self.post_observer = self.last_observer

        # Calculate likelihoods for all ensemble members
        log_likelihood_post = [
            self.calculate_log_likelihood_single_ens(obs_op_post[i, :])
            for i in range(self.n_members)
        ]
        # (divides by the SAME first-stage likelihood used to pick the ancestor, whether it came from a run at
        #  m_k or from a stored track at x_k, which keeps the weights exact)
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
        self.ess_post = float(ess_post_log)
        print(f"ess_post_log = {ess_post_log}")

        return self.cme_par_dict
