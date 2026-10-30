import numpy as np
import numpy.typing as npt
import datetime
import os
import sys

import xarray as xr

import surf.surf as S
import surf.surf_inputs as Sin

import huxt.huxt_inputs as Hin

import sunpy.coordinates.sun as sn

import astropy.units as u
from astropy.units import Quantity
from astropy.time import Time

from dotenv import load_dotenv
from pathlib import Path

import make_prior_covariance_mat as mp_cov
from wsa_reader import ReadWSAFiles

from init_sir import initialise_cme_parameter_ensemble_dict
from sir_observations import Observations
from aux_pf import AuxPF
from cme_par_ens import CmeParEns

from da_surf_setup import Setup
from sir_observation_operator import shutdown_worker_pool, ObservationOperator
from obs_bias import resolve_bias_settings
from state_transform import (StateTransform, resolve_correlation_settings, filter_state_pars,
                             weighted_corr)
from to_state_vector import ToStateVector
import time
from multiprocessing import Pool
from typing import Generator

env_path = Path('.', '.env')
load_dotenv(dotenv_path=env_path, override=True)


def print_environment_variables():
    print(f"obs_dir = {os.getenv('OBS_DIR')}")
    print(f"mo_cone_cov_dir = {os.getenv('MO_CONE_COV_DIR')}")
    print(f"donki_cov_dir = {os.getenv('DONKI_COV_DIR')}")
    print(f"mo_cme_cone_file_dir = {os.getenv('MO_CME_CONE_FILE_DIR')}")
    print(f"out_base_dir = {os.getenv('OUT_BASE_DIR')}")
    print(f"wsa_dir = {os.getenv('WSA_DIR')}")

    return None


class RunDataAssimilationRoutine:
    def __init__(self):
        self.setup_class = Setup()

        # Get SURF parameters
        surf_par_dict = self.setup_class.surf_par_dict
        self.use_model: str = surf_par_dict["use_model"]
        self.surf_init_time: datetime.datetime = surf_par_dict["surf_init_time"]
        self.vr_in: npt.NDArray[Quantity[u.km / u.s]] | Quantity[u.km / u.s] = surf_par_dict["vr_in"]
        self.vr_longs: npt.NDArray[Quantity[u.deg]] | Quantity[u.deg] = surf_par_dict["vr_longs"]
        self.vr_lats: npt.NDArray[Quantity[u.deg]] | Quantity[u.deg] = surf_par_dict["vr_lats"]
        self.br_in: npt.NDArray[Quantity[u.nT]] | Quantity[u.nT] = surf_par_dict["br_in"]
        self.br_longs: npt.NDArray[Quantity[u.deg]] | Quantity[u.deg] = surf_par_dict["br_longs"]
        self.br_lats: npt.NDArray[Quantity[u.deg]] | Quantity[u.deg] = surf_par_dict["br_lats"]
        self.lon_start: Quantity[u.deg] = surf_par_dict["lon_start"]
        self.lon_stop: Quantity[u.deg] = surf_par_dict["lon_stop"]
        self.latitude_min: Quantity[u.deg] = surf_par_dict["latitude_min"]
        self.latitude_max: Quantity[u.deg] = surf_par_dict["latitude_max"]
        self.n_lat_surf: int = surf_par_dict["n_lat_surf"]
        self.sim_time: Quantity[u.days] = surf_par_dict["sim_time"]
        self.dt_scale: int | float = surf_par_dict["dt_scale"]
        self.r_min: Quantity[u.solRad] = surf_par_dict["r_min"]
        self.cme_init_rad: Quantity[u.solRad] = surf_par_dict["cme_init_rad"]
        self.cme_fixed_duration: bool = surf_par_dict["cme_fixed_duration"]
        self.fixed_duration: Quantity[u.s] = surf_par_dict["fixed_duration"]

        if isinstance(self.vr_longs.value, float):
            self.n_lon_in: int = 1
        else:
            if not np.all(np.isnan(self.vr_longs)):
                self.n_lon_in: int = len(self.vr_longs)
            else:
                self.n_lon_in: int = len(self.vr_in)

        if isinstance(self.vr_lats.value, float):
            self.n_lat_in: int = 1
        else:
            if not np.all(np.isnan(self.vr_lats)):
                self.n_lat_in: int = len(self.vr_lats)
            else:
                self.n_lat_in: int = 1

        # Initialise observation variables
        obs_par_dict = self.setup_class.obs_par_dict
        self.n_obs: int = obs_par_dict["n_obs"]
        self.obs_radius: Quantity[u.AU] | list[Quantity[u.AU]] = obs_par_dict["obs_radius"]
        self.obs_lon: Quantity[u.deg] | list[Quantity[u.deg]] = obs_par_dict["obs_lon"]
        self.obs_lat: Quantity[u.deg] | list[Quantity[u.deg]] = obs_par_dict["obs_lat"]
        self.obs_cov: float | npt.NDArray[float] = obs_par_dict["obs_cov"]
        self.obs_times: list[datetime.datetime] = obs_par_dict["obs_times"]
        self.elong_min: float = obs_par_dict["elong_min"]
        self.elong_max: float = obs_par_dict["elong_max"]
        self.elong_min_obs: float = obs_par_dict["elong_min_obs"]
        self.elong_max_obs: float = obs_par_dict["elong_max_obs"]
        self.use_synthetic_obs: bool = obs_par_dict["use_synthetic_obs"]
        self.obs_filenames: list[str] = obs_par_dict["obs_filenames"]
        self.obs_rng_seed: int = obs_par_dict["obs_rng_seed"]
        self.ssw_event:str = obs_par_dict["ssw_event"]
        self.craft:str = obs_par_dict["craft"]
        self.img:str = obs_par_dict["img"]
        self.bias_term_bool: bool = obs_par_dict["bias_term_bool"]
        self.bias_term_5rs: float = obs_par_dict["bias_term_5rs"]
        self.bias_term_21rs: float = obs_par_dict["bias_term_21rs"]
        self.use_parallel: bool = obs_par_dict["use_parallel"]

        # Ensure all bias terms exist, if not, set bias_term_bool to False
        if (self.bias_term_5rs is None) or (self.bias_term_21rs is None):
            self.bias_term_bool = False

        # If we're using synthetic observations/ running OSSEs define true_cme_par_dict
        self.true_cme_par_dict: CmeParEns = self.setup_class.true_cme_par_dict

        if self.use_synthetic_obs:
            # Generate observations
            self.observations: list[float] = self.get_observations()
            if not isinstance(self.obs_radius, list):
                self.obs_radius = [self.obs_radius for _ in range(self.n_obs)]

            if not isinstance(self.obs_lon, list):
                self.obs_lon = [self.obs_lon for _ in range(self.n_obs)]

            if not isinstance(self.obs_lat, list):
                self.obs_lat = [self.obs_lat for _ in range(self.n_obs)]
        else:
            obs_tuple: tuple[
                list[datetime.datetime], list[Quantity[u.deg]], list[float], int
            ] = self.get_observations()

            self.obs_times: list[datetime.datetime] = obs_tuple[0]
            self.obs_lon: list[Quantity[u.deg]] = obs_tuple[1]
            self.observations: list[float] = obs_tuple[2]
            self.n_obs: int = obs_tuple[3]

            if not isinstance(self.obs_radius, list):
                self.obs_radius = [self.obs_radius for _ in range(self.n_obs)]

            if not isinstance(self.obs_lat, list):
                self.obs_lat = [self.obs_lat for _ in range(self.n_obs)]


        # Generate first-guess cme_parameters (which OSSE draws priors around)
        fg_cme_par_dict = self.setup_class.fg_cme_par_dict
        self.fg_mean_cme_t_init: datetime.datetime = fg_cme_par_dict["fg_cme_t_init"]
        self.fg_mean_cme_speed: float = fg_cme_par_dict["fg_cme_speed"]
        self.fg_mean_cme_width: float = fg_cme_par_dict["fg_cme_width"]
        self.fg_mean_cme_lon: float = fg_cme_par_dict["fg_cme_lon"]
        self.fg_mean_cme_lat: float = fg_cme_par_dict["fg_cme_lat"]
        self.fg_mean_cme_thick: float = fg_cme_par_dict["fg_cme_thick"]

        self.fg_sd_cme_t_init: float = fg_cme_par_dict["sd_cme_t_init"]
        self.fg_sd_cme_speed: float = fg_cme_par_dict["sd_cme_speed"]
        self.fg_sd_cme_width: float = fg_cme_par_dict["sd_cme_width"]
        self.fg_sd_cme_lon: float = fg_cme_par_dict["sd_cme_lon"]
        self.fg_sd_cme_lat: float = fg_cme_par_dict["sd_cme_lat"]
        self.fg_sd_cme_thick: float = fg_cme_par_dict["sd_cme_thick"]

        # Get prior CME parameters
        #  (in case of OSSE, may be perturbed from the first-guess parameters)
        prior_cme_par_dict = self.setup_class.prior_cme_cov
        self.cme_cov_type: str = prior_cme_par_dict["cme_cov_type"]
        self.scale_corr: bool = prior_cme_par_dict["scale_corr"]
        self.use_log_v: bool = prior_cme_par_dict["use_log_v"]
        self.prior_sd_t_init: float = prior_cme_par_dict["sd_t_init"]
        self.prior_sd_speed: float = prior_cme_par_dict["sd_speed"]
        self.prior_sd_width: float = prior_cme_par_dict["sd_width"]
        self.prior_sd_lon: float = prior_cme_par_dict["sd_lon"]
        self.prior_sd_lat: float = prior_cme_par_dict["sd_lat"]
        self.prior_sd_thick: float = prior_cme_par_dict["sd_thick"]

        # Get DA parameters
        da_par_dict = self.setup_class.da_par_dict #initialise_da_parameters()
        self.n_members: int = da_par_dict["n_members"]
        self.n_runs: int = da_par_dict["n_runs"]
        self.delta_aux_pf: float = da_par_dict["delta_aux_pf"]
        self.pars_in_state: list[str] = da_par_dict["pars_in_state"]
        self.time_tolerance: Quantity[u.day] = da_par_dict["time_tolerance"]

        # Observation-bias term (see obs_bias.py and Setup.get_bias_settings)
        self.bias_settings: dict = resolve_bias_settings(da_par_dict.get("bias_settings"))
        if self.bias_settings["mode"] == "estimate" and "bias" not in self.pars_in_state:
            self.pars_in_state = self.pars_in_state + ["bias"]

        # Handling of correlated parameters in the Liu-West step (see state_transform.py and
        #  Setup.get_correlation_settings). pars_in_state is still used to draw the prior; the
        #  Aux-PF updates only pars_in_filter (pars_in_state without those carried from the prior).
        self.correlation_settings: dict = resolve_correlation_settings(da_par_dict.get("correlation_settings"))
        self.pars_in_filter: list[str] = filter_state_pars(self.pars_in_state, self.correlation_settings)
        if self.correlation_settings["enabled"]:
            print(f"Correlation handling on: {self.correlation_settings}; "
                  f"parameters updated by the filter: {self.pars_in_filter}")
        #self.rng: Generator = np.random.default_rng(da_par_dict["rng_seed"])


    def generate_random_seed(
            self,
            add_const: int=None,
            mult_const: int=None,
            run_no: int=None
    ):
        """
        Function to generate random seed for each model run
        :return: random_seed: Random seed for each model run
        """
        if add_const is None:
            add_const = 42

        if mult_const is None:
            mult_const = 10000

        if run_no is None:
            run_no = 0

        if run_no >= 0:
            random_seed = add_const + (mult_const * run_no)
        else:
            random_seed = int(np.abs((add_const + (mult_const / 2.0)) + (run_no * mult_const)))

        return random_seed


    def make_state_transform(self, cme_par_dict: CmeParEns) -> StateTransform | None:
        """
        Build the StateTransform for this run's Liu-West steps from its prior ensemble
         (None when correlation handling is off, which leaves the Aux-PF exactly as before)
        :param cme_par_dict: Prior CME parameter ensemble
        :return: StateTransform or None
        """
        if not self.correlation_settings["enabled"]:
            return None
        prior_state = ToStateVector(
            cme_par_dict, pars_in_state_vector=self.pars_in_filter
        ).cme_par_dict_to_state_vector()
        transform = StateTransform(
            pars=self.pars_in_filter,
            settings=self.correlation_settings,
            r_min_rs=float(self.r_min.to(u.solRad).value),
            cme_init_rad_rs=float(self.cme_init_rad.to(u.solRad).value),
            prior_state=prior_state,
            prior_log_weights=cme_par_dict["log_weight"],
        )
        print(f"Liu-West working coordinates: {transform.working_labels()}"
              f"{' (whitened with the prior covariance)' if transform.whiten else ''}")
        return transform


    def state_correlation(self, cme_par_dict: CmeParEns) -> npt.NDArray[float]:
        """
        Weighted correlation matrix of the parameters in pars_in_state (v as ln v), for diagnosing how
         the filter handles parameter correlations
        :param cme_par_dict: CME parameter ensemble
        :return: (n_state, n_state) correlation matrix
        """
        x = ToStateVector(
            cme_par_dict, pars_in_state_vector=self.pars_in_state
        ).cme_par_dict_to_state_vector()
        if "v" in self.pars_in_state:
            i_v = self.pars_in_state.index("v")
            x[:, i_v] = np.log(np.clip(x[:, i_v], 1e-6, None))
        return weighted_corr(x, cme_par_dict["log_weight"])


    def make_output_dir(self, cme_par_dict: CmeParEns, run_no: int):
        """
        Function to make the output directory
        :return: out_dir: Output directory
        """
        base_dir = os.path.join(
            os.getenv("OUT_BASE_DIR"),
            f"ens_{self.n_members}", self.cme_cov_type
        )
        model_dir = f"model_{self.use_model.upper()}"

        if self.use_synthetic_obs:
            # (single quotes inside the f-string so this also parses on Python < 3.12)
            obs_dir = (
                f"truth_{self.true_cme_par_dict['v'].value}_{self.true_cme_par_dict['width'].value}"
                f"_{self.true_cme_par_dict['lon'].value}_{self.true_cme_par_dict['lat'].value}"
                f"_{self.true_cme_par_dict['thick'].value}"
            )
        else:
            obs_dir = (
                f"obs_{self.craft}_{self.ssw_event}_{self.img}"
            )

        prior_dir = (
            f"prior_{self.fg_mean_cme_speed:.1f}_{self.fg_mean_cme_width:.1f}"
            f"_{self.fg_mean_cme_lon:.1f}_{self.fg_mean_cme_lat:.1f}"
            f"_{self.fg_mean_cme_thick:.1f}"
        )
        run_dir = f"run_{run_no:03d}"

        # Separate folders for runs with an observation-bias term (and, in twins, the injected bias)
        if self.bias_settings["mode"] != "off":
            bias_tag = (
                f"obsbias_{self.bias_settings['mode']}_{self.bias_settings['predictor']['type']}"
                f"_prior{self.bias_settings['prior_mean']:.2f}_{self.bias_settings['prior_sd']:.2f}"
            )
            prior_dir = os.path.join(bias_tag, prior_dir)
        # Separate folders for runs with correlation handling
        if self.correlation_settings["enabled"]:
            cs = self.correlation_settings
            corr_tag = (
                f"corr_{cs['reparam']}"
                f"{('_r' + format(cs['t_ref_rs'], 'g')) if cs['reparam'] == 'tref_logv' and cs['t_ref_rs'] is not None else ''}"
                f"{'_white' if cs['whiten'] else ''}"
                f"_a{cs['shrinkage_alpha']:.2f}_floor{cs['min_jitter_sd']:.3f}"
            )
            if cs["carry_from_prior"]:
                corr_tag += "_carry-" + "-".join(cs["carry_from_prior"])
            prior_dir = os.path.join(corr_tag, prior_dir)
        if self.use_synthetic_obs and float(self.true_cme_par_dict.get("bias", 0.0)) != 0.0:
            obs_dir = f"{obs_dir}_truebias{float(self.true_cme_par_dict['bias']):.2f}"

        if self.bias_term_bool:
            bias_dir = os.path.join(
                "bias_correction", f"bias5_{self.bias_term_5rs}_bias21_{self.bias_term_21rs}"
            )
            output_dir = os.path.join(
                base_dir, model_dir, bias_dir, obs_dir,
                prior_dir, f"{self.delta_aux_pf}", run_dir
            )
        else:
            output_dir = os.path.join(
                base_dir, model_dir, obs_dir, prior_dir,
                f"{self.delta_aux_pf}", run_dir
            )
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        return output_dir

    def obs_in_elong_range(
            self,
            obs_val: float,
            inclusive_lower: bool=True,
            inclusive_upper: bool=True
    ) -> bool:
        """
        Helper function for get_observations to simplify list comprehension and determine whether
         an observation is in the desired elongation range
        :param obs_val: Elongation observation in degrees
        :param inclusive_lower: Boolean to determine whether lower bound is inclusive
        :param inclusive_upper: Boolean to determine whether upper bound is inclusive
        :return: elong_in_range_bool: bool: Boolean that shows if obs is in the range specified (True) or not (False)
        """
        # Initialise output variables
        greater_than_lower_elong: bool = False
        lower_than_upper_elong: bool = False
        elong_in_range: bool = False

        # Determine whether elongation obs. is greater than the lower bound
        if inclusive_lower:
            if obs_val >= self.elong_min:
                greater_than_lower_elong = True
        else:
            if obs_val > self.elong_min:
                greater_than_lower_elong = True

        # Determine whether elongation obs is lower than the upper bound
        if inclusive_upper:
            if obs_val <= self.elong_max:
                lower_than_upper_elong = True
        else:
            if obs_val < self.elong_max:
                lower_than_upper_elong = True

        # Determine whether elongation is within the required range
        if (greater_than_lower_elong & lower_than_upper_elong):
            elong_in_range = True

        return elong_in_range


    def get_observations(self) -> (
            list[float] | tuple[list[datetime.datetime], list[Quantity[u.deg]], list[float], int]
    ):
        """
        Function to get observations from files or generate synthetic observations
        :return: observations: Observations to be assimilated
        """

        # Initialise observations
        obs_class = Observations(
            use_model=self.use_model,
            obs_radius=self.obs_radius,
            obs_lon=self.obs_lon,
            obs_lat=self.obs_lat,
            obs_times_in_datetime=self.obs_times,
            obs_filenames=self.obs_filenames,
            use_synthetic_obs=self.use_synthetic_obs,
            true_cme_par_dict=self.true_cme_par_dict,
            obs_cov=self.obs_cov,
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
            plot_surf_output=False,
            obs_rng_seed=self.obs_rng_seed,
            ssw_event=self.ssw_event,
            craft=self.craft,
            img=self.img,
            use_parallel=self.use_parallel,
        )

        # Get observations and update obs_times, n_obs and obs_lon if necessary
        if self.use_synthetic_obs:
            obs_out = obs_class.observations
            return obs_out
        else:
            obs = obs_class.observations
            obs_times = obs_class.obs_times_in_datetime
            obs_lon = obs_class.obs_lon

            # Limit the observations to be assimilated to the range of elong_min - elong_max
            obs_times_out = [
                x for i, x in enumerate(obs_times) if self.obs_in_elong_range(obs_val=obs[i])
            ]
            obs_lon_out = [
                x for i, x in enumerate(obs_lon) if self.obs_in_elong_range(obs_val=obs[i])
            ]
            obs_out = [
                x for x in obs if ((x >= self.elong_min) & (x <= self.elong_max))
            ]


            n_obs = len(obs_out)

            assert len(obs_times_out) == n_obs
            assert len(obs_lon_out) == n_obs

            return obs_times_out, obs_lon_out, obs_out, n_obs


    def initialise_mean_cme_pars(self, rng):
        #######################################################################
        if self.setup_class.get_use_synthetic_obs():
            # Initialise CME parameters
            mean_cme_t_init = self.fg_mean_cme_t_init
            mean_cme_speed = self.fg_mean_cme_speed + (
                    self.fg_sd_cme_speed * rng.uniform(low=-1, high=1)
            )
            mean_cme_width = self.fg_mean_cme_width + (
                    self.fg_sd_cme_width * rng.uniform(low=-1, high=1)
            )
            mean_cme_lon = self.fg_mean_cme_lon + (self.fg_sd_cme_lon * rng.uniform(low=-1, high=1))
            mean_cme_lat = self.fg_mean_cme_lat
            mean_cme_thick = self.fg_mean_cme_thick
        else:
            # Initialise CME parameters
            mean_cme_t_init = self.fg_mean_cme_t_init + datetime.timedelta(
                seconds=self.fg_sd_cme_t_init * rng.uniform(low=-1, high=1)
            )
            mean_cme_t_init_str = mean_cme_t_init.strftime("%Y%m%d-%H%M")
            mean_cme_speed = self.fg_mean_cme_speed + (
                self.fg_sd_cme_speed * rng.uniform(low=-1, high=1)
            )
            mean_cme_width = self.fg_mean_cme_width + (
                    self.fg_sd_cme_width * rng.uniform(low=-1, high=1)
            )
            mean_cme_lon = self.fg_mean_cme_lon + (self.fg_sd_cme_lon * rng.uniform(low=-1, high=1))
            mean_cme_lat = self.fg_mean_cme_lat + (self.fg_sd_cme_lat * rng.uniform(low=-1, high=1))
            mean_cme_thick = self.fg_mean_cme_thick + (
                    self.fg_sd_cme_thick * rng.uniform(low=-1, high=1)
            )

        mean_cme_par_dict = {
            "mean_cme_t_init": mean_cme_t_init,
            "mean_cme_speed": mean_cme_speed,
            "mean_cme_width": mean_cme_width,
            "mean_cme_lon": mean_cme_lon,
            "mean_cme_lat": mean_cme_lat,
            "mean_cme_thick": mean_cme_thick,
        }

        return mean_cme_par_dict


    def make_cme_par_dict(self, rng):
        # Get mean CME parameters
        mean_cme_par_dict = self.initialise_mean_cme_pars(rng=rng)
        mean_cme_t_init = (
            mean_cme_par_dict["mean_cme_t_init"] - self.surf_init_time
        ).total_seconds()
        mean_cme_speed = mean_cme_par_dict["mean_cme_speed"]
        mean_cme_width = mean_cme_par_dict["mean_cme_width"]
        mean_cme_lon = mean_cme_par_dict["mean_cme_lon"]
        mean_cme_lat = mean_cme_par_dict["mean_cme_lat"]
        mean_cme_thick = mean_cme_par_dict["mean_cme_thick"]

        mean_cme_par_array = [
            mean_cme_t_init,
            mean_cme_speed,
            mean_cme_width,
            mean_cme_lon,
            mean_cme_lat,
            mean_cme_thick
        ]

        # Initialise cme_par_dict
        cme_par_dict = initialise_cme_parameter_ensemble_dict(
            n_ensemble=self.n_members, surf_init_time=self.surf_init_time
        )

        if self.cme_cov_type == "uncorr":
            print(mean_cme_par_array)
            samples = mp_cov.make_uncorr_uniform_samples(
                n_ens=self.n_members,
                mean_cme_pars=mean_cme_par_array,
                rng=rng,
                sd_t_init=self.prior_sd_t_init,
                sd_v=self.prior_sd_speed,
                sd_width=self.prior_sd_width,
                sd_lon=self.prior_sd_lon,
                sd_lat=self.prior_sd_lat,
                sd_thick=self.prior_sd_thick,
            )


        elif self.cme_cov_type == "donki":
            start_time_donki = datetime.datetime(2017, 1, 1, 0, 0, 0)
            end_time_donki = datetime.datetime(2026, 2, 1, 0, 0, 0)

            samples = mp_cov.make_donki_samples(
                n_ens=self.n_members,
                use_model=self.use_model,
                mean_cme_pars=mean_cme_par_array,
                rng=rng,
                start_time=start_time_donki,
                end_time=end_time_donki,
                scale_corr=self.scale_corr,
                vars_req=np.array([p for p in self.pars_in_state if p != "bias"]),
                sd_t_init=self.prior_sd_t_init,
                sd_v=self.prior_sd_speed,
                sd_width=self.prior_sd_width,
                sd_lon=self.prior_sd_lon,
                sd_lat=self.prior_sd_lat,
                sd_thick=self.prior_sd_thick,
                use_log_v=self.use_log_v,
                plot_cme_cov=False,
                most_acc_only="true",
                catalog="ALL",
                feature="LE"
            )

        elif self.cme_cov_type == "mo_cone":
            mo_cone_cov_dir = os.environ.get("MO_CONE_COV_DIR")

            if not os.path.exists(mo_cone_cov_dir):
                os.makedirs(mo_cone_cov_dir)

            mo_cme_cone_file_dir = os.environ.get("MO_CME_CONE_FILE_DIR")

            start_time_mo_cone = datetime.datetime(2017, 1, 1, 0, 0, 0)
            end_time_mo_cone = datetime.datetime(2026, 2, 1, 0, 0, 0)

            samples = mp_cov.make_mo_cone_samples(
                n_ens=self.n_members,
                mean_cme_pars=mean_cme_par_array,
                rng=rng,
                mo_cone_file_dir=mo_cme_cone_file_dir,
                start_time=start_time_mo_cone,
                end_time=end_time_mo_cone,
                overwrite_cov=False,
                mo_cone_cov_dir=mo_cone_cov_dir,
                scale_corr=self.scale_corr,
                vars_req=np.array([p for p in self.pars_in_state if p != "bias"]),
                sd_t_init=self.prior_sd_t_init,
                sd_v=self.prior_sd_speed,
                sd_width=self.prior_sd_width,
                sd_lon=self.prior_sd_lon,
                sd_lat=self.prior_sd_lat,
                sd_thick=self.prior_sd_thick,
                use_log_v=self.use_log_v,
                plot_cme_cov=False,
            )
        else:
            print(f"cme_cov_type provided was: {self.cme_cov_type}")
            print(f"cme_cov_type must be in {self.setup_class.allowed_cov_types()}.")
            sys.exit()

        cme_par_dict["t_init"] = [
            self.surf_init_time + datetime.timedelta(seconds=samples[0, i])
            for i in range(self.n_members)
        ]
        cme_par_dict["v"] = list(samples[1, :]) * u.km / u.s
        cme_par_dict["width"] = list(samples[2, :]) * u.deg
        cme_par_dict["lon"] = list(samples[3, :]) * u.deg
        cme_par_dict["lat"] = list(samples[4, :]) * u.deg
        cme_par_dict["thick"] = list(samples[5, :]) * u.solRad

        # Observation-bias coefficient of each member (see obs_bias.bias_units)
        bias_mode = self.bias_settings["mode"]
        if bias_mode == "estimate":
            cme_par_dict["bias"] = list(rng.normal(
                self.bias_settings["prior_mean"], self.bias_settings["prior_sd"], self.n_members
            ))
        elif bias_mode == "fixed":
            cme_par_dict["bias"] = [float(self.bias_settings["prior_mean"])] * self.n_members
        else:
            cme_par_dict["bias"] = [0.0] * self.n_members

        return cme_par_dict


    def write_elongation_file(
            self,
            output_dir: str,
            elon_saved: dict,
            log_weights: npt.NDArray[float],
            profiles: dict | None = None,
    ) -> str:
        """
        Write every ensemble member's modelled CME flank elongation at each observation to
         <output_dir>/ens_elongations.nc (overwritten after each observation).
        Variables (dimensions n_obs x n_ens unless stated):
            elon_first_stage: elongation (deg) of each prior particle after Liu-West shrinkage, i.e. the
                first-stage observation operator used to choose which particles are resampled
            elon_posterior: elongation (deg) of each resampled (posterior) particle
            parent_index: index of the prior particle each posterior particle was resampled from
            log_weight: posterior log-weight of each particle after this observation
            observation (n_obs): observed elongation (deg)
            assimilated (n_obs): whether the observation was assimilated (False -> elongations are NaN)
            resampled (n_obs): whether the ensemble was resampled at this observation (see
                aux_pf.ADAPTIVE_RESAMPLING); if not, elon_first_stage and elon_posterior are both the
                elongations of the (unchanged) particles and parent_index is 0..N-1
            ess_updated (n_obs): effective sample size of the weights w * p(y | theta) of the current
                particles, which decides whether to resample (NaN if not computed)
        Full elongation profiles (dimensions n_ens x n_profile, on the profile_time grid), written once
         they have been computed (see calculate_elongation_profiles):
            prior_profile_elon: elongation (deg) of the prior ensemble, i.e. the ensemble passed to the
                Aux-PF for the first observation
            posterior_profile_elon: elongation (deg) of the ensemble after the last observation in the
                permitted elongation range has been assimilated
            {prior,posterior}_profile_extrapolated: 1 where the value was extrapolated before the first
                tracked model point (see EXTRAPOLATE_BEFORE_TRACK), 0 where it comes from the model
            {prior,posterior}_profile_log_weight (n_ens): normalised log-weight of each member
        :param output_dir: Run output directory (as used for cme_pars.nc)
        :param elon_saved: Dictionary of arrays filled in run_data_assimilation
        :param log_weights: (n_obs, n_ens) posterior log-weights
        :param profiles: Optional dict with "time" (list of datetimes) and, for "prior" and/or
            "posterior", sub-dicts with "elon" (n_ens, n_profile), "extrapolated" (n_ens, n_profile)
            and "log_weight" (n_ens)
        :return: Path of the written file
        """
        n_obs_run = len(self.observations)
        obs_times = np.array(self.obs_times[:n_obs_run], dtype="datetime64[ns]")

        elon_ds = xr.Dataset(
            data_vars=dict(
                elon_first_stage=(["n_obs", "n_ens"], elon_saved["elon_first_stage"]),
                elon_posterior=(["n_obs", "n_ens"], elon_saved["elon_posterior"]),
                parent_index=(["n_obs", "n_ens"], elon_saved["parent_index"]),
                log_weight=(["n_obs", "n_ens"], np.asarray(log_weights, dtype=float)),
                observation=(["n_obs"], np.asarray(self.observations, dtype=float)),
                assimilated=(["n_obs"], elon_saved["assimilated"].astype(np.int8)),
                resampled=(["n_obs"], elon_saved["resampled"].astype(np.int8)),
                ess_updated=(["n_obs"], elon_saved["ess_updated"]),
            ),
            coords=dict(
                obs_no=("n_obs", np.arange(n_obs_run)),
                obs_time=("n_obs", obs_times),
                ens_no=("n_ens", np.arange(self.n_members)),
            ),
            attrs=dict(
                description="Modelled CME flank elongation of each ensemble member at each observation",
                elongation_units="deg",
                use_model=self.use_model,
                craft=str(self.craft),
            ),
        )

        # Full 3-day elongation profiles of the prior and posterior ensembles
        if profiles is not None:
            elon_ds = elon_ds.assign_coords(
                profile_time=("n_profile", np.array(profiles["time"], dtype="datetime64[ns]"))
            )
            for stage in ["prior", "posterior"]:
                if profiles.get(stage) is None:
                    continue
                elon_ds[f"{stage}_profile_elon"] = (
                    ["n_ens", "n_profile"], np.asarray(profiles[stage]["elon"], dtype=float)
                )
                elon_ds[f"{stage}_profile_extrapolated"] = (
                    ["n_ens", "n_profile"], np.asarray(profiles[stage]["extrapolated"]).astype(np.int8)
                )
                elon_ds[f"{stage}_profile_log_weight"] = (
                    ["n_ens"], np.asarray(profiles[stage]["log_weight"], dtype=float)
                )

        out_file = os.path.join(output_dir, "ens_elongations.nc")
        try:
            elon_ds.to_netcdf(out_file, mode="w")
        except (OSError, PermissionError) as err:
            # e.g. the file is open in another program (Windows locks it). Don't abort a long DA
            #  run over a diagnostic file: warn, and the next observation will try again.
            print(f"Warning: could not write {out_file}: {err}")

        return out_file


    # Full elongation profiles of the prior and posterior ensembles (written to ens_elongations.nc).
    #  Each profile costs one extra ensemble of model runs (about half an observation step).
    compute_elongation_profiles: bool = True
    profile_duration_days: float = 3.0
    profile_dt_minutes: float = 30.0

    def get_profile_times(self) -> list[datetime.datetime]:
        """
        Times at which to evaluate the full elongation profiles: from the model start time for
         profile_duration_days (the compressible SURF runs used by the observation operator are 3
         days long) every profile_dt_minutes.
        :return: List of profile datetimes
        """
        n_times = int(round(self.profile_duration_days * 1440.0 / self.profile_dt_minutes)) + 1
        return [
            self.surf_init_time + datetime.timedelta(minutes=k * self.profile_dt_minutes)
            for k in range(n_times)
        ]

    def calculate_elongation_profiles(
            self,
            cme_par_dict: CmeParEns,
            obs_index: int = 0,
    ) -> dict:
        """
        Run every ensemble member once and evaluate its modelled CME flank elongation over the full
         profile time grid, with the same observation operator (imager, tracking, extrapolation
         before the first tracked point) as used for assimilation.
        NB for a real craft the observer position is taken at the first profile time (the
         observation operator locates the craft at its first time); for twin experiments it is the
         observer position of observation obs_index.
        :param cme_par_dict: Ensemble of CME parameters to evaluate
        :param obs_index: Observation whose observer position is used (twin experiments)
        :return: dict with "elon" and "extrapolated" (n_ens, n_profile) and "log_weight" (n_ens)
        """
        profile_times = self.get_profile_times()
        print(f"Computing elongation profiles at {len(profile_times)} times for {self.n_members} members")

        obs_op_class = ObservationOperator(
            use_model=self.use_model,
            cme_par_dict=cme_par_dict,
            obs_time_in_datetime=profile_times,
            obs_cov=self.obs_cov,
            craft=self.craft,
            obs_radius=self.obs_radius[obs_index],
            obs_lon=self.obs_lon[obs_index],
            obs_lat=self.obs_lat[obs_index],
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
            plot_surf_output=False,
            bias_term_bool=self.bias_term_bool,
            bias_term_5rs=self.bias_term_5rs,
            bias_term_21rs=self.bias_term_21rs,
            use_parallel=self.use_parallel,
            write_diagnostics=False,
        )

        if self.use_model in ["compress_surf", "compress_surf3d"]:
            cme_flanks = obs_op_class.get_cme_flanks()
            elon = np.vstack([np.asarray(f["el"].values, dtype=float) for f in cme_flanks])
            extrap = np.vstack([
                np.asarray(f["extrapolated"].values, dtype=bool) if "extrapolated" in f.columns
                else np.zeros(len(profile_times), dtype=bool)
                for f in cme_flanks
            ])
        else:
            elon = np.asarray(obs_op_class.make_obs_op(), dtype=float)
            extrap = np.zeros(elon.shape, dtype=bool)

        return {
            "elon": elon,
            "extrapolated": extrap,
            "log_weight": np.asarray(cme_par_dict["log_weight"], dtype=float),
        }

    # Observations must be at least this far (deg) above the smallest elongation the model domain
    #  can show (the inner boundary r_min seen from the observer); see get_min_model_elongation
    elong_model_margin_deg: float = 1.0

    def get_min_model_elongation(self) -> npt.NDArray[float]:
        """
        Smallest elongation (deg) at which the model can produce a CME flank at each observation time:
        the model's inner boundary r_min seen from the observer, arcsin(r_min / r_observer).
        Below this the synthetic J-map is masked (no solution inside r_min), so an observation there can
        only be matched by members whose CME is already further out - i.e. the earliest/fastest
        launches - and assimilating it biases the ensemble towards early launches (all other members
        return NaN and are discarded).
        Observer distance: the spacecraft's ephemeris for real craft, otherwise obs_radius.
        :return: Array (n_obs) of minimum model elongations in degrees (NaN if unknown)
        """
        n_obs_run = len(self.observations)
        r_min_rs = self.r_min.to(u.solRad).value

        r_obs_rs = None
        craft = (self.craft or "").lower()
        if craft not in ["", "twin", "none"]:
            try:
                craft_obs = S.Observer(craft.upper(), Time(list(self.obs_times[:n_obs_run])))
                r_obs_rs = np.asarray(craft_obs.r.to(u.solRad).value, dtype=float)
            except Exception as err:
                print(f"get_min_model_elongation: could not get {craft} ephemeris ({err}); using obs_radius")

        if r_obs_rs is None:
            r_obs_rs = np.array([
                (r if isinstance(r, Quantity) else r * u.AU).to(u.solRad).value
                for r in (self.obs_radius if isinstance(self.obs_radius, list)
                          else [self.obs_radius] * n_obs_run)
            ][:n_obs_run], dtype=float)

        with np.errstate(invalid="ignore"):
            elong_min_model = np.rad2deg(np.arcsin(np.clip(r_min_rs / r_obs_rs, 0.0, 1.0)))

        return elong_min_model

    def run_data_assimilation(self, run_start=0):
        """
        Function to run the data assimilation routine
        :return: None
        """
        # Negative run numbers count downwards (run_start, run_start - 1, ...).
        #  (Previously self.n_runs itself was negated, so the progress message showed a negative
        #  run count and a second call to this method flipped the sign back and ran the wrong range.)
        if run_start < 0:
            step = -1
        else:
            step = 1

        n_runs_req = step * self.n_runs

        for run_no in range(run_start, run_start + n_runs_req, step):
            print(f"run_no = {run_no}")
            if run_no < 0:
                self.fg_sd_cme_t_init: float = 0
                self.fg_sd_cme_speed: float = 0
                self.fg_sd_cme_width: float = 0
                self.fg_sd_cme_lon: float = 0
                self.fg_sd_cme_lat: float = 0
                self.fg_sd_cme_thick: float = 0

            # Initialise random generator
            rand_seed = self.generate_random_seed(
                add_const=42,
                mult_const=10000,
                run_no=run_no
            )
            rng: Generator = np.random.default_rng(rand_seed)

            cme_par_dict = self.make_cme_par_dict(rng=rng)
            cme_saved_pars = np.zeros((self.n_obs + 1, self.n_members, 7))
            # Observation-bias coefficient of each member, prior (row 0) and after each observation
            bias_saved = np.zeros((self.n_obs + 1, self.n_members))
            bias_saved[0, :] = np.asarray(cme_par_dict.get("bias", 0.0), dtype=float)

            # Correlation handling: working coordinates fixed from this run's prior ensemble
            state_transform = self.make_state_transform(cme_par_dict)
            # Weighted correlation matrix of the state parameters (v as ln v), prior and after each obs
            n_state = len(self.pars_in_state)
            corr_saved = np.full((self.n_obs + 1, n_state, n_state), np.nan)
            corr_saved[0] = self.state_correlation(cme_par_dict)

            # Modelled elongation of every ensemble member at every observation (NaN where the
            #  observation was not assimilated, or the member's CME was not seen)
            n_obs_run = len(self.observations)
            elon_saved = {
                "elon_first_stage": np.full((n_obs_run, self.n_members), np.nan),
                "elon_posterior": np.full((n_obs_run, self.n_members), np.nan),
                "parent_index": np.full((n_obs_run, self.n_members), -1, dtype=int),
                "assimilated": np.zeros(n_obs_run, dtype=bool),
                "resampled": np.zeros(n_obs_run, dtype=bool),
                "ess_updated": np.full(n_obs_run, np.nan),
            }

            # Save prior state
            cme_saved_pars[0, :, 0] = [
                (
                        cme_par_dict["t_init"][i] - cme_par_dict["surf_init_time"]
                ).total_seconds()
                for i in range(self.n_members)
            ]
            cme_saved_pars[0, :, 1] = cme_par_dict["v"].to(u.km / u.s).value
            cme_saved_pars[0, :, 2] = cme_par_dict["width"].to(u.deg).value

            cme_saved_pars[0, :, 3] = cme_par_dict["lon"].to(u.deg).value
            lon_cond = cme_saved_pars[0, :, 3] > 180
            cme_saved_pars[0, lon_cond, 3] = (
                    cme_saved_pars[0, lon_cond, 3] - 360
            )

            cme_saved_pars[0, :, 4] = cme_par_dict["lat"].to(u.deg).value
            cme_saved_pars[0, :, 5] = cme_par_dict["thick"].to(u.solRad).value
            cme_saved_pars[0, :, 6] = cme_par_dict["log_weight"]

            # Make output directory
            output_dir = self.make_output_dir(
                cme_par_dict=cme_par_dict, run_no=run_no
            )

            # Lowest elongation each observation may have to be assimilable by the model
            elong_min_model = self.get_min_model_elongation()
            elong_min_usable = np.fmax(elong_min_model + self.elong_model_margin_deg, self.elong_min_obs)
            print(
                f"Minimum usable elongation (model inner boundary {self.r_min} + "
                f"{self.elong_model_margin_deg} deg margin): "
                f"{np.nanmin(elong_min_usable):.2f}-{np.nanmax(elong_min_usable):.2f} deg"
            )

            # Full elongation profiles of the prior ensemble (the ensemble passed to the Aux-PF for
            #  the first assimilated observation: skipped observations do not change it)
            profiles = None
            if self.compute_elongation_profiles:
                profiles = {"time": self.get_profile_times(), "prior": None, "posterior": None}
                try:
                    profiles["prior"] = self.calculate_elongation_profiles(cme_par_dict, obs_index=0)
                except Exception as err:
                    print(f"Warning: could not compute the prior elongation profiles: {err}")

            for yi, obs in enumerate(self.observations):
                print(f"run_no = {run_no}/{self.n_runs}, obs = {yi}/{len(self.observations)}")
                # Initial conditions for running the DA, that the observation time is after
                #  the first-guess of the CME launch and that the observation is in the
                #  set [elong_min_obs, elong_max_obs]
                fg_cme_init_cond = self.obs_times[yi] > self.fg_mean_cme_t_init
                obs_in_elon_range_cond = (
                    (obs >= self.elong_min_obs) and (obs <= self.elong_max_obs)
                )
                # ... and above the smallest elongation the model domain can show from this observer
                obs_above_model_min_cond = bool(
                    np.isnan(elong_min_usable[yi]) or (obs >= elong_min_usable[yi])
                )
                if obs_in_elon_range_cond and not obs_above_model_min_cond:
                    print(
                        f"Skipping obs {yi} ({obs:.2f} deg): below the model's minimum visible elongation "
                        f"{elong_min_model[yi]:.2f} deg + {self.elong_model_margin_deg} deg margin "
                        f"(inner boundary r_min = {self.r_min})"
                    )
                obs_in_elon_range_cond = obs_in_elon_range_cond and obs_above_model_min_cond

                if fg_cme_init_cond and obs_in_elon_range_cond:
                    aux_pf_class = AuxPF(
                        use_model=self.use_model,
                        cme_par_dict=cme_par_dict,
                        obs=obs,
                        obs_cov=self.obs_cov,
                        obs_radius=self.obs_radius[yi],
                        obs_lon=self.obs_lon[yi],
                        obs_lat=self.obs_lat[yi],
                        obs_time=self.obs_times[yi],
                        craft=self.craft,
                        true_cme_par_dict=self.true_cme_par_dict,
                        infl_fact=1,
                        pars_in_state=self.pars_in_filter,
                        vr_in=self.vr_in,
                        vr_longs=self.vr_longs,
                        vr_lats=self.vr_lats,
                        n_lat_surf=self.n_lat_surf,
                        lon_start=self.lon_start,
                        lon_stop=self.lon_stop,
                        time_tolerance=self.time_tolerance,
                        dt_scale=self.dt_scale,
                        delta_aux_pf=self.delta_aux_pf,
                        r_min=self.r_min,
                        cme_init_rad=self.cme_init_rad,
                        cme_fixed_duration = self.cme_fixed_duration,
                        fixed_duration=self.fixed_duration,
                        bias_term_bool=self.bias_term_bool,
                        bias_term_5rs=self.bias_term_5rs,
                        bias_term_21rs=self.bias_term_21rs,
                        use_parallel=self.use_parallel,
                        rng=rng,
                        state_transform=state_transform
                    )

                    cme_par_dict = aux_pf_class.aux_pf()

                    # Record each member's modelled elongation at this observation
                    elon_saved["elon_first_stage"][yi, :] = aux_pf_class.elon_first_stage
                    elon_saved["elon_posterior"][yi, :] = aux_pf_class.elon_posterior
                    elon_saved["parent_index"][yi, :] = aux_pf_class.parent_index
                    # (False if the Aux-PF left the ensemble unchanged, e.g. because a member's CME
                    #  produced no elongation; see aux_pf.SKIP_UPDATE_IF_ANY_MEMBER_NAN)
                    elon_saved["assimilated"][yi] = not getattr(aux_pf_class, "update_skipped", False)
                    elon_saved["resampled"][yi] = bool(getattr(aux_pf_class, "resampled", True))
                    elon_saved["ess_updated"][yi] = getattr(aux_pf_class, "ess_updated", np.nan)
                    if getattr(aux_pf_class, "update_skipped", False):
                        print(f"Observation {yi} not assimilated: {aux_pf_class.skip_reason}")
                elif fg_cme_init_cond:
                    # (these two messages were previously swapped)
                    print("Observation is outside the permitted elongation range for the model")
                elif obs_in_elon_range_cond:
                    print("Obs time before CME is expected to be in SURF model domain")
                else:
                    print("Obs time before CME launch and obs outside permitted elongation range")

                # Standardise the units and remove the astropy units
                cme_saved_pars[yi + 1, :, 0] = [
                    (
                        cme_par_dict["t_init"][i] - cme_par_dict["surf_init_time"]
                    ).total_seconds()
                    for i in range(self.n_members)
                ]
                cme_saved_pars[yi + 1, :, 1] = cme_par_dict["v"].to(u.km / u.s).value
                cme_saved_pars[yi + 1, :, 2] = cme_par_dict["width"].to(u.deg).value

                cme_saved_pars[yi + 1, :, 3] = cme_par_dict["lon"].to(u.deg).value
                lon_cond = cme_saved_pars[yi + 1, :, 3] > 180
                cme_saved_pars[yi + 1, lon_cond, 3] = (
                        cme_saved_pars[yi + 1, lon_cond, 3] - 360
                )

                cme_saved_pars[yi + 1, :, 4] = cme_par_dict["lat"].to(u.deg).value
                cme_saved_pars[yi + 1, :, 5] = cme_par_dict["thick"].to(u.solRad).value
                cme_saved_pars[yi + 1, :, 6] = cme_par_dict["log_weight"]
                bias_saved[yi + 1, :] = np.asarray(cme_par_dict.get("bias", 0.0), dtype=float)
                corr_saved[yi + 1] = self.state_correlation(cme_par_dict)

                # Write the elongations after every observation, so a run that stops part-way
                #  still leaves everything computed so far on disk
                self.write_elongation_file(
                    output_dir=output_dir,
                    elon_saved=elon_saved,
                    # (observations not yet reached are NaN rather than the array's initial zeros)
                    log_weights=np.where(
                        (np.arange(n_obs_run) <= yi)[:, None],
                        cme_saved_pars[1:n_obs_run + 1, :, 6],
                        np.nan
                    ),
                    profiles=profiles,
                )

            # Full elongation profiles of the posterior ensemble (after the last observation in the
            #  permitted elongation range has been assimilated)
            if profiles is not None:
                assimilated_ind = np.flatnonzero(elon_saved["assimilated"])
                last_obs_ind = int(assimilated_ind[-1]) if assimilated_ind.size else 0
                try:
                    profiles["posterior"] = self.calculate_elongation_profiles(
                        cme_par_dict, obs_index=last_obs_ind
                    )
                except Exception as err:
                    print(f"Warning: could not compute the posterior elongation profiles: {err}")
                self.write_elongation_file(
                    output_dir=output_dir,
                    elon_saved=elon_saved,
                    log_weights=cme_saved_pars[1:n_obs_run + 1, :, 6],
                    profiles=profiles,
                )

            for par_ind in range(7):
                if par_ind == 6:
                    print(f"weights_post = {np.exp(cme_saved_pars[-1, :, par_ind])}")
                print(f"mean_cme_saved_pars[{par_ind}] = {np.mean(cme_saved_pars[:, :, par_ind], axis=1)}")

            # Save cme_parameters into a .nc file
            # Make an xarray object
            if self.use_model in ["surf", "huxt"]:
                cme_par_ds = xr.Dataset(
                    data_vars=dict(
                        model_init_time=self.surf_init_time,
                        cme_init_rad=self.cme_init_rad,
                        r_min=self.r_min,
                        ambient_vr=(["n_lon"], self.vr_in),
                        t_init=(["n_obs", "n_ens"], cme_saved_pars[:, :, 0]),
                        v=(["n_obs", "n_ens"], cme_saved_pars[:, :, 1]),
                        width=(["n_obs", "n_ens"], cme_saved_pars[:, :, 2]),
                        lon=(["n_obs", "n_ens"], cme_saved_pars[:, :, 3]),
                        lat=(["n_obs", "n_ens"], cme_saved_pars[:, :, 4]),
                        thick=(["n_obs", "n_ens"], cme_saved_pars[:, :, 5]),
                        log_weight=(["n_obs", "n_ens"], cme_saved_pars[:, :, 6]),
                    ),
                    coords=dict(
                        obs_no=("n_obs", range(self.n_obs + 1)),
                        ens_no=("n_ens", range(self.n_members)),
                        surf_lon=("n_lon", (2 * np.pi / self.n_lon_in) * np.arange(self.n_lon_in)),
                    ),
                )
            elif self.use_model in ["compress_surf"]:
                cme_par_ds = xr.Dataset(
                    data_vars=dict(
                        model_init_time=self.surf_init_time,
                        cme_init_rad=self.cme_init_rad,
                        r_min=self.r_min,
                        ambient_vr=(["n_lon"], self.vr_in),
                        t_init=(["n_obs", "n_ens"], cme_saved_pars[:, :, 0]),
                        v=(["n_obs", "n_ens"], cme_saved_pars[:, :, 1]),
                        width=(["n_obs", "n_ens"], cme_saved_pars[:, :, 2]),
                        lon=(["n_obs", "n_ens"], cme_saved_pars[:, :, 3]),
                        lat=(["n_obs", "n_ens"], cme_saved_pars[:, :, 4]),
                        thick=(["n_obs", "n_ens"], cme_saved_pars[:, :, 5]),
                        log_weight=(["n_obs", "n_ens"], cme_saved_pars[:, :, 6]),
                    ),
                    coords=dict(
                        obs_no=("n_obs", range(self.n_obs + 1)),
                        ens_no=("n_ens", range(self.n_members)),
                        surf_lon=("n_lon", (2 * np.pi / self.n_lon_in) * np.arange(self.n_lon_in)),
                    ),
                )
            elif self.use_model in ["compress_surf3d"]:
                print(np.shape(self.vr_in))
                print(f"{self.n_lon_in}, {self.n_lat_in}")
                cme_par_ds = xr.Dataset(
                    data_vars=dict(
                        model_init_time=self.surf_init_time,
                        cme_init_rad=self.cme_init_rad,
                        r_min=self.r_min,
                        ambient_vr=(["n_lat", "n_lon"], self.vr_in),
                        vr_longs=(["n_lon"], self.vr_longs),
                        vr_lats=(["n_lat"], self.vr_lats),
                        t_init=(["n_obs", "n_ens"], cme_saved_pars[:, :, 0]),
                        v=(["n_obs", "n_ens"], cme_saved_pars[:, :, 1]),
                        width=(["n_obs", "n_ens"], cme_saved_pars[:, :, 2]),
                        lon=(["n_obs", "n_ens"], cme_saved_pars[:, :, 3]),
                        lat=(["n_obs", "n_ens"], cme_saved_pars[:, :, 4]),
                        thick=(["n_obs", "n_ens"], cme_saved_pars[:, :, 5]),
                        log_weight=(["n_obs", "n_ens"], cme_saved_pars[:, :, 6]),
                    ),
                    coords=dict(
                        obs_no=("n_obs", range(self.n_obs + 1)),
                        ens_no=("n_ens", range(self.n_members)),
                        surf_lon=("n_lon", (2 * np.pi / self.n_lon_in) * np.arange(self.n_lon_in)),
                        surf_lat=("n_lat", range(self.n_lat_in)),
                    ),
                )

            else:
                sys.exit(f"Saving output for use_model = {self.use_model} is not supported")

            # Observation-bias coefficients and settings (see obs_bias.py)
            cme_par_ds["bias"] = (["n_obs", "n_ens"], bias_saved)
            cme_par_ds.attrs.update({
                "bias_mode": self.bias_settings["mode"],
                "bias_predictor": self.bias_settings["predictor"]["type"],
                "bias_e_ref_deg": self.bias_settings["predictor"]["e_ref_deg"],
                "bias_e_scale_deg": self.bias_settings["predictor"]["e_scale_deg"],
                "bias_prior_mean": float(self.bias_settings["prior_mean"]),
                "bias_prior_sd": float(self.bias_settings["prior_sd"]),
                "bias_prior_source": self.bias_settings["prior_source"],
                "true_bias": float(self.true_cme_par_dict.get("bias", 0.0)) if self.use_synthetic_obs else np.nan,
            })

            # Parameter correlations (v as ln v) and the correlation-handling settings (state_transform.py)
            cme_par_ds["state_corr"] = (["n_obs", "state_par", "state_par_2"], corr_saved)
            cme_par_ds = cme_par_ds.assign_coords(
                state_par=("state_par", list(self.pars_in_state)),
                state_par_2=("state_par_2", list(self.pars_in_state)),
            )
            cs = self.correlation_settings
            cme_par_ds.attrs.update({
                "corr_enabled": int(cs["enabled"]),
                "corr_reparam": cs["reparam"],
                "corr_t_ref_rs": float(cs["t_ref_rs"]) if cs["t_ref_rs"] is not None else np.nan,
                "corr_whiten": int(bool(cs["whiten"])),
                "corr_prior_cov_shrinkage": float(cs["prior_cov_shrinkage"]),
                "corr_shrinkage_alpha": float(cs["shrinkage_alpha"]),
                "corr_min_jitter_sd": float(cs["min_jitter_sd"]),
                "corr_carry_from_prior": ",".join(cs["carry_from_prior"]),
                "pars_in_filter": ",".join(self.pars_in_filter),
            })

            print(cme_par_ds)
            outParFile = os.path.join(output_dir, "cme_pars.nc")
            cme_par_ds.to_netcdf(outParFile, mode='w')

        return None


def main():
    print_environment_variables()

    # Reuse the Setup instance built by RunDataAssimilationRoutine
    #  (each Setup() re-reads/re-maps the WSA boundary conditions)
    run_da_class = RunDataAssimilationRoutine()
    setup_class = run_da_class.setup_class
    da_par_dict = setup_class.da_par_dict

    run_start = da_par_dict["run_start"]

    try:
        if setup_class.get_use_synthetic_obs():
            run_da_class.run_data_assimilation(run_start=run_start)
        else:
            run_da_class.run_data_assimilation(run_start=-1)
    finally:
        # Close the persistent worker pool used by the observation operator
        shutdown_worker_pool()

    return None

if __name__ == "__main__":
    main()