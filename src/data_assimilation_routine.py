import argparse

import numpy as np
import numpy.typing as npt
import datetime
import filecmp
import os
import shutil
import sys

import pandas as pd
import xarray as xr

import surf.surf as S

import astropy.units as u
from astropy.units import Quantity
from astropy.time import Time

import make_prior_covariance_mat as mp_cov

from init_sir import initialise_cme_parameter_ensemble_dict
from sir_observations import Observations
from aux_pf import AuxPF
from cme_par_ens import CmeParEns

from dataclasses import replace
from typing import Any

from da_surf_setup import Setup, load_environment, events_file, load_events
from aux_pf import PFSettings
from model_run_reuse import ElongationStore
from sir_observation_operator import ObservationOperator, ObsOpSettings, WorkerPool
from obs_bias import resolve_bias_settings
from state_transform import (
    StateTransform,
    resolve_correlation_settings,
    filter_state_pars,
    weighted_corr,
)
from to_state_vector import ToStateVector
from output_names import (
    bias_dir_name,
    true_bias_suffix,
    twin_sampling_dir_name,
    corr_dir_name,
    check_path_length,
    PathLimits,
)


def print_environment_variables() -> None:
    print(f"obs_dir = {os.getenv('OBS_DIR')}")
    print(f"mo_cone_cov_dir = {os.getenv('MO_CONE_COV_DIR')}")
    print(f"donki_cov_dir = {os.getenv('DONKI_COV_DIR')}")
    print(f"mo_cme_cone_file_dir = {os.getenv('MO_CME_CONE_FILE_DIR')}")
    print(f"out_base_dir = {os.getenv('OUT_BASE_DIR')}")
    print(f"wsa_dir = {os.getenv('WSA_DIR')}")

    return None


class RunDataAssimilationRoutine:
    def __init__(self, setup_class: type[Setup] = Setup) -> None:
        """
        :param setup_class: Setup class giving the run's settings (da_surf_setup.Setup, or a subclass
            of it, e.g. for a twin experiment)
        """
        self.setup_class: Setup = setup_class()

        # Settings of the filter and the observation operator, and the worker pool shared by every
        #  observation operator of the run (shut down at the end of run_data_assimilation). The
        #  observation-bias predictor comes from the bias settings, and is also needed for the
        #  synthetic observations made below.
        da_par_dict: dict[str, Any] = self.setup_class.da_par_dict
        self.bias_settings: dict = resolve_bias_settings(
            da_par_dict.get("bias_settings")
        )
        self.pf_settings: PFSettings = da_par_dict.get("pf_settings") or PFSettings()
        self.obs_op_settings: ObsOpSettings = replace(
            da_par_dict.get("obs_op_settings") or ObsOpSettings(),
            bias_predictor=self.bias_settings["bias_predictor"],
        )
        self.worker_pool: WorkerPool = WorkerPool()

        # Get SURF parameters
        surf_par_dict: dict[str, Any] = self.setup_class.surf_par_dict
        self.use_model: str = surf_par_dict["use_model"]
        self.surf_init_time: datetime.datetime = surf_par_dict["surf_init_time"]
        self.vr_in: Quantity[u.km / u.s] = surf_par_dict["vr_in"]
        self.vr_longs: Quantity[u.deg] = surf_par_dict["vr_longs"]
        self.vr_lats: Quantity[u.deg] = surf_par_dict["vr_lats"]
        self.lon_start: Quantity[u.deg] = surf_par_dict["lon_start"]
        self.lon_stop: Quantity[u.deg] = surf_par_dict["lon_stop"]
        self.n_lat_surf: int = surf_par_dict["n_lat_surf"]
        # Latitude range of the SURF3d grid (NaN: set from the observing craft's field of view)
        self.lat_min: Quantity[u.deg] = surf_par_dict["latitude_min"]
        self.lat_max: Quantity[u.deg] = surf_par_dict["latitude_max"]
        self.sim_time: Quantity[u.days] = surf_par_dict["sim_time"]
        self.dt_scale: int | float = surf_par_dict["dt_scale"]
        self.r_min: Quantity[u.solRad] = surf_par_dict["r_min"]
        self.cme_init_rad: Quantity[u.solRad] = surf_par_dict["cme_init_rad"]
        self.cme_fixed_duration: bool = surf_par_dict["cme_fixed_duration"]
        self.fixed_duration: Quantity[u.s] = surf_par_dict["fixed_duration"]

        # Number of longitudes and latitudes of the ambient wind boundary condition (for cme_pars.nc)
        if isinstance(self.vr_longs.value, float):
            self.n_lon_in: int = 1
        elif not np.all(np.isnan(self.vr_longs)):
            self.n_lon_in: int = len(self.vr_longs)
        else:
            self.n_lon_in: int = len(self.vr_in)

        if isinstance(self.vr_lats.value, float) or np.all(np.isnan(self.vr_lats)):
            self.n_lat_in: int = 1
        else:
            self.n_lat_in: int = len(self.vr_lats)

        # Initialise observation variables
        obs_par_dict: dict[str, Any] = self.setup_class.obs_par_dict
        self.n_obs: int = obs_par_dict["n_obs"]
        self.obs_radius: Quantity[u.AU] | list[Quantity[u.AU]] = obs_par_dict[
            "obs_radius"
        ]
        self.obs_lon: Quantity[u.deg] | list[Quantity[u.deg]] = obs_par_dict["obs_lon"]
        self.obs_lat: Quantity[u.deg] | list[Quantity[u.deg]] = obs_par_dict["obs_lat"]
        self.obs_cov: float | npt.NDArray[np.float64] = obs_par_dict["obs_cov"]
        self.obs_times: list[datetime.datetime] = obs_par_dict["obs_times"]
        self.elong_min: float = obs_par_dict["elong_min"]
        self.elong_max: float = obs_par_dict["elong_max"]
        self.elong_min_obs: float = obs_par_dict["elong_min_obs"]
        self.elong_max_obs: float = obs_par_dict["elong_max_obs"]
        self.use_synthetic_obs: bool = obs_par_dict["use_synthetic_obs"]
        self.obs_filenames: list[str] = obs_par_dict["obs_filenames"]
        self.obs_rng_seed: int = obs_par_dict["obs_rng_seed"]
        self.ssw_event: str = obs_par_dict["ssw_event"]
        self.craft: str = obs_par_dict["craft"]
        self.img: str = obs_par_dict["img"]
        self.use_parallel: bool = obs_par_dict["use_parallel"]

        # If we're using synthetic observations/ running OSSEs define true_cme_par_dict
        self.true_cme_par_dict: CmeParEns = self.setup_class.true_cme_par_dict

        # Twin experiments with several runs (Setup.get_twin_run_sampling, from events.yaml): each
        #  run's first guess drawn from the prior around the truth, and/or new observation noise per
        #  run. Both False for real observations
        twin_sampling: dict[str, bool] = self.setup_class.get_twin_run_sampling()
        self.sample_first_guess: bool = bool(twin_sampling["sample_first_guess"])
        self.new_obs_noise_per_run: bool = bool(twin_sampling["new_obs_noise_per_run"])
        # Whether this run's first guess is perturbed (False for negative run numbers)
        self.perturb_first_guess: bool = True
        # This run's first guess (set by make_cme_par_dict)
        self.run_first_guess: dict[str, Any] = {}
        # Noise-free synthetic observations and the noise sd (set by get_observations)
        self.unperturbed_observations: list[float] = []
        self.obs_noise_sd: float = 0.0

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
        fg_cme_par_dict: dict[str, Any] = self.setup_class.fg_cme_par_dict
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
        prior_cme_par_dict: dict[str, Any] = self.setup_class.prior_cme_cov
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
        self.n_members: int = da_par_dict["n_members"]
        self.n_runs: int = da_par_dict["n_runs"]
        self.delta_aux_pf: float = da_par_dict["delta_aux_pf"]
        self.pars_in_state: list[str] = da_par_dict["pars_in_state"]
        self.time_tolerance: Quantity[u.day] = da_par_dict["time_tolerance"]

        # Observation-bias term (see obs_bias.py and Setup.get_bias_settings; resolved above)
        if (
            self.bias_settings["mode"] == "estimate"
            and "bias" not in self.pars_in_state
        ):
            self.pars_in_state = self.pars_in_state + ["bias"]

        # Handling of correlated parameters in the Liu-West step (see state_transform.py and
        #  Setup.get_correlation_settings). pars_in_state is still used to draw the prior; the
        #  Aux-PF updates only pars_in_filter (pars_in_state without those carried from the prior).
        self.correlation_settings: dict = resolve_correlation_settings(
            da_par_dict.get("correlation_settings")
        )
        self.pars_in_filter: list[str] = filter_state_pars(
            self.pars_in_state, self.correlation_settings
        )
        if self.correlation_settings["enabled"]:
            print(
                f"Correlation handling on: {self.correlation_settings}; "
                f"parameters updated by the filter: {self.pars_in_filter}"
            )

    def generate_random_seed(
        self, add_const: int = None, mult_const: int = None, run_no: int = None
    ) -> int:
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
            random_seed: int = add_const + (mult_const * run_no)
        else:
            random_seed = int(
                np.abs((add_const + (mult_const / 2.0)) + (run_no * mult_const))
            )

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
        prior_state: npt.NDArray[np.float64] = ToStateVector(
            cme_par_dict, pars_in_state_vector=self.pars_in_filter
        ).cme_par_dict_to_state_vector()
        transform: StateTransform = StateTransform(
            pars=self.pars_in_filter,
            settings=self.correlation_settings,
            r_min_rs=float(self.r_min.to(u.solRad).value),
            cme_init_rad_rs=float(self.cme_init_rad.to(u.solRad).value),
            prior_state=prior_state,
            prior_log_weights=cme_par_dict["log_weight"],
        )
        print(
            f"Liu-West working coordinates: {transform.working_labels()}"
            f"{' (whitened with the prior covariance)' if transform.whiten else ''}"
        )
        return transform

    def state_correlation(self, cme_par_dict: CmeParEns) -> npt.NDArray[np.float64]:
        """
        Weighted correlation matrix of the parameters in pars_in_state (v as ln v), for diagnosing how
         the filter handles parameter correlations
        :param cme_par_dict: CME parameter ensemble
        :return: (n_state, n_state) correlation matrix
        """
        x: npt.NDArray[np.float64] = ToStateVector(
            cme_par_dict, pars_in_state_vector=self.pars_in_state
        ).cme_par_dict_to_state_vector()
        if "v" in self.pars_in_state:
            i_v: int = self.pars_in_state.index("v")
            x[:, i_v] = np.log(np.clip(x[:, i_v], 1e-6, None))
        return weighted_corr(x, cme_par_dict["log_weight"])

    def make_output_dir(self, cme_par_dict: CmeParEns, run_no: int) -> str:
        """
        Function to make the output directory
        :return: out_dir: Output directory
        """
        base_dir: str = os.path.join(
            os.getenv("OUT_BASE_DIR"), f"ens_{self.n_members}_{self.cme_cov_type}"
        )
        model_dir: str = f"model_{self.use_model.upper()}"

        if self.use_synthetic_obs:
            # (single quotes inside the f-string so this also parses on Python < 3.12)
            obs_dir: str = (
                f"truth_{self.true_cme_par_dict['v'].value}_{self.true_cme_par_dict['width'].value}"
                f"_{self.true_cme_par_dict['lon'].value}_{self.true_cme_par_dict['lat'].value}"
                f"_{self.true_cme_par_dict['thick'].value}"
            )
        else:
            obs_dir = f"obs_{self.craft}_{self.ssw_event}_{self.img}"

        prior_dir: str = (
            f"prior_{self.fg_mean_cme_speed:.1f}_{self.fg_mean_cme_width:.1f}"
            f"_{self.fg_mean_cme_lon:.1f}_{self.fg_mean_cme_lat:.1f}"
            f"_{self.fg_mean_cme_thick:.1f}"
        )
        run_dir: str = f"run_{run_no:03d}"

        # Separate folders for twin runs that sample the first guess and/or the observation noise
        sampling_tag: str = (
            twin_sampling_dir_name(self.sample_first_guess, self.new_obs_noise_per_run)
            if self.use_synthetic_obs
            else ""
        )
        if sampling_tag:
            prior_dir = f"{sampling_tag}_{prior_dir}"
        # Separate folders for runs whose first-stage weights use the current particles
        #  (PFSettings.first_stage = "current"; the results differ from the usual shrunk particles).
        #  Reusing model runs (with reuse_observer_tol_deg = 0) does not change the results, so
        #  needs no folder of its own.
        if self.pf_settings.first_stage != "shrunk":
            prior_dir = f"fsc_{prior_dir}"

        # Separate folders for runs with an observation-bias term (and, in twins, the injected bias).
        #  Short names (output_names.py): Windows limits a full path to 259 characters, and netCDF
        #  files in a longer path cannot be written ("PermissionError: [Errno 13]").
        if self.bias_settings["mode"] != "off":
            bias_tag: str = bias_dir_name(
                self.bias_settings["mode"],
                self.bias_settings["predictor"]["type"],
                self.bias_settings["prior_mean"],
                self.bias_settings["prior_sd"],
            )
            prior_dir = f"{bias_tag}_{prior_dir}"
        # Separate folders for runs with correlation handling
        if self.correlation_settings["enabled"]:
            prior_dir = os.path.join(
                corr_dir_name(self.correlation_settings), prior_dir
            )
        if (
            self.use_synthetic_obs
            and float(self.true_cme_par_dict.get("bias", 0.0)) != 0.0
        ):
            obs_dir = f"{obs_dir}{true_bias_suffix(self.true_cme_par_dict['bias'])}"

        output_dir: str = os.path.join(
            base_dir,
            model_dir,
            obs_dir,
            f"{prior_dir}_{self.delta_aux_pf}",
            run_dir,
        )
        # Stop now, before any model runs, if the run's files could not be written; warn if the
        #  analysis outputs that osse_ana.py writes into the run folder would be too long
        check_path_length(
            os.path.join(output_dir, "ens_elongations.nc"), what="run output file"
        )
        check_path_length(
            output_dir,
            what="run folder",
            allowance=PathLimits.RUN_DIR_FILE_ALLOWANCE + 1,
            error=False,
        )
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.copy_events_file(output_dir)

        return output_dir

    @staticmethod
    def copy_events_file(output_dir: str) -> None:
        """
        Copy events.yaml (truth and first guesses, latitude range, model start, ... of the events and
        twin experiments) into the experiment folder, the folder above run_NNN, so osse_ana.py can
        read the parameters a run was made with. If the folder already has a copy that differs
        (events.yaml was edited between runs of the experiment), that copy is kept and this run's is
        written as events_run_NNN.yaml
        :param output_dir: The run folder (.../run_NNN)
        """
        experiment_dir: str = os.path.dirname(os.path.normpath(output_dir))
        source: str = str(events_file())
        target: str = os.path.join(experiment_dir, "events.yaml")
        if os.path.isfile(target) and not filecmp.cmp(source, target, shallow=False):
            target = os.path.join(
                experiment_dir,
                f"events_{os.path.basename(os.path.normpath(output_dir))}.yaml",
            )
            print(
                f"events.yaml differs from the copy already in {experiment_dir}; this run's copy is"
                f" saved as {os.path.basename(target)}"
            )
        try:
            shutil.copyfile(source, target)
        except OSError as err:
            print(f"Warning: could not copy events.yaml to {target} ({err})")

    def obs_in_elong_range(
        self, obs_val: float, inclusive_lower: bool = True, inclusive_upper: bool = True
    ) -> bool:
        """
        Helper function for get_observations to simplify list comprehension and determine whether
         an observation is in the desired elongation range
        :param obs_val: Elongation observation in degrees
        :param inclusive_lower: Boolean to determine whether lower bound is inclusive
        :param inclusive_upper: Boolean to determine whether upper bound is inclusive
        :return: elong_in_range_bool: bool: Boolean that shows if obs is in the range specified (True) or not (False)
        """
        above_lower: bool = (
            (obs_val >= self.elong_min)
            if inclusive_lower
            else (obs_val > self.elong_min)
        )
        below_upper: bool = (
            (obs_val <= self.elong_max)
            if inclusive_upper
            else (obs_val < self.elong_max)
        )

        return bool(above_lower and below_upper)

    def get_observations(
        self,
    ) -> (
        list[float]
        | tuple[list[datetime.datetime], list[Quantity[u.deg]], list[float], int]
    ):
        """
        Function to get observations from files or generate synthetic observations
        :return: observations: Observations to be assimilated
        """

        # Initialise observations
        obs_class: Observations = Observations(
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
            lat_min=self.lat_min,
            lat_max=self.lat_max,
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
            obs_op_settings=self.obs_op_settings,
            worker_pool=self.worker_pool,
        )

        # Get observations and update obs_times, n_obs and obs_lon if necessary
        if self.use_synthetic_obs:
            obs_out: list[float] = obs_class.observations
            self.unperturbed_observations = list(obs_class.unperturbed_observations)
            self.obs_noise_sd = float(obs_class.obs_noise_sd)
            return obs_out
        else:
            obs: list[float] = obs_class.observations
            obs_times: list[datetime.datetime] = obs_class.obs_times_in_datetime
            obs_lon: list[Quantity] = obs_class.obs_lon

            # Limit the observations to be assimilated to the range of elong_min - elong_max
            keep: list[bool] = [self.obs_in_elong_range(obs_val=x) for x in obs]
            obs_times_out: list[datetime.datetime] = [
                x for x, k in zip(obs_times, keep) if k
            ]
            obs_lon_out: list[Quantity] = [x for x, k in zip(obs_lon, keep) if k]
            obs_out = [x for x, k in zip(obs, keep) if k]

            return obs_times_out, obs_lon_out, obs_out, len(obs_out)

    def initialise_mean_cme_pars(self, rng: np.random.Generator) -> dict[str, Any]:
        #######################################################################
        if self.setup_class.get_use_synthetic_obs() and self.sample_first_guess:
            # Twin runs that sample their first guess: one draw from the prior around the twin's
            #  first guess, which is the truth (Setup.get_twin_run_sampling); negative run numbers
            #  start from the truth itself
            fg_array: list[float] = [
                (self.fg_mean_cme_t_init - self.surf_init_time).total_seconds(),
                self.fg_mean_cme_speed,
                self.fg_mean_cme_width,
                self.fg_mean_cme_lon,
                self.fg_mean_cme_lat,
                self.fg_mean_cme_thick,
            ]
            if self.perturb_first_guess:
                truth_speed: float = float(fg_array[1])
                fg_array = [
                    float(x)
                    for x in self.sample_prior(fg_array, n_ens=1, rng=rng)[:, 0]
                ]
                if self.cme_cov_type == "uncorr":
                    # The uncorr prior of the speed is uniform within (1 +- sd) x the first guess, not
                    #  symmetric about it: draw the first guess so that the truth is a uniform draw
                    #  from the prior around it, v_true = (1 - sd + 2 sd u) v_fg with u uniform in
                    #  [0, 1] (else the truth can lie outside the run's prior)
                    sd_v: float = float(self.prior_sd_speed)
                    fg_array[1] = truth_speed / (
                        1.0 - sd_v + 2.0 * sd_v * float(rng.uniform(0.0, 1.0))
                    )
            mean_cme_t_init: datetime.datetime = (
                self.surf_init_time + datetime.timedelta(seconds=float(fg_array[0]))
            )
            mean_cme_speed: float = float(fg_array[1])
            mean_cme_width: float = float(fg_array[2])
            mean_cme_lon: float = float(fg_array[3])
            mean_cme_lat: float = float(fg_array[4])
            mean_cme_thick: float = float(fg_array[5])
        elif self.setup_class.get_use_synthetic_obs():
            # Initialise CME parameters
            mean_cme_t_init = self.fg_mean_cme_t_init
            mean_cme_speed = self.fg_mean_cme_speed + (
                self.fg_sd_cme_speed * rng.uniform(low=-1, high=1)
            )
            mean_cme_width = self.fg_mean_cme_width + (
                self.fg_sd_cme_width * rng.uniform(low=-1, high=1)
            )
            mean_cme_lon = self.fg_mean_cme_lon + (
                self.fg_sd_cme_lon * rng.uniform(low=-1, high=1)
            )
            mean_cme_lat = self.fg_mean_cme_lat
            mean_cme_thick = self.fg_mean_cme_thick
        else:
            # Initialise CME parameters
            mean_cme_t_init = self.fg_mean_cme_t_init + datetime.timedelta(
                seconds=self.fg_sd_cme_t_init * rng.uniform(low=-1, high=1)
            )
            mean_cme_t_init_str: str = mean_cme_t_init.strftime("%Y%m%d-%H%M")
            mean_cme_speed = self.fg_mean_cme_speed + (
                self.fg_sd_cme_speed * rng.uniform(low=-1, high=1)
            )
            mean_cme_width = self.fg_mean_cme_width + (
                self.fg_sd_cme_width * rng.uniform(low=-1, high=1)
            )
            mean_cme_lon = self.fg_mean_cme_lon + (
                self.fg_sd_cme_lon * rng.uniform(low=-1, high=1)
            )
            mean_cme_lat = self.fg_mean_cme_lat + (
                self.fg_sd_cme_lat * rng.uniform(low=-1, high=1)
            )
            mean_cme_thick = self.fg_mean_cme_thick + (
                self.fg_sd_cme_thick * rng.uniform(low=-1, high=1)
            )

        mean_cme_par_dict: dict[str, Any] = {
            "mean_cme_t_init": mean_cme_t_init,
            "mean_cme_speed": mean_cme_speed,
            "mean_cme_width": mean_cme_width,
            "mean_cme_lon": mean_cme_lon,
            "mean_cme_lat": mean_cme_lat,
            "mean_cme_thick": mean_cme_thick,
        }

        return mean_cme_par_dict

    def sample_prior(
        self,
        mean_cme_par_array: list[float],
        n_ens: int,
        rng: np.random.Generator,
    ) -> npt.NDArray[np.float64]:
        """
        Draw n_ens CME parameter sets from the prior (cme_cov_type and the prior sds) around
        mean_cme_par_array
        :param mean_cme_par_array: [t_init (s after surf_init_time), v, width, lon, lat, thick]
        :param n_ens: Number of draws
        :param rng: Random number generator
        :return: samples (6, n_ens), rows as mean_cme_par_array
        """
        if self.cme_cov_type == "uncorr":
            print(mean_cme_par_array)
            samples: npt.NDArray[np.float64] = mp_cov.make_uncorr_uniform_samples(
                n_ens=n_ens,
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
            start_time_donki: datetime.datetime = datetime.datetime(2017, 1, 1, 0, 0, 0)
            end_time_donki: datetime.datetime = datetime.datetime(2026, 2, 1, 0, 0, 0)

            samples = mp_cov.make_donki_samples(
                n_ens=n_ens,
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
                feature="LE",
            )

        elif self.cme_cov_type == "mo_cone":
            mo_cone_cov_dir: str | None = os.environ.get("MO_CONE_COV_DIR")

            if not os.path.exists(mo_cone_cov_dir):
                os.makedirs(mo_cone_cov_dir)

            mo_cme_cone_file_dir: str | None = os.environ.get("MO_CME_CONE_FILE_DIR")

            start_time_mo_cone: datetime.datetime = datetime.datetime(
                2017, 1, 1, 0, 0, 0
            )
            end_time_mo_cone: datetime.datetime = datetime.datetime(2026, 2, 1, 0, 0, 0)

            samples = mp_cov.make_mo_cone_samples(
                n_ens=n_ens,
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

        return samples

    def make_cme_par_dict(self, rng: np.random.Generator) -> CmeParEns:
        # Get mean CME parameters
        mean_cme_par_dict: dict[str, Any] = self.initialise_mean_cme_pars(rng=rng)
        self.run_first_guess = dict(mean_cme_par_dict)
        mean_cme_t_init: float = (
            mean_cme_par_dict["mean_cme_t_init"] - self.surf_init_time
        ).total_seconds()
        mean_cme_speed: float = mean_cme_par_dict["mean_cme_speed"]
        mean_cme_width: float = mean_cme_par_dict["mean_cme_width"]
        mean_cme_lon: float = mean_cme_par_dict["mean_cme_lon"]
        mean_cme_lat: float = mean_cme_par_dict["mean_cme_lat"]
        mean_cme_thick: float = mean_cme_par_dict["mean_cme_thick"]

        mean_cme_par_array: list[float] = [
            mean_cme_t_init,
            mean_cme_speed,
            mean_cme_width,
            mean_cme_lon,
            mean_cme_lat,
            mean_cme_thick,
        ]

        # Initialise cme_par_dict
        cme_par_dict: CmeParEns = initialise_cme_parameter_ensemble_dict(
            n_ensemble=self.n_members, surf_init_time=self.surf_init_time
        )

        samples: npt.NDArray[np.float64] = self.sample_prior(
            mean_cme_par_array, n_ens=self.n_members, rng=rng
        )

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
        bias_mode: str = self.bias_settings["mode"]
        if bias_mode == "estimate":
            cme_par_dict["bias"] = list(
                rng.normal(
                    self.bias_settings["prior_mean"],
                    self.bias_settings["prior_sd"],
                    self.n_members,
                )
            )
        elif bias_mode == "fixed":
            cme_par_dict["bias"] = [
                float(self.bias_settings["prior_mean"])
            ] * self.n_members
        else:
            cme_par_dict["bias"] = [0.0] * self.n_members

        return cme_par_dict

    def write_elongation_file(
        self,
        output_dir: str,
        elon_saved: dict,
        log_weights: npt.NDArray[np.float64],
        profiles: dict | None = None,
    ) -> str:
        """
        Write every ensemble member's modelled CME flank elongation at each observation to
         <output_dir>/ens_elongations.nc (overwritten after each observation).
        Variables (dimensions n_obs x n_ens unless stated):
            elon_first_stage: elongation (deg) of each prior particle after Liu-West shrinkage, i.e. the
                first-stage observation operator used to choose which particles are resampled (with
                PFSettings.first_stage = "current", attrs["first_stage"], of the current particles)
            elon_posterior: elongation (deg) of each resampled (posterior) particle
            parent_index: index of the prior particle each posterior particle was resampled from
            log_weight: posterior log-weight of each particle after this observation
            observation (n_obs): observed elongation (deg)
            assimilated (n_obs): whether the observation was assimilated (False -> elongations are NaN)
            resampled (n_obs): whether the ensemble was resampled at this observation (see
                PFSettings.adaptive_resampling); if not, elon_first_stage and elon_posterior are both the
                elongations of the (unchanged) particles and parent_index is 0..N-1
            ess_updated (n_obs): effective sample size of the weights w * p(y | theta) of the current
                particles, which decides whether to resample (NaN if not computed)
            n_model_runs (n_obs): ensembles of model runs made for the observation
            n_reused_runs (n_obs): ensembles whose elongations were read from earlier model runs
                instead (PFSettings.reuse_model_runs)
        Full elongation profiles (dimensions n_ens x n_profile, on the profile_time grid), written once
         they have been computed (see calculate_elongation_profiles):
            prior_profile_elon: elongation (deg) of the prior ensemble, i.e. the ensemble passed to the
                Aux-PF for the first observation
            posterior_profile_elon: elongation (deg) of the ensemble after the last observation in the
                permitted elongation range has been assimilated
            {prior,posterior}_profile_extrapolated: 1 where the value was extrapolated before the first
                tracked model point (see ObsOpSettings.extrapolate_before_track), 0 where it comes from the model
            {prior,posterior}_profile_log_weight (n_ens): normalised log-weight of each member
        :param output_dir: Run output directory (as used for cme_pars.nc)
        :param elon_saved: Dictionary of arrays filled in run_data_assimilation
        :param log_weights: (n_obs, n_ens) posterior log-weights
        :param profiles: Optional dict with "time" (list of datetimes) and, for "prior" and/or
            "posterior", sub-dicts with "elon" (n_ens, n_profile), "extrapolated" (n_ens, n_profile)
            and "log_weight" (n_ens)
        :return: Path of the written file
        """
        n_obs_run: int = len(self.observations)
        obs_times: npt.NDArray[np.datetime64] = np.array(
            self.obs_times[:n_obs_run], dtype="datetime64[ns]"
        )
        pf_settings: PFSettings = getattr(self, "pf_settings", None) or PFSettings()

        elon_ds: xr.Dataset = xr.Dataset(
            data_vars=dict(
                elon_first_stage=(["n_obs", "n_ens"], elon_saved["elon_first_stage"]),
                elon_posterior=(["n_obs", "n_ens"], elon_saved["elon_posterior"]),
                parent_index=(["n_obs", "n_ens"], elon_saved["parent_index"]),
                log_weight=(["n_obs", "n_ens"], np.asarray(log_weights, dtype=float)),
                observation=(["n_obs"], np.asarray(self.observations, dtype=float)),
                assimilated=(["n_obs"], elon_saved["assimilated"].astype(np.int8)),
                resampled=(["n_obs"], elon_saved["resampled"].astype(np.int8)),
                ess_updated=(["n_obs"], elon_saved["ess_updated"]),
                n_model_runs=(
                    ["n_obs"],
                    np.asarray(
                        elon_saved.get("n_model_runs", np.zeros(n_obs_run)), dtype=int
                    ),
                ),
                n_reused_runs=(
                    ["n_obs"],
                    np.asarray(
                        elon_saved.get("n_reused_runs", np.zeros(n_obs_run)), dtype=int
                    ),
                ),
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
                first_stage=str(pf_settings.first_stage),
                reuse_model_runs=int(bool(pf_settings.reuse_model_runs)),
                reuse_observer_tol_deg=float(pf_settings.reuse_observer_tol_deg),
            ),
        )

        # Full 3-day elongation profiles of the prior and posterior ensembles
        if profiles is not None:
            elon_ds = elon_ds.assign_coords(
                profile_time=(
                    "n_profile",
                    np.array(profiles["time"], dtype="datetime64[ns]"),
                )
            )
            for stage in ["prior", "posterior"]:
                if profiles.get(stage) is None:
                    continue
                elon_ds[f"{stage}_profile_elon"] = (
                    ["n_ens", "n_profile"],
                    np.asarray(profiles[stage]["elon"], dtype=float),
                )
                elon_ds[f"{stage}_profile_extrapolated"] = (
                    ["n_ens", "n_profile"],
                    np.asarray(profiles[stage]["extrapolated"]).astype(np.int8),
                )
                elon_ds[f"{stage}_profile_log_weight"] = (
                    ["n_ens"],
                    np.asarray(profiles[stage]["log_weight"], dtype=float),
                )
                # Ensemble stage of the profiles (0 = prior, k + 1 = after observation k)
                if "stage" in profiles[stage]:
                    elon_ds.attrs[f"{stage}_profile_stage"] = int(
                        profiles[stage]["stage"]
                    )
                # Each member's CME arrival from the same model runs
                if "arrival_time" in profiles[stage]:
                    elon_ds[f"{stage}_profile_arrival_time"] = (
                        ["n_ens"],
                        np.array(
                            [
                                (
                                    np.datetime64(t)
                                    if isinstance(t, datetime.datetime)
                                    else np.datetime64("NaT")
                                )
                                for t in profiles[stage]["arrival_time"]
                            ],
                            dtype="datetime64[ns]",
                        ),
                    )
                    elon_ds[f"{stage}_profile_arrival_speed"] = (
                        ["n_ens"],
                        np.asarray(profiles[stage]["arrival_speed"], dtype=float),
                    )
                    elon_ds[f"{stage}_profile_arrival_available"] = (
                        ["n_ens"],
                        np.asarray(profiles[stage]["arrival_available"]).astype(
                            np.int8
                        ),
                    )
                    elon_ds.attrs["profile_arrival_object"] = str(
                        self.profile_arrival_object
                    )
                    elon_ds.attrs["profile_sim_time_days"] = float(
                        self.profile_sim_time_days
                    )

        out_file: str = os.path.join(output_dir, "ens_elongations.nc")
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
    # The same model runs also give each member's CME arrival at this body, for osse_ana.py (which
    #  then need not rerun these ensembles); the runs last profile_sim_time_days (osse_ana.py's
    #  arrival window, arrival_sim_time_days, should match it for the arrivals to be reused)
    profile_arrival_object: str = "EARTH"
    profile_sim_time_days: float = 4.0
    profile_duration_days: float = 3.0
    profile_dt_minutes: float = 30.0

    def get_profile_times(self) -> list[datetime.datetime]:
        """
        Times at which to evaluate the full elongation profiles: from the model start time for
         profile_duration_days (the compressible SURF runs used by the observation operator are 3
         days long) every profile_dt_minutes.
        :return: List of profile datetimes
        """
        n_times: int = (
            int(round(self.profile_duration_days * 1440.0 / self.profile_dt_minutes))
            + 1
        )
        return [
            self.surf_init_time
            + datetime.timedelta(minutes=k * self.profile_dt_minutes)
            for k in range(n_times)
        ]

    def calculate_elongation_profiles(
        self,
        cme_par_dict: CmeParEns,
        obs_index: int = 0,
        arrival_object: str | None = None,
        model_sim_time: Quantity[u.day] | None = None,
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
        :param arrival_object: If given (e.g. "EARTH"), each member's CME arrival at this body is taken
            from the same model run (cme_arrival.member_arrival)
        :param model_sim_time: Run length of the models (None = the usual: 3 days for the
            compressible SURF models, sim_time for the others); long enough for the arrivals
        :return: dict with "elon" and "extrapolated" (n_ens, n_profile) and "log_weight" (n_ens) and,
            with arrival_object, "arrival_time" (n_ens; datetime or None), "arrival_speed" (n_ens;
            km/s) and "arrival_available" (n_ens; False where the body is outside the model's
            latitude grid)
        """
        profile_times: list[datetime.datetime] = self.get_profile_times()
        print(
            f"Computing elongation profiles at {len(profile_times)} times for {self.n_members} members"
        )

        obs_op_class: ObservationOperator = ObservationOperator(
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
            lat_min=self.lat_min,
            lat_max=self.lat_max,
            lon_start=self.lon_start,
            lon_stop=self.lon_stop,
            sim_time=self.sim_time if model_sim_time is None else model_sim_time,
            dt_scale=self.dt_scale,
            r_min=self.r_min,
            cme_init_rad=self.cme_init_rad,
            cme_fixed_duration=self.cme_fixed_duration,
            fixed_duration=self.fixed_duration,
            plot_surf_output=False,
            use_parallel=self.use_parallel,
            write_diagnostics=False,
            settings=self.obs_op_settings,
            worker_pool=self.worker_pool,
            compress_sim_time=model_sim_time,
            arrival_object=arrival_object,
        )

        cme_flanks: list[pd.DataFrame] | None = None
        if self.use_model in ["compress_surf", "compress_surf3d"]:
            cme_flanks = obs_op_class.get_cme_flanks()
            elon: npt.NDArray[np.float64] = np.vstack(
                [np.asarray(f["el"].values, dtype=float) for f in cme_flanks]
            )
            extrap: npt.NDArray[np.bool_] = np.vstack(
                [
                    (
                        np.asarray(f["extrapolated"].values, dtype=bool)
                        if "extrapolated" in f.columns
                        else np.zeros(len(profile_times), dtype=bool)
                    )
                    for f in cme_flanks
                ]
            )
        else:
            # Without the bias term, as for the compressible models above (osse_ana.py adds it
            #  when plotting the profiles)
            elon = np.asarray(obs_op_class.make_obs_op(include_bias=False), dtype=float)
            extrap = np.zeros(elon.shape, dtype=bool)
            cme_flanks = getattr(obs_op_class, "last_cme_flanks", None)

        out: dict[str, Any] = {
            "elon": elon,
            "extrapolated": extrap,
            "log_weight": np.asarray(cme_par_dict["log_weight"], dtype=float),
        }
        if arrival_object is not None:
            arrivals: list[dict[str, Any]] = [
                (f.attrs.get("arrival") if cme_flanks is not None else None)
                or {"time": None, "speed": np.nan, "hit": False, "available": False}
                for f in (cme_flanks or [None] * len(elon))
            ]
            out["arrival_time"] = [a["time"] for a in arrivals]
            out["arrival_speed"] = np.array([a["speed"] for a in arrivals], dtype=float)
            out["arrival_available"] = np.array(
                [a["available"] for a in arrivals], dtype=bool
            )
        return out

    # Observations must be at least this far (deg) above the smallest elongation the model domain
    #  can show (the inner boundary r_min seen from the observer); see get_min_model_elongation
    elong_model_margin_deg: float = 1.0

    def later_observations(self, obs_index: int) -> list[dict[str, Any]]:
        """
        The observations after obs_index, with their observer positions, for evaluating them from the
        model runs made at obs_index (PFSettings.reuse_model_runs)
        :param obs_index: Index of the observation being assimilated
        :return: dicts with "index", "time", "radius", "lon" and "lat"
        """
        return [
            {
                "index": j,
                "time": self.obs_times[j],
                "radius": self.obs_radius[j],
                "lon": self.obs_lon[j],
                "lat": self.obs_lat[j],
            }
            for j in range(obs_index + 1, len(self.observations))
        ]

    def get_min_model_elongation(self) -> npt.NDArray[np.float64]:
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
        n_obs_run: int = len(self.observations)
        r_min_rs: float = self.r_min.to(u.solRad).value

        r_obs_rs: npt.NDArray[np.float64] | None = None
        craft: str = (self.craft or "").lower()
        if craft not in ["", "twin", "none"]:
            try:
                craft_obs: S.Observer = S.Observer(
                    craft.upper(), Time(list(self.obs_times[:n_obs_run]))
                )
                r_obs_rs = np.asarray(craft_obs.r.to(u.solRad).value, dtype=float)
            except Exception as err:
                print(
                    f"get_min_model_elongation: could not get {craft} ephemeris ({err}); using obs_radius"
                )

        if r_obs_rs is None:
            r_obs_rs = np.array(
                [
                    (r if isinstance(r, Quantity) else r * u.AU).to(u.solRad).value
                    for r in (
                        self.obs_radius
                        if isinstance(self.obs_radius, list)
                        else [self.obs_radius] * n_obs_run
                    )
                ][:n_obs_run],
                dtype=float,
            )

        with np.errstate(invalid="ignore"):
            elong_min_model: npt.NDArray[np.float64] = np.rad2deg(
                np.arcsin(np.clip(r_min_rs / r_obs_rs, 0.0, 1.0))
            )

        return elong_min_model

    def saved_cme_pars(self, cme_par_dict: CmeParEns) -> npt.NDArray[np.float64]:
        """
        The ensemble's CME parameters and log-weights without units, as saved in cme_pars.nc
        :param cme_par_dict: CME parameter ensemble
        :return: (n_members, 7) array: t_init (s since the model start), v (km/s), width (deg),
            lon (deg, -180 to 180), lat (deg), thick (solar radii), log_weight
        """
        row: npt.NDArray[np.float64] = np.zeros((self.n_members, 7))
        row[:, 0] = [
            (cme_par_dict["t_init"][i] - cme_par_dict["surf_init_time"]).total_seconds()
            for i in range(self.n_members)
        ]
        row[:, 1] = cme_par_dict["v"].to(u.km / u.s).value
        row[:, 2] = cme_par_dict["width"].to(u.deg).value
        row[:, 3] = cme_par_dict["lon"].to(u.deg).value
        row[row[:, 3] > 180, 3] -= 360
        row[:, 4] = cme_par_dict["lat"].to(u.deg).value
        row[:, 5] = cme_par_dict["thick"].to(u.solRad).value
        row[:, 6] = cme_par_dict["log_weight"]
        return row

    def make_cme_par_dataset(
        self, cme_saved_pars: npt.NDArray[np.float64]
    ) -> xr.Dataset:
        """
        The CME parameters of every member before (obs_no 0) and after each observation, with the model
        set-up, for cme_pars.nc
        :param cme_saved_pars: (n_obs + 1, n_members, 7) array of saved_cme_pars rows
        :return: Dataset
        """
        if self.use_model not in ["surf", "huxt", "compress_surf", "compress_surf3d"]:
            sys.exit(f"Saving output for use_model = {self.use_model} is not supported")
        is_3d: bool = self.use_model == "compress_surf3d"

        data_vars: dict[str, Any] = dict(
            model_init_time=self.surf_init_time,
            cme_init_rad=self.cme_init_rad,
            r_min=self.r_min,
            ambient_vr=(["n_lat", "n_lon"] if is_3d else ["n_lon"], self.vr_in),
        )
        if is_3d:
            data_vars["vr_longs"] = (["n_lon"], self.vr_longs)
            data_vars["vr_lats"] = (["n_lat"], self.vr_lats)
        for ind, name in enumerate(
            ["t_init", "v", "width", "lon", "lat", "thick", "log_weight"]
        ):
            data_vars[name] = (["n_obs", "n_ens"], cme_saved_pars[:, :, ind])

        coords: dict[str, Any] = dict(
            obs_no=("n_obs", range(self.n_obs + 1)),
            ens_no=("n_ens", range(self.n_members)),
            surf_lon=("n_lon", (2 * np.pi / self.n_lon_in) * np.arange(self.n_lon_in)),
        )
        if is_3d:
            coords["surf_lat"] = ("n_lat", range(self.n_lat_in))

        return xr.Dataset(data_vars=data_vars, coords=coords)

    def run_data_assimilation(self, run_start: int = 0) -> None:
        """
        Function to run the data assimilation routine
        :return: None
        """
        # Negative run numbers count downwards (run_start, run_start - 1, ...)
        step: int = -1 if run_start < 0 else 1

        for run_no in range(run_start, run_start + step * self.n_runs, step):
            print(f"run_no = {run_no}")
            if run_no < 0:
                self.fg_sd_cme_t_init: float = 0
                self.fg_sd_cme_speed: float = 0
                self.fg_sd_cme_width: float = 0
                self.fg_sd_cme_lon: float = 0
                self.fg_sd_cme_lat: float = 0
                self.fg_sd_cme_thick: float = 0

            # Initialise random generator
            rand_seed: int = self.generate_random_seed(
                add_const=42, mult_const=10000, run_no=run_no
            )
            rng: np.random.Generator = np.random.default_rng(rand_seed)

            # Twin runs: whether the first guess is perturbed, and this run's observation noise
            #  (its own random stream, so the CME draws are the same with or without it)
            self.perturb_first_guess = run_no >= 0
            if self.sample_first_guess and run_no < 0:
                print(
                    "Note: negative run numbers start from the truth itself; use run_start >= 0 for"
                    " first guesses drawn from the prior"
                )
            if self.new_obs_noise_per_run:
                obs_rng: np.random.Generator = np.random.default_rng(
                    [int(self.obs_rng_seed), int(rand_seed)]
                )
                self.observations = list(
                    np.asarray(self.unperturbed_observations, dtype=float)
                    + obs_rng.normal(
                        0.0, self.obs_noise_sd, len(self.unperturbed_observations)
                    )
                )

            cme_par_dict: CmeParEns = self.make_cme_par_dict(rng=rng)
            cme_saved_pars: npt.NDArray[np.float64] = np.zeros(
                (self.n_obs + 1, self.n_members, 7)
            )
            # Observation-bias coefficient of each member, prior (row 0) and after each observation
            bias_saved: npt.NDArray[np.float64] = np.zeros(
                (self.n_obs + 1, self.n_members)
            )
            bias_saved[0, :] = np.asarray(cme_par_dict.get("bias", 0.0), dtype=float)

            # Correlation handling: working coordinates fixed from this run's prior ensemble
            state_transform: StateTransform | None = self.make_state_transform(
                cme_par_dict
            )
            # Weighted correlation matrix of the state parameters (v as ln v), prior and after each obs
            n_state: int = len(self.pars_in_state)
            corr_saved: npt.NDArray[np.float64] = np.full(
                (self.n_obs + 1, n_state, n_state), np.nan
            )
            corr_saved[0] = self.state_correlation(cme_par_dict)

            # Modelled elongation of every ensemble member at every observation (NaN where the
            #  observation was not assimilated, or the member's CME was not seen)
            n_obs_run: int = len(self.observations)
            elon_saved: dict[str, npt.NDArray[Any]] = {
                "elon_first_stage": np.full((n_obs_run, self.n_members), np.nan),
                "elon_posterior": np.full((n_obs_run, self.n_members), np.nan),
                "parent_index": np.full((n_obs_run, self.n_members), -1, dtype=int),
                "assimilated": np.zeros(n_obs_run, dtype=bool),
                "resampled": np.zeros(n_obs_run, dtype=bool),
                "ess_updated": np.full(n_obs_run, np.nan),
                # Ensembles of model runs made at each observation, and ensembles read from earlier
                #  runs instead (PFSettings.reuse_model_runs)
                "n_model_runs": np.zeros(n_obs_run, dtype=int),
                "n_reused_runs": np.zeros(n_obs_run, dtype=int),
            }

            # Elongations of this run's model runs, for reuse (PFSettings.reuse_model_runs)
            elon_store: ElongationStore | None = (
                ElongationStore() if self.pf_settings.reuse_model_runs else None
            )

            # Save prior state
            cme_saved_pars[0] = self.saved_cme_pars(cme_par_dict)

            # Make output directory
            output_dir: str = self.make_output_dir(
                cme_par_dict=cme_par_dict, run_no=run_no
            )

            # Lowest elongation each observation may have to be assimilable by the model
            elong_min_model: npt.NDArray[np.float64] = self.get_min_model_elongation()
            elong_min_usable: npt.NDArray[np.float64] = np.fmax(
                elong_min_model + self.elong_model_margin_deg, self.elong_min_obs
            )
            print(
                f"Minimum usable elongation (model inner boundary {self.r_min} + "
                f"{self.elong_model_margin_deg} deg margin): "
                f"{np.nanmin(elong_min_usable):.2f}-{np.nanmax(elong_min_usable):.2f} deg"
            )

            # Full elongation profiles of the prior ensemble (the ensemble passed to the Aux-PF for
            #  the first assimilated observation: skipped observations do not change it)
            profiles: dict[str, Any] | None = None
            if self.compute_elongation_profiles:
                profiles = {
                    "time": self.get_profile_times(),
                    "prior": None,
                    "posterior": None,
                }
                try:
                    profiles["prior"] = self.calculate_elongation_profiles(
                        cme_par_dict,
                        obs_index=0,
                        arrival_object=self.profile_arrival_object,
                        model_sim_time=self.profile_sim_time_days * u.day,
                    )
                    profiles["prior"]["stage"] = 0
                except Exception as err:
                    print(
                        f"Warning: could not compute the prior elongation profiles: {err}"
                    )

            for yi, obs in enumerate(self.observations):
                print(
                    f"run_no = {run_no}/{self.n_runs}, obs = {yi}/{len(self.observations)}"
                )
                # Initial conditions for running the DA, that the observation time is after
                #  the first-guess of the CME launch and that the observation is in the
                #  set [elong_min_obs, elong_max_obs]
                fg_cme_init_cond: bool = self.obs_times[yi] > self.fg_mean_cme_t_init
                obs_in_elon_range_cond: bool = (obs >= self.elong_min_obs) and (
                    obs <= self.elong_max_obs
                )
                # ... and above the smallest elongation the model domain can show from this observer
                obs_above_model_min_cond: bool = bool(
                    np.isnan(elong_min_usable[yi]) or (obs >= elong_min_usable[yi])
                )
                if obs_in_elon_range_cond and not obs_above_model_min_cond:
                    print(
                        f"Skipping obs {yi} ({obs:.2f} deg): below the model's minimum visible elongation "
                        f"{elong_min_model[yi]:.2f} deg + {self.elong_model_margin_deg} deg margin "
                        f"(inner boundary r_min = {self.r_min})"
                    )
                obs_in_elon_range_cond = (
                    obs_in_elon_range_cond and obs_above_model_min_cond
                )

                if fg_cme_init_cond and obs_in_elon_range_cond:
                    aux_pf_class: AuxPF = AuxPF(
                        use_model=self.use_model,
                        cme_par_dict=cme_par_dict,
                        obs=obs,
                        obs_cov=self.obs_cov,
                        obs_radius=self.obs_radius[yi],
                        obs_lon=self.obs_lon[yi],
                        obs_lat=self.obs_lat[yi],
                        obs_time=self.obs_times[yi],
                        craft=self.craft,
                        pars_in_state=self.pars_in_filter,
                        vr_in=self.vr_in,
                        vr_longs=self.vr_longs,
                        vr_lats=self.vr_lats,
                        n_lat_surf=self.n_lat_surf,
                        lat_min=self.lat_min,
                        lat_max=self.lat_max,
                        lon_start=self.lon_start,
                        lon_stop=self.lon_stop,
                        time_tolerance=self.time_tolerance,
                        dt_scale=self.dt_scale,
                        delta_aux_pf=self.delta_aux_pf,
                        r_min=self.r_min,
                        cme_init_rad=self.cme_init_rad,
                        cme_fixed_duration=self.cme_fixed_duration,
                        fixed_duration=self.fixed_duration,
                        use_parallel=self.use_parallel,
                        rng=rng,
                        state_transform=state_transform,
                        pf_settings=self.pf_settings,
                        obs_op_settings=self.obs_op_settings,
                        worker_pool=self.worker_pool,
                        elon_store=elon_store,
                        obs_index=yi,
                        later_obs=self.later_observations(yi) if elon_store else None,
                        run_until=self.obs_times[n_obs_run - 1] if elon_store else None,
                    )

                    cme_par_dict = aux_pf_class.aux_pf()
                    elon_saved["n_model_runs"][yi] = aux_pf_class.n_model_runs
                    elon_saved["n_reused_runs"][yi] = aux_pf_class.n_reused_runs

                    # Record each member's modelled elongation at this observation
                    elon_saved["elon_first_stage"][
                        yi, :
                    ] = aux_pf_class.elon_first_stage
                    elon_saved["elon_posterior"][yi, :] = aux_pf_class.elon_posterior
                    elon_saved["parent_index"][yi, :] = aux_pf_class.parent_index
                    # (False if the Aux-PF left the ensemble unchanged, e.g. because a member's CME
                    #  produced no elongation; see PFSettings.skip_update_if_any_member_nan)
                    elon_saved["assimilated"][yi] = not getattr(
                        aux_pf_class, "update_skipped", False
                    )
                    elon_saved["resampled"][yi] = bool(
                        getattr(aux_pf_class, "resampled", True)
                    )
                    elon_saved["ess_updated"][yi] = getattr(
                        aux_pf_class, "ess_updated", np.nan
                    )
                    if getattr(aux_pf_class, "update_skipped", False):
                        print(
                            f"Observation {yi} not assimilated: {aux_pf_class.skip_reason}"
                        )
                elif fg_cme_init_cond:
                    print(
                        "Observation is outside the permitted elongation range for the model"
                    )
                elif obs_in_elon_range_cond:
                    print("Obs time before CME is expected to be in SURF model domain")
                else:
                    print(
                        "Obs time before CME launch and obs outside permitted elongation range"
                    )

                cme_saved_pars[yi + 1] = self.saved_cme_pars(cme_par_dict)
                bias_saved[yi + 1, :] = np.asarray(
                    cme_par_dict.get("bias", 0.0), dtype=float
                )
                corr_saved[yi + 1] = self.state_correlation(cme_par_dict)

                # Write the elongations after every observation, so a run that stops part-way
                #  still leaves everything computed so far on disk
                self.write_elongation_file(
                    output_dir=output_dir,
                    elon_saved=elon_saved,
                    # (observations not yet reached are NaN rather than the array's initial zeros)
                    log_weights=np.where(
                        (np.arange(n_obs_run) <= yi)[:, None],
                        cme_saved_pars[1 : n_obs_run + 1, :, 6],
                        np.nan,
                    ),
                    profiles=profiles,
                )

            print(
                f"run_no = {run_no}: {int(elon_saved['n_model_runs'].sum())} ensembles of model runs"
                f" for the observations, {int(elon_saved['n_reused_runs'].sum())} read from earlier"
                f" runs (first stage: {self.pf_settings.first_stage})"
            )

            # Full elongation profiles of the posterior ensemble (after the last observation in the
            #  permitted elongation range has been assimilated)
            if profiles is not None:
                assimilated_ind: npt.NDArray[np.int_] = np.flatnonzero(
                    elon_saved["assimilated"]
                )
                last_obs_ind: int = (
                    int(assimilated_ind[-1]) if assimilated_ind.size else 0
                )
                try:
                    profiles["posterior"] = self.calculate_elongation_profiles(
                        cme_par_dict,
                        obs_index=last_obs_ind,
                        arrival_object=self.profile_arrival_object,
                        model_sim_time=self.profile_sim_time_days * u.day,
                    )
                    # The ensemble after the last assimilated observation (stage = obs index + 1)
                    profiles["posterior"]["stage"] = (
                        last_obs_ind + 1 if assimilated_ind.size else 0
                    )
                except Exception as err:
                    print(
                        f"Warning: could not compute the posterior elongation profiles: {err}"
                    )
                self.write_elongation_file(
                    output_dir=output_dir,
                    elon_saved=elon_saved,
                    log_weights=cme_saved_pars[1 : n_obs_run + 1, :, 6],
                    profiles=profiles,
                )

            for par_ind in range(7):
                if par_ind == 6:
                    print(f"weights_post = {np.exp(cme_saved_pars[-1, :, par_ind])}")
                print(
                    f"mean_cme_saved_pars[{par_ind}] = {np.mean(cme_saved_pars[:, :, par_ind], axis=1)}"
                )

            # Save the CME parameters, bias coefficients and settings to cme_pars.nc
            cme_par_ds: xr.Dataset = self.make_cme_par_dataset(cme_saved_pars)

            # Observation-bias coefficients and settings (see obs_bias.py)
            cme_par_ds["bias"] = (["n_obs", "n_ens"], bias_saved)
            cme_par_ds.attrs.update(
                {
                    "bias_mode": self.bias_settings["mode"],
                    "bias_predictor": self.bias_settings["predictor"]["type"],
                    "bias_e_ref_deg": self.bias_settings["predictor"]["e_ref_deg"],
                    "bias_e_scale_deg": self.bias_settings["predictor"]["e_scale_deg"],
                    "bias_prior_mean": float(self.bias_settings["prior_mean"]),
                    "bias_prior_sd": float(self.bias_settings["prior_sd"]),
                    "bias_prior_source": self.bias_settings["prior_source"],
                    "true_bias": (
                        float(self.true_cme_par_dict.get("bias", 0.0))
                        if self.use_synthetic_obs
                        else np.nan
                    ),
                }
            )
            # Truth CME of a twin experiment, so the run can be analysed (osse_ana.py) without the
            #  settings it was made with
            if self.use_synthetic_obs:
                truth: CmeParEns = self.true_cme_par_dict
                cme_par_ds.attrs.update(
                    {
                        "twin_experiment": self.setup_class.get_twin_experiment(),
                        "true_t_init": pd.Timestamp(truth["t_init"]).isoformat(),
                        "true_v": float(truth["v"].to(u.km / u.s).value),
                        "true_width": float(truth["width"].to(u.deg).value),
                        "true_lon": float(truth["lon"].to(u.deg).value),
                        "true_lat": float(truth["lat"].to(u.deg).value),
                        "true_thick": float(truth["thick"].to(u.solRad).value),
                        "obs_cov": float(np.ravel(np.asarray(self.obs_cov, float))[0]),
                        "sample_first_guess": int(self.sample_first_guess),
                        "new_obs_noise_per_run": int(self.new_obs_noise_per_run),
                    }
                )
                # This run's first guess (the centre of its prior ensemble)
                fg: dict[str, Any] = self.run_first_guess
                if fg:
                    cme_par_ds.attrs.update(
                        {
                            "first_guess_t_init": pd.Timestamp(
                                fg["mean_cme_t_init"]
                            ).isoformat(),
                            "first_guess_v": float(fg["mean_cme_speed"]),
                            "first_guess_width": float(fg["mean_cme_width"]),
                            "first_guess_lon": float(fg["mean_cme_lon"]),
                            "first_guess_lat": float(fg["mean_cme_lat"]),
                            "first_guess_thick": float(fg["mean_cme_thick"]),
                        }
                    )

            # Parameter correlations (v as ln v) and the correlation-handling settings (state_transform.py)
            cme_par_ds["state_corr"] = (
                ["n_obs", "state_par", "state_par_2"],
                corr_saved,
            )
            cme_par_ds = cme_par_ds.assign_coords(
                state_par=("state_par", list(self.pars_in_state)),
                state_par_2=("state_par_2", list(self.pars_in_state)),
            )
            cs: dict[str, Any] = self.correlation_settings
            cme_par_ds.attrs.update(
                {
                    "corr_enabled": int(cs["enabled"]),
                    "corr_reparam": cs["reparam"],
                    "corr_t_ref_rs": (
                        float(cs["t_ref_rs"]) if cs["t_ref_rs"] is not None else np.nan
                    ),
                    "corr_whiten": int(bool(cs["whiten"])),
                    "corr_prior_cov_shrinkage": float(cs["prior_cov_shrinkage"]),
                    "corr_shrinkage_alpha": float(cs["shrinkage_alpha"]),
                    "corr_min_jitter_sd": float(cs["min_jitter_sd"]),
                    "corr_carry_from_prior": ",".join(cs["carry_from_prior"]),
                    "pars_in_filter": ",".join(self.pars_in_filter),
                }
            )

            print(cme_par_ds)
            outParFile: str = os.path.join(output_dir, "cme_pars.nc")
            cme_par_ds.to_netcdf(outParFile, mode="w")

        return None


def command_line_setup(
    event: str | None = None,
    run_start: int | None = None,
    n_runs: int | None = None,
    base: type[Setup] = Setup,
) -> type[Setup]:
    """
    Setup class with the settings given on the command line in place of those in da_surf_setup.py
    :param event: Event or twin experiment in events.yaml (None = Setup.get_ssw_event)
    :param run_start: First run number (None = initialise_da_parameters' run_start)
    :param n_runs: Number of runs (None = initialise_da_parameters' n_runs)
    :param base: Setup class to start from
    :return: Setup class
    """

    class CommandLineSetup(base):
        def get_ssw_event(self) -> str:
            return super().get_ssw_event() if event is None else event

        def initialise_da_parameters(self) -> dict[str, Any]:
            da_par_dict: dict[str, Any] = super().initialise_da_parameters()
            if run_start is not None:
                da_par_dict["run_start"] = run_start
            if n_runs is not None:
                da_par_dict["n_runs"] = n_runs
            return da_par_dict

    return CommandLineSetup


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Command-line options (any left out keep the values in da_surf_setup.py)"""
    parser = argparse.ArgumentParser(
        description="Run the data assimilation for an event or twin experiment of events.yaml. "
        "Settings not given here come from da_surf_setup.Setup."
    )
    parser.add_argument(
        "--event",
        help="Event or twin experiment in events.yaml, e.g. 12A or twin_base "
        "(default: Setup.get_ssw_event)",
    )
    parser.add_argument(
        "--run-start",
        type=int,
        help="First run number (twin experiments; 0 or more draws each run's first guess from the "
        "prior). Real observations always use run -1",
    )
    parser.add_argument(
        "--n-runs", type=int, help="Number of runs (run_start, run_start + 1, ...)"
    )
    parser.add_argument(
        "--list-events",
        action="store_true",
        help="List the events and twin experiments in events.yaml and exit",
    )
    args = parser.parse_args(argv)
    if args.n_runs is not None and args.n_runs < 1:
        parser.error("--n-runs must be 1 or more")
    return args


def main(argv: list[str] | None = None) -> None:
    args: argparse.Namespace = parse_args(argv)
    events: dict[str, dict[str, Any]] = load_events()
    if args.list_events:
        print("\n".join(events))
        return None
    if args.event is not None and args.event not in events:
        sys.exit(
            f"Unknown event {args.event!r}: add it to {events_file()} "
            f"(events there: {', '.join(events)})"
        )

    load_environment()
    print_environment_variables()

    # Reuse the Setup instance built by RunDataAssimilationRoutine
    #  (each Setup() re-reads/re-maps the WSA boundary conditions)
    run_da_class: RunDataAssimilationRoutine = RunDataAssimilationRoutine(
        setup_class=command_line_setup(args.event, args.run_start, args.n_runs)
    )
    setup_class: Setup = run_da_class.setup_class
    da_par_dict: dict[str, Any] = setup_class.da_par_dict
    print(f"event = {setup_class.get_ssw_event_craft()[0]}")

    run_start: int = da_par_dict["run_start"]
    if args.run_start is not None and not setup_class.get_use_synthetic_obs():
        print("--run-start is ignored for real observations (run -1)")

    try:
        if setup_class.get_use_synthetic_obs():
            run_da_class.run_data_assimilation(run_start=run_start)
        else:
            run_da_class.run_data_assimilation(run_start=-1)
    finally:
        # Close the persistent worker pool used by the observation operator
        run_da_class.worker_pool.shutdown()

    return None


if __name__ == "__main__":
    main()
