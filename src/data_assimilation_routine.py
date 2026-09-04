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

from multiprocessing import Pool

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
        self.vr_in: npt.NDArray[Quantity[u.km / u.s]] = surf_par_dict["vr_in"]
        self.n_lon: int = len(self.vr_in)
        self.lon_start: Quantity[u.deg] = surf_par_dict["lon_start"]
        self.lon_stop: Quantity[u.deg] = surf_par_dict["lon_stop"]
        self.sim_time: Quantity[u.days] = surf_par_dict["sim_time"]
        self.dt_scale: int | float = surf_par_dict["dt_scale"]
        self.r_min: Quantity[u.solRad] = surf_par_dict["r_min"]
        self.cme_init_rad: Quantity[u.solRad] = surf_par_dict["cme_init_rad"]
        self.cme_fixed_duration: bool = surf_par_dict["cme_fixed_duration"]
        self.fixed_duration: Quantity[u.s] = surf_par_dict["fixed_duration"]

        # Initialise observation variables
        obs_par_dict = self.setup_class.obs_par_dict
        self.n_obs: int = obs_par_dict["n_obs"]
        self.obs_radius: Quantity[u.AU] | list[Quantity[u.AU]] = obs_par_dict["obs_radius"]
        self.obs_lon: Quantity[u.deg] | list[Quantity[u.deg]] = obs_par_dict["obs_lon"]
        self.obs_lat: Quantity[u.deg] | list[Quantity[u.deg]] = obs_par_dict["obs_lat"]
        self.obs_cov: float | npt.NDArray[float] = obs_par_dict["obs_cov"]
        self.obs_times: list[datetime.datetime] = obs_par_dict["obs_times"]
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
        self.bias_term_5rs: float = obs_par_dict["bias_term_5rs"]
        self.bias_term_21rs: float = obs_par_dict["bias_term_21rs"]

        # Ensure all bias terms exist, if not, set bias_term_bool to False
        if (self.bias_term_5rs is None) or (self.bias_term_21rs is None):
            self.bias_term_bool = False

        # If we're using synthetic observations/ running OSSEs define true_cme_par_dict
        self.true_cme_par_dict: CmeParEns = self.setup_class.true_cme_par_dict

        if self.use_synthetic_obs:
            # Generate observations
            self.observations: list[float] = self.get_observations()
            if not isinstance(self.obs_radius, list):
                print("Hi")
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

        model_dir = f"model_{self.use_model.upper()}"

        if self.use_synthetic_obs:
            obs_dir = (
                f"truth_{self.true_cme_par_dict["v"].value}_{self.true_cme_par_dict["width"].value}"
                f"_{self.true_cme_par_dict["lon"].value}_{self.true_cme_par_dict["lat"].value}"
                f"_{self.true_cme_par_dict["thick"].value}"
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
            obs_lon=self.obs_lon,
            obs_times_in_datetime=self.obs_times,
            obs_filenames=self.obs_filenames,
            use_synthetic_obs=self.use_synthetic_obs,
            true_cme_par_dict=self.true_cme_par_dict,
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
            obs_out = obs_class.observations
            obs_times_out = obs_class.obs_times_in_datetime
            obs_lon_out = obs_class.obs_lon
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
                mean_cme_pars=mean_cme_par_array,
                rng=rng,
                start_time=start_time_donki,
                end_time=end_time_donki,
                scale_corr=self.scale_corr,
                vars_req=np.array(self.pars_in_state),
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
                vars_req=np.array(self.pars_in_state),
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
            print(f"cme_cov_type must be in {allowed_cme_cov_types}.")
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

        return cme_par_dict


    def run_data_assimilation(self, run_start=0):
        """
        Function to run the data assimilation routine
        :return: None
        """
        if run_start < 0:
            self.n_runs = -self.n_runs
            step = -1
        else:
            step = 1

        if self.use_parallel:
            n_runs_req = self.n_runs
        else:
            n_runs_req = self.n_runs

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

            for yi, obs in enumerate(self.observations):
                print(f"run_no = {run_no}/{self.n_runs}, obs = {yi}/{len(self.observations)}")

                aux_pf_class = AuxPF(
                    use_model=self.use_model,
                    cme_par_dict=cme_par_dict,
                    obs=obs,
                    obs_cov=self.obs_cov,
                    obs_radius=self.obs_radius[yi],
                    obs_lon=self.obs_lon[yi],
                    obs_lat=self.obs_lat[yi],
                    obs_time=self.obs_times[yi],
                    true_cme_par_dict=self.true_cme_par_dict,
                    infl_fact=1,
                    pars_in_state=self.pars_in_state,
                    vr_in=self.vr_in,
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
                    rng=rng
                )

                cme_par_dict = aux_pf_class.aux_pf()

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

            for par_ind in range(7):
                if par_ind == 6:
                    print(f"weights_post = {np.exp(cme_saved_pars[-1, :, par_ind])}")
                print(f"mean_cme_saved_pars[{par_ind}] = {np.mean(cme_saved_pars[:, :, par_ind], axis=1)}")

            # Save cme_parameters into a .nc file
            # Make an xarray object
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
                    surf_lon=("n_lon", (2 * np.pi / self.n_lon) * np.arange(self.n_lon)),
                ),
            )
            print(cme_par_ds)
            outParFile = os.path.join(output_dir, "cme_pars.nc")
            cme_par_ds.to_netcdf(outParFile, mode='w')

        return None


def main():
    print_environment_variables()
    setup_class = Setup()
    run_da_class = RunDataAssimilationRoutine()

    run_start = 75

    if setup_class.get_use_synthetic_obs():
        run_da_class.run_data_assimilation(run_start=run_start)
    else:
        run_da_class.run_data_assimilation(run_start=-1)

    return None

if __name__ == "__main__":
    main()