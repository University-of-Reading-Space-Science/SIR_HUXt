import numpy as np
import numpy.typing as npt
import datetime
import os
import sys
import pandas as pd
import xarray as xr
from typing import TypedDict
import json

import huxt.huxt as H
import huxt.huxt_analysis as HA
from mypy.build import TypedDict
from scipy.special.cython_special import log_wright_bessel

import sir_huxt_mono_obs as shmo
import sunpy.coordinates.sun as sn

import astropy.units as u
from astropy.units import Quantity
from astropy.time import Time

import matplotlib.pyplot as plt
from cme_par_ens import CmeParEns
import seaborn as sns
import colorcet as cc
import pytest
from cme_par_dict_structure import required_dict_keys

# import numpy as np
# import numpy.typing as npt
# from astropy.units import Quantity
# import datetime

from init_sir import initialise_cme_parameter_ensemble_dict
from sir_observations import Observations
from aux_pf import AuxPF
from cme_par_ens import CmeParEns

def initialise_huxt_parameters():
    """
    Function to initialise the Huxt parameters for simulation
    :return:
    """
    huxt_init_time: datetime.datetime = datetime.datetime(2008, 1, 1, 0, 0, 0)
    vr_in: npt.NDArray[Quantity[u.km / u.s]] = np.zeros(128) + 400 * u.km / u.s
    lon_start: Quantity[u.deg] = 290 * u.deg
    lon_stop: Quantity[u.deg] = 430 * u.deg
    sim_time: Quantity[u.day] = 3 * u.day
    dt_scale: int = 1
    cme_init_rad: Quantity[u.solRad] = 12 * u.solRad
    r_min: Quantity[u.solRad] = 30 * u.solRad
    cme_fixed_duration: bool = True
    fixed_duration: Quantity[u.s] = 12 * 3600 * u.s

    out_huxt_par = {
        "huxt_init_time": huxt_init_time,
        "vr_in": vr_in,
        "lon_start": lon_start,
        "lon_stop": lon_stop,
        "sim_time": sim_time,
        "dt_scale": dt_scale,
        "r_min": r_min,
        "cme_init_rad": cme_init_rad,
        "cme_fixed_duration": cme_fixed_duration,
        "fixed_duration": fixed_duration
    }

    return out_huxt_par


def initialise_true_cme_par_dict(huxt_init_time: datetime.datetime):
    true_cme_par_dict = initialise_cme_parameter_ensemble_dict(1, huxt_init_time)

    # Initialise true CME parameters
    true_cme_t_init: datetime.datetime = datetime.datetime(2008, 1, 1, 1, 0, 0)
    true_cme_speed: float = 495
    true_cme_width: float = 37.4
    true_cme_lon: float = 0
    true_cme_lat: float = 0
    true_cme_thick: float = 0

    true_cme_par_dict["t_init"]: datetime.datetime = true_cme_t_init
    true_t_init_str: str = true_cme_par_dict["t_init"].strftime("%Y%m%d-%H%M")

    true_cme_par_dict["v"]: Quantity[u.km / u.s] = true_cme_speed * u.km / u.s
    true_cme_par_dict["width"]: Quantity[u.deg] = true_cme_width * u.deg
    true_cme_par_dict["lon"]: Quantity[u.deg] = true_cme_lon * u.deg
    true_cme_par_dict["lat"]: Quantity[u.deg] = true_cme_lat * u.deg
    true_cme_par_dict["thick"]: Quantity[u.solRad] = true_cme_thick * u.solRad

    return true_cme_par_dict


def initialise_fg_cme_parameters():
    # Initialise CME parameters
    fg_cme_t_init: datetime.datetime = datetime.datetime(2008, 1, 1, 1, 0, 0)
    fg_cme_t_init_str: str = fg_cme_t_init.strftime("%Y%m%d-%H%M")
    fg_cme_speed: float = 470
    fg_cme_width: float = 37.0
    fg_cme_lon: float = -4
    fg_cme_lat: float = 0
    fg_cme_thick: float = 0

    sd_cme_t_init: float = 0  # In seconds
    sd_cme_speed: float = 50  # In km/s
    sd_cme_width: float = 5  # In deg
    sd_cme_lon: float = 5  # In deg
    sd_cme_lat: float = 0  # In deg
    sd_cme_thick: float = 0  # In solRad

    fg_cme_par_dict = {
        "fg_cme_t_init": fg_cme_t_init,
        "fg_cme_speed": fg_cme_speed,
        "fg_cme_width": fg_cme_width,
        "fg_cme_lon": fg_cme_lon,
        "fg_cme_lat": fg_cme_lat,
        "fg_cme_thick": fg_cme_thick,
        "sd_cme_t_init": sd_cme_t_init,
        "sd_cme_speed": sd_cme_speed,
        "sd_cme_width": sd_cme_width,
        "sd_cme_lon": sd_cme_lon,
        "sd_cme_lat": sd_cme_lat,
        "sd_cme_thick": sd_cme_thick
    }

    return fg_cme_par_dict


def initialise_prior_cme_sd():
    # Initialise CME parameters for uniform distribution
    sd_t_init: float = 0  # In seconds
    sd_speed: float = 0.1 # Multiplicative variance
    sd_width: float = 5  # In deg
    sd_lon: float = 5  # In deg
    sd_lat: float = 0  # In deg
    sd_thick: float = 0  # In solRad

    prior_cme_par_dict = {
        "sd_t_init": sd_t_init,
        "sd_speed": sd_speed,
        "sd_width": sd_width,
        "sd_lon": sd_lon,
        "sd_lat": sd_lat,
        "sd_thick": sd_thick
    }

    return prior_cme_par_dict


def initialise_observation_parameters():
    n_obs: int = 4

    obs_lon: Quantity[u.deg] = 300 * u.deg
    obs_lat: Quantity[u.deg] = 0 * u.deg

    obs_cov: list[float] = [0.15]  # * np.eye(len(obs))

    obs_times: list[datetime.datetime] = [
        datetime.datetime(2008, 1, 1, 9, 0, 0) + datetime.timedelta(hours=1 * i)
        for i in range(1, 1 + n_obs)
    ]

    use_synthetic_obs: bool = True
    obs_rng_seed: int = 4096
    obs_filenames: list[str] = None

    obs_par_dict = {
        "n_obs": n_obs,
        "obs_lon": obs_lon,
        "obs_lat": obs_lat,
        "obs_cov": obs_cov,
        "obs_times": obs_times,
        "use_synthetic_obs": use_synthetic_obs,
        "obs_filenames": obs_filenames,
        "obs_rng_seed": obs_rng_seed,
    }

    return obs_par_dict


def initialise_da_parameters():
    n_members: int = 50
    n_runs: int = 1

    delta_aux_pf: float = 0.98
    pars_in_state_vector: list[str] = ["v", "width", "lon"]
    time_tolerance: Quantity[u.day] = 0.5 * u.day

    da_par_dict = {
        "n_members": n_members,
        "n_runs": n_runs,
        "delta_aux_pf": delta_aux_pf,
        "pars_in_state": pars_in_state_vector,
        "time_tolerance": time_tolerance
    }

    return da_par_dict


class RunDataAssimilationRoutine:
    def __init__(self):
        # Get HUXt parameters
        huxt_par_dict = initialise_huxt_parameters()
        self.huxt_init_time: datetime.datetime = huxt_par_dict["huxt_init_time"]
        self.vr_in: npt.NDArray[Quantity[u.km / u.s]] = huxt_par_dict["vr_in"]
        self.lon_start: Quantity[u.deg] = huxt_par_dict["lon_start"]
        self.lon_stop: Quantity[u.deg] = huxt_par_dict["lon_stop"]
        self.sim_time: Quantity[u.days] = huxt_par_dict["sim_time"]
        self.dt_scale: int | float = huxt_par_dict["dt_scale"]
        self.r_min: Quantity[u.solRad] = huxt_par_dict["r_min"]
        self.cme_init_rad: Quantity[u.solRad] = huxt_par_dict["cme_init_rad"]
        self.cme_fixed_duration: bool = huxt_par_dict["cme_fixed_duration"]
        self.fixed_duration: Quantity[u.s] = huxt_par_dict["fixed_duration"]

        # Initialise observation variables
        obs_par_dict = initialise_observation_parameters()
        self.n_obs: int = obs_par_dict["n_obs"]
        self.obs_lon: Quantity[u.deg] = obs_par_dict["obs_lon"]
        self.obs_lat: Quantity[u.deg] = obs_par_dict["obs_lat"]
        self.obs_cov: float | npt.NDArray[float] = obs_par_dict["obs_cov"]
        self.obs_times: list[datetime.datetime] = obs_par_dict["obs_times"]
        self.use_synthetic_obs: bool = obs_par_dict["use_synthetic_obs"]
        self.obs_filenames: list[str] = obs_par_dict["obs_filenames"]
        self.obs_rng_seed: int = obs_par_dict["obs_rng_seed"]

        # If we're using synthetic observations/ running OSSEs define true_cme_par_dict
        if self.use_synthetic_obs or (self.obs_filenames is None):
            self.true_cme_par_dict: CmeParEns = initialise_true_cme_par_dict(
                huxt_init_time=self.huxt_init_time
            )
        else:
            self.true_cme_par_dict: CmeParEns = None

        # Generate observations
        self.observations: list[float] = self.get_observations()

        # Generate prior cme_parameters
        fg_cme_par_dict = initialise_fg_cme_parameters()
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

        # Get prior CME standard deviations
        prior_cme_par_dict = initialise_prior_cme_sd()
        self.prior_sd_t_init: float = prior_cme_par_dict["sd_t_init"]
        self.prior_sd_speed: float = prior_cme_par_dict["sd_speed"]
        self.prior_sd_width: float = prior_cme_par_dict["sd_width"]
        self.prior_sd_lon: float = prior_cme_par_dict["sd_lon"]
        self.prior_sd_lat: float = prior_cme_par_dict["sd_lat"]
        self.prior_sd_thick: float = prior_cme_par_dict["sd_thick"]

        # Get DA parameters
        da_par_dict = initialise_da_parameters()
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

        random_seed = add_const + (mult_const * run_no)

        return random_seed


    def make_output_dir(self, cme_par_dict: CmeParEns, run_no: int):
        """
        Function to make the output directory
        :return: out_dir: Output directory
        """
        current_dir = os.path.dirname(__file__)
        parent_dir = os.path.join(current_dir, "..")
        abs_par_dir = os.path.abspath(parent_dir)

        base_dir = os.path.join(abs_par_dir, "output", "24_obs")

        truth_dir = (
            f"truth_{self.true_cme_par_dict["v"].value}_{self.true_cme_par_dict["width"].value}"
            f"_{self.true_cme_par_dict["lon"].value}_{self.true_cme_par_dict["lat"].value}"
            f"_{self.true_cme_par_dict["thick"].value}"
        )
        prior_dir = (
            f"prior_{self.fg_mean_cme_speed}_{self.fg_mean_cme_width}"
            f"_{self.fg_mean_cme_lon}_{self.fg_mean_cme_lat}"
            f"_{self.fg_mean_cme_thick}"
        )
        run_dir = f"run_{run_no:03d}"

        output_dir = os.path.join(
            base_dir, truth_dir, prior_dir, f"{self.delta_aux_pf}", run_dir
        )
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        return output_dir


    def get_observations(self) -> list[float]:
        """
        Function to get observations from files or generate synthetic observations
        :return: observations: Observations to be assimilated
        """
        # Initialise observations
        obs_class = Observations(
            obs_lon=self.obs_lon,
            obs_times_in_datetime=self.obs_times,
            obs_filenames=None,
            use_synthetic_obs=self.use_synthetic_obs,
            true_cme_par_dict=self.true_cme_par_dict,
            obs_cov=self.obs_cov,
            huxt_init_time=self.huxt_init_time,
            vr_in=self.vr_in,
            lon_start=self.lon_start,
            lon_stop=self.lon_stop,
            sim_time=self.sim_time,
            dt_scale=self.dt_scale,
            r_min=self.r_min,
            cme_init_rad=self.cme_init_rad,
            cme_fixed_duration=self.cme_fixed_duration,
            fixed_duration=self.fixed_duration,
            plot_huxt_output=False,
            obs_rng_seed=self.obs_rng_seed,
        )

        return obs_class.observations


    def initialise_mean_cme_pars(self, rng):
        #######################################################################
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
            mean_cme_par_dict["mean_cme_t_init"] - self.huxt_init_time
        ).total_seconds()
        mean_cme_speed = mean_cme_par_dict["mean_cme_speed"]
        mean_cme_width = mean_cme_par_dict["mean_cme_width"]
        mean_cme_lon = mean_cme_par_dict["mean_cme_lon"]
        mean_cme_lat = mean_cme_par_dict["mean_cme_lat"]
        mean_cme_thick = mean_cme_par_dict["mean_cme_thick"]

        # Initialise cme_par_dict
        cme_par_dict = initialise_cme_parameter_ensemble_dict(
            n_ensemble=self.n_members,huxt_init_time=self.huxt_init_time
        )

        low_cme_t_init = mean_cme_t_init - self.prior_sd_t_init
        high_cme_t_init = mean_cme_t_init + self.prior_sd_t_init
        cme_par_dict["t_init"] = [
            self.huxt_init_time + datetime.timedelta(
                seconds=rng.uniform(low=low_cme_t_init, high=high_cme_t_init)
            ) for _ in range(self.n_members)
        ]

        low_cme_speed = (1 - self.prior_sd_speed) * mean_cme_speed
        high_cme_speed = (1 + self.prior_sd_speed) * mean_cme_speed
        cme_par_dict["v"] = [
            rng.uniform(low=low_cme_speed, high=high_cme_speed) for _ in range(self.n_members)
        ] * u.km / u.s

        low_cme_width = mean_cme_width - self.prior_sd_width
        high_cme_width = mean_cme_width + self.prior_sd_width
        cme_par_dict["width"] = [
            rng.uniform(low=low_cme_width, high=high_cme_width) for _ in range(self.n_members)
        ] * u.deg

        low_cme_lon = mean_cme_lon - self.prior_sd_lon
        high_cme_lon = mean_cme_lon + self.prior_sd_lon
        cme_par_dict["lon"] = [
            rng.uniform(low=low_cme_lon, high=high_cme_lon) for _ in range(self.n_members)
        ] * u.deg

        low_cme_lat = mean_cme_lat - self.prior_sd_lat
        high_cme_lat = mean_cme_lat + self.prior_sd_lat
        cme_par_dict["lat"] = [
            rng.uniform(low=low_cme_lat, high=high_cme_lat) for _ in range(self.n_members)
        ] * u.deg

        low_cme_thick = mean_cme_thick - self.prior_sd_thick
        high_cme_thick = mean_cme_thick + self.prior_sd_thick
        cme_par_dict["thick"] = [
            rng.uniform(low=low_cme_thick, high=high_cme_thick) for _ in range(self.n_members)
        ] * u.solRad

        return cme_par_dict


    def run_data_assimilation(self, run_start=0):
        """
        Function to run the data assimilation routine
        :return: None
        """
        for run_no in range(run_start, run_start + self.n_runs):
            print(f"run_no = {run_no}")

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
                        cme_par_dict["t_init"][i] - cme_par_dict["huxt_init_time"]
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
            #print(f"obs={self.observations}")
            for yi, obs in enumerate(self.observations):
                print(f"run_no = {run_no}/{self.n_runs}, obs = {yi}/{len(self.observations)}")
                aux_pf_class = AuxPF(
                    cme_par_dict=cme_par_dict,
                    obs=obs,
                    obs_cov=self.obs_cov,
                    obs_lon=self.obs_lon,
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
                    rng=rng
                )

                cme_par_dict = aux_pf_class.aux_pf()

                # Standardise the units and remove the astropy units
                cme_saved_pars[yi + 1, :, 0] = [
                    (
                        cme_par_dict["t_init"][i] - cme_par_dict["huxt_init_time"]
                    ).total_seconds()
                    for i in range(self.n_members)
                ]
                # print(cme_par_dict['width'])
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

                # for par_ind in [1, 2, 3]:
                #     print(f"cme_saved_pars[{yi + 1}] = {cme_saved_pars[yi + 1, :, par_ind]}")

            for par_ind in [1, 2, 3]:
                print(f"mean_cme_saved_pars[{par_ind}] = {np.mean(cme_saved_pars[:, :, par_ind], axis=1)}")

            # Save cme_parameters into a .nc file
            # Make an xarray object
            cme_par_ds = xr.Dataset(
                data_vars=dict(
                    model_init_time=self.huxt_init_time,
                    cme_init_rad=self.cme_init_rad,
                    r_min=self.r_min,
                    ambient_vr=(["n_lon"], self.vr_in),
                    t_init=(["n_obs", "n_ens"], cme_saved_pars[:, :, 0]),
                    v=(["n_obs", "n_ens"], cme_saved_pars[:, :, 1]),
                    width=(["n_obs", "n_ens"], cme_saved_pars[:, :, 2]),
                    lon=(["n_obs", "n_ens"], cme_saved_pars[:, :, 3]),
                    lat=(["n_obs", "n_ens"], cme_saved_pars[:, :, 4]),
                    thick=(["n_obs", "n_ens"], cme_saved_pars[:, :, 5]),
                ),
                coords=dict(
                    obs_no=("n_obs", range(self.n_obs + 1)),
                    ens_no=("n_ens", range(self.n_members)),
                    huxt_lon=("n_lon", (2 * np.pi / 128.0) * np.arange(128)),
                ),
            )
            print(cme_par_ds)
            outParFile = os.path.join(output_dir, "cme_pars.nc")
            cme_par_ds.to_netcdf(outParFile, mode='w')

        return None


def main():
    run_da_class = RunDataAssimilationRoutine()
    run_da_class.run_data_assimilation(run_start=0)

    return None

if __name__ == "__main__":
    main()