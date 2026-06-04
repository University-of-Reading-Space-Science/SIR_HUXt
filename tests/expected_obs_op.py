import huxt.huxt
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

import sunpy.coordinates.sun as sn

import astropy.units as u
from astropy.units import Quantity
from astropy.time import Time
from sir_observation_operator import ObservationOperator

import matplotlib.pyplot as plt
from cme_par_ens import CmeParEns
import seaborn as sns
import colorcet as cc
import pytest
from cme_par_dict_structure import required_dict_keys

class ExpectedObservationOperator:
    def __init__(self):
        self.n_members: int = 5

        self.exp_cme_launch_time: list[Quantity[u.s]] = [
            68245.2,
            69568.3636363636,
            67096.8,
            81722.432432432,
            66235.5
        ] * u.s

        self.exp_cme_speed: list[Quantity[u.km / u.s]] = [
            500, 550, 750, 740, 1200
        ] * u.km / u.s

        self.exp_cme_width: list[Quantity[u.deg]] = [
            50, 30, 48, 56, 12
        ] * u.deg

        self.exp_cme_lon: list[Quantity[u.deg]] = [
            -34, -120, 49, -120, -175
        ] * u.deg

        self.exp_cme_lat: list[Quantity[u.deg]] = [
            2, 4, -49, -12, 8
        ] * u.deg

        self.exp_cme_thick: list[Quantity[u.solRad]] = [
            3, 12, 5, 7, 8
        ] * u.solRad

    def expected_cme_launch_time(self) -> list[Quantity[u.s]]:
        return self.exp_cme_launch_time

    def expected_cme_speed(self) -> list[Quantity[u.km / u.s]]:
        return self.exp_cme_speed

    def expected_cme_width(self) -> list[Quantity[u.deg]]:
        return self.exp_cme_width

    def expected_cme_lon(self) -> list[Quantity[u.deg]]:
        return self.exp_cme_lon

    def expected_cme_lat(self) -> list[Quantity[u.deg]]:
        return self.exp_cme_lat

    def expected_cme_thick(self) -> list[Quantity[u.solRad]]:
        return self.exp_cme_thick

    def expected_cme_parameters(self) -> tuple[
        list[Quantity[u.s]],
        list[Quantity[u.km / u.s]],
        list[Quantity[u.deg]],
        list[Quantity[u.deg]],
        list[Quantity[u.deg]],
        list[Quantity[u.solRad]]
    ]:
        return (
            self.exp_cme_launch_time, self.exp_cme_speed, self.exp_cme_width,
            self.exp_cme_lon, self.exp_cme_lat, self.exp_cme_thick
        )


class ExpectedCMEFlankSingle:
    def __init__(self):
        # For test CME flanks, set timestep to 10 minutes
        #  (for ease of checking the rounding of the times)
        self.huxt_init_time = datetime.datetime(year=2021, month=1, day=1, hour=0, minute=0, second=0)
        self.deltaT: Quantity[u.s] = 6000 * u.s
        self.n_timesteps_day: int = 144
        self.n_members: int = 5

        self.times: Time = Time([
            self.huxt_init_time + datetime.timedelta(seconds=i * self.deltaT.value)
            for i in range(self.n_timesteps_day)
        ])

        self.exp_flank_el0: npt.NDArray[float] = np.array(
            [4 + (i / 5.0) for i in range(self.n_timesteps_day)]
        )
        self.exp_flank_el1: npt.NDArray[float] = np.array(
            [3 + (i / 5.0) for i in range(self.n_timesteps_day)]
        )
        self.exp_flank_el2: npt.NDArray[float] = np.array(
            [4.5 + (i / 5.0) for i in range(self.n_timesteps_day)]
        )
        self.exp_flank_el3: npt.NDArray[float] = np.array(
            [3.9 + (i / 5.0) for i in range(self.n_timesteps_day)]
        )
        self.exp_flank_el4: npt.NDArray[float] = np.array(
            [4.2 + (i / 5.0) for i in range(self.n_timesteps_day)]
        )

        self.exp_flank_r0: npt.NDArray[float] = np.array(
            [0.05 + (i / 400.0) for i in range(self.n_timesteps_day)]
        )
        self.exp_flank_r1: npt.NDArray[float] = np.array(
            [0.07 + (i / 400.0) for i in range(self.n_timesteps_day)]
        )
        self.exp_flank_r2: npt.NDArray[float] = np.array(
            [0.06 + (i / 400.0) for i in range(self.n_timesteps_day)]
        )
        self.exp_flank_r3: npt.NDArray[float] = np.array(
            [0.08 + (i / 400.0) for i in range(self.n_timesteps_day)]
        )
        self.exp_flank_r4: npt.NDArray[float] = np.array(
            [0.075 + (i / 400.0) for i in range(self.n_timesteps_day)]
        )

        self.exp_flank_lon0: npt.NDArray[float] = np.array(
            [(-5.0 + (i / 14.4)) * np.pi / 180.0 for i in range(self.n_timesteps_day)]
        )
        self.exp_flank_lon1: npt.NDArray[float] = np.array(
            [(-5.5 + (i / 14.4)) * np.pi / 180.0 for i in range(self.n_timesteps_day)]
        )
        self.exp_flank_lon2: npt.NDArray[float] = np.array(
            [(-4.5 + (i / 14.4)) * np.pi / 180.0 for i in range(self.n_timesteps_day)]
        )
        self.exp_flank_lon3: npt.NDArray[float] = np.array(
            [(-4.0 + (i / 14.4)) * np.pi / 180.0 for i in range(self.n_timesteps_day)]
        )
        self.exp_flank_lon4: npt.NDArray[float] = np.array(
            [(-6.0 + (i / 14.4)) * np.pi / 180.0 for i in range(self.n_timesteps_day)]
        )


    def expected_flank_el(self, ens_member: int) -> npt.NDArray[float]:
        # Get the required variable for this key
        var_name = f"exp_flank_el{ens_member}"
        var_attr = getattr(self, var_name)

        # Call the function and place it's output into cme_par_array
        flank_el = var_attr

        return flank_el

    def expected_flank_r(self, ens_member: int) -> npt.NDArray[float]:
        # Get the required variable for this key
        var_name = f"exp_flank_r{ens_member}"
        var_attr = getattr(self, var_name)

        # Call the function and place it's output into cme_par_array
        flank_r = var_attr

        return flank_r

    def expected_flank_lon(self, ens_member: int) -> npt.NDArray[float]:
        # Get the required variable for this key
        var_name = f"exp_flank_lon{ens_member}"
        var_attr = getattr(self, var_name)

        # Call the function and place it's output into cme_par_array
        flank_lon = var_attr

        return flank_lon


    def init_flank2(self, ens_member: int) -> pd.DataFrame:
        # Initialise flank dataframe
        flank: pd.DataFrame = pd.DataFrame(
            index=np.arange(self.times.size),
            columns=['time', 'el', 'r', 'lon']
        )
        flank.loc[:, 'time'] = self.times.jd

        flank.loc[:, 'el'] = self.expected_flank_el(ens_member)
        flank.loc[:, 'r'] = self.expected_flank_r(ens_member)
        flank.loc[:, 'lon'] = self.expected_flank_lon(ens_member)

        return flank


    def expected_flanks(self) -> list[pd.DataFrame]:
        # Make list of flank dataframes
        flanks: list[pd.DataFrame] = [
            self.init_flank2(ens_member=i) for i in range(self.n_members)
        ]

        return flanks


    def expected_obs_op(self, obs_op_tests) -> tuple[list[float], list[datetime.datetime], list[int]]:
        """if hasattr(request, 'param'):
            exp_no = request.param
        else:
            exp_no = None"""

        # Initialise flank dataframe
        flanks: list[pd.DataFrame] = self.expected_flanks()

        # Extract the observation times for the experiment number specified
        obs_times, index_output = obs_op_tests

        #print(f"obs_times = {obs_times}")
        if type(index_output) is int:
            hx = [
                [flanks[m].loc[index_output, 'el']]
                for m in range(self.n_members)
            ]
        else:
            hx = [
                [flanks[m].loc[i, 'el'] for i in index_output]
                for m in range(self.n_members)
            ]
        print(f"hx = {hx}")

        return hx

if __name__ == "__main__":
    exp_class = ExpectedCMEFlankSingle()
    hx = exp_class.expected_obs_op()

