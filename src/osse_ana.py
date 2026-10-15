"""
osse_ana.py

Analysis of data-assimilation output (OSSEs and real events). ``main()`` loads the ``cme_pars.nc``
file(s) of the configured event / craft / runs, prints the prior and posterior mean CME parameters,
runs the arrival-time analysis and plots each parameter against observation number.

Edit the settings at the top of ``main()`` (event_list, craft_list, use_model, n_ens, runs) and the
per-event block (prior directory, observed arrival time, observation times) to choose what to analyse.

The helpers are split into:
    osse_stats.py   : weighted median and NaN-safe statistics
    osse_plots.py   : plotting functions
    osse_arrival.py : CME arrival-time analysis
and are re-exported here for backwards compatibility.
"""
import datetime
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
import xarray as xr
from dotenv import load_dotenv

from sir_observations import Observations

# Re-exported for backwards compatibility (these functions were previously defined here)
from osse_stats import weighted_median, numpy_nan_mean, numpy_nan_median, numpy_nan_std  # noqa: F401
from osse_plots import (  # noqa: F401
    plot_par_values_over_single_run, plot_par_values_over_mult_runs,
    plot_par_histograms_over_mult_runs, plot_sample_cov,
)
from osse_arrival import (  # noqa: F401
    get_prior_post_cme_pars_from_ds, get_true_cme_arrival_time, get_cme_stats_prior_post, get_cme_arrival_time,
)

env_path = Path('.', '.env')
load_dotenv(dotenv_path=env_path, override=True)


def print_environment_variables() -> None:
    """Print the directories configured in the .env file."""
    print(f"obs_dir = {os.getenv('OBS_DIR')}")
    print(f"mo_cone_cov_dir = {os.getenv('MO_CONE_COV_DIR')}")
    print(f"donki_cov_dir = {os.getenv('DONKI_COV_DIR')}")
    print(f"mo_cme_cone_file_dir = {os.getenv('MO_CME_CONE_FILE_DIR')}")
    print(f"out_base_dir = {os.getenv('OUT_BASE_DIR')}")
    print(f"wsa_dir = {os.getenv('WSA_DIR')}")

    return None


def main() -> None:
    """Analyse the configured DA output (see module docstring)."""
    # ---------------------------------------------------------------------------------
    # Settings
    # ---------------------------------------------------------------------------------
    nRuns = 1
    start_run = -1
    t_initTruth = None
    vTruth = None
    widthTruth = None
    lonTruth = None
    latTruth = None

    obs_dir = os.getenv('OBS_DIR')

    # e.g. ["ssw_007", "ssw_008", "ssw_009", "ssw_012"], ["09A"] or ["twin"]
    event_list = ["09A"]
    craft_list = ["sta"]#, "stb"]
    img_list = ["diff"] # ["norm", "diff"]
    par_type = "donki"
    use_model = "compress_surf3d"
    bias_folder_name = "bias5_2.5_bias21_2.5"
    n_ens = 50

    all_comb_list = [
        (x, y, z) for x in event_list for y in craft_list for z in img_list
    ]
    assert (use_model in ["surf", "compress_surf", "compress_surf3d", "huxt"])

    for (event, craft, img) in all_comb_list:
        print(f"\nevent: {event}, craft: {craft}, img: {img}")

        # -----------------------------------------------------------------------------
        # Per-event settings: prior directory, observed arrival time, observation times
        # -----------------------------------------------------------------------------

        if craft == "sta":
            craft_name = "STEREO-A"
        elif craft == "stb":
            craft_name = "STEREO-B"

        if event == "ssw_007":
            if par_type == "donki":
                prior_dir = "prior_1498_150_-63_-15_0"
            else:
                prior_dir = "prior_1010_66_-30_0_0"
            real_arrival_time = datetime.datetime(2012, 9, 3, 11, 23, 0)
            real_arr_str = real_arrival_time.strftime("%Y/%m/%d %H:%M")
            cme_hit_object = "EARTH"
            fig_title = f"CME-1: {craft_name} data assimilated"
            if craft == "sta":
                obs_times = [
                    datetime.datetime(2012, 9, 1, 0, 9, 1), datetime.datetime(2012, 9, 1, 0, 49, 1),
                    datetime.datetime(2012, 9, 1, 1, 29, 1), datetime.datetime(2012, 9, 1, 2, 9, 1),
                    datetime.datetime(2012, 9, 1, 2, 49, 1), datetime.datetime(2012, 9, 1, 3, 29, 1),
                    datetime.datetime(2012, 9, 1, 4, 9, 1), datetime.datetime(2012, 9, 1, 4, 49, 1),
                    datetime.datetime(2012, 9, 1, 5, 29, 1), datetime.datetime(2012, 9, 1, 6, 9, 1),
                    datetime.datetime(2012, 9, 1, 6, 49, 1), datetime.datetime(2012, 9, 1, 7, 29, 1),
                    datetime.datetime(2012, 9, 1, 8, 9, 1), datetime.datetime(2012, 9, 1, 8, 49, 1),
                    datetime.datetime(2012, 9, 1, 9, 29, 1), datetime.datetime(2012, 9, 1, 10, 9, 1),
                    datetime.datetime(2012, 9, 1, 10, 49, 1), datetime.datetime(2012, 9, 1, 11, 29, 1),
                    datetime.datetime(2012, 9, 1, 12, 9, 1), datetime.datetime(2012, 9, 1, 12, 49, 1),
                    datetime.datetime(2012, 9, 1, 13, 29, 1), datetime.datetime(2012, 9, 1, 14, 9, 1),
                    datetime.datetime(2012, 9, 1, 14, 49, 1), datetime.datetime(2012, 9, 1, 15, 29, 1),
                    datetime.datetime(2012, 9, 1, 16, 9, 1), datetime.datetime(2012, 9, 1, 16, 49, 1),
                    datetime.datetime(2012, 9, 1, 17, 29, 1), datetime.datetime(2012, 9, 1, 18, 9, 1),
                    datetime.datetime(2012, 9, 1, 18, 49, 1), datetime.datetime(2012, 9, 1, 19, 29, 1)
                ]
            elif craft == "stb":
                obs_times = [
                    datetime.datetime(2012, 8, 31, 23, 29, 1), datetime.datetime(2012, 9, 1, 0, 9, 1),
                    datetime.datetime(2012, 9, 1, 0, 49, 1), datetime.datetime(2012, 9, 1, 1, 29, 1),
                    datetime.datetime(2012, 9, 1, 2, 9, 1), datetime.datetime(2012, 9, 1, 2, 49, 1),
                    datetime.datetime(2012, 9, 1, 3, 29, 1), datetime.datetime(2012, 9, 1, 4, 9, 1),
                    datetime.datetime(2012, 9, 1, 4, 49, 1), datetime.datetime(2012, 9, 1, 5, 29, 1),
                    datetime.datetime(2012, 9, 1, 6, 9, 1), datetime.datetime(2012, 9, 1, 6, 49, 1),
                    datetime.datetime(2012, 9, 1, 7, 29, 1), datetime.datetime(2012, 9, 1, 8, 9, 1),
                    datetime.datetime(2012, 9, 1, 8, 49, 1), datetime.datetime(2012, 9, 1, 9, 29, 1),
                    datetime.datetime(2012, 9, 1, 10, 9, 1), datetime.datetime(2012, 9, 1, 10, 49, 1)
                ]

        elif event == "ssw_008":
            if par_type == "donki":
                prior_dir = "prior_1160_170_30_5_0"
            else:
                prior_dir = "prior_872_110_20_4_0"
            real_arrival_time = datetime.datetime(2012, 9, 30, 22, 13, 0)
            cme_hit_object = "EARTH"
            fig_title = f"CME-2: {craft_name} data assimilated"
            if craft == "sta":
                obs_times = [
                    datetime.datetime(2012, 9, 28, 3, 29, 1), datetime.datetime(2012, 9, 28, 4, 9, 1),
                    datetime.datetime(2012, 9, 28, 4, 49, 1), datetime.datetime(2012, 9, 28, 5, 29, 1),
                    datetime.datetime(2012, 9, 28, 6, 9, 1), datetime.datetime(2012, 9, 28, 6, 49, 1),
                    datetime.datetime(2012, 9, 28, 7, 29, 1), datetime.datetime(2012, 9, 28, 8, 9, 1),
                    datetime.datetime(2012, 9, 28, 8, 49, 1), datetime.datetime(2012, 9, 28, 9, 29, 1),
                    datetime.datetime(2012, 9, 28, 10, 9, 1), datetime.datetime(2012, 9, 28, 10, 49, 1),
                    datetime.datetime(2012, 9, 28, 11, 29, 1), datetime.datetime(2012, 9, 28, 12, 9, 1),
                    datetime.datetime(2012, 9, 28, 12, 49, 1), datetime.datetime(2012, 9, 28, 13, 29, 1),
                    datetime.datetime(2012, 9, 28, 14, 9, 1), datetime.datetime(2012, 9, 28, 14, 49, 1),
                    datetime.datetime(2012, 9, 28, 15, 29, 1), datetime.datetime(2012, 9, 28, 16, 9, 1),
                    datetime.datetime(2012, 9, 28, 16, 49, 1)
                ]
            elif craft == "stb":
                obs_times = [
                    datetime.datetime(2012, 9, 28, 8, 9, 1), datetime.datetime(2012, 9, 28, 8, 49, 1),
                    datetime.datetime(2012, 9, 28, 9, 29, 1), datetime.datetime(2012, 9, 28, 10, 9, 1),
                    datetime.datetime(2012, 9, 28, 10, 49, 1), datetime.datetime(2012, 9, 28, 11, 29, 1),
                    datetime.datetime(2012, 9, 28, 12, 9, 1), datetime.datetime(2012, 9, 28, 12, 49, 1),
                    datetime.datetime(2012, 9, 28, 13, 29, 1), datetime.datetime(2012, 9, 28, 14, 9, 1),
                    datetime.datetime(2012, 9, 28, 14, 49, 1), datetime.datetime(2012, 9, 28, 15, 29, 1),
                    datetime.datetime(2012, 9, 28, 16, 9, 1), datetime.datetime(2012, 9, 28, 16, 49, 1),
                    datetime.datetime(2012, 9, 28, 17, 29, 1), datetime.datetime(2012, 9, 28, 18, 9, 1),
                    datetime.datetime(2012, 9, 28, 18, 49, 1), datetime.datetime(2012, 9, 28, 19, 29, 1),
                    datetime.datetime(2012, 9, 28, 20, 9, 1), datetime.datetime(2012, 9, 28, 20, 49, 1),
                    datetime.datetime(2012, 9, 28, 21, 29, 1), datetime.datetime(2012, 9, 28, 22, 9, 1),
                    datetime.datetime(2012, 9, 28, 22, 49, 1), datetime.datetime(2012, 9, 28, 23, 29, 1),
                    datetime.datetime(2012, 9, 29, 0, 9, 1), datetime.datetime(2012, 9, 29, 0, 49, 1),
                    datetime.datetime(2012, 9, 29, 1, 29, 1)
                ]


        elif event == "ssw_009":
            if par_type=="donki":
                prior_dir= "prior_650_94_10_-28_0"
            else:
                prior_dir = "prior_698_84_9_-24_0"
            real_arrival_time = datetime.datetime(2012, 10, 8, 4, 31, 0)
            cme_hit_object = "EARTH"
            fig_title = f"CME-3: {craft_name} data assimilated"
            if craft == "sta":
                obs_times = [
                    datetime.datetime(2012, 10, 5, 9, 29, 1), datetime.datetime(2012, 10, 5, 10, 9, 1),
                    datetime.datetime(2012, 10, 5, 10, 49, 1), datetime.datetime(2012, 10, 5, 11, 29, 1),
                    datetime.datetime(2012, 10, 5, 12, 9, 1), datetime.datetime(2012, 10, 5, 12, 49, 1),
                    datetime.datetime(2012, 10, 5, 13, 29, 1), datetime.datetime(2012, 10, 5, 14, 9, 1),
                    datetime.datetime(2012, 10, 5, 14, 49, 1), datetime.datetime(2012, 10, 5, 15, 29, 1),
                    datetime.datetime(2012, 10, 5, 16, 9, 1), datetime.datetime(2012, 10, 5, 16, 49, 1),
                    datetime.datetime(2012, 10, 5, 17, 29, 1), datetime.datetime(2012, 10, 5, 18, 9, 1),
                    datetime.datetime(2012, 10, 5, 18, 49, 1), datetime.datetime(2012, 10, 5, 19, 29, 1),
                    datetime.datetime(2012, 10, 5, 20, 9, 1), datetime.datetime(2012, 10, 5, 20, 49, 1),
                    datetime.datetime(2012, 10, 5, 21, 29, 1), datetime.datetime(2012, 10, 5, 22, 9, 1),
                    datetime.datetime(2012, 10, 5, 22, 49, 1), datetime.datetime(2012, 10, 5, 23, 29, 1),
                    datetime.datetime(2012, 10, 6, 0, 9, 1), datetime.datetime(2012, 10, 6, 0, 49, 1),
                    datetime.datetime(2012, 10, 6, 1, 29, 1), datetime.datetime(2012, 10, 6, 2, 9, 1),
                    datetime.datetime(2012, 10, 6, 2, 49, 1)
                ]
            elif craft == "stb":
                obs_times = [
                    datetime.datetime(2012, 10, 5, 10, 49, 1), datetime.datetime(2012, 10, 5, 11, 29, 1),
                    datetime.datetime(2012, 10, 5, 12, 9, 1), datetime.datetime(2012, 10, 5, 12, 49, 1),
                    datetime.datetime(2012, 10, 5, 13, 29, 1), datetime.datetime(2012, 10, 5, 14, 9, 1),
                    datetime.datetime(2012, 10, 5, 14, 49, 1), datetime.datetime(2012, 10, 5, 15, 29, 1),
                    datetime.datetime(2012, 10, 5, 16, 9, 1), datetime.datetime(2012, 10, 5, 16, 49, 1),
                    datetime.datetime(2012, 10, 5, 17, 29, 1), datetime.datetime(2012, 10, 5, 18, 9, 1),
                    datetime.datetime(2012, 10, 5, 18, 49, 1), datetime.datetime(2012, 10, 5, 19, 29, 1),
                    datetime.datetime(2012, 10, 5, 20, 9, 1), datetime.datetime(2012, 10, 5, 20, 49, 1),
                    datetime.datetime(2012, 10, 5, 21, 29, 1), datetime.datetime(2012, 10, 5, 22, 9, 1),
                    datetime.datetime(2012, 10, 5, 22, 49, 1), datetime.datetime(2012, 10, 5, 23, 29, 1),
                    datetime.datetime(2012, 10, 6, 0, 9, 1), datetime.datetime(2012, 10, 6, 0, 49, 1),
                    datetime.datetime(2012, 10, 6, 1, 29, 1), datetime.datetime(2012, 10, 6, 2, 9, 1),
                    datetime.datetime(2012, 10, 6, 3, 29, 1), datetime.datetime(2012, 10, 6, 4, 9, 1)
                ]

        elif event == "ssw_012":
            if par_type=="donki":
                prior_dir = "prior_725_60_90_5_0"
                cme_hit_object = "EARTH"
            else:
                prior_dir = "prior_664_94_22_20_0"
                cme_hit_object = "EARTH"
            real_arrival_time = datetime.datetime(2012, 11, 23, 21, 10, 0)
            fig_title = f"CME-4: {craft_name} data assimilated"

            if craft == "sta":
                obs_times = [
                    datetime.datetime(2012, 11, 20, 16, 49, 1), datetime.datetime(2012, 11, 20, 17, 29, 1),
                    datetime.datetime(2012, 11, 20, 18, 9, 1), datetime.datetime(2012, 11, 20, 18, 49, 1),
                    datetime.datetime(2012, 11, 20, 19, 29, 1), datetime.datetime(2012, 11, 20, 20, 9, 1),
                    datetime.datetime(2012, 11, 20, 20, 49, 1), datetime.datetime(2012, 11, 20, 21, 29, 1),
                    datetime.datetime(2012, 11, 20, 22, 9, 1), datetime.datetime(2012, 11, 20, 22, 49, 1),
                    datetime.datetime(2012, 11, 20, 23, 29, 1), datetime.datetime(2012, 11, 21, 0, 9, 1),
                    datetime.datetime(2012, 11, 21, 0, 49, 1), datetime.datetime(2012, 11, 21, 1, 29, 1),
                    datetime.datetime(2012, 11, 21, 2, 9, 1), datetime.datetime(2012, 11, 21, 2, 49, 1),
                    datetime.datetime(2012, 11, 21, 3, 29, 1), datetime.datetime(2012, 11, 21, 4, 9, 1),
                    datetime.datetime(2012, 11, 21, 4, 49, 1), datetime.datetime(2012, 11, 21, 5, 29, 1),
                    datetime.datetime(2012, 11, 21, 6, 9, 1), datetime.datetime(2012, 11, 21, 6, 49, 1),
                    datetime.datetime(2012, 11, 21, 7, 29, 1), datetime.datetime(2012, 11, 21, 8, 9, 1),
                    datetime.datetime(2012, 11, 21, 8, 49, 1), datetime.datetime(2012, 11, 21, 9, 29, 1),
                    datetime.datetime(2012, 11, 21, 10, 9, 1)
                ]
            elif craft == "stb":
                obs_times = [
                    datetime.datetime(2012, 11, 20, 19, 29, 1), datetime.datetime(2012, 11, 20, 20, 9, 1),
                    datetime.datetime(2012, 11, 20, 20, 49, 1), datetime.datetime(2012, 11, 20, 21, 29, 1),
                    datetime.datetime(2012, 11, 20, 22, 9, 1), datetime.datetime(2012, 11, 20, 22, 49, 1),
                    datetime.datetime(2012, 11, 21, 0, 49, 1), datetime.datetime(2012, 11, 21, 1, 29, 1),
                    datetime.datetime(2012, 11, 21, 2, 9, 1), datetime.datetime(2012, 11, 21, 2, 49, 1),
                    datetime.datetime(2012, 11, 21, 3, 29, 1), datetime.datetime(2012, 11, 21, 4, 9, 1),
                    datetime.datetime(2012, 11, 21, 4, 49, 1), datetime.datetime(2012, 11, 21, 5, 29, 1),
                    datetime.datetime(2012, 11, 21, 6, 9, 1), datetime.datetime(2012, 11, 21, 6, 49, 1),
                    datetime.datetime(2012, 11, 21, 7, 29, 1), datetime.datetime(2012, 11, 21, 8, 9, 1),
                    datetime.datetime(2012, 11, 21, 8, 49, 1), datetime.datetime(2012, 11, 21, 9, 29, 1),
                    datetime.datetime(2012, 11, 21, 10, 9, 1), datetime.datetime(2012, 11, 21, 10, 49, 1),
                    datetime.datetime(2012, 11, 21, 11, 29, 1), datetime.datetime(2012, 11, 21, 12, 9, 1),
                    datetime.datetime(2012, 11, 21, 12, 49, 1), datetime.datetime(2012, 11, 21, 13, 29, 1),
                    datetime.datetime(2012, 11, 21, 14, 9, 1), datetime.datetime(2012, 11, 21, 14, 49, 1),
                    datetime.datetime(2012, 11, 21, 15, 29, 1)
                ]
        elif event == "01B":
            obs_file_name = "CME_01B__20100407__PA265_good_hi2.dat"
            obs_file_path = os.path.join(obs_dir, obs_file_name)
            prior_dir = "prior_258.0_50.0_-74.0_18.0_0.0"
            obs_class = Observations(
                use_model=use_model,
                use_synthetic_obs=False,
                obs_filenames=obs_file_path,
            )
            obs_times, obs_lon, obs_lat, obs = obs_class.l5_read_obs_from_file()
            cme_hit_object = "EARTH"

            real_arrival_time = datetime.datetime(2010, 4, 11, 13, 4, 0)
            real_arrival_speed = 0
            fig_title = f"CME-{event}"
        elif event == "02A":
            obs_file_name = "CME_02A__20201025__PA272_good_hi2.dat"
            obs_file_path = os.path.join(obs_dir, obs_file_name)
            prior_dir = "prior_159.0_36.0_-15.0_-10.0_0.0"
            obs_class = Observations(
                use_model=use_model,
                use_synthetic_obs=False,
                obs_filenames=obs_file_path,
            )
            obs_times, obs_lon, obs_lat, obs = obs_class.l5_read_obs_from_file()
            cme_hit_object = "EARTH"

            real_arrival_time = datetime.datetime(2020, 11, 1, 11, 54, 0)
            real_arrival_speed = 0
            fig_title = f"CME-{event}"
        elif event == "03A":
            obs_file_name = "CME_03A__20211101__PA268_good_hi2.dat"
            obs_file_path = os.path.join(obs_dir, obs_file_name)
            prior_dir = "prior_1188.0_102.0_5.0_18.0_0.0"

            obs_class = Observations(
                use_model=use_model,
                use_synthetic_obs=False,
                obs_filenames=obs_file_path,
            )
            obs_times, obs_lon, obs_lat, obs = obs_class.l5_read_obs_from_file()
            cme_hit_object = "EARTH"

            real_arrival_time = datetime.datetime(2021, 11, 3, 19, 49, 0)
            real_arrival_speed = 0
            fig_title = f"CME-{event}"
        elif event == "04B":
            obs_file_name = "CME_04B__20100523__PA271_good_hi2.dat"
            obs_file_path = os.path.join(obs_dir, obs_file_name)
            prior_dir = "prior_542.0_76.0_24.0_1.0_0.0"

            obs_class = Observations(
                use_model=use_model,
                use_synthetic_obs=False,
                obs_filenames=obs_file_path,
            )
            obs_times, obs_lon, obs_lat, obs = obs_class.l5_read_obs_from_file()

            cme_hit_object = "EARTH"
            real_arrival_time = datetime.datetime(2010, 5, 28, 2, 58, 0)
            real_arrival_speed = 0
            fig_title = f"CME-{event}"
        elif event == "05B":
            obs_file_name = "CME_05B__20091215__PA266_good_hi2.dat"
            obs_file_path = os.path.join(obs_dir, obs_file_name)
            prior_dir = "prior_542.0_48.0_-7.0_10.0_0.0"

            obs_class = Observations(
                use_model=use_model,
                use_synthetic_obs=False,
                obs_filenames=obs_file_path,
            )
            obs_times, obs_lon, obs_lat, obs = obs_class.l5_read_obs_from_file()
            cme_hit_object = "EARTH"

            real_arrival_time = datetime.datetime(2009, 12, 19, 10, 0, 0)
            real_arrival_speed = 0
            fig_title = f"CME-{event}"
        elif event == "09A":
            obs_file_name = "CME_09A__20211027__PA268_good_hi2.dat"
            obs_file_path = os.path.join(obs_dir, obs_file_name)
            prior_dir = "prior_1368.0_100.0_2.0_-23.0_0.0"

            obs_class = Observations(
                use_model=use_model,
                use_synthetic_obs=False,
                obs_filenames=obs_file_path,
            )
            obs_times, obs_lon, obs_lat, obs = obs_class.l5_read_obs_from_file()
            cme_hit_object = "EARTH"

            real_arrival_time = datetime.datetime(2021, 10, 31, 9, 13, 0)
            real_arrival_speed = 0
            fig_title = f"CME-{event}"
        elif event == "10A":
            obs_file_name = "CME_10A__20201206__PA266_good_hi2.dat"
            obs_file_path = os.path.join(obs_dir, obs_file_name)
            prior_dir = "prior_1436.0_82.0_8.0_-18.0_0.0"

            obs_class = Observations(
                use_model=use_model,
                use_synthetic_obs=False,
                obs_filenames=obs_file_path,
            )
            obs_times, obs_lon, obs_lat, obs = obs_class.l5_read_obs_from_file()
            cme_hit_object = "EARTH"

            real_arrival_time = datetime.datetime(2020, 12, 10, 2, 10, 0)
            real_arrival_speed = 0
            fig_title = f"CME-{event}"
        elif event == "13B":
            obs_file_name = "CME_13B__20110524__PA271_good_hi2.dat"
            obs_file_path = os.path.join(obs_dir, obs_file_name)
            prior_dir = "prior_598.0_52.0_6.0_-10.0_0.0"

            obs_class = Observations(
                use_model=use_model,
                use_synthetic_obs=False,
                obs_filenames=obs_file_path,
            )
            obs_times, obs_lon, obs_lat, obs = obs_class.l5_read_obs_from_file()
            cme_hit_object = "EARTH"

            real_arrival_time = datetime.datetime(2011, 5, 28, 0, 14, 0)
            real_arrival_speed = 0
            fig_title = f"CME-{event}"
        elif event == "14B":
            obs_file_name = "CME_14B__20110710__PA274_good_hi2.dat"
            obs_file_path = os.path.join(obs_dir, obs_file_name)
            prior_dir = "prior_552.0_42.0_14.0_-11.0_0.0"

            obs_class = Observations(
                use_model=use_model,
                use_synthetic_obs=False,
                obs_filenames=obs_file_path,
            )
            obs_times, obs_lon, obs_lat, obs = obs_class.l5_read_obs_from_file()
            cme_hit_object = "EARTH"

            real_arrival_time = datetime.datetime(2011, 7, 14, 12, 0, 0)
            real_arrival_speed = 0
            fig_title = f"CME-{event}"
        elif event == "15B":
            obs_file_name = "CME_15B__20111027__PA276_good_hi2.dat"
            obs_file_path = os.path.join(obs_dir, obs_file_name)
            prior_dir = "prior_589.0_78.0_-7.0_24.0_0.0"

            obs_class = Observations(
                use_model=use_model,
                use_synthetic_obs=False,
                obs_filenames=obs_file_path,
            )
            obs_times, obs_lon, obs_lat, obs = obs_class.l5_read_obs_from_file()
            cme_hit_object = "EARTH"

            real_arrival_time = datetime.datetime(2011, 11, 1, 8, 9, 0)
            real_arrival_speed = 0
            fig_title = f"CME-{event}"
        elif event == "twin":
            true_cme_t_init = datetime.datetime(2012, 1, 1, 1, 0, 0, 0)
            true_cme_speed = 495.0
            true_cme_width = 37.4
            true_cme_lon = 0.0
            true_cme_lat = 0.0
            true_cme_thick = 0.0
            truth_dir = "truth_495.0_37.4_0.0_0.0_0.0"
            prior_dir = "prior_495.0_37.4_0.0_-2.9_0.0"

            vTruth = 495
            widthTruth = 37.4
            lonTruth = 0

        else:
            sys.exit(f"Unknown event {event}.")
        # Observation directory name, as written by data_assimilation_routine.make_output_dir:
        #  truth_<...> for twin experiments, otherwise obs_<craft>_<event>_<img>, where for the
        #  "NNA"/"NNB" events the craft follows the event letter (A -> STEREO-A, B -> STEREO-B)
        if event == "twin":
            obs_dir_name = truth_dir
        else:
            if event.startswith("ssw"):
                event_craft = craft
            else:
                event_craft = {"A": "sta", "B": "stb"}[event[-1]]
            obs_dir_name = f"obs_{event_craft}_{event}_{img}"

        # -----------------------------------------------------------------------------
        # Directory holding run_NNN/cme_pars.nc (alternatives kept in the strings below)
        # -----------------------------------------------------------------------------
        """baseFilePath = os.path.join(
            "C:\\", "Users", "ss905122", "PycharmProjects", "SIR_HUXt", "output",
            "WSA_v", f"{use_model.upper()}", f"ens_{n_ens}", "mo_cone", obs_dir_name,  # "New folder",
            prior_dir, "0.98",
        )"""
        baseFilePath = os.path.join(
            "C:\\", "Users", "ss905122", "PycharmProjects", "SIR_HUXt", "output",
            "compress_huxt", "test", "WSA_v", f"ens_{n_ens}", "mo_cone",
            f"model_{use_model.upper()}",
            obs_dir_name, # "New folder",
            prior_dir, "0.98",
        )
        """
        baseFilePath = os.path.join(
            "C:\\", "Users", "ss905122", "PycharmProjects", "SIR_HUXt", "output",
            "compress_huxt", "dares", "SURF_parallel", f"ens_{n_ens}", "uncorr", f"model_{use_model.upper()}",
            truth_dir, prior_dir, "0.98",
        )"""
        """baseFilePath = os.path.join(
            "C:\\", "Users", "ss905122", "PycharmProjects", "SIR_HUXt", "output3",
            "WSA_v", f"ens_{n_ens}", "mo_cone", obs_dir_name,  # "New folder",
            prior_dir, "0.98",
        )"""

        # -----------------------------------------------------------------------------
        # Load the runs, print the mean parameters, arrival-time analysis and plots
        # -----------------------------------------------------------------------------
        # Read in all nc files and concatenate them into a single xarray object
        for ir, runNo in enumerate(range(start_run, start_run + nRuns)):
            print(runNo)

            filePath = os.path.join(
                baseFilePath, f"run_{runNo:03d}", "cme_pars.nc"
            )
            with xr.load_dataset(filePath) as dsTemp:
                dsTemp.expand_dims(dim={"run": 1})
                dsTemp["run"] = (("run",), [runNo])

                if ir == 0:
                    ds = dsTemp.copy()
                else:
                    ds = xr.concat([ds, dsTemp], "run")

        print(ds)
        cme_t_init_mean_sec = ds["t_init"].values.mean(axis=1)
        cme_t_init_mean = [
                pd.to_datetime(ds["model_init_time"].values) + datetime.timedelta(seconds=i)
                for i in cme_t_init_mean_sec
        ]
        cme_speed_mean = ds["v"].values.mean(axis=1)
        cme_width_mean = ds["width"].values.mean(axis=1)
        cme_lon_mean = ds["lon"].values.mean(axis=1)
        cme_lat_mean = ds["lat"].values.mean(axis=1)

        print("=======================================================")
        print(f"Mean CME launch time = Prior: {cme_t_init_mean[0]} - Post: {cme_t_init_mean[-1]}")
        print(f"Mean CME speed = Prior: {cme_speed_mean[0]:.2f} - Post: {cme_speed_mean[-1]:.2f}")
        print(f"Mean CME width = Prior: {cme_width_mean[0]:.2f} - Post: {cme_width_mean[-1]:.2f}")
        print(f"Mean CME lon = Prior: {cme_lon_mean[0]:.2f} - Post: {cme_lon_mean[-1]:.2f}")
        print(f"Mean CME lat = Prior: {cme_lat_mean[0]:.2f} - Post: {cme_lat_mean[-1]:.2f}")
        print("=========================================================")

        if event == "twin":
            real_arrival_time, real_arrival_speed = get_true_cme_arrival_time(
                ds=ds,
                use_model=use_model,
                true_cme_t_init=true_cme_t_init,
                true_cme_speed=true_cme_speed,
                true_cme_width=true_cme_width,
                true_cme_lat=true_cme_lat,
                true_cme_lon=true_cme_lon,
                true_cme_thick=true_cme_thick,
                cme_hit_object="EARTH"
            )
        print(f"real_arrival_time = {real_arrival_time}")
        print(f"real_arrival_speed = {real_arrival_speed}")

        if nRuns > n_ens:
            nProc = int(os.cpu_count() - 1)
            if nProc < 1:
                nProc = 1
            pool = Pool(processes=nProc)
            try:
                for ir, runNo in enumerate(range(start_run, start_run + nRuns)):
                    print("HI")
                    pool.apply_async(
                        get_cme_arrival_time,
                        kwds={
                            'ds': ds,
                            'runNo': runNo,
                            'use_model': use_model,
                            'real_arrival_time':real_arrival_time,
                            'cme_hit_object': "EARTH",
                            'base_file_path': baseFilePath,
                            'parallel_ens': False,
                        },
                    )
                pool.close()
                pool.join()
            except BaseException:
                # Release the worker processes if anything fails (or on Ctrl+C)
                pool.terminate()
                pool.join()
                raise
        else:
            for ir, runNo in enumerate(range(start_run, start_run + nRuns)):
                print("HI2")
                get_cme_arrival_time(
                    ds=ds,
                    runNo=runNo,
                    use_model=use_model,
                    real_arrival_time=real_arrival_time,
                    cme_hit_object="EARTH",
                    base_file_path=baseFilePath,
                    parallel_ens=True,
                )

        n_obs = len(ds["obs_no"][:])
        # (optionally: plot_sample_cov(ds, 0, i, vars_req=[...]) for each observation i)

        for ir, runNo in enumerate(range(start_run, start_run + nRuns)):
            if event == "twin":
                fig_title = f"Twin experiment, Run no. = {runNo}"
                obs_times = None

            plot_par_values_over_single_run(
                ds,
                xVarName="obs_no",
                yVarName="t_init",
                xLabel="Time",
                obs_times=obs_times,
                tick_cadence=3,
                fig_title=fig_title,
                parNameLegend="CME initial time",
                yTruth=vTruth,
                runNo=runNo,
                parUnits="",
                n_obs=n_obs,
                base_file_path=baseFilePath,
                craft=craft,
            )

            plot_par_values_over_single_run(
                ds,
                xVarName="obs_no",
                yVarName="v",
                xLabel="Time",
                obs_times=obs_times,
                tick_cadence=3,
                fig_title=fig_title,
                parNameLegend="CME speed",
                yTruth=vTruth,
                runNo=runNo,
                parUnits="km/s",
                n_obs=n_obs,
                base_file_path=baseFilePath,
                craft=craft,
            )
            plot_par_values_over_single_run(
                ds,
                xVarName="obs_no",
                yVarName="width",
                xLabel="Time",
                parNameLegend="CME Width",
                obs_times=obs_times,
                tick_cadence=3,
                fig_title=fig_title,
                yTruth=widthTruth,
                runNo=runNo,
                parUnits=r"$^\circ$",
                n_obs=n_obs,
                base_file_path=baseFilePath,
                craft=craft,
            )
            plot_par_values_over_single_run(
                ds,
                xVarName="obs_no",
                yVarName="lon",
                xLabel="Time",
                fig_title=fig_title,
                parNameLegend="CME Longitude",
                obs_times=obs_times,
                tick_cadence=3,
                yTruth=lonTruth,
                runNo=runNo,
                parUnits=r"$^\circ$",
                n_obs=n_obs,
                base_file_path=baseFilePath,
                craft=craft,
            )
            plot_par_values_over_single_run(
                ds,
                xVarName="obs_no",
                yVarName="lat",
                xLabel="Time",
                fig_title=fig_title,
                parNameLegend="CME Latitude",
                obs_times=obs_times,
                tick_cadence=3,
                yTruth=lonTruth,
                runNo=runNo,
                parUnits=r"$^\circ$",
                n_obs=n_obs,
                base_file_path=baseFilePath,
                craft=craft,
            )

    # (multi-run plots: see plot_par_values_over_mult_runs and plot_par_histograms_over_mult_runs)

    return None

if __name__ == "__main__":
    main()