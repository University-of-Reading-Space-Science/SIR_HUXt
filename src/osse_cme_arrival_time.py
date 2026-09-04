import glob
import os
import sys
import json
from collections import defaultdict, ChainMap

from IPython.core.pylabtools import figsize
from astropy.time import Time
import astropy.units as u
import h5py
import matplotlib.pyplot as plt
import datetime
import matplotlib as mpl
import numpy as np

import huxt.huxt as H
import surf.surf as S
import xarray as xr
import scipy.stats as st

import sunpy.coordinates.sun as sn
import seaborn as sns
import pandas as pd
import holoviews as hv
import colorcet as cc
import weightedstats as wst

from colorcet.plotting import sine_combs
from sunpy.coordinates.sun import orientation
from multiprocessing import Pool


def flatten_nested(nested):
    flat = []
    stack = [iter(nested)]
    while stack:
        for item in stack[-1]:
            if isinstance(item, list):
                stack.append(iter(item))
                break
            flat.append(item)
        else:
            stack.pop()
    return flat


def main():
    use_model = "compress_surf"
    event = "twin"
    n_ens = 50
    start_run = 1
    n_runs = 99

    if event == "twin":
        true_cme_t_init = datetime.datetime(2012, 1, 1, 1, 0, 0, 0)
        true_cme_speed = 495.0
        true_cme_width = 37.4
        true_cme_lon = 0.0
        true_cme_lat = 0.0
        true_cme_thick = 0.0
        truth_dir = "truth_495.0_37.4_0.0_0.0_0.0"
        prior_dir = "prior_495.0_37.4_0.0_-2.9_0.0"

    base_file_path = os.path.join(
        "C:\\", "Users", "ss905122", "PycharmProjects", "SIR_HUXt", "output",
        "compress_huxt", "parallel", "WSA_v", f"ens_{n_ens}", "uncorr", f"model_{use_model.upper()}",
        truth_dir, prior_dir, "0.98",
    )

    cme_dict_list = []
    for runNo in range(start_run, start_run + n_runs):
        out_dir_path = os.path.join(base_file_path, f"run_{runNo:03d}")
        json_file_path = os.path.join(out_dir_path, f"cme_arrival_stats.json")

        with open(json_file_path, "r") as f:
            cme_arr_dict = json.load(f)

        cme_dict_list.append(cme_arr_dict)

    dict1 = {}
    for dict2 in cme_dict_list:
        dict1 = dict(ChainMap(
            {
                key: [dict1[key], dict2[key]]
                if key in dict2 else dict1[key] for key in dict1
            },
            {key: dict2[key] for key in dict2 if key not in dict1}
        ))
    # res = defaultdict(list)
    # for d in cme_dict_list:
    #     for k, v in d.items():
    #         res[k].extend(v)

    #print(f"dict_val = {dict(dict1)}")
    #print(np.shape(dict1["prior_cme_arrival_s_from_20000101"]))

    for key in dict1:
        dict1[key] = flatten_nested(dict1[key][:])

    print(dict1)
    real_arrival_time = datetime.datetime(2012,1,3,23,24,18)
    base_time = datetime.datetime(2000,1, 1, 0,0,0)

    seconds_real_arr_from_base = (real_arrival_time - base_time).total_seconds()

    prior_arrival_times_flat = np.array(dict1["prior_cme_arrival_s_from_20000101"]).flatten()
    post_arrival_times_flat = np.array(dict1["post_cme_arrival_s_from_20000101"]).flatten()

    mean_prior_arrival_times_flat = np.array(dict1["mean_prior_arrival"]).flatten()
    mean_post_arrival_times_flat = np.array(dict1["mean_post_arrival"]).flatten()

    median_prior_arrival_times_flat = np.array(dict1["median_prior_arrival"]).flatten()
    median_post_arrival_times_flat = np.array(dict1["median_post_arrival"]).flatten()

    #print(prior_arrival_times_flat)
    hours_prior_arrival_time_err = [
        (prior_arrival_times_flat[i] - seconds_real_arr_from_base) / 3600.0
        for i in range(n_runs * n_ens)
    ]
    hours_post_arrival_time_err = [
        (post_arrival_times_flat[i] - seconds_real_arr_from_base) / 3600.0
        for i in range(n_runs * n_ens)
    ]
    #print(hours_prior_arrival_time_err)

    hours_mean_prior_arrival_time_err = [
        (mean_prior_arrival_times_flat[i] - seconds_real_arr_from_base) / 3600.0
        for i in range(n_runs)
    ]
    hours_mean_post_arrival_time_err = [
        (mean_post_arrival_times_flat[i] - seconds_real_arr_from_base) / 3600.0
        for i in range(n_runs)
    ]

    hours_median_prior_arrival_time_err = [
        (median_prior_arrival_times_flat[i] - seconds_real_arr_from_base) / 3600.0
        for i in range(n_runs)
    ]
    hours_median_post_arrival_time_err = [
        (median_post_arrival_times_flat[i] - seconds_real_arr_from_base) / 3600.0
        for i in range(n_runs)
    ]
    max_y_axis = 1000
    fig, ax = plt.subplots(figsize=(16, 12), nrows=2, ncols=1, sharex=True, layout="constrained")
    ax[0].hist(hours_prior_arrival_time_err, bins=np.arange(-20, 10.25, 0.5))
    ax[0].set_title(f"Prior CME Arrival Error", fontsize=24)
    ax[0].set_ylim([0, max_y_axis])
    ax[0].tick_params(axis='both', which='major', labelsize=16)
    ax[0].set_ylabel("No. of particles", fontsize=20)
    ax[0].plot([0, 0], [0, max_y_axis], color='red', linestyle='dashed')

    ax[1].hist(hours_post_arrival_time_err, bins=np.arange(-20, 10.25, 0.5))
    ax[1].set_title(f"Posterior CME Arrival Error", fontsize=24)
    ax[1].set_ylim([0, max_y_axis])
    ax[1].set_ylabel("No. of particles", fontsize=20)
    ax[1].tick_params(axis='both', which='major', labelsize=16)
    # if len(weights_post_plot) > 0:
    #    ax[2].hist(post_arrival_error_hours, weights=weights_post_plot)
    # ax[2].set_title(f"Weighted posterior CME Arrival Error at {cme_hit_object}")
    # ax[2].set_ylim([0, 1])
    ax[-1].set_xlabel("Arrival time error (hours)", fontsize=20)
    ax[-1].set_xlim([-5, 15])
    ax[1].plot([0, 0], [0, max_y_axis], color='red', linestyle='dashed')
    plt.show()

    max_y_axis = 20
    fig, ax = plt.subplots(figsize=(16, 12), nrows=2, ncols=1, sharex=True, layout="constrained")
    ax[0].hist(hours_mean_prior_arrival_time_err, bins=np.arange(-20, 10.25, 0.5))
    ax[0].set_title(f"Mean Prior CME Arrival Error", fontsize=24)
    ax[0].set_ylim([0, max_y_axis])
    ax[0].tick_params(axis='both', which='major', labelsize=16)
    ax[0].set_ylabel("No. of particles", fontsize=20)
    ax[0].plot([0, 0], [0, max_y_axis], color='red', linestyle='dashed')

    ax[1].hist(hours_mean_post_arrival_time_err, bins=np.arange(-20, 10.25, 0.5))
    ax[1].set_title(f"Mean Posterior CME Arrival Error", fontsize=24)
    ax[1].set_ylim([0, max_y_axis])
    ax[1].set_ylabel("No. of particles", fontsize=20)
    ax[1].tick_params(axis='both', which='major', labelsize=16)
    # if len(weights_post_plot) > 0:
    #    ax[2].hist(post_arrival_error_hours, weights=weights_post_plot)
    # ax[2].set_title(f"Weighted posterior CME Arrival Error at {cme_hit_object}")
    # ax[2].set_ylim([0, 1])
    ax[-1].set_xlabel("Arrival time error (hours)", fontsize=20)
    ax[-1].set_xlim([-5, 15])
    ax[1].plot([0, 0], [0, max_y_axis], color='red', linestyle='dashed')
    plt.show()

    max_y_axis = 20
    fig, ax = plt.subplots(figsize=(16, 12), nrows=2, ncols=1, sharex=True, layout="constrained")
    ax[0].hist(hours_median_prior_arrival_time_err, bins=np.arange(-20, 10.25, 0.5))
    ax[0].set_title(f"Median Prior CME Arrival Error", fontsize=24)
    ax[0].set_ylim([0, max_y_axis])
    ax[0].tick_params(axis='both', which='major', labelsize=16)
    ax[0].set_ylabel("No. of particles", fontsize=20)
    ax[0].plot([0, 0], [0, max_y_axis], color='red', linestyle='dashed')

    ax[1].hist(hours_median_post_arrival_time_err, bins=np.arange(-20, 10.25, 0.5))
    ax[1].set_title(f"Median Posterior CME Arrival Error", fontsize=24)
    ax[1].set_ylim([0, max_y_axis])
    ax[1].set_ylabel("No. of particles", fontsize=20)
    ax[1].tick_params(axis='both', which='major', labelsize=16)
    # if len(weights_post_plot) > 0:
    #    ax[2].hist(post_arrival_error_hours, weights=weights_post_plot)
    # ax[2].set_title(f"Weighted posterior CME Arrival Error at {cme_hit_object}")
    # ax[2].set_ylim([0, 1])
    ax[-1].set_xlabel("Arrival time error (hours)", fontsize=20)
    ax[-1].set_xlim([-5, 15])
    ax[1].plot([0, 0], [0, max_y_axis], color='red', linestyle='dashed')
    plt.show()

    # prior_arrival_speed_flat = np.array(dict1["prior_cme_arrival_s_from_20000101"]).flatten()
    # post_arrival_speed_flat = np.array(dict1["post_cme_arrival_s_from_20000101"]).flatten()

    real_speed = 519.3605577633269
    mean_prior_arrival_speed_flat = np.array(dict1["mean_prior_speed"]).flatten()
    mean_post_arrival_speed_flat = np.array(dict1["mean_post_speed"]).flatten()

    median_prior_arrival_speed_flat = np.array(dict1["median_prior_speed"]).flatten()
    median_post_arrival_speed_flat = np.array(dict1["median_post_speed"]).flatten()

    max_y_axis = 20
    fig, ax = plt.subplots(figsize=(16, 12), nrows=2, ncols=1, sharex=True, layout="constrained")
    ax[0].hist(mean_prior_arrival_speed_flat - real_speed, bins=np.arange(-50, 50, 5))
    ax[0].set_title(f"Mean Prior CME Arrival Speed Error", fontsize=24)
    ax[0].set_ylim([0, max_y_axis])
    ax[0].tick_params(axis='both', which='major', labelsize=16)
    ax[0].set_ylabel("No. of particles", fontsize=20)
    ax[0].plot([0, 0], [0, max_y_axis], color='red', linestyle='dashed')

    ax[1].hist(mean_post_arrival_speed_flat - real_speed, bins=np.arange(-50, 50, 5))
    ax[1].set_title(f"Mean Posterior CME Arrival Speed Error", fontsize=24)
    ax[1].set_ylim([0, max_y_axis])
    ax[1].set_ylabel("No. of particles", fontsize=20)
    ax[1].tick_params(axis='both', which='major', labelsize=16)
    # if len(weights_post_plot) > 0:
    #    ax[2].hist(post_arrival_error_hours, weights=weights_post_plot)
    # ax[2].set_title(f"Weighted posterior CME Arrival Error at {cme_hit_object}")
    # ax[2].set_ylim([0, 1])
    ax[-1].set_xlabel("Arrival time error (hours)", fontsize=20)
    #ax[-1].set_xlim([-5, 15])
    ax[1].plot([0, 0], [0, max_y_axis], color='red', linestyle='dashed')
    plt.show()
    return None


if __name__ == "__main__":
    main()