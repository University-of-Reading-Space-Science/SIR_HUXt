import glob
import os
import sys
import json

#from EmuHUXt.SURF.surf.notebooks.SURF_examples import cr_lon_init
# from IPython.core.pylabtools import figsize
# from SURF.surf.notebooks.SURF_examples import vr_longs
from astropy.time import Time
import astropy.units as u
import h5py
import matplotlib.pyplot as plt
import datetime
import matplotlib as mpl
import numpy as np
import numpy.typing as npt

import huxt.huxt as H
import surf.surf as S
import xarray as xr
from sir_observations import Observations
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

#hv.extension('matplotlib')

from dotenv import load_dotenv
from pathlib import Path

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

def weighted_median(values, weights, axis=0):
    """
    Compute the weighted median along a specified axis.

    Parameters
    ----------
    values : ndarray
        Input data.
    weights : ndarray
        Weights, same shape as values.
    axis : int, optional
        Axis along which to compute the weighted median.

    Returns
    -------
    ndarray
        Weighted median with the specified axis removed.
    """
    values = np.asarray(values)
    weights = np.asarray(weights)

    if values.shape != weights.shape:
        raise ValueError("values and weights must have the same shape")

    if np.any(weights < 0):
        raise ValueError("weights must be non-negative")

    # Sort values and corresponding weights along the axis
    sort_idx = np.argsort(values, axis=axis)
    sorted_values = np.take_along_axis(values, sort_idx, axis=axis)
    sorted_weights = np.take_along_axis(weights, sort_idx, axis=axis)

    # Cumulative weights
    cum_weights = np.cumsum(sorted_weights, axis=axis)

    # Half total weight
    cutoff = np.sum(sorted_weights, axis=axis, keepdims=True) / 2

    # First index where cumulative weight exceeds cutoff
    median_idx = np.argmax(cum_weights >= cutoff, axis=axis)

    # Extract weighted median values
    result = np.take_along_axis(
        sorted_values,
        np.expand_dims(median_idx, axis=axis),
        axis=axis,
    )

    result = np.squeeze(result, axis=axis)
    if np.ndim(result) == 0:
        result = float(result)

    return result


def plot_par_values_over_single_run(
        ds,
        xVarName,
        yVarName,
        xLabel,
        fig_title,
        parNameLegend,
        yTruth=None,
        obs_times=None,
        tick_cadence=None,
        craft=None,
        runNo=0,
        parUnits='',
        n_obs=24,
        base_file_path=None
):
    if (obs_times is not None) and (tick_cadence is not None):
        assert tick_cadence > 0

    fig, ax = plt.subplots(
        figsize=(10, 6), nrows=1, ncols=1, layout="constrained"
    )
    xPlot = ds[xVarName][:].values

    yTrue = yTruth
    try:
        ds[yVarName][runNo, :, :].values
        ds["log_weight"][runNo, :, :].values
    except:
        yPlot = ds[yVarName][:, :].values
        weights = np.exp(ds["log_weight"][:, :].values)
    else:
        yPlot = ds[yVarName][runNo, :, :].values
        weights = np.exp(ds["log_weight"][runNo, :, :].values)

    if (obs_times is not None) and (tick_cadence is not None):
        if len(obs_times) > tick_cadence:
            ind_req = range(0, len(obs_times), tick_cadence)
            x_ticks = [obs_times[i].strftime("%Y/%m/%d %H:%m") for i in ind_req]
        elif len(obs_times) > 1:
            ind_req = [0, -1]
            x_ticks = [obs_times[i].strftime("%Y/%m/%d %H:%m") for i in ind_req]
        else:
            ind_req = [0]
            x_ticks = [obs_times[i].strftime("%Y/%m/%d %H:%m") for i in ind_req]

    y_weighted_median = weighted_median(yPlot, weights=weights, axis=1)
    y_median = np.median(yPlot, axis=1)

    y_mean = np.mean(yPlot, axis=1)
    y_weighted_mean = np.average(yPlot, weights=weights, axis=1)

    nEns = len(ds["ens_no"].values)
    for ir in range(nEns):
        if ir == 0:
            ax.plot(xPlot, yPlot[:, ir], color='pink', alpha=0.5, label=f"{parNameLegend}")
        else:
            ax.plot(xPlot, yPlot[:, ir], color='pink', alpha=0.5)
    ax.plot(xPlot, y_median, color='red', label=f"Median {parNameLegend}")
    #ax.plot(xPlot, y_weighted_median, color="red", label=f"Weighted Median {parNameLegend}")
    ax.plot(xPlot, y_mean, color='blue', label=f"Mean {parNameLegend}")
    #ax.plot(xPlot, y_weighted_mean, color='blue', label=f"Weighted Mean {parNameLegend}")
    if yTruth is not None:
        ax.plot(
            [xPlot.min(), xPlot.max()], [yTrue, yTrue],
            color='k', linestyle='dashed', label=f"True {parNameLegend}"
        )
    ax.legend(fontsize=16)

    ax.set_title(f"{fig_title}", fontsize=24)
    ax.set_xlim(0, n_obs)
    ax.set_xlabel(f"{xLabel}", fontsize=20)
    ax.set_ylabel(f"{parNameLegend} ({parUnits})", fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=16)
    if (obs_times is not None) and (tick_cadence is not None):
        ax.set_xticks(ind_req)
        ax.set_xticklabels(x_ticks, rotation=30)

    if base_file_path is None:
        plt.show()
    else:
        if craft is None:
            filePath = os.path.join(
                base_file_path, f"run_{runNo:03d}", f"{parNameLegend.replace(' ', '_')}_vs_obs_no.png"
            )
        else:
            filePath = os.path.join(
                base_file_path, f"run_{runNo:03d}",
                f"{parNameLegend.replace(' ', '_')}_vs_obs_no_{craft}_assim.png"
            )
        plt.savefig(filePath)

    return None

def plot_par_values_over_mult_runs(
    ds, xVarName, yVarName, yTruth, xLabel, parNameLegend, parUnits=''
):
    fig, ax = plt.subplots(1, 1)
    xPlot = ds[xVarName][:].values

    yTrue = yTruth
    yPlot = np.median(ds[yVarName], axis=2)#.values

    nRuns = len(ds["run"].values)
    for ir in range(nRuns):
        if ir == 0:
            ax.plot(xPlot, yPlot[ir, :], color='pink', alpha=0.5, label=f"{parNameLegend}")
        else:
            ax.plot(xPlot, yPlot[ir, :], color='pink', alpha=0.5)
    ax.plot(xPlot, np.median(yPlot, axis=0), color='red', label=f"Median {parNameLegend} over model runs")
    ax.plot([xPlot.min(), xPlot.max()], [yTrue, yTrue], color='k', linestyle='dashed', label=f"True {parNameLegend}")
    ax.legend()
    ax.set_xlabel(f"{xLabel}")
    ax.set_ylabel(f"{parNameLegend} ({parUnits})")
    plt.show()

    return None

def get_prior_post_cme_pars_from_ds(ds, runNo=0):
    """
    Function to retrieve prior and posterior cme parameters: [t_init, v, width, lon, lat, thick]
      from dataset (and the posterior logarithm of the weights)
    :param ds: Dataset containing all CME parameters from data assimilation run
    :param runNo: Run index to extract if ds contains multiple runs (previously undefined here)
    :return: prior_dict, Dictionary containing the prior CME parameters extracted
    :return: posterior_dict, Dictionary containing the posterior CME parameters extracted plus the log_weights
    """
    # Get the number of ensemble members from ds
    n_members = len(ds["ens_no"].values)

    # Extract the prior and posterior variables, accounting for scenarios
    #  where multiple runs are being performed
    try:
        prior_t_init_in = ds["t_init"][0, :].values
        prior_v_in = ds["v"][0, :].values
        prior_width_in = ds["width"][0, :].values
        prior_lon_in = ds["lon"][0, :].values
        prior_lat_in = ds["lat"][0, :].values
        prior_thick_in = ds["thick"][0, :].values

        post_t_init_in = ds["t_init"][-1, :].values
        post_v_in = ds["v"][-1, :].values
        post_width_in = ds["width"][-1, :].values
        post_lon_in = ds["lon"][-1, :].values
        post_lat_in = ds["lat"][-1, :].values
        post_thick_in = ds["thick"][-1, :].values

        log_weights_in = ds["log_weight"][-1, :].values
    except:
        prior_t_init_in = ds["t_init"][runNo, 0, :].values
        prior_v_in = ds["v"][runNo, 0, :].values
        prior_width_in = ds["width"][runNo, 0, :].values
        prior_lon_in = ds["lon"][runNo, 0, :].values
        prior_lat_in = ds["lat"][runNo, 0, :].values
        prior_thick_in = ds["thick"][runNo, 0, :].values

        # Posterior is the final observation index (previously index 0, i.e. the prior)
        post_t_init_in = ds["t_init"][runNo, -1, :].values
        post_v_in = ds["v"][runNo, -1, :].values
        post_width_in = ds["width"][runNo, -1, :].values
        post_lon_in = ds["lon"][runNo, -1, :].values
        post_lat_in = ds["lat"][runNo, -1, :].values
        post_thick_in = ds["thick"][runNo, -1, :].values

        log_weights_in = ds["log_weight"][runNo, -1, :].values
    else:
        prior_t_init_in = ds["t_init"][0, :].values
        prior_v_in = ds["v"][0, :].values
        prior_width_in = ds["width"][0, :].values
        prior_lon_in = ds["lon"][0, :].values
        prior_lat_in = ds["lat"][0, :].values
        prior_thick_in = ds["thick"][0, :].values

        post_t_init_in = ds["t_init"][-1, :].values
        post_v_in = ds["v"][-1, :].values
        post_width_in = ds["width"][-1, :].values
        post_lon_in = ds["lon"][-1, :].values
        post_lat_in = ds["lat"][-1, :].values
        post_thick_in = ds["thick"][-1, :].values

        log_weights_in = ds["log_weight"][-1, :].values

    sec_to_cme_prior = [
        float(prior_t_init_in[i]) * u.s for i in range(n_members)
    ]
    v_prior = [prior_v_in[i] * u.km / u.s for i in range(n_members)]
    width_prior = [prior_width_in[i] * u.deg for i in range(n_members)]
    lon_prior = [prior_lon_in[i] * u.deg for i in range(n_members)]
    lat_prior = [prior_lat_in[i] * u.deg for i in range(n_members)]
    thick_prior = [prior_thick_in[i] * u.solRad for i in range(n_members)]

    sec_to_cme_post = [
        float(post_t_init_in[i]) * u.s for i in range(n_members)
    ]
    v_post = [post_v_in[i] * u.km / u.s for i in range(n_members)]
    width_post = [post_width_in[i] * u.deg for i in range(n_members)]
    lon_post = [post_lon_in[i] * u.deg for i in range(n_members)]
    lat_post = [post_lat_in[i] * u.deg for i in range(n_members)]
    thick_post = [post_thick_in[i] * u.solRad for i in range(n_members)]
    log_weights_post = [log_weights_in[i] for i in range(n_members)]
    weights_post = np.array([
        np.exp(log_weights_post[i]) for i in range(n_members)
    ])

    # Make dictionaries for output
    prior_dict_out = {
        "t_init": sec_to_cme_prior,
        "v": v_prior,
        "width": width_prior,
        "lon": lon_prior,
        "lat": lat_prior,
        "thick": thick_prior,
    }

    post_dict_out = {
        "t_init": sec_to_cme_post,
        "v": v_post,
        "width": width_post,
        "lon": lon_post,
        "lat": lat_post,
        "thick": thick_post,
        "log_weights": log_weights_post,
        "weights": weights_post,
    }

    return prior_dict_out, post_dict_out


def get_true_cme_arrival_time(
        ds,
        use_model,
        true_cme_t_init,
        true_cme_speed,
        true_cme_width,
        true_cme_lat,
        true_cme_lon,
        true_cme_thick,
        cme_hit_object="EARTH"
):
    print(ds["model_init_time"].values)
    try:
        start_time = datetime.datetime.strptime(
            str(ds["model_init_time"].values), '%Y-%m-%dT%H:%M:%S.000000000'
        )
        r_min_value = ds["r_min"].values * u.solRad
        v_bound_val = ds["ambient_vr"].values * u.km / u.s
    except:
        start_time = datetime.datetime.strptime(
            str(ds["model_init_time"].values[0]), '%Y-%m-%dT%H:%M:%S.000000000'
        )
        r_min_value = ds["r_min"].values[0] * u.solRad
        v_bound_val = ds["ambient_vr"].values[0] * u.km / u.s
    else:
        start_time = datetime.datetime.strptime(
            str(ds["model_init_time"].values), '%Y-%m-%dT%H:%M:%S.000000000'
        )
        r_min_value = ds["r_min"].values * u.solRad
        v_bound_val = ds["ambient_vr"].values * u.km / u.s

    print(start_time)
    start_time_astro = Time(start_time, format='datetime')
    cr_num: int = np.trunc(sn.carrington_rotation_number(start_time))

    if use_model in ["surf", "compress_surf"]:
        ert = S.Observer("EARTH", start_time_astro)
    elif use_model in ["huxt"]:
        ert = H.Observer("EARTH", start_time_astro)
    else:
        sys.exit("Unknown use_model name, expected either 'surf', 'compress_surf' or 'huxt'")

    lon_start = 300 * u.deg
    lon_end = 420 * u.deg

    # Initialise list
    cme_arrival_prior = []
    cme_arr_prior_speed = []
    cme_arrival_post = []
    cme_arr_post_speed = []

    n_members = len(ds["ens_no"].values)

    if use_model in ["surf"]:
        model: SURF = S.SURF(
            v_boundary=v_bound_val,#ds["ambient_vr"].values * u.km / u.s,
            cr_num=cr_num,
            cr_lon_init=ert.lon_c.to(u.deg),
            latitude=ert.lat.to(u.deg),
            lon_start=lon_start.to(u.rad),
            lon_stop=lon_end.to(u.rad),
            simtime=5 * u.day,
            dt_scale=20,
            r_min=r_min_value,#ds["r_min"].values * u.solRad,
            solver='huxt',
            # accel_limit=accel_limit
        )
    elif use_model in ["compress_surf"]:
        model: SURF = S.SURF(
            v_boundary=v_bound_val,#ds["ambient_vr"].values * u.km / u.s,
            cr_num=cr_num,
            cr_lon_init=ert.lon_c.to(u.deg),
            latitude=ert.lat.to(u.deg),
            lon_start=lon_start.to(u.rad),
            lon_stop=lon_end.to(u.rad),
            simtime=5 * u.day,
            dt_scale=20,
            r_min=r_min_value,#ds["r_min"].values * u.solRad,
            solver='hydro',
            # accel_limit=accel_limit
        )
    elif use_model in ["huxt"]:
        model: HUXt = H.HUXt(
            v_boundary=v_bound_val,#ds["ambient_vr"].values * u.km / u.s,
            cr_num=cr_num,
            cr_lon_init=ert.lon_c.to(u.deg),
            latitude=ert.lat.to(u.deg),
            lon_start=lon_start.to(u.rad),
            lon_stop=lon_end.to(u.rad),
            simtime=5 * u.day,
            dt_scale=20,
            r_min=r_min_value,#ds["r_min"].values * u.solRad,
            # accel_limit=accel_limit
        )
    else:
        sys.exit("Unknown use_model name, expected either 'surf', 'compress_surf' or 'huxt'")

    # Generate CME object
    sec_to_cme_true = float(
        (true_cme_t_init - start_time).total_seconds()
    ) * u.s
    v_true = float(true_cme_speed) * u.km / u.s
    width_true = float(true_cme_width) * u.deg
    lon_true =  float(true_cme_lon) * u.deg
    lat_true =  float(true_cme_lat) * u.deg
    thick_true = float(true_cme_thick) * u.solRad

    if use_model in ["surf", "compress_surf"]:
        cme_object: S.ConeCME = S.ConeCME(
            t_launch=sec_to_cme_true,
            v=v_true,
            width=width_true,
            longitude=lon_true,
            latitude=lat_true,
            thickness=thick_true,
            cme_fixed_duration=True,
            fixed_duration=12 * 3600 * u.s
        )

    elif use_model in ["huxt"]:
        cme_object: H.ConeCME = H.ConeCME(
            t_launch=sec_to_cme_true,
            v=v_true,
            width=width_true,
            longitude=lon_true,
            latitude=lat_true,
            thickness=thick_true,
            cme_fixed_duration=True,
            fixed_duration=12 * 3600 * u.s
        )
    else:
        sys.exit("Unknown use_model name, expected either 'surf', 'compress_surf' or 'huxt'")

    # Make true CME arrival time
    model.solve([cme_object])
    cme = model.cmes[0]
    true_stats = cme.compute_arrival_at_body(cme_hit_object)
    # print(f"prior_stats_{i}={prior_stats}")

    if true_stats['hit']:
        cme_arrival_time = true_stats['t_arrive'].datetime
        cme_arrival_speed = (true_stats["v"].to(u.km / u.s)).value
    else:
        cme_arrival_time = np.nan
        cme_arrival_speed = np.nan

    print(f"True CME arrival time at {cme_hit_object}: {cme_arrival_time}")
    print(f"True CME arrival speed at {cme_hit_object}: {cme_arrival_speed}")

    return cme_arrival_time, cme_arrival_speed


def get_cme_stats_prior_post(
    ens_no, n_members, model, prior_cme, post_cme, cme_hit_object
):
    """
    Function to output CME stats + arrival time and speed, primarily for use with parallel application
    :param ens_no: Current ensemble member
    :param n_members: Total number of members
    :param model: SURF/HUXt model to solve for
    :param prior_cme: prior_cme_object corresponding to ensemble member, ens_no
    :param post_cme: posterior_cme_object corresponding to ensemble member, ens_no
    :param cme_hit_object: Object that CME is projected to hit
    :return:
    """
    if np.mod(ens_no, 1) == 0:
        print(f"{ens_no}/{n_members}")

    # Make prior CME arrival times
    print(f"prior_cme: {prior_cme}")
    model.solve([prior_cme])
    cme = model.cmes[0]
    prior_stats = cme.compute_arrival_at_body(cme_hit_object)
    #print(f"prior_stats_{ens_no}={prior_stats}")

    if prior_stats['hit']:
        cme_arrival_prior = prior_stats['t_arrive'].datetime
        cme_arr_prior_speed = (prior_stats["v"].to(u.km / u.s)).value
        prior_count_hit = 1
    else:
        cme_arrival_prior = np.nan
        cme_arr_prior_speed = np.nan
        prior_count_hit = 0

    # Make posterior CME arrival times
    model.solve([post_cme])
    cme = model.cmes[0]
    post_stats = cme.compute_arrival_at_body(cme_hit_object)
    #print(f"post_stats_{ens_no}={post_stats}")

    if post_stats['hit']:
        cme_arrival_post = post_stats['t_arrive'].datetime
        cme_arr_post_speed = (post_stats["v"].to(u.km / u.s)).value
        post_count_hit = 1
    else:
        cme_arrival_post = np.nan
        cme_arr_post_speed = np.nan
        post_count_hit = 0

    print(
        f"cme_arrival_prior={cme_arrival_prior},\ncme_arr_prior_speed={cme_arr_prior_speed},\n"
        f" prior_count_hit={prior_count_hit},\n prior_stats={prior_stats},\n,"
        f"cme_arrival_post={cme_arrival_post},\n cme_arr_post_speed={cme_arr_post_speed},\n"
        f" post_count_hit={post_count_hit},\n post_stats={post_stats}"
    )
    return (
        cme_arrival_prior, cme_arr_prior_speed, prior_count_hit, prior_stats,
        cme_arrival_post, cme_arr_post_speed, post_count_hit, post_stats
    )


def numpy_nan_mean(
        arr: npt.NDArray | list,
        axis: int=0
) -> float:
    """
    Helper function to ensure that all-nan slices in nanmean are appropriately accounted for
    :param arr: array or list to calculate mean for
    :return: mean_arr: Mean of array or list
    """
    if np.isnan(arr).all():
        mean_arr = np.nan
    else:
        mean_arr = np.nanmean(arr, axis=axis)

    return mean_arr


def numpy_nan_median(
        arr: npt.NDArray | list,
        axis: int=0
) -> float:
    """
    Helper function to ensure that all-nan slices in nanmedian are appropriately accounted for
    :param arr:
    :return: median_arr: Median of array or list
    """
    if np.isnan(arr).all():
        median_arr = np.nan
    else:
        median_arr = np.nanmedian(arr, axis=axis)

    return median_arr


def numpy_nan_std(
        arr: npt.NDArray | list,
        axis: int=0
) -> float:
    """
    Helper function to ensure that all-nan slices in nanstd are appropriately accounted for
    :param arr:
    :return: std_arr: Standard deviation of array or list
    """
    if np.isnan(arr).all():
        std_arr = np.nan
    else:
        std_arr = np.nanstd(arr, axis=axis)

    return std_arr


def get_cme_arrival_time(
        ds,
        use_model,
        runNo=0,
        real_arrival_time=None,
        cme_hit_object="EARTH",
        base_file_path=None,
        parallel_ens=False,
):
    print(ds["model_init_time"].values)
    try:
        start_time = datetime.datetime.strptime(
            str(ds["model_init_time"].values), '%Y-%m-%dT%H:%M:%S.000000000'
        )
        r_min_value = ds["r_min"].values * u.solRad
        v_bound_val = ds["ambient_vr"].values * u.km / u.s
    except:
        start_time = datetime.datetime.strptime(
            str(ds["model_init_time"].values[0]), '%Y-%m-%dT%H:%M:%S.000000000'
        )
        r_min_value = ds["r_min"].values[0] * u.solRad
        v_bound_val = ds["ambient_vr"].values[0] * u.km / u.s
    else:
        start_time = datetime.datetime.strptime(
            str(ds["model_init_time"].values), '%Y-%m-%dT%H:%M:%S.000000000'
        )
        r_min_value = ds["r_min"].values * u.solRad
        v_bound_val = ds["ambient_vr"].values * u.km / u.s

    #start_time = datetime.datetime.strptime(str(ds["model_init_time"].values), '%Y-%m-%dT%H:%M:%S.000000000')
    print(start_time)
    start_time_astro = Time(start_time, format='datetime')
    cr_num: int = np.trunc(sn.carrington_rotation_number(start_time))

    if use_model in ["surf", "compress_surf", "compress_surf3d"]:
        ert = S.Observer("EARTH", start_time_astro)
    elif use_model in ["huxt"]:
        ert = H.Observer("EARTH", start_time_astro)
    else:
        sys.exit("Unknown use_model name, expected either 'surf', 'compress_surf' or 'huxt'")

    lon_start = 300 * u.deg
    lon_end = 420 * u.deg

    n_members = len(ds["ens_no"].values)

    for i in range(n_members):
        print(f"{i}: {ds['v'][:, i].values}")
    sys.exit()

    # Initialise list
    cme_arrival_prior = [None for _ in range(n_members)]
    cme_arr_prior_speed = [None for _ in range(n_members)]
    cme_arrival_post = [None for _ in range(n_members)]
    cme_arr_post_speed = [None for _ in range(n_members)]

    if use_model in ["huxt", "surf", "compress_surf"]:
        try:
            v_bound_in = ds["ambient_vr"].values[runNo, :, :] * u.km / u.s
            r_min_in = ds["r_min"].values[runNo, :, :] * u.solRad
        except:
            v_bound_in = ds["ambient_vr"].values * u.km / u.s
            r_min_in = ds["r_min"].values * u.solRad
        else:
            v_bound_in = ds["ambient_vr"].values[runNo, :, :] * u.km / u.s
            r_min_in = ds["r_min"].values[runNo, :, :] * u.solRad
    elif use_model in ["compress_surf3d"]:
        try:
            v_bound_in = ds["ambient_vr"].values[runNo, :, :] * u.km / u.s
            vr_longs_in = ds["vr_longs"].values[runNo, :, :] * u.deg
            vr_lats_in = ds["vr_lats"].values[runNo, :, :] * u.deg
            r_min_in = ds["r_min"].values[runNo, :, :] * u.solRad
        except:
            v_bound_in = ds["ambient_vr"].values * u.km / u.s
            vr_longs_in = ds["vr_longs"].values * u.deg
            vr_lats_in = ds["vr_lats"].values * u.deg
            r_min_in = ds["r_min"].values * u.solRad
        else:
            v_bound_in = ds["ambient_vr"].values[runNo, :, :] * u.km / u.s
            vr_longs_in = ds["vr_longs"].values[runNo, :, :] * u.deg
            vr_lats_in = ds["vr_lats"].values[runNo, :, :] * u.deg
            r_min_in = ds["r_min"].values[runNo, :, :] * u.solRad

    if use_model in ["surf"]:
        model: SURF = S.SURF(
            v_boundary=v_bound_in,
            cr_num=cr_num,
            cr_lon_init=ert.lon_c.to(u.deg),
            latitude=ert.lat.to(u.deg),
            lon_start=lon_start.to(u.rad),
            lon_stop=lon_end.to(u.rad),
            simtime=5 * u.day,
            dt_scale=20,
            r_min=r_min_in,
            solver='huxt',
            # accel_limit=accel_limit
        )
    elif use_model in ["compress_surf"]:
        model: SURF = S.SURF(
            v_boundary=v_bound_in,
            cr_num=cr_num,
            cr_lon_init=ert.lon_c.to(u.deg),
            latitude=ert.lat.to(u.deg),
            lon_start=lon_start.to(u.rad),
            lon_stop=lon_end.to(u.rad),
            simtime=3 * u.day,
            dt_scale=20,
            r_min=r_min_in,
            solver='hydro',
            # accel_limit=accel_limit
        )
    elif use_model in ["compress_surf3d"]:
        # TODO: Need to remove hard-coded numbers here
        model: SURF3d = S.SURF3d(
            v_map=v_bound_in,
            v_map_long=vr_longs_in,
            v_map_lat=vr_lats_in,
            cr_num=cr_num,
            cr_lon_init=ert.lon_c.to(u.deg),
            latitude_min=-15 * u.deg,
            latitude_max=15 * u.deg,
            lon_start=lon_start.to(u.rad),
            lon_stop=lon_end.to(u.rad),
            simtime=3 * u.day,
            dt_scale=20,
            r_min=r_min_in,
            solver='hydro',
        )
    elif use_model in ["huxt"]:
        model: HUXt = H.HUXt(
            v_boundary=v_bound_in,
            cr_num=cr_num,
            cr_lon_init=ert.lon_c.to(u.deg),
            latitude=ert.lat.to(u.deg),
            lon_start=lon_start.to(u.rad),
            lon_stop=lon_end.to(u.rad),
            simtime=3 * u.day,
            dt_scale=20,
            r_min=r_min_in,
            # accel_limit=accel_limit
        )
    else:
        sys.exit("Unknown use_model name, expected either 'surf', 'compress_surf', 'compress_surf3d' or 'huxt'")

    # Generate CME object
    # print(f"cme_launch_time = {cme_launch_time}")

    # Extract the CME parameters from the xr.Dataset
    prior_dict, post_dict = get_prior_post_cme_pars_from_ds(ds, runNo=runNo)
    sec_to_cme_prior = np.array(prior_dict["t_init"]) * u.s
    v_prior = np.array(prior_dict["v"])
    width_prior = np.array(prior_dict["width"])
    lon_prior = np.array(prior_dict["lon"])
    lat_prior = np.array(prior_dict["lat"])
    thick_prior = np.array(prior_dict["thick"])

    sec_to_cme_post = post_dict["t_init"]
    v_post = post_dict["v"]
    width_post = post_dict["width"]
    lon_post = post_dict["lon"]
    lat_post = post_dict["lat"]
    thick_post = post_dict["thick"]
    log_weights_post = post_dict["log_weights"]
    weights_post = post_dict["weights"]

    print(f"t_init: {np.shape(sec_to_cme_prior.value)}, {sec_to_cme_prior}")
    print(f"v_prior: {np.shape(v_prior.value)}, {v_prior}")
    print(f"width_prior: {np.shape(width_prior.value)}, {width_prior}")
    print(f"lon_prior: {np.shape(lon_prior)}, {lon_prior}")
    print(f"lat_prior: {np.shape(lat_prior)}, {lat_prior}")
    print(f"thick_prior: {np.shape(thick_prior)}, {thick_prior}")

    print(f"v_post: {v_post}")
    print(f"width_post: {width_post}")

    if use_model in ["surf", "compress_surf", "compress_surf3d"]:
        prior_cme_objects: list[S.ConeCME] = [
            S.ConeCME(
                t_launch=sec_to_cme_prior[i],
                v=v_prior[i],
                width=width_prior[i],
                longitude=lon_prior[i],
                latitude=lat_prior[i],
                thickness=thick_prior[i],
                cme_fixed_duration=True,
                fixed_duration=12 * 3600 * u.s
            ) for i in range(n_members)
        ]

        post_cme_objects: list[S.ConeCME] = [
            S.ConeCME(
                t_launch=sec_to_cme_post[i],
                v=v_post[i],
                width=width_post[i],
                longitude=lon_post[i],
                latitude=lat_post[i],
                thickness=thick_post[i],
                cme_fixed_duration=True,
                fixed_duration=12 * 3600 * u.s
            ) for i in range(n_members)
        ]
    elif use_model in ["huxt"]:
        prior_cme_objects: list[H.ConeCME] = [
            H.ConeCME(
                t_launch=sec_to_cme_prior[i],
                v=v_prior[i],
                width=width_prior[i],
                longitude=lon_prior[i],
                latitude=lat_prior[i],
                thickness=thick_prior[i],
                cme_fixed_duration=True,
                fixed_duration=12 * 3600 * u.s
            ) for i in range(n_members)
        ]

        post_cme_objects: list[H.ConeCME] = [
            H.ConeCME(
                t_launch=sec_to_cme_post[i],
                v=v_post[i],
                width=width_post[i],
                longitude=lon_post[i],
                latitude=lat_post[i],
                thickness=thick_post[i],
                cme_fixed_duration=True,
                fixed_duration=12 * 3600 * u.s
            ) for i in range(n_members)
        ]
    else:
        sys.exit("Unknown use_model name, expected either 'surf', 'compress_surf' or 'huxt'")

    prior_count_hit_list = np.zeros(n_members)
    post_count_hit_list = np.zeros(n_members)
    prior_stat_dict = {}
    post_stat_dict = {}

    if parallel_ens:
        def log_result(result):
            # This is called whenever self.get_cme_flank_single_ens_member(i)
            #  returns a result in parallel computation.
            # result_list is modified only by the main process, not the pool workers.
            result_list.append(result)

        print("Use parallel execution on ensemble.")

        # Initialise variables required for parallel computations
        result_list = []
        obj_result = [None] * n_members

        n_proc = int(os.cpu_count() - 1)
        if n_proc < 1:
            n_proc = 1
        pool = Pool(processes=n_proc)

        for i in range(n_members):
            obj_result[i] = pool.apply_async(
                get_cme_stats_prior_post,
                kwds={
                    'ens_no': i,
                    'n_members': n_members,
                    'model': model,
                    'prior_cme': prior_cme_objects[i],
                    'post_cme': post_cme_objects[i],
                    'cme_hit_object': cme_hit_object,
                },
                callback=log_result,
            )

        pool.close()
        pool.join()

        for i in range(n_members):
            obs_tuple = obj_result[i].get()

            cme_arrival_prior[i] = obs_tuple[0]
            cme_arr_prior_speed[i] = obs_tuple[1]
            prior_count_hit_list[i] = obs_tuple[2]
            prior_stat_dict[f"{i}"] = obs_tuple[3]

            cme_arrival_post[i] = obs_tuple[4]
            cme_arr_post_speed[i] = obs_tuple[5]
            post_count_hit_list[i] = obs_tuple[6]
            post_stat_dict[f"{i}"] = obs_tuple[7]
            """(
                cme_arrival_prior, cme_arr_prior_speed, prior_count_hit, prior_stats,
                cme_arrival_post, cme_arr_post_speed, post_count_hit, post_stats
            )"""

        prior_count_hit = np.sum(prior_count_hit_list)
        post_count_hit = np.sum(post_count_hit_list)

    else:
        print("Use serial execution on ensemble.")


        #prior_count_hit = 0
        #post_count_hit = 0
        for i in range(n_members):
            obs_tuple = get_cme_stats_prior_post(
                ens_no=i,
                n_members=n_members,
                model=model,
                prior_cme=prior_cme_objects[i],
                post_cme=post_cme_objects[i],
                cme_hit_object=cme_hit_object,
            )

            cme_arrival_prior[i] = obs_tuple[0]
            cme_arr_prior_speed[i] = obs_tuple[1]
            prior_count_hit_list[i] = obs_tuple[2]
            prior_stat_dict[f"{i}"] = obs_tuple[3]

            cme_arrival_post[i] = obs_tuple[4]
            cme_arr_post_speed[i] = obs_tuple[5]
            post_count_hit_list[i] = obs_tuple[6]
            post_stat_dict[f"{i}"] = obs_tuple[7]

        #     # Make prior CME arrival times
        #     model.solve([prior_cme_objects[i]])
        #     cme = model.cmes[0]
        #     prior_stats = cme.compute_arrival_at_body(cme_hit_object)
        #     #print(f"prior_stats_{i}={prior_stats}")
        #
        #     prior_stat_dict[f"{i}"] = prior_stats
        #
        #     if prior_stats['hit']:
        #         cme_arrival_prior.append(prior_stats['t_arrive'].datetime)
        #         cme_arr_prior_speed.append((prior_stats["v"].to(u.km / u.s)).value)
        #         prior_count_hit = prior_count_hit + 1
        #     else:
        #         cme_arrival_prior.append(np.nan)
        #         cme_arr_prior_speed.append(np.nan)
        #
        #     # Make posterior CME arrival times
        #     model.solve([post_cme_objects[i]])
        #     cme = model.cmes[0]
        #     post_stats = cme.compute_arrival_at_body(cme_hit_object)
        #     post_stat_dict[f"{i}"] = post_stats
        #
        #     #print(f"post_stats_{i} = {stats}")
        #     if post_stats['hit']:
        #         cme_arrival_post.append(post_stats['t_arrive'].datetime)
        #         cme_arr_post_speed.append((post_stats["v"].to(u.km / u.s)).value)
        #
        #         post_count_hit = post_count_hit + 1
        #     else:
        #         cme_arrival_post.append(np.nan)
        #         cme_arr_post_speed.append(np.nan)
        prior_count_hit = np.sum(prior_count_hit_list)
        post_count_hit = np.sum(post_count_hit_list)

    prior_hit_rate = prior_count_hit / n_members
    post_hit_rate = post_count_hit / n_members

    print(f"Cme_arrival_prior = {cme_arrival_prior}")
    print(f"Cme_arr_prior_speed = {cme_arr_prior_speed}")
    print(f"cme_arrival_post = {cme_arrival_post}")
    print(f"cme_arr_post_speed = {cme_arr_post_speed}")

    base_time = datetime.datetime(2000, 1, 1, 0, 0, 0)
    seconds_arrival_prior = np.array([
        (cme_arrival_prior[i] - base_time).total_seconds()
        if isinstance(cme_arrival_prior[i], datetime.datetime) else np.nan
        for i in range(n_members)
    ])
    seconds_arrival_post = np.array([
        (cme_arrival_post[i] - base_time).total_seconds()
        if isinstance(cme_arrival_post[i], datetime.datetime) else np.nan
        for i in range(n_members)
    ])
    cme_arr_prior_speed = np.array(cme_arr_prior_speed)
    cme_arr_post_speed = np.array(cme_arr_post_speed)

    print(f"prior_CME_hit_rate = {prior_hit_rate * 100}%")
    if prior_hit_rate > 1e-7:
        mean_prior_arr_sec = np.nanmean(seconds_arrival_prior)
        mean_prior_arr_speed = np.nanmean(cme_arr_prior_speed)

        median_prior_arr_sec = np.nanmedian(seconds_arrival_prior)
        median_prior_arr_speed = np.nanmedian(cme_arr_prior_speed)

        sd_prior_arr_sec = np.nanstd(seconds_arrival_prior)
        sd_prior_arr_hr = sd_prior_arr_sec / 3600.0

        mean_prior_arrival = base_time + datetime.timedelta(
            seconds=mean_prior_arr_sec
        )
        median_prior_arrival = base_time + datetime.timedelta(
            seconds=median_prior_arr_sec
        )
        sd_prior_arr_speed = np.nanstd(cme_arr_prior_speed)

        print(f"\nMean_prior_arrival = {mean_prior_arrival} +/- {sd_prior_arr_hr} hours")
        print(f"Mean_prior_speed = {mean_prior_arr_speed} +/- {sd_prior_arr_speed} km/s")
        print(f"\nMedian_prior_arrival = {median_prior_arrival}")
        print(f"Median_prior_speed = {median_prior_arr_speed} km/s")
    else:
        print(f"Prior CME not detected at {cme_hit_object}")
        mean_prior_arr_sec = np.nan
        sd_prior_arr_sec = np.nan
        mean_prior_arr_speed = np.nan
        sd_prior_arr_speed = np.nan
        median_prior_arr_sec = np.nan
        median_prior_arr_speed = np.nan

    print(f"\nPost CME_hit_rate = {post_hit_rate * 100}%")
    if post_hit_rate > 1e-7:
        ind_req = ~np.isnan(seconds_arrival_post)
        renorm_weights_post = weights_post[ind_req] / np.sum(weights_post[ind_req])

        ################################################################
        # Calculate unweighted statistics
        mean_post_arr_sec = np.nanmean(seconds_arrival_post)
        median_post_arr_sec = np.nanmedian(seconds_arrival_post)
        sd_post_arr_sec = np.nanstd(seconds_arrival_post)
        sd_post_arr_hr = sd_post_arr_sec / 3600.0

        mean_post_arrival = base_time + datetime.timedelta(
            seconds=mean_post_arr_sec
        )
        median_post_arrival = base_time + datetime.timedelta(
            seconds=median_post_arr_sec
        )
        mean_post_arr_speed = np.nanmean(cme_arr_post_speed)
        median_post_arr_speed = np.nanmedian(cme_arr_post_speed)
        sd_post_arr_speed = np.nanstd(cme_arr_post_speed)
        #################################################################

        print(f"\nMean_post_arrival = {mean_post_arrival} +/- {sd_post_arr_hr} hours")
        print(f"Mean_post_speed = {mean_post_arr_speed} +/- {sd_post_arr_speed} km/s")
        print(f"\nMedian_post_arrival_time = {median_post_arrival}")
        print(f"Median_post_speed = {median_post_arr_speed}")

        if np.sum(renorm_weights_post) != 0:
            #################################################################
            # Calculate weighted statistics
            weighted_mean_post_arr_sec = np.average(seconds_arrival_post[ind_req], weights=renorm_weights_post)
            weighted_mean_post_arrival = base_time + datetime.timedelta(
                seconds=weighted_mean_post_arr_sec
            )

            weighted_median_post_arr_sec = weighted_median(
                seconds_arrival_post[ind_req], weights=renorm_weights_post
            )

            weighted_median_post_arrival = base_time + datetime.timedelta(
                seconds=weighted_median_post_arr_sec
            )

            weighted_mean_post_arr_speed = np.average(cme_arr_post_speed[ind_req], weights=renorm_weights_post)
            weighted_median_post_arr_speed = weighted_median(
                cme_arr_post_speed[ind_req], weights=renorm_weights_post
            )
            # Calculate (biased) weighted posterior variance
            weighted_var_post_arr_sec_biased = np.average(
                (seconds_arrival_post[ind_req] - weighted_mean_post_arr_sec) ** 2,
                weights=renorm_weights_post
            )
            weighted_var_post_arr_speed_biased = np.average(
                (cme_arr_post_speed[ind_req] - weighted_mean_post_arr_speed) ** 2,
                weights=renorm_weights_post
            )

            # Transform to unbiased variance estimator
            n_non_nan_samples = np.sum(ind_req)
            b_factor = n_non_nan_samples / (n_non_nan_samples - 1)
            weighted_var_post_arr_sec_unbiased = b_factor * weighted_var_post_arr_sec_biased
            weighted_var_post_arr_speed_unbiased = b_factor * weighted_var_post_arr_speed_biased

            # Calculate weighted standard deviation
            weighted_sd_post_arr_sec = np.sqrt(weighted_var_post_arr_sec_unbiased)
            weighted_sd_post_arr_hr = weighted_sd_post_arr_sec / 3600.0
            weighted_sd_post_arr_speed = np.sqrt(weighted_var_post_arr_speed_unbiased)
            ##################################################################

            print(f"\nWeighted_mean_post_arrival = {weighted_mean_post_arrival} +/- {weighted_sd_post_arr_hr} hours")
            print(f"Weighted_mean_post_speed = {weighted_mean_post_arr_speed} +/- {weighted_sd_post_arr_speed} km/s")
            print(f"\nWeighted_median_arrival_time = {weighted_median_post_arrival}")
            print(f"Weighted_median_post_speed = {weighted_median_post_arr_speed}")
        else:
            weighted_mean_post_arr_sec = np.nan
            weighted_median_post_arr_sec = np.nan
            weighted_sd_post_arr_sec = np.nan
            weighted_mean_post_arr_speed = np.nan
            weighted_median_post_arr_speed = np.nan
            weighted_sd_post_arr_speed = np.nan

    else:
        print(f"Posterior CME not detected at {cme_hit_object}")
        weighted_mean_post_arr_sec = np.nan
        weighted_median_post_arr_sec = np.nan
        weighted_sd_post_arr_sec = np.nan
        weighted_mean_post_arr_speed = np.nan
        weighted_median_post_arr_speed = np.nan
        weighted_sd_post_arr_speed = np.nan

        mean_post_arr_sec = np.nan
        median_post_arr_sec = np.nan
        sd_post_arr_sec = np.nan
        mean_post_arr_speed = np.nan
        median_post_arr_speed = np.nan
        sd_post_arr_speed = np.nan

        renorm_weights_post = np.nan


    if real_arrival_time is not None:
        prior_arrival_error_hours = [
            ((pri_arr - real_arrival_time).total_seconds()) / 3600.0
            for pri_arr in cme_arrival_prior if isinstance(pri_arr, datetime.datetime)
        ]
        post_arrival_error_hours = [
            ((post_arr - real_arrival_time).total_seconds()) / 3600.0
            for post_arr in cme_arrival_post if isinstance(post_arr, datetime.datetime)
        ]

        if len(prior_arrival_error_hours) > 0:
            mean_prior_arrival_error_hours = np.nanmean(prior_arrival_error_hours)
            median_prior_arrival_error_hours = np.nanmedian(prior_arrival_error_hours)
        else:
            mean_prior_arrival_error_hours = np.nan
            median_prior_arrival_error_hours = np.nan

        if len(post_arrival_error_hours) > 0:
            mean_post_arrival_error_hours = np.nanmean(post_arrival_error_hours)
            median_post_arrival_error_hours = np.nanmedian(post_arrival_error_hours)
        else:
            mean_post_arrival_error_hours = np.nan
            median_post_arrival_error_hours = np.nan

        weights_post_plot = [
            w
            for i, w in enumerate(weights_post)
            if isinstance(cme_arrival_post[i], datetime.datetime)
        ]

        if len(weights_post_plot) > 0:
            weights_post_plot = weights_post_plot / np.sum(weights_post_plot)
            print(np.max(weights_post_plot))
            print(f"weights_post_plot - renorm_weights_post = {weights_post_plot-renorm_weights_post}")

            weighted_mean_post_arrival_error_hours = np.average(
                post_arrival_error_hours, weights=weights_post_plot
            )
            weighted_median_post_arrival_error_hours = weighted_median(
                post_arrival_error_hours, weights=weights_post_plot
            )
        else:
            weighted_mean_post_arrival_error_hours = np.nan
            weighted_median_post_arrival_error_hours = np.nan

        print(f"\n Observed CME arrival time at {cme_hit_object}: {real_arrival_time}")
        print(f"\nPrior mean_arrival_time_error = {mean_prior_arrival_error_hours} hours")
        print(f"Post mean_arrival_time_error = {mean_post_arrival_error_hours} hours")
        print(f"\nPrior Median_arrival_time_error = {median_prior_arrival_error_hours} hours")
        print(f"Post Median_arrival_time_error = {median_post_arrival_error_hours} hours")
        print(f"\nWeighted Mean_arrival_time_error = {weighted_mean_post_arrival_error_hours} hours")
        print(f"Weighted Median_arrival_time_error = {weighted_median_post_arrival_error_hours} hours")

        max_y_axis = 10 * np.ceil(n_members / 40)
        fig, ax = plt.subplots(figsize=(16, 12), nrows=2, ncols=1, sharex=True, layout="constrained")
        ax[0].hist(prior_arrival_error_hours, bins=np.arange(-10, 20.5, 1))
        ax[0].set_title(f"Prior CME Arrival Error at {cme_hit_object}", fontsize=24)
        ax[0].set_ylim([0, max_y_axis])
        ax[0].tick_params(axis='both', which='major', labelsize=16)
        ax[0].set_ylabel("No. of particles", fontsize=20)

        ax[1].hist(post_arrival_error_hours, bins=np.arange(-10, 20.5, 1))
        ax[1].set_title(f"Posterior CME Arrival Error at {cme_hit_object}", fontsize=24)
        ax[1].set_ylim([0, max_y_axis])
        ax[1].set_ylabel("No. of particles", fontsize=20)
        ax[1].tick_params(axis='both', which='major', labelsize=16)
        #if len(weights_post_plot) > 0:
        #    ax[2].hist(post_arrival_error_hours, weights=weights_post_plot)
        #ax[2].set_title(f"Weighted posterior CME Arrival Error at {cme_hit_object}")
        #ax[2].set_ylim([0, 1])
        ax[-1].set_xlabel("Arrival time error (hours)", fontsize=20)

        if base_file_path is None:
            plt.show()
        else:
            out_dir_path = os.path.join(base_file_path, f"run_{runNo:03d}")
            filePath = os.path.join(
                out_dir_path, "histogram_arrival_time_error.png"
            )
            plt.savefig(filePath)

            # if np.isnan(weighted_median_post_arr_sec):
            #     weight_med_prior = np.nan
            #     weight_med_post = np.nan
            # else:
            #     weight_med_prior = float(weighted_median_post_arr_sec)
            #     weight_med_post = float(weighted_median_post_arr_sec)

            out_dict = {
                "prior_cme_arrival_s_from_20000101": list(seconds_arrival_prior),
                "post_cme_arrival_s_from_20000101": list(seconds_arrival_post),
                "prior_arrival_speed": list(cme_arr_prior_speed),
                "post_arrival_speed": list(cme_arr_post_speed),
                "prior_cme_hit_rate": prior_hit_rate,
                "post_cme_hit_rate": post_hit_rate,
                "mean_prior_arrival": float(mean_prior_arr_sec),
                "mean_post_arrival": float(mean_post_arr_sec),
                "mean_prior_speed": float(mean_prior_arr_speed),
                "mean_post_speed": float(mean_post_arr_speed),
                "median_prior_arrival": float(median_prior_arr_sec),
                "median_post_arrival": float(median_post_arr_sec),
                "median_prior_speed": float(median_prior_arr_speed),
                "median_post_speed": float(median_post_arr_speed),
                "weighted_mean_post_arrival": float(weighted_mean_post_arr_sec),
                "weighted_mean_post_speed": float(weighted_mean_post_arr_speed),
                "weighted_median_post_arrival": float(weighted_median_post_arr_sec),
                "weighted_median_post_speed": float(weighted_median_post_arr_speed),
                "sd_prior_arrival": float(sd_prior_arr_sec),
                "sd_post_arrival": float(sd_post_arr_sec),
                "sd_prior_speed": float(sd_prior_arr_speed),
                "sd_post_speed": float(sd_post_arr_speed),
                "weighted_sd_post_arrival": float(weighted_sd_post_arr_sec),
                "weighted_sd_post_speed": float(weighted_sd_post_arr_speed),
            }
            json_file_path = os.path.join(out_dir_path, f"cme_arrival_stats.json")
            with open(json_file_path, "w") as f:
                json.dump(out_dict, f)

            # json_file_path = os.path.join(out_dir_path, f"prior_stats.json")
            # with open(json_file_path, "w") as f:
            #     json.dump(prior_stat_dict, f)
            #
            # json_file_path = os.path.join(out_dir_path, f"post_stats.json")
            # with open(json_file_path, "w") as f:
            #     json.dump(post_stat_dict, f)

    return None


def plot_par_histograms_over_mult_runs(
    ds, ax, xVarName, yVarName, xTruth, yTruth, xLabel, yLabel,
        plotPrior=False, cmap=cc.m_gouldian, grid_size=10,
        xMin=np.nan, xMax=np.nan, yMin=np.nan, yMax=np.nan, cbarMin=np.nan, cbarMax=np.nan
):
    ax.set_facecolor(cmap(0))
    # Extract the relevant values to plot
    if plotPrior:
        xValues = ds[xVarName][:, 0, :].values
        yValues = ds[yVarName][:, 0, :].values
    else:
        xValues = ds[xVarName][:, -1, :].values
        yValues = ds[yVarName][:, -1, :].values

    # Add constraints on colorbar if provided
    axMinMaxCond = np.isnan(xMin) or np.isnan(xMax) or np.isnan(yMin) or np.isnan(yMax)
    if np.isnan(cbarMin + cbarMax):
        if axMinMaxCond:
            im = ax.hexbin(
                xValues, yValues,
                gridsize=grid_size,
                cmap=cmap
            )
        else:
            im = ax.hexbin(
                xValues, yValues,
                gridsize=grid_size, extent=(xMin, xMax, yMin, yMax),
                cmap=cmap
            )
    else:
        if axMinMaxCond:
            im = ax.hexbin(
                xValues, yValues,
                gridsize=grid_size,
                cmap=cmap, vmin=cbarMin, vmax=cbarMax
            )
        else:
            im = ax.hexbin(
                xValues, yValues,
                gridsize=grid_size, extent=(xMin, xMax, yMin, yMax),
                cmap=cmap, vmin=cbarMin, vmax=cbarMax
            )

    if not axMinMaxCond:
        ax.plot(
            [xMin, xMax], [yTruth, yTruth],
            color='r', linestyle='dashed', label=f"True {yVarName}"
        )
        ax.plot(
            [xTruth, xTruth], [yMin, yMax],
            color='r', linestyle='dashed', label=f"True {yVarName}"
        )
    #plt.colorbar(im, label='count', ax=ax, orientation='vertical')

    # Set limits if provided
    if not np.isnan(xMin + xMax):
        ax.set_xlim(xMin, xMax)

    if not np.isnan(yMin + yMax):
        ax.set_ylim(yMin, yMax)

    ax.set_xlabel(f"{xLabel}", fontsize=28)
    ax.set_ylabel(f"{yLabel}",fontsize=28)
    ax.tick_params(axis='both', which='major', labelsize=22)
    if plotPrior:
        ax.set_title(f'Prior {xVarName} against {yVarName}', fontsize=32)
    else:
        ax.set_title(f'Posterior {xVarName} against {yVarName}', fontsize=32)

    return im


def plot_sample_cov(
        ds, run_no, obs_no, vars_req=["v", "lon", "width"]
):
    # Get number of ensemble members
    n_runs = len(ds["run"][:])
    n_ens = len(ds["ens_no"][:])
    n_var_req = len(vars_req)

    if run_no == "all":
        # Initialise an array to hold ensemble at run_no and obs_no specified
        var_array = np.zeros((n_ens * n_runs, n_var_req))

        # Get relevant run and obs_no required
        for iv, v_name in enumerate(vars_req):
            try:
                var_array[:, iv] = ds[v_name][:, obs_no, :].values.flatten()
            except:
                var_array[:, iv] = ds[v_name][obs_no, :].values
            else:
                var_array[:, iv] = ds[v_name][:, obs_no, :].values.flatten()
        df = pd.DataFrame(data=var_array, columns=vars_req, index=range(n_ens * n_runs))
    else:
        # Initialise an array to hold ensemble at run_no and obs_no specified
        var_array = np.zeros((n_ens, n_var_req))

        # Get relevant run and obs_no required
        for iv, v_name in enumerate(vars_req):
            try:
                var_array[:, iv] = ds[v_name][run_no, obs_no, :].values
            except:
                var_array[:, iv] = ds[v_name][obs_no, :].values
            else:
                var_array[:, iv] = ds[v_name][run_no, obs_no, :].values

        df = pd.DataFrame(data=var_array, columns=vars_req, index=range(n_ens))

    # Calculate covariance
    #print(df)
    # Basic correlogram
    # sns.pairplot(df)
    # plt.show()

    tick_labels = [
        i for i in df.columns
    ]
    # if use_log_v:
    #     tick_labels[cme_par_get_indices("v")] = "log(v)"

    fig, ax = plt.subplots(1, 1)
    sns.heatmap(
        df.corr(), annot=True, cmap=plt.cm.RdBu_r,
        vmin=-1, vmax=1, ax=ax,
        xticklabels=tick_labels, yticklabels=tick_labels
    )
    ax.set_title(f"Correlation matrix for obs_no = {obs_no}, run_no = {run_no}")
    plt.show()

    # sns.heatmap(df.cov(), annot=True, cmap=plt.cm.RdBu_r, vmin=-1600, vmax=1600)
    # plt.show()
    #cov_var_array = np.cov(var_array, rowvar=False)

    #print(cov_var_array)

    return None


def main():
    nRuns = 1
    start_run = -1
    t_initTruth = None
    vTruth = None
    widthTruth = None
    lonTruth = None
    latTruth = None

    obs_dir = os.getenv('OBS_DIR')
    #indep_cov\truth_20080101 - 0000_495_37.4_0_0_0\prior_20080101 - 0100_495_37.4_0_0_0\nEns - 5_8_300.0 deg_0.0 deg

    # baseFilePath = os.path.join(
    #     "C:\\", "Users", "ss905122", "PycharmProjects", "SIR_HUXt", "output2",
    #     "mo_cone",# "New folder",
    #     "truth_495.0_37.4_0.0_0.0_0.0", "prior_470_37.0_-4_0_0","0.98",
    # )
    # event_list = ["ssw_008"]#, "ssw_009", "ssw_012"]#["ssw_007", "ssw_008", "ssw_009", "ssw_012"]
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
        #event = "ssw_007"
        #craft = "stb"
        #img = "diff"

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
            obs_dir_name = "obs_stb_01B_diff"
            obs_class = Observations(
                use_model=use_model,
                use_synthetic_obs=False,
                obs_filenames=obs_file_path,
            )
            obs_times, obs_lon, obs = obs_class.l5_read_obs_from_file()
            cme_hit_object = "EARTH"

            real_arrival_time = datetime.datetime(2010, 4, 11, 13, 4, 0)
            real_arrival_speed = 0
            fig_title = f"CME-{event}"
        elif event == "02A":
            obs_file_name = "CME_02A__20201025__PA272_good_hi2.dat"
            obs_file_path = os.path.join(obs_dir, obs_file_name)
            prior_dir = "prior_159.0_36.0_-15.0_-10.0_0.0"
            obs_dir_name = "obs_stb_02A_diff"
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
            obs_dir_name = "obs_stb_03A_diff"

            obs_class = Observations(
                use_model=use_model,
                use_synthetic_obs=False,
                obs_filenames=obs_file_path,
            )
            obs_times, obs_lon, obs = obs_class.l5_read_obs_from_file()
            cme_hit_object = "EARTH"

            real_arrival_time = datetime.datetime(2021, 11, 3, 19, 49, 0)
            real_arrival_speed = 0
            fig_title = f"CME-{event}"
        elif event == "04B":
            obs_file_name = "CME_04B__20100523__PA271_good_hi2.dat"
            obs_file_path = os.path.join(obs_dir, obs_file_name)
            prior_dir = "prior_542.0_76.0_24.0_1.0_0.0"
            obs_dir_name = "obs_stb_04B_diff"

            obs_class = Observations(
                use_model=use_model,
                use_synthetic_obs=False,
                obs_filenames=obs_file_path,
            )
            obs_times, obs_lon, obs = obs_class.l5_read_obs_from_file()

            cme_hit_object = "EARTH"
            real_arrival_time = datetime.datetime(2010, 5, 28, 2, 58, 0)
            real_arrival_speed = 0
            fig_title = f"CME-{event}"
        elif event == "05B":
            obs_file_name = "CME_05B__20091215__PA266_good_hi2.dat"
            obs_file_path = os.path.join(obs_dir, obs_file_name)
            obs_dir_name = "obs_stb_05B_diff"
            prior_dir = "prior_542.0_48.0_-7.0_10.0_0.0"

            obs_class = Observations(
                use_model=use_model,
                use_synthetic_obs=False,
                obs_filenames=obs_file_path,
            )
            obs_times, obs_lon, obs = obs_class.l5_read_obs_from_file()
            cme_hit_object = "EARTH"

            real_arrival_time = datetime.datetime(2009, 12, 19, 10, 0, 0)
            real_arrival_speed = 0
            fig_title = f"CME-{event}"
        elif event == "09A":
            obs_file_name = "CME_09A__20211027__PA268_good_hi2.dat"
            obs_file_path = os.path.join(obs_dir, obs_file_name)
            prior_dir = "prior_1368.0_100.0_2.0_-23.0_0.0"
            obs_dir_name = "obs_sta_09A_diff"

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
            obs_dir_name = "obs_stb_10A_diff"

            obs_class = Observations(
                use_model=use_model,
                use_synthetic_obs=False,
                obs_filenames=obs_file_path,
            )
            obs_times, obs_lon, obs = obs_class.l5_read_obs_from_file()
            cme_hit_object = "EARTH"

            real_arrival_time = datetime.datetime(2020, 12, 10, 2, 10, 0)
            real_arrival_speed = 0
            fig_title = f"CME-{event}"
        elif event == "13B":
            obs_file_name = "CME_13B__20110524__PA271_good_hi2.dat"
            obs_file_path = os.path.join(obs_dir, obs_file_name)
            prior_dir = "prior_598.0_52.0_6.0_-10.0_0.0"
            obs_dir_name = "obs_stb_13B_diff"

            obs_class = Observations(
                use_model=use_model,
                use_synthetic_obs=False,
                obs_filenames=obs_file_path,
            )
            obs_times, obs_lon, obs = obs_class.l5_read_obs_from_file()
            cme_hit_object = "EARTH"

            real_arrival_time = datetime.datetime(2011, 5, 28, 0, 14, 0)
            real_arrival_speed = 0
            fig_title = f"CME-{event}"
        elif event == "14B":
            obs_file_name = "CME_14B__20110710__PA274_good_hi2.dat"
            obs_file_path = os.path.join(obs_dir, obs_file_name)
            prior_dir = "prior_552.0_42.0_14.0_-11.0_0.0"
            obs_dir_name = "obs_stb_14B_diff"

            obs_class = Observations(
                use_model=use_model,
                use_synthetic_obs=False,
                obs_filenames=obs_file_path,
            )
            obs_times, obs_lon, obs = obs_class.l5_read_obs_from_file()
            cme_hit_object = "EARTH"

            real_arrival_time = datetime.datetime(2011, 7, 14, 12, 0, 0)
            real_arrival_speed = 0
            fig_title = f"CME-{event}"
        elif event == "15B":
            obs_file_name = "CME_15B__20111027__PA276_good_hi2.dat"
            obs_file_path = os.path.join(obs_dir, obs_file_name)
            prior_dir = "prior_589.0_78.0_-7.0_24.0_0.0"
            obs_dir_name = "obs_stb_15B_diff"

            obs_class = Observations(
                use_model=use_model,
                use_synthetic_obs=False,
                obs_filenames=obs_file_path,
            )
            obs_times, obs_lon, obs = obs_class.l5_read_obs_from_file()
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
        #obs_dir_name = f"obs_{craft}_{event}_{img}"
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
        #sys.exit()

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
        # for i in range(n_obs):
        #     plot_sample_cov(
        #         ds, 0, i, vars_req=["t_init", "v", "width", "lon", "lat"]
        #     )
        # # plot_sample_cov(
        #     ds, "all", -1, vars_req=["v", "lon", "width"]
        # )

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

    # colours_for_plots = sns.color_palette(cc.glasbey, n_colors=nRuns)
    # plot_par_values_over_mult_runs(
    #     ds,
    #     "obs_no",
    #     "v",
    #     vTruth,
    #     "Observation number",
    #     "CME speed",
    #     parUnits="km/s"
    # )
    # plot_par_values_over_mult_runs(
    #     ds,
    #     "obs_no",
    #     "width",
    #     widthTruth,
    #     "Observation number",
    #     "CME Width",
    #     parUnits=r"$^\circ$"
    # )
    # plot_par_values_over_mult_runs(
    #     ds,
    #     "obs_no",
    #     "lon",
    #     lonTruth,
    #     "Observation number",
    #     "CME Longitude",
    #     parUnits=r"$^\circ$"
    # )
    #
    # fig, axes = plt.subplots(figsize=(25, 17), nrows=2, ncols=3, layout="constrained")
    # axes = axes.flatten()
    # xNames = ["v", "v", "width", "v", "v", "width"]
    # yNames = ["width", "lon", "lon", "width", "lon", "lon"]
    #
    # xTruth = [vTruth, vTruth, widthTruth, vTruth, vTruth, widthTruth]
    # yTruth = [widthTruth, lonTruth, lonTruth, widthTruth, lonTruth, lonTruth]
    #
    # xLab = ["CME speed (km/s)", "CME speed (km/s)", "CME Width (deg)", "CME speed (km/s)","CME speed (km/s)", "CME width (deg)"]
    # yLab = ["CME Width (deg)", "CME Longitude (deg)", "CME Longitude (deg)", "CME Width (deg)", "CME Longitude (deg)", "CME Longitude (deg)"]
    #
    # xMin = [400, 400, 25, 400, 400, 25]
    # yMin = [25, -15, -15, 25, -15, -15]
    #
    # xMax = [650, 650, 50, 650, 650, 50]
    # yMax = [50, 15, 15, 50, 15, 15]
    #
    # grid_size = 15
    # """xMin = [380, 380, 24, 380, 380, 24]
    # yMin = [24, -14, -14, 24, -14, -14]
    #
    # xMax = [630, 630, 52, 630, 630, 52]
    # yMax = [52, 14, 14, 52, 14, 14]"""
    #
    # cbarMin = 0
    # cbarMax = 200
    #
    # for i in range(6):
    #     if i < 3:
    #         im = plot_par_histograms_over_mult_runs(
    #             ds, axes[i], xNames[i], yNames[i],
    #             xTruth[i], yTruth[i], xLab[i], yLab[i], grid_size=grid_size,
    #             plotPrior=True, cmap=cc.m_gouldian, cbarMin=cbarMin, cbarMax=cbarMax,
    #             xMin=xMin[i], xMax=xMax[i], yMin=yMin[i], yMax=yMax[i]
    #         )
    #     else:
    #         im = plot_par_histograms_over_mult_runs(
    #             ds, axes[i], xNames[i], yNames[i],
    #             xTruth[i], yTruth[i], xLab[i], yLab[i], grid_size=grid_size,
    #             plotPrior=False, cmap=cc.m_gouldian, cbarMin=cbarMin, cbarMax=cbarMax,
    #             xMin=xMin[i], xMax=xMax[i], yMin=yMin[i], yMax=yMax[i]
    #         )
    # cbar = fig.colorbar(im, ax=axes, orientation='vertical', fraction=.1)
    # cbar.ax.tick_params(labelsize=24)
    # cbar.set_label("Count", fontsize=30)
    # plt.show()

    return None

if __name__ == "__main__":
    main()