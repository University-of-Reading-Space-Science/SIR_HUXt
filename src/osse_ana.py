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

def safe_savefig(fig, out_path, **kwargs):
    """
    Save a figure, without stopping the analysis if the file cannot be overwritten. On Windows a file
    that is open in another program (an image viewer, an IDE preview, a sync or antivirus tool) is
    locked, and savefig then fails with "OSError: [Errno 22] Invalid argument" or PermissionError. In
    that case the figure is saved next to it with a timestamp added to the name, and a warning printed.
    :param fig: Matplotlib figure
    :param out_path: Path to save to
    :param kwargs: Passed to fig.savefig (e.g. dpi)
    :return: Path the figure was actually saved to (None if it could not be saved)
    """
    try:
        fig.savefig(out_path, **kwargs)
        return out_path
    except (OSError, PermissionError) as err:
        stem, ext = os.path.splitext(out_path)
        alt_path = f"{stem}_{datetime.datetime.now():%Y%m%dT%H%M%S}{ext}"
        print(f"Warning: could not write {out_path} ({err}); it may be open in another program. "
              f"Saving to {alt_path} instead.")
        try:
            fig.savefig(alt_path, **kwargs)
            return alt_path
        except (OSError, PermissionError) as err2:
            print(f"Warning: could not save the figure ({err2}).")
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
            x_ticks = [obs_times[i].strftime("%Y/%m/%d %H:%M") for i in ind_req]
        elif len(obs_times) > 1:
            ind_req = [0, -1]
            x_ticks = [obs_times[i].strftime("%Y/%m/%d %H:%M") for i in ind_req]
        else:
            ind_req = [0]
            x_ticks = [obs_times[i].strftime("%Y/%m/%d %H:%M") for i in ind_req]

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
        safe_savefig(plt.gcf(), filePath)

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


def get_transit_time_to_r_min(ds, cme_speed):
    """
    Time for a CME to travel from cme_init_rad (where t_init is defined) to the model's inner
    boundary r_min, matching ObservationOperator.extract_cme_launch_time_and_speed
    :param ds: Dataset from the DA run (contains r_min and cme_init_rad in solar radii)
    :param cme_speed: CME speed(s) with velocity units
    :return: Transit time(s) in seconds
    """
    r_min = float(np.ravel(ds["r_min"].values)[0]) * u.solRad
    cme_init_rad = float(np.ravel(ds["cme_init_rad"].values)[0]) * u.solRad
    return ((r_min - cme_init_rad) / cme_speed).to(u.s)


def get_arrival_cme(model, use_model, cme_hit_object="EARTH"):
    """
    Return the solved ConeCME used for arrival statistics.
    For compress_surf3d the model is a stack of 2D SURF runs, one per latitude. The CME must be
    taken from the latitude slice nearest the target body's latitude; SURFlat[0] is the most
    southerly slice (e.g. -7.7 deg), not the one containing Earth.
    :param model: Solved SURF/SURF3d/HUXt model
    :param use_model: Model name
    :param cme_hit_object: Body the CME arrival is computed for
    :return: ConeCME object of the relevant (latitude slice of the) model
    """
    if use_model in ["compress_surf3d"]:
        times = model.SURFlat[0].time_init + model.SURFlat[0].time_out
        body = S.Observer(cme_hit_object.upper(), times)
        body_lat = np.nanmean(body.lat.to(u.deg).value)
        slice_lats = np.array([sl.latitude.to(u.deg).value for sl in model.SURFlat])
        i_lat = int(np.argmin(np.abs(slice_lats - body_lat)))
        return model.SURFlat[i_lat].cmes[0]
    return model.cmes[0]


def get_arrival_speed_from_front(cme, stats, n_side=2):
    """
    Estimate the CME front speed at arrival as dr/dt of the front along the hit longitude.
    Needed for the compressible SURF solver, which tracks the CME front radius but never fills
    the tracer-particle speeds (cme_particles_v stays NaN), so compute_arrival_at_body returns
    v = NaN for every member.
    :param cme: Solved ConeCME
    :param stats: Output of cme.compute_arrival_at_body
    :param n_side: Number of output time steps either side of the arrival step to fit
    :return: Arrival speed in km/s (float), NaN if it cannot be estimated
    """
    if not stats["hit"]:
        return np.nan
    hit_id = int(stats["hit_id"])
    hit_lon = stats["lon"].to(u.rad).value
    t_list, r_list = [], []
    for i in range(hit_id - n_side, hit_id + n_side + 1):
        coord = cme.coords.get(i)
        if (coord is None) or (len(coord["r"]) == 0):
            continue
        front = np.asarray(coord["front_id"].value) == 1.0
        if np.sum(front) < 2:
            continue
        # longitudes relative to the CME nose, in (-pi, pi], so the front is monotonic in lon
        lon_rel = np.angle(np.exp(1j * (coord["lon"].to(u.rad).value[front] - cme.longitude.to(u.rad).value)))
        target = np.angle(np.exp(1j * (hit_lon - cme.longitude.to(u.rad).value)))
        order = np.argsort(lon_rel)
        r_at = np.interp(target, lon_rel[order], coord["r"].to(u.km).value[front][order],
                         left=np.nan, right=np.nan)
        if np.isfinite(r_at):
            t_list.append(coord["time"].jd * 86400.0)
            r_list.append(r_at)
    if len(t_list) < 2:
        return np.nan
    slope = np.polyfit(np.array(t_list) - t_list[0], np.array(r_list), 1)[0]
    return float(slope)


def get_arrival_time_and_speed(model, use_model, cme_hit_object):
    """
    Solve-independent helper: compute arrival stats for the relevant CME, with a speed fallback
    :return: (arrival datetime or NaN, arrival speed km/s or NaN, hit count 0/1, stats dict)
    """
    cme = get_arrival_cme(model, use_model, cme_hit_object)
    stats = cme.compute_arrival_at_body(cme_hit_object)
    if stats["hit"]:
        arr_time = stats["t_arrive"].datetime
        arr_speed = (stats["v"].to(u.km / u.s)).value
        if not np.isfinite(arr_speed):
            arr_speed = get_arrival_speed_from_front(cme, stats)
        return arr_time, arr_speed, 1, stats
    return np.nan, np.nan, 0, stats


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
    v_true = float(true_cme_speed) * u.km / u.s
    # Launch at r_min, not at cme_init_rad (see get_transit_time_to_r_min)
    sec_to_cme_true = float(
        (true_cme_t_init - start_time).total_seconds()
    ) * u.s + get_transit_time_to_r_min(ds, v_true)
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
    cme_arrival_time, cme_arrival_speed, _, true_stats = get_arrival_time_and_speed(
        model, use_model, cme_hit_object
    )

    print(f"True CME arrival time at {cme_hit_object}: {cme_arrival_time}")
    print(f"True CME arrival speed at {cme_hit_object}: {cme_arrival_speed}")

    return cme_arrival_time, cme_arrival_speed


def get_cme_stats_prior_post(
    ens_no, n_members, model, use_model, prior_cme, post_cme, cme_hit_object
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
    cme_arrival_prior, cme_arr_prior_speed, prior_count_hit, prior_stats = get_arrival_time_and_speed(
        model, use_model, cme_hit_object
    )

    # Make posterior CME arrival times
    model.solve([post_cme])
    cme_arrival_post, cme_arr_post_speed, post_count_hit, post_stats = get_arrival_time_and_speed(
        model, use_model, cme_hit_object
    )

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

    # (A leftover debugging block here printed each member's speeds and then called sys.exit(),
    #  so the arrival-time analysis below never ran.)

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
    # u.Quantity keeps the units (np.array of a list of Quantities returns a plain float array,
    #  so v_prior.value below and the ConeCME unit checks failed)
    sec_to_cme_prior = u.Quantity([t.to(u.s) for t in prior_dict["t_init"]])
    v_prior = u.Quantity(prior_dict["v"])
    width_prior = u.Quantity(prior_dict["width"])
    lon_prior = u.Quantity(prior_dict["lon"])
    lat_prior = u.Quantity(prior_dict["lat"])
    thick_prior = u.Quantity(prior_dict["thick"])

    # t_init in the DA output is the time the CME is at cme_init_rad, but these models start at r_min,
    #  so (as in ObservationOperator.extract_cme_launch_time_and_speed) launch each CME at
    #  t_init + (r_min - cme_init_rad) / v. Previously this offset was omitted, so every prior and
    #  posterior arrival time (and the twin "truth" arrival) was too early by 9.5 Rs / v (~3 h at 600 km/s).
    sec_to_cme_prior = sec_to_cme_prior + get_transit_time_to_r_min(ds, v_prior)
    sec_to_cme_post = [
        t + get_transit_time_to_r_min(ds, v)
        for t, v in zip(post_dict["t_init"], post_dict["v"])
    ]

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

        # Use no more workers than there are ensemble members, leaving one core free
        n_proc = max(1, min(int((os.cpu_count() or 2) - 1), int(n_members)))
        pool = Pool(processes=n_proc)

        for i in range(n_members):
            obj_result[i] = pool.apply_async(
                get_cme_stats_prior_post,
                kwds={
                    'ens_no': i,
                    'n_members': n_members,
                    'model': model,
                    'use_model': use_model,
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
                use_model=use_model,
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

            # Transform to unbiased variance estimator. For normalised (reliability) weights the
            #  correction is 1 / (1 - sum(w^2)); n / (n - 1) is only right for equal weights.
            sum_w2 = np.sum(renorm_weights_post ** 2)
            b_factor = 1.0 / (1.0 - sum_w2) if sum_w2 < 1.0 else np.nan
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
            safe_savefig(plt.gcf(), filePath)

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


def plot_elongation_profiles(
        base_file_path,
        runNo=0,
        obs_sd=None,
        fig_title=None,
        craft=None,
        save_fig=True,
):
    """
    Plot the modelled CME flank elongation of every ensemble member for the prior and posterior,
    together with the observations, against observation time.

    Reads <base_file_path>/run_<runNo>/ens_elongations.nc, written by
    RunDataAssimilationRoutine.write_elongation_file. Two panels with shared axes:
      - Prior: elongation of each (Liu-West shrunk) prior particle at each observation time
        (elon_first_stage), with the ensemble median.
      - Posterior: elongation of each resampled particle (elon_posterior), with the
        weight-weighted median (weights = exp(log_weight)).
    The observations are overplotted in both panels (with +/- 1 sd error bars if obs_sd is given).
    Observations that were not assimilated are NaN in the file and leave gaps in the lines.

    :param base_file_path: Directory containing the run_<runNo> output folders
    :param runNo: Run number to plot
    :param obs_sd: Observation error standard deviation in degrees (sqrt(obs_cov)); None for no error bars
    :param fig_title: Figure title (defaults to "Elongation profiles, run <runNo>")
    :param craft: Optional craft name, appended to the output file name as in the other plots
    :param save_fig: If True save to <run dir>/elongation_profiles[_<craft>_assim].png, else plt.show()
    :return: (fig, ax), or (None, None) if the elongation file does not exist
    """
    import matplotlib.dates as mdates

    run_dir = os.path.join(base_file_path, f"run_{runNo:03d}")
    elon_file = os.path.join(run_dir, "ens_elongations.nc")
    if not os.path.isfile(elon_file):
        print(f"plot_elongation_profiles: {elon_file} not found (run predates elongation output?)")
        return None, None

    with xr.load_dataset(elon_file) as elon_ds:
        obs_time = pd.to_datetime(elon_ds["obs_time"].values)
        observation = elon_ds["observation"].values
        assimilated = elon_ds["assimilated"].values.astype(bool)
        elon_prior = elon_ds["elon_first_stage"].values  # (n_obs, n_ens)
        elon_post = elon_ds["elon_posterior"].values  # (n_obs, n_ens)
        log_weight = elon_ds["log_weight"].values  # (n_obs, n_ens)

    n_obs, n_ens = elon_prior.shape

    # Ensemble summaries per observation time (NaN where nothing was assimilated / CME not seen)
    prior_median = np.full(n_obs, np.nan)
    post_wmedian = np.full(n_obs, np.nan)
    for k in range(n_obs):
        pri = elon_prior[k, :]
        if np.any(np.isfinite(pri)):
            prior_median[k] = np.nanmedian(pri)

        post = elon_post[k, :]
        good = np.isfinite(post) & np.isfinite(log_weight[k, :])
        if np.any(good):
            lw = log_weight[k, good]
            w = np.exp(lw - np.max(lw))
            post_wmedian[k] = weighted_median(post[good], weights=w / np.sum(w))

    # Colours: categorical slot 1 (blue) for the prior, slot 2 (orange) for the posterior;
    #  observations in neutral dark ink so they never read as a third model series
    col_prior = "#2a78d6"
    col_post = "#eb6834"
    col_obs = "#1a1a19"
    col_grid = "#d9d8d2"

    fig, ax = plt.subplots(
        figsize=(12, 8), nrows=2, ncols=1, sharex=True, sharey=True, layout="constrained"
    )

    panels = [
        (ax[0], elon_prior, prior_median, col_prior, "Prior", "Prior member", "Prior median"),
        (ax[1], elon_post, post_wmedian, col_post, "Posterior", "Posterior member", "Posterior weighted median"),
    ]
    for axi, elon, summary, col, panel_name, member_label, summary_label in panels:
        # Individual members: thin, translucent lines (one legend entry for all of them)
        for m in range(n_ens):
            axi.plot(
                obs_time, elon[:, m], color=col, linewidth=0.8, alpha=0.25,
                label=member_label if m == 0 else None, zorder=1
            )
        # Ensemble summary: 2 px line with markers
        axi.plot(
            obs_time, summary, color=col, linewidth=2.0, marker="o", markersize=4,
            label=summary_label, zorder=2
        )
        # Observations: dark markers (hollow where the observation was not assimilated)
        if obs_sd is not None:
            axi.errorbar(
                obs_time[assimilated], observation[assimilated], yerr=obs_sd,
                fmt="none", ecolor=col_obs, elinewidth=1.0, capsize=2, zorder=3
            )
        axi.plot(
            obs_time[assimilated], observation[assimilated], linestyle="none", marker="o",
            markersize=7, markerfacecolor=col_obs, markeredgecolor="white", markeredgewidth=1.0,
            label="Observation (assimilated)", zorder=4
        )
        if np.any(~assimilated):
            axi.plot(
                obs_time[~assimilated], observation[~assimilated], linestyle="none", marker="o",
                markersize=7, markerfacecolor="none", markeredgecolor=col_obs, markeredgewidth=1.0,
                label="Observation (not assimilated)", zorder=4
            )

        axi.set_title(panel_name, fontsize=18, loc="left")
        axi.set_ylabel("Elongation (deg)", fontsize=16)
        axi.tick_params(axis="both", which="major", labelsize=13)
        axi.grid(True, color=col_grid, linewidth=0.6)
        axi.set_axisbelow(True)
        for spine in ["top", "right"]:
            axi.spines[spine].set_visible(False)
        axi.legend(fontsize=12, frameon=False, loc="upper left")

    ax[-1].set_xlabel("Observation time (UT)", fontsize=16)
    ax[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y/%m/%d %H:%M"))
    plt.setp(ax[-1].get_xticklabels(), rotation=30, ha="right")

    if fig_title is None:
        fig_title = f"Elongation profiles, run {runNo}"
    fig.suptitle(fig_title, fontsize=20)

    if save_fig:
        if craft is None:
            out_path = os.path.join(run_dir, "elongation_profiles.png")
        else:
            out_path = os.path.join(run_dir, f"elongation_profiles_{craft}_assim.png")
        out_path = safe_savefig(fig, out_path, dpi=150)
        print(f"Saved elongation profile plot to {out_path}")
        plt.close(fig)
    else:
        plt.show()

    return fig, ax


def weighted_quantiles(values, weights, quantiles):
    """
    Weighted quantiles of a 1D sample (linear interpolation of the weighted CDF at the midpoints
    of each sample's weight). NaN values are ignored. Censored values may be given as -inf (known
    to be below every finite value) or +inf (known to be above); a quantile that falls among (or
    next to) censored values cannot be located and is returned as NaN.
    :param values: 1D sample
    :param weights: Non-negative weights, same length as values
    :param quantiles: Quantile(s) in [0, 1]
    :return: Array of quantiles (NaN if no usable values with positive weight)
    """
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    quantiles = np.atleast_1d(np.asarray(quantiles, dtype=float))
    good = ~np.isnan(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(good):
        return np.full(quantiles.shape, np.nan)
    v = values[good]
    w = weights[good]
    order = np.argsort(v)
    v, w = v[order], w[order]
    cdf = (np.cumsum(w) - 0.5 * w) / np.sum(w)
    with np.errstate(invalid="ignore"):
        q = np.interp(quantiles, cdf, v)
    q[~np.isfinite(q)] = np.nan
    return q


def plot_full_elongation_profiles(
        base_file_path,
        runNo=0,
        obs_sd=None,
        fig_title=None,
        craft=None,
        save_fig=True,
        include_extrapolated=False,
        min_weight_fraction=0.5,
        band_quantiles=(0.05, 0.95),
        focus_on_window=True,
        window_pad_hours=3.0,
        shade_window=False,
):
    """
    Plot the full (3-day) modelled CME flank elongation profiles of the prior and posterior
    ensembles against the observations.

    Reads prior_profile_* and posterior_profile_* from <base_file_path>/run_<runNo>/ens_elongations.nc
    (written by RunDataAssimilationRoutine.calculate_elongation_profiles). Prior = the ensemble passed
    to the Aux-PF for the first observation; posterior = the ensemble after the last observation in
    the permitted elongation range. Two panels with shared axes, each showing:
      - every member's profile (thin translucent lines; dashed where the value was extrapolated
        before the member's first tracked model point),
      - the weighted median and a weighted band_quantiles band (weights = exp(log_weight)). A member
        with no value before its first (or after its last) plotted value is counted as below (above)
        the field of view rather than dropped, so members leaving the 35 deg field of view do not pull
        the median down; a quantile that falls among such members is not drawn,
      - the observations (filled = assimilated, hollow = not assimilated, +/- obs_sd error bars),
      - the assimilation window (first to last assimilated observation) shaded.
    The per-observation plot (plot_elongation_profiles) is unchanged.

    :param base_file_path: Directory containing the run_<runNo> output folders
    :param runNo: Run number to plot
    :param obs_sd: Observation error standard deviation in degrees (sqrt(obs_cov)); None for no error bars
    :param fig_title: Figure title (defaults to "Full elongation profiles, run <runNo>")
    :param craft: Optional craft name, appended to the output file name as in the other plots
    :param save_fig: If True save to <run dir>/elongation_profiles_full[_<craft>_assim].png, else plt.show()
    :param include_extrapolated: If False, extrapolated values are dropped from the lines and summaries
    :param min_weight_fraction: Minimum fraction of the ensemble weight with a known position (a value,
        or known to be below/above the field of view) for the median and band to be drawn at a time
    :param band_quantiles: Lower and upper weighted quantiles of the shaded band
    :param focus_on_window: If True, zoom the axes on the assimilation window (first to last assimilated
        observation, +/- window_pad_hours); the y-range is set from the observations and the ensemble
        within that time range. If False (or nothing was assimilated) show the full 3-day profiles.
    :param window_pad_hours: Padding (hours) either side of the assimilation window when focusing
    :param shade_window: If True, shade the assimilation window (first to last assimilated observation)
    :return: (fig, ax), or (None, None) if the file or the profiles do not exist
    """
    import matplotlib.dates as mdates

    run_dir = os.path.join(base_file_path, f"run_{runNo:03d}")
    elon_file = os.path.join(run_dir, "ens_elongations.nc")
    if not os.path.isfile(elon_file):
        print(f"plot_full_elongation_profiles: {elon_file} not found")
        return None, None

    with xr.load_dataset(elon_file) as elon_ds:
        if "profile_time" not in elon_ds.coords:
            print(f"plot_full_elongation_profiles: no elongation profiles in {elon_file} "
                  f"(run predates profile output or compute_elongation_profiles was off)")
            return None, None
        profile_time = pd.to_datetime(elon_ds["profile_time"].values)
        obs_time = pd.to_datetime(elon_ds["obs_time"].values)
        observation = elon_ds["observation"].values
        assimilated = elon_ds["assimilated"].values.astype(bool)
        stages = {}
        for stage in ["prior", "posterior"]:
            if f"{stage}_profile_elon" in elon_ds:
                stages[stage] = dict(
                    elon=elon_ds[f"{stage}_profile_elon"].values.astype(float),  # (n_ens, n_profile)
                    extrap=elon_ds[f"{stage}_profile_extrapolated"].values.astype(bool),
                    log_weight=elon_ds[f"{stage}_profile_log_weight"].values.astype(float),
                )

    if not stages:
        print(f"plot_full_elongation_profiles: no prior/posterior profiles in {elon_file}")
        return None, None

    # Colours as in plot_elongation_profiles: prior blue, posterior orange, observations dark ink
    col_prior = "#2a78d6"
    col_post = "#eb6834"
    col_obs = "#1a1a19"
    col_grid = "#d9d8d2"
    col_window = "#ecebe6"

    panel_info = [
        ("prior", col_prior, "Prior (before the first observation)"),
        ("posterior", col_post, "Posterior (after the last assimilated observation)"),
    ]
    panel_info = [p for p in panel_info if p[0] in stages]

    fig, ax = plt.subplots(
        figsize=(12, 4.5 * len(panel_info)), nrows=len(panel_info), ncols=1,
        sharex=True, sharey=True, layout="constrained", squeeze=False
    )
    ax = ax[:, 0]

    lo_q, hi_q = band_quantiles
    for axi, (stage, col, panel_name) in zip(ax, panel_info):
        elon = stages[stage]["elon"].copy()
        extrap = stages[stage]["extrap"]
        log_w = stages[stage]["log_weight"]
        n_ens, n_prof = elon.shape

        # Normalised linear weights (non-finite log-weights -> zero weight)
        lw = np.where(np.isfinite(log_w), log_w, -np.inf)
        w = np.exp(lw - np.max(lw)) if np.any(np.isfinite(lw)) else np.ones(n_ens)
        w = w / np.sum(w)

        if not include_extrapolated:
            elon[extrap] = np.nan
        elon_model = np.where(extrap, np.nan, elon)  # model-tracked part only
        elon_extrap = np.where(extrap, elon, np.nan)  # extrapolated part only

        # Assimilation window
        if shade_window and np.any(assimilated):
            axi.axvspan(
                obs_time[assimilated].min(), obs_time[assimilated].max(),
                color=col_window, zorder=0, label="Assimilation window"
            )

        # Members: model-tracked solid, extrapolated dashed (one legend entry each)
        for m in range(n_ens):
            axi.plot(profile_time, elon_model[m, :], color=col, linewidth=0.8, alpha=0.25,
                     label=f"{stage.capitalize()} member" if m == 0 else None, zorder=1)
            if include_extrapolated and np.any(np.isfinite(elon_extrap[m, :])):
                axi.plot(profile_time, elon_extrap[m, :], color=col, linewidth=0.8, alpha=0.25,
                         linestyle=(0, (3, 2)),
                         label="Extrapolated (before tracked)" if m == 0 else None, zorder=1)

        # Summary statistics with censoring: before a member's first plotted value it has not yet
        #  entered the field of view (or been tracked) -> below (-inf); after its last value it has
        #  left the field of view -> above (+inf). Gaps between values stay unknown (NaN).
        #  (Previously members without a value were simply dropped, so the median jumped down
        #  whenever the fastest members left the field of view.)
        elon_cens = elon.copy()
        for m in range(n_ens):
            idx = np.flatnonzero(np.isfinite(elon[m, :]))
            if idx.size:
                elon_cens[m, :idx[0]] = -np.inf
                elon_cens[m, idx[-1] + 1:] = np.inf

        med = np.full(n_prof, np.nan)
        band_lo = np.full(n_prof, np.nan)
        band_hi = np.full(n_prof, np.nan)
        for k in range(n_prof):
            known = ~np.isnan(elon_cens[:, k])
            if np.sum(w[known]) >= min_weight_fraction:
                band_lo[k], med[k], band_hi[k] = weighted_quantiles(
                    elon_cens[:, k], w, [lo_q, 0.5, hi_q]
                )
        axi.fill_between(
            profile_time, band_lo, band_hi, color=col, alpha=0.18, linewidth=0,
            label=f"Weighted {int(round(100 * lo_q))}-{int(round(100 * hi_q))}% range", zorder=2
        )
        axi.plot(profile_time, med, color=col, linewidth=2.0, label="Weighted median", zorder=3)

        # Observations
        if obs_sd is not None:
            axi.errorbar(
                obs_time, observation, yerr=obs_sd,
                fmt="none", ecolor=col_obs, elinewidth=1.0, capsize=2, zorder=4
            )
        axi.plot(
            obs_time[assimilated], observation[assimilated], linestyle="none", marker="o",
            markersize=6, markerfacecolor=col_obs, markeredgecolor="white", markeredgewidth=1.0,
            label="Observation (assimilated)", zorder=5
        )
        if np.any(~assimilated):
            axi.plot(
                obs_time[~assimilated], observation[~assimilated], linestyle="none", marker="o",
                markersize=6, markerfacecolor="none", markeredgecolor=col_obs, markeredgewidth=1.0,
                label="Observation (not assimilated)", zorder=5
            )

        axi.set_title(panel_name, fontsize=18, loc="left")
        axi.set_ylabel("Elongation (deg)", fontsize=16)
        axi.tick_params(axis="both", which="major", labelsize=13)
        axi.grid(True, color=col_grid, linewidth=0.6)
        axi.set_axisbelow(True)
        for spine in ["top", "right"]:
            axi.spines[spine].set_visible(False)
        axi.legend(fontsize=11, frameon=False, loc="upper left")

    # x-range: the assimilation window (+/- padding), or the full profile time range
    if focus_on_window and np.any(assimilated):
        pad = pd.Timedelta(hours=window_pad_hours)
        t_lo = obs_time[assimilated].min() - pad
        t_hi = obs_time[assimilated].max() + pad
    else:
        t_lo, t_hi = profile_time.min(), profile_time.max()
    ax[0].set_xlim(t_lo, t_hi)

    # y-range: observations and the plotted ensemble values inside the x-range, ignoring the
    #  0 deg pre-launch values
    in_prof = (profile_time >= t_lo) & (profile_time <= t_hi)
    in_obs = (obs_time >= t_lo) & (obs_time <= t_hi) & np.isfinite(observation)
    vals = [observation[in_obs] + (obs_sd or 0.0), observation[in_obs] - (obs_sd or 0.0)]
    for st in stages.values():
        e = st["elon"][:, in_prof]
        if not include_extrapolated:
            e = np.where(st["extrap"][:, in_prof], np.nan, e)
        vals.append(e[np.isfinite(e) & (e > 0)])
    all_vals = np.concatenate(vals)
    if all_vals.size:
        y_lo, y_hi = np.min(all_vals), np.max(all_vals)
        y_pad = 0.05 * max(y_hi - y_lo, 1.0)
        ax[0].set_ylim(max(0.0, y_lo - y_pad) if focus_on_window else 0.0, y_hi + y_pad)

    ax[-1].set_xlabel("Time (UT)", fontsize=16)
    ax[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y/%m/%d %H:%M"))
    plt.setp(ax[-1].get_xticklabels(), rotation=30, ha="right")

    if fig_title is None:
        fig_title = f"Full elongation profiles, run {runNo}"
    fig.suptitle(fig_title, fontsize=20)

    if save_fig:
        if craft is None:
            out_path = os.path.join(run_dir, "elongation_profiles_full.png")
        else:
            out_path = os.path.join(run_dir, f"elongation_profiles_full_{craft}_assim.png")
        out_path = safe_savefig(fig, out_path, dpi=150)
        print(f"Saved full elongation profile plot to {out_path}")
        plt.close(fig)
    else:
        plt.show()

    return fig, ax


def compute_elongation_profiles_per_obs(
        base_file_path,
        runNo=-1,
        stages="assimilated",
        overwrite=False,
):
    """
    Compute the full (3-day) modelled CME flank elongation profile of every ensemble member for the
    ensemble at each observation: the prior (stage 0) and the ensemble after observation k (stage k+1).

    The ensembles are rebuilt from <run dir>/cme_pars.nc and run through the same observation operator
    the data assimilation used (RunDataAssimilationRoutine.calculate_elongation_profiles), so the model
    set-up, imager, search band and tracking match what the filter compared with the observations.
    The routine is set up from the current da_surf_setup.py, which must match the run (the number of
    members and the model start time are checked).

    An ensemble whose particles are identical to one already computed (e.g. an observation where the
    filter only updated the weights, or one that was not assimilated) is not rerun: its profiles are
    copied, and only its weights differ.

    Output: <run dir>/ens_elongation_profiles_per_obs.nc with
        elon_profile (n_stage, n_ens, n_profile): elongation (deg), NaN where not visible/not computed
        extrapolated (n_stage, n_ens, n_profile): 1 where extrapolated before the first tracked point
        log_weight (n_stage, n_ens): normalised log-weight of each member at that stage
        computed (n_stage): 1 if profiles are available for that stage
        reused_from (n_stage): stage whose model runs were reused (-1 if run for this stage)
        stage_obs_no (n_stage): observation index after which the ensemble is taken (-1 = prior)
        coords: stage, ens_no, profile_time, stage_obs_time

    :param base_file_path: Directory containing the run_<runNo> output folders
    :param runNo: Run number
    :param stages: "assimilated" (prior + ensembles after each assimilated observation) or "all"
    :param overwrite: If False and the output file exists, return it without recomputing
    :return: Path of the output file, or None if the inputs are missing
    """
    from data_assimilation_routine import RunDataAssimilationRoutine
    from init_sir import initialise_cme_parameter_ensemble_dict
    from sir_observation_operator import shutdown_worker_pool

    run_dir = os.path.join(base_file_path, f"run_{runNo:03d}")
    par_file = os.path.join(run_dir, "cme_pars.nc")
    elon_file = os.path.join(run_dir, "ens_elongations.nc")
    out_file = os.path.join(run_dir, "ens_elongation_profiles_per_obs.nc")
    if os.path.isfile(out_file) and not overwrite:
        print(f"compute_elongation_profiles_per_obs: using existing {out_file}")
        return out_file
    for f in [par_file, elon_file]:
        if not os.path.isfile(f):
            print(f"compute_elongation_profiles_per_obs: {f} not found")
            return None

    par_ds = xr.load_dataset(par_file)
    elon_ds = xr.load_dataset(elon_file)
    n_stage = par_ds.sizes["n_obs"]  # prior + one row per observation
    n_ens = par_ds.sizes["n_ens"]
    assimilated = elon_ds["assimilated"].values.astype(bool)
    obs_time = pd.to_datetime(elon_ds["obs_time"].values)

    run_da = RunDataAssimilationRoutine()
    if run_da.n_members != n_ens:
        raise ValueError(
            f"da_surf_setup.py has n_members = {run_da.n_members} but the run has {n_ens} members; "
            f"set da_surf_setup.py to the run's settings before computing its profiles."
        )
    model_init_time = pd.to_datetime(np.ravel(par_ds["model_init_time"].values)[0]).to_pydatetime()
    if abs((model_init_time - run_da.surf_init_time).total_seconds()) > 1.0:
        print(f"Warning: run model start {model_init_time} differs from da_surf_setup.py "
              f"({run_da.surf_init_time}); profiles use the setup's model start.")

    if stages == "all":
        stage_list = list(range(n_stage))
    else:
        stage_list = [0] + [k + 1 for k in range(min(len(assimilated), n_stage - 1)) if assimilated[k]]

    profile_times = run_da.get_profile_times()
    n_prof = len(profile_times)
    elon = np.full((n_stage, n_ens, n_prof), np.nan)
    extrap = np.zeros((n_stage, n_ens, n_prof), dtype=np.int8)
    computed = np.zeros(n_stage, dtype=np.int8)
    reused_from = np.full(n_stage, -1, dtype=int)
    log_weight = np.asarray(par_ds["log_weight"].values, dtype=float)

    par_names = ["t_init", "v", "width", "lon", "lat", "thick"]
    cache = {}
    try:
        for st in stage_list:
            pars = np.array([np.asarray(par_ds[p].values[st], dtype=float) for p in par_names])
            key = np.round(pars, 6).tobytes()
            if key in cache:
                src = cache[key]
                elon[st], extrap[st] = elon[src], extrap[src]
                reused_from[st] = src
                computed[st] = 1
                print(f"Stage {st}: particles unchanged since stage {src}, reusing its profiles")
                continue

            cme_par_dict = initialise_cme_parameter_ensemble_dict(
                n_ensemble=n_ens, surf_init_time=run_da.surf_init_time
            )
            cme_par_dict["t_init"] = [
                model_init_time + datetime.timedelta(seconds=float(s)) for s in pars[0]
            ]
            cme_par_dict["v"] = list(pars[1]) * u.km / u.s
            cme_par_dict["width"] = list(pars[2]) * u.deg
            cme_par_dict["lon"] = list(pars[3]) * u.deg
            cme_par_dict["lat"] = list(pars[4]) * u.deg
            cme_par_dict["thick"] = list(pars[5]) * u.solRad
            lw = log_weight[st]
            lw = np.where(np.isfinite(lw), lw, -1e31)
            cme_par_dict["log_weight"] = list(lw)
            cme_par_dict["weight"] = list(np.exp(lw - np.max(lw)))

            label = "prior" if st == 0 else f"after observation {st - 1} ({obs_time[st - 1]})"
            print(f"Stage {st}: computing elongation profiles for the ensemble {label}")
            prof = run_da.calculate_elongation_profiles(cme_par_dict, obs_index=max(st - 1, 0))
            elon[st] = prof["elon"]
            extrap[st] = np.asarray(prof["extrapolated"]).astype(np.int8)
            computed[st] = 1
            cache[key] = st
    finally:
        shutdown_worker_pool()

    stage_obs_time = np.array(
        [np.datetime64("NaT")] + [np.datetime64(t) for t in obs_time[:n_stage - 1]],
        dtype="datetime64[ns]"
    )
    out_ds = xr.Dataset(
        data_vars=dict(
            elon_profile=(["stage", "ens_no", "n_profile"], elon),
            extrapolated=(["stage", "ens_no", "n_profile"], extrap),
            log_weight=(["stage", "ens_no"], log_weight),
            computed=(["stage"], computed),
            reused_from=(["stage"], reused_from),
            stage_obs_no=(["stage"], np.arange(n_stage) - 1),
        ),
        coords=dict(
            stage=("stage", np.arange(n_stage)),
            ens_no=("ens_no", np.arange(n_ens)),
            profile_time=("n_profile", np.array(profile_times, dtype="datetime64[ns]")),
            stage_obs_time=("stage", stage_obs_time),
        ),
        attrs=dict(
            description="Full elongation profiles of the ensemble at each observation; stage 0 = prior, "
                        "stage k+1 = ensemble after observation k",
            elongation_units="deg",
        ),
    )
    out_ds.to_netcdf(out_file, mode="w")
    print(f"Saved per-observation elongation profiles to {out_file} "
          f"({int(computed.sum())} stages, {len(cache)} ensembles run)")
    return out_file


def _weighted_profile_summary(elon, log_w, min_weight_fraction=0.5, band_quantiles=(0.05, 0.95)):
    """
    Weighted median and band of an ensemble of elongation profiles, treating a member as below the
    field of view before its first value and above it after its last (see plot_full_elongation_profiles)
    :param elon: (n_ens, n_profile) elongations, NaN where not available
    :param log_w: (n_ens) log-weights
    :return: (lower band, median, upper band, normalised linear weights)
    """
    n_ens, n_prof = elon.shape
    lw = np.where(np.isfinite(log_w), log_w, -np.inf)
    w = np.exp(lw - np.max(lw)) if np.any(np.isfinite(lw)) else np.ones(n_ens)
    w = w / np.sum(w)
    elon_cens = elon.copy()
    for m in range(n_ens):
        idx = np.flatnonzero(np.isfinite(elon[m, :]))
        if idx.size:
            elon_cens[m, :idx[0]] = -np.inf
            elon_cens[m, idx[-1] + 1:] = np.inf
    lo_q, hi_q = band_quantiles
    med, b_lo, b_hi = (np.full(n_prof, np.nan) for _ in range(3))
    for k in range(n_prof):
        known = ~np.isnan(elon_cens[:, k])
        if np.sum(w[known]) >= min_weight_fraction:
            b_lo[k], med[k], b_hi[k] = weighted_quantiles(elon_cens[:, k], w, [lo_q, 0.5, hi_q])
    return b_lo, med, b_hi, w


def plot_elongation_profiles_per_obs(
        base_file_path,
        runNo=-1,
        obs_sd=None,
        fig_title=None,
        craft=None,
        include_extrapolated=False,
        focus_on_window=True,
        window_pad_hours=3.0,
        min_weight_fraction=0.5,
        band_quantiles=(0.05, 0.95),
        weight_member_lines=True,
):
    """
    Plot, for the prior and for the ensemble after each observation, every member's full elongation
    profile with the weighted median and band, against the observations. One figure per stage, saved in
    <run dir>/elongation_profiles_per_obs/. Observations already assimilated at that stage are filled,
    later ones hollow, and the observation just assimilated is circled and marked with a vertical line.
    Reads ens_elongation_profiles_per_obs.nc (see compute_elongation_profiles_per_obs).

    :param base_file_path: Directory containing the run_<runNo> output folders
    :param runNo: Run number
    :param obs_sd: Observation error standard deviation (deg) for error bars; None for none
    :param fig_title: Figure title prefix
    :param craft: Optional craft name, added to the file names as in the other plots
    :param include_extrapolated: Include values extrapolated before each member's first tracked point
    :param focus_on_window: Zoom on the assimilation window (+/- window_pad_hours) if True
    :param window_pad_hours: Padding either side of the assimilation window (hours)
    :param min_weight_fraction: Minimum weight fraction with a known position to draw median/band
    :param band_quantiles: Weighted quantiles of the shaded band
    :param weight_member_lines: If True, member line opacity scales with its weight
    :return: List of saved file paths
    """
    import matplotlib.dates as mdates

    run_dir = os.path.join(base_file_path, f"run_{runNo:03d}")
    prof_file = os.path.join(run_dir, "ens_elongation_profiles_per_obs.nc")
    elon_file = os.path.join(run_dir, "ens_elongations.nc")
    if not (os.path.isfile(prof_file) and os.path.isfile(elon_file)):
        print(f"plot_elongation_profiles_per_obs: {prof_file} or {elon_file} not found "
              f"(run compute_elongation_profiles_per_obs first)")
        return []

    with xr.load_dataset(prof_file) as p_ds:
        elon_all = p_ds["elon_profile"].values.astype(float)
        extrap_all = p_ds["extrapolated"].values.astype(bool)
        lw_all = p_ds["log_weight"].values.astype(float)
        computed = p_ds["computed"].values.astype(bool)
        profile_time = pd.to_datetime(p_ds["profile_time"].values)
    with xr.load_dataset(elon_file) as e_ds:
        obs_time = pd.to_datetime(e_ds["obs_time"].values)
        observation = e_ds["observation"].values.astype(float)
        assimilated = e_ds["assimilated"].values.astype(bool)
        resampled = e_ds["resampled"].values.astype(bool) if "resampled" in e_ds else None

    col_ens = "#2a78d6"
    col_obs = "#1a1a19"
    col_now = "#eb6834"
    col_grid = "#d9d8d2"

    # Common axes for all stages
    if focus_on_window and np.any(assimilated):
        pad = pd.Timedelta(hours=window_pad_hours)
        t_lo, t_hi = obs_time[assimilated].min() - pad, obs_time[assimilated].max() + pad
    else:
        t_lo, t_hi = profile_time.min(), profile_time.max()
    in_prof = (profile_time >= t_lo) & (profile_time <= t_hi)
    in_obs = (obs_time >= t_lo) & (obs_time <= t_hi) & np.isfinite(observation)
    vals = [observation[in_obs]]
    for st in np.flatnonzero(computed):
        e = np.where(extrap_all[st] & ~include_extrapolated, np.nan, elon_all[st])[:, in_prof]
        vals.append(e[np.isfinite(e) & (e > 0)])
    all_vals = np.concatenate(vals) if vals else np.array([])
    pad_e = (obs_sd or 0.0) + 0.05 * max(np.ptp(all_vals) if all_vals.size else 1.0, 1.0)
    y_lim = (max(0.0, np.min(all_vals) - pad_e), np.max(all_vals) + pad_e) if all_vals.size else (0, 35)

    out_dir = os.path.join(run_dir, "elongation_profiles_per_obs")
    os.makedirs(out_dir, exist_ok=True)
    saved = []
    for st in np.flatnonzero(computed):
        k_now = st - 1  # observation just assimilated (-1 for the prior)
        elon = elon_all[st].copy()
        if not include_extrapolated:
            elon[extrap_all[st]] = np.nan
        b_lo, med, b_hi, w = _weighted_profile_summary(
            elon, lw_all[st], min_weight_fraction=min_weight_fraction, band_quantiles=band_quantiles
        )

        fig, ax = plt.subplots(figsize=(12, 6), layout="constrained")
        w_rel = w / np.max(w)
        for m in range(elon.shape[0]):
            alpha = 0.08 + 0.5 * w_rel[m] if weight_member_lines else 0.25
            ax.plot(profile_time, elon[m], color=col_ens, lw=0.8, alpha=alpha,
                    label="Member (opacity = weight)" if (m == 0 and weight_member_lines) else
                    ("Member" if m == 0 else None), zorder=1)
        ax.fill_between(profile_time, b_lo, b_hi, color=col_ens, alpha=0.18, lw=0,
                        label=f"Weighted {int(round(100 * band_quantiles[0]))}-"
                              f"{int(round(100 * band_quantiles[1]))}% range", zorder=2)
        ax.plot(profile_time, med, color=col_ens, lw=2.0, label="Weighted median", zorder=3)

        done = assimilated & (np.arange(len(observation)) <= k_now)
        later = ~done & np.isfinite(observation)
        if obs_sd is not None:
            ax.errorbar(obs_time[done], observation[done], yerr=obs_sd, fmt="none", ecolor=col_obs,
                        elinewidth=1.0, capsize=2, zorder=4)
        ax.plot(obs_time[done], observation[done], ls="none", marker="o", ms=6, mfc=col_obs,
                mec="white", mew=1.0, label="Observation (assimilated so far)", zorder=5)
        ax.plot(obs_time[later], observation[later], ls="none", marker="o", ms=6, mfc="none",
                mec="#8c8b85", mew=1.0, label="Observation (later / not assimilated)", zorder=5)
        if k_now >= 0:
            ax.axvline(obs_time[k_now], color=col_now, lw=1.2, ls=(0, (4, 2)), zorder=0)
            ax.plot(obs_time[k_now], observation[k_now], ls="none", marker="o", ms=13, mfc="none",
                    mec=col_now, mew=2.0, label="Observation just assimilated", zorder=6)

        ess = 1.0 / np.sum(w ** 2)
        if k_now < 0:
            title = "Prior ensemble"
        else:
            how = ""
            if resampled is not None:
                how = ", resampled" if resampled[k_now] else ", weights updated only"
            title = (f"After observation {k_now} ({obs_time[k_now]:%d %b %H:%M}, "
                     f"{observation[k_now]:.2f}°{how})")
        ax.set_title(f"{title}   ESS = {ess:.1f}/{len(w)}", loc="left", fontsize=14)
        if fig_title:
            fig.suptitle(fig_title, fontsize=16)
        ax.set_xlim(t_lo, t_hi)
        ax.set_ylim(*y_lim)
        ax.set_ylabel("Elongation (deg)", fontsize=13)
        ax.set_xlabel("Time (UT)", fontsize=13)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H:%M"))
        ax.grid(True, color=col_grid, lw=0.6)
        ax.set_axisbelow(True)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.legend(fontsize=10, frameon=False, loc="upper left")

        stem = f"stage_{st:03d}" + ("_prior" if k_now < 0 else f"_obs_{k_now:03d}")
        if craft is not None:
            stem += f"_{craft}_assim"
        out_path = os.path.join(out_dir, stem + ".png")
        out_path = safe_savefig(fig, out_path, dpi=130)
        plt.close(fig)
        saved.append(out_path)

    print(f"Saved {len(saved)} per-observation elongation profile plots to {out_dir}")
    return saved


def make_arrival_model(ds, use_model, sim_time=4.0 * u.day, dt_scale=20):
    """
    Build the model used to compute CME arrival times, with the same set-up as get_cme_arrival_time
    (longitudes 300-420 deg, SURF3d latitudes -15 to +15 deg, dt_scale 20) but a configurable run
    length, from a single run's cme_pars.nc dataset.
    :param ds: Dataset loaded from one run's cme_pars.nc
    :param use_model: "surf", "compress_surf", "compress_surf3d" or "huxt"
    :param sim_time: Model run length (CMEs arriving after the end of the run count as misses)
    :param dt_scale: Model output time step scale
    :return: (model, model start datetime)
    """
    start_time = pd.to_datetime(np.ravel(ds["model_init_time"].values)[0]).to_pydatetime()
    start_time_astro = Time(start_time, format="datetime")
    cr_num: int = np.trunc(sn.carrington_rotation_number(start_time))
    r_min_in = float(np.ravel(ds["r_min"].values)[0]) * u.solRad
    v_bound_in = ds["ambient_vr"].values * u.km / u.s
    lon_start = 300 * u.deg
    lon_end = 420 * u.deg

    if use_model in ["surf", "compress_surf", "compress_surf3d"]:
        ert = S.Observer("EARTH", start_time_astro)
    elif use_model in ["huxt"]:
        ert = H.Observer("EARTH", start_time_astro)
    else:
        raise ValueError(f"Unknown use_model {use_model}")

    if use_model in ["surf", "compress_surf"]:
        model = S.SURF(
            v_boundary=v_bound_in, cr_num=cr_num, cr_lon_init=ert.lon_c.to(u.deg),
            latitude=ert.lat.to(u.deg), lon_start=lon_start.to(u.rad), lon_stop=lon_end.to(u.rad),
            simtime=sim_time, dt_scale=dt_scale, r_min=r_min_in,
            solver="huxt" if use_model == "surf" else "hydro",
        )
    elif use_model in ["compress_surf3d"]:
        model = S.SURF3d(
            v_map=v_bound_in, v_map_long=ds["vr_longs"].values * u.deg,
            v_map_lat=ds["vr_lats"].values * u.deg, cr_num=cr_num, cr_lon_init=ert.lon_c.to(u.deg),
            latitude_min=-15 * u.deg, latitude_max=15 * u.deg,
            lon_start=lon_start.to(u.rad), lon_stop=lon_end.to(u.rad),
            simtime=sim_time, dt_scale=dt_scale, r_min=r_min_in, solver="hydro",
        )
    else:
        model = H.HUXt(
            v_boundary=v_bound_in, cr_num=cr_num, cr_lon_init=ert.lon_c.to(u.deg),
            latitude=ert.lat.to(u.deg), lon_start=lon_start.to(u.rad), lon_stop=lon_end.to(u.rad),
            simtime=sim_time, dt_scale=dt_scale, r_min=r_min_in,
        )
    return model, start_time


def _arrival_for_cme(model, use_model, cme, cme_hit_object="EARTH"):
    """
    Solve the model for one CME and return its arrival at cme_hit_object
    (worker function for compute_arrival_times_per_obs)
    :return: (arrival datetime or None, arrival speed in km/s or NaN)
    """
    model.solve([cme])
    arr_time, arr_speed, hit, _ = get_arrival_time_and_speed(model, use_model, cme_hit_object)
    if not hit:
        return None, np.nan
    return arr_time, float(arr_speed)


def compute_arrival_times_per_obs(
        base_file_path,
        runNo=-1,
        use_model="compress_surf3d",
        stages="assimilated",
        cme_hit_object="EARTH",
        sim_time_days=4.0,
        parallel=True,
        overwrite=False,
):
    """
    Compute every ensemble member's CME arrival time and speed at cme_hit_object for the ensemble at
    each observation: the prior (stage 0) and the ensemble after observation k (stage k+1), rebuilt from
    <run dir>/cme_pars.nc. The model set-up matches get_cme_arrival_time (make_arrival_model), with a
    run length of sim_time_days. Each CME is launched at r_min at t_init + (r_min - cme_init_rad) / v.
    An ensemble whose particles are identical to one already computed (only its weights differ, e.g. an
    observation where the filter did not resample) reuses those arrival times.

    Output: <run dir>/cme_arrival_per_obs.nc with
        arrival_time (n_stage, n_ens): arrival time (NaT if the CME does not arrive within the run)
        arrival_speed (n_stage, n_ens): arrival speed (km/s)
        log_weight (n_stage, n_ens): normalised log-weight of each member at that stage
        computed (n_stage), reused_from (n_stage), stage_obs_no (n_stage)
        coords: stage, ens_no, stage_obs_time

    :param base_file_path: Directory containing the run_<runNo> output folders
    :param runNo: Run number
    :param use_model: Model used in the run
    :param stages: "assimilated" (prior + after each assimilated observation) or "all"
    :param cme_hit_object: Body to compute the arrival at
    :param sim_time_days: Model run length in days from the model start
    :param parallel: Solve the ensemble members in parallel
    :param overwrite: If False and the output file exists, return it without recomputing
    :return: Path of the output file, or None if the inputs are missing
    """
    run_dir = os.path.join(base_file_path, f"run_{runNo:03d}")
    par_file = os.path.join(run_dir, "cme_pars.nc")
    elon_file = os.path.join(run_dir, "ens_elongations.nc")
    out_file = os.path.join(run_dir, "cme_arrival_per_obs.nc")
    if os.path.isfile(out_file) and not overwrite:
        print(f"compute_arrival_times_per_obs: using existing {out_file}")
        return out_file
    if not os.path.isfile(par_file):
        print(f"compute_arrival_times_per_obs: {par_file} not found")
        return None

    par_ds = xr.load_dataset(par_file)
    n_stage = par_ds.sizes["n_obs"]
    n_ens = par_ds.sizes["n_ens"]
    if os.path.isfile(elon_file):
        elon_ds = xr.load_dataset(elon_file)
        assimilated = elon_ds["assimilated"].values.astype(bool)
        obs_time = pd.to_datetime(elon_ds["obs_time"].values)
    else:
        # Older runs: every observation stage, no observation times
        assimilated = np.ones(n_stage - 1, dtype=bool)
        obs_time = pd.to_datetime([pd.NaT] * (n_stage - 1))

    if stages == "all":
        stage_list = list(range(n_stage))
    else:
        stage_list = [0] + [k + 1 for k in range(min(len(assimilated), n_stage - 1)) if assimilated[k]]

    model, start_time = make_arrival_model(par_ds, use_model, sim_time=sim_time_days * u.day)

    arr_sec = np.full((n_stage, n_ens), np.nan)  # seconds after model start
    arr_speed = np.full((n_stage, n_ens), np.nan)
    computed = np.zeros(n_stage, dtype=np.int8)
    reused_from = np.full(n_stage, -1, dtype=int)
    log_weight = np.asarray(par_ds["log_weight"].values, dtype=float)
    par_names = ["t_init", "v", "width", "lon", "lat", "thick"]
    cache = {}

    pool = None
    if parallel:
        n_proc = max(1, min(int((os.cpu_count() or 2) - 1), n_ens))
        pool = Pool(processes=n_proc)
    try:
        for st in stage_list:
            pars = np.array([np.asarray(par_ds[p].values[st], dtype=float) for p in par_names])
            key = np.round(pars, 6).tobytes()
            if key in cache:
                src = cache[key]
                arr_sec[st], arr_speed[st] = arr_sec[src], arr_speed[src]
                reused_from[st] = src
                computed[st] = 1
                print(f"Stage {st}: particles unchanged since stage {src}, reusing its arrival times")
                continue

            v = pars[1] * u.km / u.s
            t_launch = (pars[0] * u.s) + get_transit_time_to_r_min(par_ds, v)
            cme_cls = H.ConeCME if use_model in ["huxt"] else S.ConeCME
            cmes = [
                cme_cls(
                    t_launch=t_launch[i], v=v[i], width=pars[2][i] * u.deg,
                    longitude=pars[3][i] * u.deg, latitude=pars[4][i] * u.deg,
                    thickness=pars[5][i] * u.solRad,
                    cme_fixed_duration=True, fixed_duration=12 * 3600 * u.s,
                ) for i in range(n_ens)
            ]
            label = "prior" if st == 0 else f"after observation {st - 1}"
            print(f"Stage {st}: computing arrival times for the ensemble {label}")
            if pool is not None:
                jobs = [pool.apply_async(_arrival_for_cme, (model, use_model, c, cme_hit_object)) for c in cmes]
                results = [j.get() for j in jobs]
            else:
                results = [_arrival_for_cme(model, use_model, c, cme_hit_object) for c in cmes]
            for i, (t_arr, v_arr) in enumerate(results):
                if t_arr is not None:
                    arr_sec[st, i] = (t_arr - start_time).total_seconds()
                    arr_speed[st, i] = v_arr
            computed[st] = 1
            cache[key] = st
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    arrival_time = np.array([
        [np.datetime64(start_time + datetime.timedelta(seconds=s)) if np.isfinite(s) else np.datetime64("NaT")
         for s in row] for row in arr_sec
    ], dtype="datetime64[ns]")
    stage_obs_time = np.array(
        [np.datetime64("NaT")] + [np.datetime64(t) for t in obs_time[:n_stage - 1]],
        dtype="datetime64[ns]"
    )
    out_ds = xr.Dataset(
        data_vars=dict(
            arrival_time=(["stage", "ens_no"], arrival_time),
            arrival_speed=(["stage", "ens_no"], arr_speed),
            log_weight=(["stage", "ens_no"], log_weight),
            computed=(["stage"], computed),
            reused_from=(["stage"], reused_from),
            stage_obs_no=(["stage"], np.arange(n_stage) - 1),
        ),
        coords=dict(
            stage=("stage", np.arange(n_stage)),
            ens_no=("ens_no", np.arange(n_ens)),
            stage_obs_time=("stage", stage_obs_time),
        ),
        attrs=dict(
            description="CME arrival time and speed of each member for the ensemble at each observation; "
                        "stage 0 = prior, stage k+1 = ensemble after observation k",
            cme_hit_object=cme_hit_object,
            sim_time_days=sim_time_days,
            model_init_time=str(start_time),
        ),
    )
    out_ds.to_netcdf(out_file, mode="w")
    print(f"Saved per-observation arrival times to {out_file} "
          f"({int(computed.sum())} stages, {len(cache)} ensembles run)")
    return out_file


def plot_arrival_time_histograms_per_obs(
        base_file_path,
        runNo=-1,
        real_arrival_time=None,
        fig_title=None,
        craft=None,
        bin_hours=3.0,
        band_quantiles=(0.05, 0.95),
        n_overlay=4,
        save_fig=True,
):
    """
    Collate the weighted arrival-time distributions of the ensemble at every observation (from
    cme_arrival_per_obs.nc, see compute_arrival_times_per_obs) into one figure:
      - top: weighted histograms (probability per bin) of the prior (grey), the final ensemble (orange)
        and up to n_overlay intermediate stages (lines shaded from light to dark with observation number);
      - bottom: the full histogram at every stage as a heat map (stage on x, arrival time on y), with
        the weighted median and band_quantiles range, and the observed arrival time if given.
    Members whose CME does not arrive within the model run are left out of the histograms (the
    probability shown is per arriving weight); the fraction of weight that arrives is printed per stage.
    :param base_file_path: Directory containing the run_<runNo> output folders
    :param runNo: Run number
    :param real_arrival_time: Observed (or twin truth) arrival datetime; None to omit
    :param fig_title: Figure title
    :param craft: Optional craft name, added to the file name as in the other plots
    :param bin_hours: Histogram bin width (hours)
    :param band_quantiles: Weighted quantiles drawn as the range in the bottom panel
    :param n_overlay: Number of intermediate stages drawn in the top panel
    :param save_fig: If True save to <run dir>/arrival_time_histograms_per_obs[_<craft>_assim].png
    :return: (fig, ax), or (None, None) if the input file does not exist
    """
    import matplotlib.dates as mdates

    run_dir = os.path.join(base_file_path, f"run_{runNo:03d}")
    in_file = os.path.join(run_dir, "cme_arrival_per_obs.nc")
    if not os.path.isfile(in_file):
        print(f"plot_arrival_time_histograms_per_obs: {in_file} not found "
              f"(run compute_arrival_times_per_obs first)")
        return None, None
    with xr.load_dataset(in_file) as a_ds:
        arr = np.asarray(a_ds["arrival_time"].values, dtype="datetime64[ns]")
        lw_all = a_ds["log_weight"].values.astype(float)
        computed = a_ds["computed"].values.astype(bool)
        stage_obs_time = pd.to_datetime(a_ds["stage_obs_time"].values)

    stages = np.flatnonzero(computed)
    # Arrival times as seconds (NaN for CMEs that do not arrive; NaT must not be cast to a number)
    secs = arr.astype("datetime64[s]").astype("int64").astype(float)
    secs[np.isnat(arr)] = np.nan
    if not np.any(np.isfinite(secs[stages])):
        print("plot_arrival_time_histograms_per_obs: no member arrives in any stage")
        return None, None
    t_ref_s = float(np.nanmin(secs[stages]))
    t_ref = np.datetime64(int(t_ref_s), "s")
    hours = (secs - t_ref_s) / 3600.0
    h_min = np.floor(np.nanmin(hours[stages]) / bin_hours) * bin_hours - bin_hours
    h_max = np.ceil(np.nanmax(hours[stages]) / bin_hours) * bin_hours + bin_hours
    real_h = None
    if isinstance(real_arrival_time, (datetime.datetime, np.datetime64, pd.Timestamp)) \
            and not pd.isna(real_arrival_time):
        real_h = (pd.Timestamp(real_arrival_time) - pd.Timestamp(t_ref)).total_seconds() / 3600.0
        h_min = min(h_min, np.floor(real_h / bin_hours) * bin_hours - bin_hours)
        h_max = max(h_max, np.ceil(real_h / bin_hours) * bin_hours + bin_hours)
    edges = np.arange(h_min, h_max + bin_hours / 2, bin_hours)
    to_time = lambda h: pd.Timestamp(t_ref) + pd.to_timedelta(h, unit="h")

    def stage_hist(st):
        lw = np.where(np.isfinite(lw_all[st]), lw_all[st], -np.inf)
        w = np.exp(lw - np.max(lw)) if np.any(np.isfinite(lw)) else np.ones(lw.size)
        w = w / np.sum(w)
        ok = np.isfinite(hours[st])
        frac_arrive = float(np.sum(w[ok]))
        hist = np.histogram(hours[st][ok], bins=edges, weights=w[ok])[0]
        hist = hist / hist.sum() if hist.sum() > 0 else hist
        q = weighted_quantiles(hours[st][ok], w[ok], [band_quantiles[0], 0.5, band_quantiles[1]]) \
            if np.any(ok) else np.full(3, np.nan)
        return hist, q, frac_arrive

    hists, quants, fracs = {}, {}, {}
    for st in stages:
        hists[st], quants[st], fracs[st] = stage_hist(st)
        label = "prior" if st == 0 else f"after obs {st - 1}"
        med = to_time(quants[st][1]) if np.isfinite(quants[st][1]) else "none arrive"
        print(f"Stage {st:3d} ({label}): weighted median arrival {med}, "
              f"arriving weight {100 * fracs[st]:.0f}%")

    col_prior = "#8c8b85"
    col_post = "#eb6834"
    col_real = "#1a1a19"
    centres = 0.5 * (edges[:-1] + edges[1:])
    fig, ax = plt.subplots(2, 1, figsize=(13, 11), layout="constrained",
                           gridspec_kw=dict(height_ratios=[1, 1.3]))

    # Top: overlaid histograms
    axt = ax[0]
    s_first, s_last = stages[0], stages[-1]
    mids = stages[1:-1]
    if mids.size > n_overlay:
        mids = mids[np.round(np.linspace(0, mids.size - 1, n_overlay)).astype(int)]
    cmap = plt.get_cmap("Blues")
    for j, st in enumerate(mids):
        c = cmap(0.35 + 0.6 * j / max(1, mids.size - 1))
        when = "" if pd.isna(stage_obs_time[st]) else f" ({stage_obs_time[st]:%d %b %H:%M})"
        axt.step(to_time(centres), hists[st], where="mid", color=c, lw=1.4,
                 label=f"After obs {st - 1}{when}")
    axt.fill_between(to_time(centres), hists[s_first], step="mid", color=col_prior, alpha=0.35,
                     label="Prior" if s_first == 0 else f"Stage {s_first}")
    axt.fill_between(to_time(centres), hists[s_last], step="mid", color=col_post, alpha=0.45,
                     label=f"Final (after obs {s_last - 1})" if s_last > 0 else "Final")
    if real_h is not None:
        axt.axvline(to_time(real_h), color=col_real, lw=2, ls=(0, (4, 2)), label="Observed arrival")
    axt.set_ylabel(f"Probability per {bin_hours:g} h bin", fontsize=13)
    axt.set_title("Weighted arrival-time distributions", loc="left", fontsize=14)
    axt.legend(fontsize=9.5, frameon=False, loc="upper left", ncol=2)
    axt.xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H:%M"))

    # Bottom: heat map of every stage's histogram, with weighted median and range
    axb = ax[1]
    mat = np.full((edges.size - 1, stages.size), np.nan)
    for j, st in enumerate(stages):
        mat[:, j] = np.where(hists[st] > 0, hists[st], np.nan)
    x_edges = np.arange(stages.size + 1) - 0.5
    y_edges = to_time(edges)
    pcm = axb.pcolormesh(x_edges, y_edges, mat, cmap="Blues", shading="flat",
                         vmin=0, vmax=np.nanmax(mat))
    fig.colorbar(pcm, ax=axb, label=f"Probability per {bin_hours:g} h bin", pad=0.01)
    xs = np.arange(stages.size)
    q = np.array([quants[st] for st in stages])
    axb.plot(xs, to_time(q[:, 1]), color=col_post, lw=2, marker="o", ms=3, label="Weighted median")
    axb.plot(xs, to_time(q[:, 0]), color=col_post, lw=1, ls=(0, (3, 2)),
             label=f"Weighted {int(round(100 * band_quantiles[0]))}-{int(round(100 * band_quantiles[1]))}%")
    axb.plot(xs, to_time(q[:, 2]), color=col_post, lw=1, ls=(0, (3, 2)))
    if real_h is not None:
        axb.axhline(to_time(real_h), color=col_real, lw=2, ls=(0, (4, 2)), label="Observed arrival")
    labels = ["Prior" if st == 0 else
              (f"{st - 1}\n{stage_obs_time[st]:%H:%M}" if not pd.isna(stage_obs_time[st]) else f"{st - 1}")
              for st in stages]
    step = max(1, int(np.ceil(stages.size / 25)))
    axb.set_xticks(xs[::step])
    axb.set_xticklabels(labels[::step], fontsize=8.5)
    axb.set_xlabel("Ensemble after observation (number, time UT)", fontsize=13)
    axb.set_ylabel("Arrival time (UT)", fontsize=13)
    axb.yaxis.set_major_formatter(mdates.DateFormatter("%d %b %H:%M"))
    axb.set_title("Arrival-time distribution of the ensemble at every observation", loc="right", fontsize=14)
    axb.legend(fontsize=10, frameon=False, loc="upper right")
    # Fraction of weight that arrives within the run, where some does not
    if any(fracs[st] < 0.999 for st in stages):
        trans = axb.get_xaxis_transform()
        for j, st in enumerate(stages):
            if fracs[st] < 0.999:
                axb.text(xs[j], 0.985, f"{100 * fracs[st]:.0f}%", transform=trans, fontsize=7,
                         ha="center", va="top", color="#b03030")
        axb.text(0.0, 1.0, "red %: weight of members arriving within the model run",
                 transform=axb.transAxes, fontsize=8, color="#b03030", ha="left", va="bottom")

    fig.suptitle(fig_title if fig_title else f"CME arrival times, run {runNo}", fontsize=16)
    if save_fig:
        name = "arrival_time_histograms_per_obs" + (f"_{craft}_assim" if craft is not None else "")
        out_path = os.path.join(run_dir, name + ".png")
        out_path = safe_savefig(fig, out_path, dpi=150)
        print(f"Saved arrival-time histograms to {out_path}")
        plt.close(fig)
    else:
        plt.show()
    return fig, ax


def compute_rank_pit(
        base_file_path,
        run_nos,
        obs_sd=np.sqrt(0.5),
        include_obs_error=True,
):
    """
    Rank statistics of the observations within the forecast ensemble, for every observation and run.
    The forecast for observation k is the ensemble before observation k is assimilated: elongations
    elon_first_stage[k] from ens_elongations.nc (the current particles with adaptive resampling; for
    older runs the Liu-West shrunk particles, a = 0.99, which are almost identical) with the weights
    log_weight[k] from cme_pars.nc (row k = weights after observation k - 1, row 0 = prior).
    Because the members are weighted, the rank is computed as the probability integral transform
        PIT = sum_i w_i Phi((y - h_i) / obs_sd)     (include_obs_error=True, Hamill 2001)
        PIT = sum_i w_i [1(h_i < y) + 0.5 1(h_i = y)]   (include_obs_error=False)
    PIT is uniform on [0, 1] for a reliable ensemble; the rank histogram is its histogram. An
    observation is only ranked if every member has an elongation at that time.
    :param base_file_path: Directory containing the run_<runNo> output folders
    :param run_nos: Run numbers to include
    :param obs_sd: Observation error standard deviation (deg)
    :param include_obs_error: Account for observation error when ranking (recommended)
    :return: dict with "pit" (n_runs, n_obs), "observation" (n_runs, n_obs), "obs_time" (n_obs),
        "assimilated" (n_runs, n_obs), "n_members", "run_nos", "n_skipped" (observations with NaN members)
    """
    run_nos = list(run_nos)
    records = []
    for run_no in run_nos:
        run_dir = os.path.join(base_file_path, f"run_{run_no:03d}")
        elon_file = os.path.join(run_dir, "ens_elongations.nc")
        par_file = os.path.join(run_dir, "cme_pars.nc")
        if not (os.path.isfile(elon_file) and os.path.isfile(par_file)):
            print(f"compute_rank_pit: run {run_no} skipped ({elon_file} or {par_file} not found)")
            continue
        with xr.load_dataset(elon_file) as e_ds:
            h = e_ds["elon_first_stage"].values.astype(float)  # (n_obs, n_ens)
            y = e_ds["observation"].values.astype(float)
            assim = e_ds["assimilated"].values.astype(bool)
            t = pd.to_datetime(e_ds["obs_time"].values)
        with xr.load_dataset(par_file) as p_ds:
            lw_rows = p_ds["log_weight"].values.astype(float)  # (n_obs + 1, n_ens)
        records.append((run_no, h, y, assim, t, lw_rows))

    if not records:
        return None
    n_obs = max(r[1].shape[0] for r in records)
    n_ens = records[0][1].shape[1]
    pit = np.full((len(records), n_obs), np.nan)
    obs_all = np.full((len(records), n_obs), np.nan)
    assim_all = np.zeros((len(records), n_obs), dtype=bool)
    obs_time = pd.to_datetime([pd.NaT] * n_obs)
    n_skipped = 0
    for j, (run_no, h, y, assim, t, lw_rows) in enumerate(records):
        if j == 0 or obs_time.isna().all():
            obs_time = pd.to_datetime(list(t) + [pd.NaT] * (n_obs - len(t)))
        for k in range(h.shape[0]):
            obs_all[j, k] = y[k]
            assim_all[j, k] = assim[k]
            if not np.isfinite(y[k]):
                continue
            if not np.all(np.isfinite(h[k])):
                if np.any(np.isfinite(h[k])):
                    n_skipped += 1
                continue
            lw = lw_rows[k] if k < lw_rows.shape[0] else lw_rows[-1]
            lw = np.where(np.isfinite(lw), lw, -np.inf)
            w = np.exp(lw - np.max(lw)) if np.any(np.isfinite(lw)) else np.ones(n_ens)
            w = w / np.sum(w)
            if include_obs_error:
                pit[j, k] = float(np.sum(w * st.norm.cdf((y[k] - h[k]) / obs_sd)))
            else:
                pit[j, k] = float(np.sum(w * ((h[k] < y[k]) + 0.5 * (h[k] == y[k]))))
    return dict(pit=pit, observation=obs_all, obs_time=obs_time, assimilated=assim_all,
                n_members=n_ens, run_nos=[r[0] for r in records], n_skipped=n_skipped)


def pit_to_rank(pit, n_members):
    """
    Convert the (weighted) probability integral transform to an ensemble rank 0..n_members:
    rank = floor(PIT * (n_members + 1)), capped at n_members. For equally weighted members ranked
    without observation error this is exactly the number of members below the observation (the
    classic Talagrand rank); with weights and observation error it is its natural generalisation.
    :param pit: Array of PIT values in [0, 1] (NaN allowed)
    :param n_members: Ensemble size N
    :return: Float array of ranks 0..N (NaN where pit is NaN)
    """
    pit = np.asarray(pit, dtype=float)
    rank = np.floor(pit * (n_members + 1))
    rank = np.minimum(rank, n_members)
    rank[~np.isfinite(pit)] = np.nan
    return rank


def plot_rank_histograms_per_obs(
        base_file_path,
        run_nos,
        obs_sd=np.sqrt(0.5),
        include_obs_error=True,
        fig_title=None,
        craft=None,
        n_cols=6,
):
    """
    Rank (Talagrand) histograms of the observations within the forecast ensemble: normalised count
    (fraction of cases) against rank 0..N, the number of ensemble members below the observation.
    The rank is computed from the weighted probability integral transform (see compute_rank_pit and
    pit_to_rank), so it accounts for unequal member weights and, if include_obs_error, observation error.
      - rank_histograms_per_obs: one rank histogram per observation time, over the runs in run_nos
        (with a single run each panel holds one entry, marking where the observation fell);
      - rank_histogram_summary: the rank histogram pooled over all observations and runs, and the rank
        of each observation against time.
    The dashed line, 1 / (N + 1), is the normalised count expected for a reliable ensemble. A U shape
    means the ensemble is too narrow, a dome too wide, and a slope a bias: here a low rank means the
    observed elongation is below the ensemble (modelled CMEs ahead of / faster than the observed one).
    Figures are saved in the run folder for a single run, otherwise in base_file_path, and the ranks
    are written to rank_pit_per_obs.csv alongside.
    :param base_file_path: Directory containing the run_<runNo> output folders
    :param run_nos: Run numbers to include
    :param obs_sd: Observation error standard deviation (deg)
    :param include_obs_error: Account for observation error when ranking
    :param fig_title: Figure title
    :param craft: Optional craft name, added to the file names as in the other plots
    :param n_cols: Columns of the per-observation panel grid
    :return: dict from compute_rank_pit with an added "rank" array, or None
    """
    import matplotlib.dates as mdates
    from matplotlib.ticker import MaxNLocator

    res = compute_rank_pit(base_file_path, run_nos, obs_sd=obs_sd, include_obs_error=include_obs_error)
    if res is None:
        print("plot_rank_histograms_per_obs: no runs with ens_elongations.nc and cme_pars.nc")
        return None
    n_mem = res["n_members"]
    pit, obs_time = res["pit"], res["obs_time"]
    rank = pit_to_rank(pit, n_mem)
    res["rank"] = rank
    n_runs, n_obs = pit.shape
    ranked = np.flatnonzero(np.any(np.isfinite(rank), axis=0))
    if ranked.size == 0:
        print("plot_rank_histograms_per_obs: no observation could be ranked")
        return res

    out_dir = (os.path.join(base_file_path, f"run_{res['run_nos'][0]:03d}") if len(res["run_nos"]) == 1
               else base_file_path)
    suffix = f"_{craft}_assim" if craft is not None else ""
    rank_label = (f"Rank (members below observation, 0-{n_mem})"
                  + (", with obs. error" if include_obs_error else ""))
    col_bar = "#2a78d6"
    col_exp = "#1a1a19"
    col_grid = "#d9d8d2"
    ranks_axis = np.arange(n_mem + 1)
    expected = 1.0 / (n_mem + 1)

    def norm_counts(vals):
        counts = np.bincount(vals.astype(int), minlength=n_mem + 1)[:n_mem + 1].astype(float)
        return counts / counts.sum() if counts.sum() > 0 else counts

    # Ranks to a csv for reference
    rows = []
    for j, run_no in enumerate(res["run_nos"]):
        for k in range(n_obs):
            rows.append(dict(run=run_no, obs_no=k, obs_time=obs_time[k], observation=res["observation"][j, k],
                             assimilated=bool(res["assimilated"][j, k]), pit=pit[j, k], rank=rank[j, k]))
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "rank_pit_per_obs.csv"), index=False)

    # 1. One rank histogram per observation time
    n_rows = int(np.ceil(ranked.size / n_cols))
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(2.6 * n_cols, 2.2 * n_rows + 1.2),
                            sharex=True, sharey=True, layout="constrained", squeeze=False)
    for i, k in enumerate(ranked):
        ax = axs.flat[i]
        vals = rank[:, k][np.isfinite(rank[:, k])]
        ax.bar(ranks_axis, norm_counts(vals), width=0.9, color=col_bar)
        ax.axhline(expected, color=col_exp, lw=1, ls=(0, (4, 2)))
        when = "" if pd.isna(obs_time[k]) else f" {obs_time[k]:%d %b %H:%M}"
        y_txt = np.nanmean(res["observation"][:, k])
        ax.set_title(f"obs {k}{when}\n{y_txt:.2f}°, n={vals.size}", fontsize=9, loc="left")
        ax.grid(True, axis="y", color=col_grid, lw=0.5)
        ax.set_axisbelow(True)
    for ax in axs.flat[ranked.size:]:
        ax.set_visible(False)
    for ax in axs[-1, :]:
        ax.set_xlabel("Rank", fontsize=9)
        ax.set_xlim(-0.6, n_mem + 0.6)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    for ax in axs[:, 0]:
        ax.set_ylabel("Normalised count", fontsize=9)
    runs_txt = f"{len(res['run_nos'])} run" + ("s" if len(res["run_nos"]) > 1 else "")
    fig.suptitle((fig_title + ": " if fig_title else "") +
                 f"rank histogram at each observation time ({runs_txt}, {n_mem} members; "
                 f"dashed = reliable, 1/{n_mem + 1})", fontsize=13)
    safe_savefig(fig, os.path.join(out_dir, f"rank_histograms_per_obs{suffix}.png"), dpi=130)
    plt.close(fig)

    # 2. Pooled rank histogram, and rank against observation time
    fig, ax = plt.subplots(1, 2, figsize=(14, 5), layout="constrained", gridspec_kw=dict(width_ratios=[1, 1.6]))
    vals = rank[np.isfinite(rank)]
    ax[0].bar(ranks_axis, norm_counts(vals), width=0.9, color=col_bar)
    ax[0].axhline(expected, color=col_exp, lw=1.2, ls=(0, (4, 2)), label=f"Reliable ensemble (1/{n_mem + 1})")
    ax[0].set_xlabel(rank_label, fontsize=11)
    ax[0].set_ylabel("Normalised count", fontsize=11)
    ax[0].set_xlim(-0.6, n_mem + 0.6)
    ax[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    ax[0].set_title(f"Pooled over {np.isfinite(rank).any(axis=0).sum()} observations, {runs_txt} "
                    f"(n = {vals.size})", loc="left", fontsize=11)
    ax[0].legend(frameon=False, fontsize=9)
    ax[0].grid(True, axis="y", color=col_grid, lw=0.5)
    ax[0].set_axisbelow(True)

    t_ok = not obs_time[ranked].isna().any()
    x_all = obs_time if t_ok else np.arange(n_obs)
    ax[1].axhspan(0.05 * (n_mem + 1) - 0.5, 0.95 * (n_mem + 1) - 0.5, color="#ecebe6", zorder=0,
                  label="Central 90% of a reliable ensemble")
    ax[1].axhline(n_mem / 2, color=col_grid, lw=1, zorder=0)
    cmap = plt.get_cmap("tab10")
    for j, run_no in enumerate(res["run_nos"]):
        ok = np.isfinite(rank[j])
        ax[1].plot(np.asarray(x_all)[ok], rank[j, ok], marker="o", ms=4, lw=0.8,
                   color=col_bar if n_runs == 1 else cmap(j % 10), alpha=0.9 if n_runs <= 10 else 0.4,
                   label=f"run {run_no}" if n_runs <= 10 else None)
    ax[1].set_ylim(-0.5, n_mem + 0.5)
    ax[1].yaxis.set_major_locator(MaxNLocator(integer=True))
    ax[1].set_ylabel(rank_label, fontsize=11)
    ax[1].set_xlabel("Observation time (UT)" if t_ok else "Observation number", fontsize=11)
    if t_ok:
        ax[1].xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H:%M"))
    ax[1].set_title(f"Rank of each observation (0: below all members, {n_mem}: above all members)",
                    loc="left", fontsize=11)
    ax[1].legend(frameon=False, fontsize=9, loc="best")
    ax[1].grid(True, color=col_grid, lw=0.5)
    ax[1].set_axisbelow(True)
    if res["n_skipped"]:
        ax[1].text(0.0, -0.18, f"{res['n_skipped']} observation(s) not ranked: some members had no elongation",
                   transform=ax[1].transAxes, fontsize=8, color="#b03030")
    fig.suptitle((fig_title + ": " if fig_title else "") + "rank histogram of the forecast ensemble", fontsize=13)
    safe_savefig(fig, os.path.join(out_dir, f"rank_histogram_summary{suffix}.png"), dpi=140)
    plt.close(fig)
    print(f"Saved rank histograms to {out_dir}")
    return res


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
    # Compute and plot the ensemble's full elongation profiles at every observation (one extra
    #  ensemble of model runs per distinct ensemble, i.e. per resampling step)
    make_profiles_per_obs = True
    # Compute every member's CME arrival time for the ensemble at every observation and plot the
    #  distributions as histograms. The model runs for arrival_sim_time_days from its start time; CMEs
    #  arriving later count as misses (4 days covers arrivals up to ~4 days after the model start).
    make_arrivals_per_obs = False
    arrival_sim_time_days = 4.0
    # Rank (Talagrand) histograms of the observations within the forecast ensemble
    make_rank_histograms = True
    bias_folder_name = "bias5_2.5_bias21_2.5"
    n_ens = 20

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
            "compress_huxt", "test3", "WSA_v", f"ens_{n_ens}", "mo_cone",
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
                yTruth=t_initTruth,  # (was vTruth: the speed truth was drawn on the launch-time plot)
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
                yTruth=latTruth,  # (was lonTruth)
                runNo=runNo,
                parUnits=r"$^\circ$",
                n_obs=n_obs,
                base_file_path=baseFilePath,
                craft=craft,
            )

            # Prior/posterior ensemble elongation profiles against the observations
            #  (obs_sd = sqrt(obs_cov); obs_cov = [0.5] deg^2 in da_surf_setup)
            plot_elongation_profiles(
                base_file_path=baseFilePath,
                runNo=runNo,
                obs_sd=np.sqrt(0.5),
                fig_title=fig_title,
                craft=craft,
            )

            # Full 3-day prior/posterior ensemble elongation profiles against the observations
            plot_full_elongation_profiles(
                base_file_path=baseFilePath,
                runNo=runNo,
                obs_sd=np.sqrt(0.5),
                fig_title=fig_title,
                craft=craft,
            )

            # Full elongation profiles of the ensemble at every observation (reruns the model for
            #  each distinct ensemble; the result is cached in ens_elongation_profiles_per_obs.nc)
            if make_profiles_per_obs:
                compute_elongation_profiles_per_obs(base_file_path=baseFilePath, runNo=runNo)
                plot_elongation_profiles_per_obs(
                    base_file_path=baseFilePath,
                    runNo=runNo,
                    obs_sd=np.sqrt(0.5),
                    fig_title=fig_title,
                    craft=craft,
                )

            # CME arrival times of the ensemble at every observation, collated into histograms
            #  (one ensemble of model runs per distinct ensemble; cached in cme_arrival_per_obs.nc)
            if make_arrivals_per_obs:
                compute_arrival_times_per_obs(
                    base_file_path=baseFilePath,
                    runNo=runNo,
                    use_model=use_model,
                    cme_hit_object="EARTH",
                    sim_time_days=arrival_sim_time_days,
                )
                plot_arrival_time_histograms_per_obs(
                    base_file_path=baseFilePath,
                    runNo=runNo,
                    real_arrival_time=real_arrival_time,
                    fig_title=fig_title,
                    craft=craft,
                )

        # Rank histograms of the observations within the forecast ensemble, at each observation time
        #  (over all runs) and pooled
        if make_rank_histograms:
            plot_rank_histograms_per_obs(
                base_file_path=baseFilePath,
                run_nos=list(range(start_run, start_run + nRuns)),
                obs_sd=np.sqrt(0.5),
                fig_title=fig_title,
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