import os
import argparse
import sys
import traceback
import json

from astropy.time import Time
import astropy.units as u
import matplotlib.pyplot as plt
import datetime
import numpy as np

import huxt.huxt as H
import surf.surf as S
import xarray as xr
from sir_observations import Observations
from output_names import (
    bias_dir_name,
    true_bias_suffix,
    twin_sampling_dir_name,
    read_path,
    write_path,
    check_path_length,
    path_too_long,
    resolve_output_path,
    PathLimits,
)
import scipy.stats as st

import sunpy.coordinates.sun as sn
import pandas as pd
from multiprocessing import Pool

from da_surf_setup import (
    load_environment,
    get_event,
    get_twin_cme,
    is_twin_experiment,
    event_craft,
    load_events,
    events_file,
)
from sir_observation_operator import available_cpus
from cme_arrival import (
    get_arrival_cme,
    get_arrival_speed_from_front,
    get_arrival_time_and_speed,
)


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
        print(
            f"Warning: could not write {out_path} ({err}); it may be open in another program. "
            f"Saving to {alt_path} instead."
        )
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
    parUnits="",
    n_obs=24,
    base_file_path=None,
):
    if (obs_times is not None) and (tick_cadence is not None):
        assert tick_cadence > 0

    fig, ax = plt.subplots(figsize=(10, 6), nrows=1, ncols=1, layout="constrained")
    xPlot = ds[xVarName][:].values

    yTrue = yTruth
    # ds may hold several runs (run, obs, member): take run runNo, by its run number
    if "run" in ds[yVarName].dims:
        run_values = list(np.ravel(ds["run"].values))
        ds_run = ds.isel(run=run_values.index(runNo) if runNo in run_values else 0)
    else:
        ds_run = ds
    yPlot = ds_run[yVarName].values
    weights = np.exp(ds_run["log_weight"].values)

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
            ax.plot(
                xPlot, yPlot[:, ir], color="pink", alpha=0.5, label=f"{parNameLegend}"
            )
        else:
            ax.plot(xPlot, yPlot[:, ir], color="pink", alpha=0.5)
    ax.plot(xPlot, y_median, color="red", label=f"Median {parNameLegend}")
    ax.plot(xPlot, y_mean, color="blue", label=f"Mean {parNameLegend}")
    if yTruth is not None:
        ax.plot(
            [xPlot.min(), xPlot.max()],
            [yTrue, yTrue],
            color="k",
            linestyle="dashed",
            label=f"True {parNameLegend}",
        )
    ax.legend(fontsize=16)

    ax.set_title(f"{fig_title}", fontsize=24)
    ax.set_xlim(0, n_obs)
    ax.set_xlabel(f"{xLabel}", fontsize=20)
    ax.set_ylabel(f"{parNameLegend} ({parUnits})", fontsize=20)
    ax.tick_params(axis="both", which="major", labelsize=16)
    if (obs_times is not None) and (tick_cadence is not None):
        ax.set_xticks(ind_req)
        ax.set_xticklabels(x_ticks, rotation=30)

    if base_file_path is None:
        plt.show()
    else:
        if craft is None:
            filePath = os.path.join(
                base_file_path,
                f"run_{runNo:03d}",
                f"{parNameLegend.replace(' ', '_')}_vs_obs_no.png",
            )
        else:
            filePath = os.path.join(
                base_file_path,
                f"run_{runNo:03d}",
                f"{parNameLegend.replace(' ', '_')}_vs_obs_no_{craft}_assim.png",
            )
        safe_savefig(plt.gcf(), filePath)

    return None


def get_prior_post_cme_pars_from_ds(ds, runNo=0):
    """
    Function to retrieve prior and posterior cme parameters: [t_init, v, width, lon, lat, thick]
      from dataset (and the posterior logarithm of the weights)
    :param ds: Dataset containing all CME parameters from data assimilation run
    :param runNo: Run index to extract if ds contains multiple runs
    :return: prior_dict, Dictionary containing the prior CME parameters extracted
    :return: posterior_dict, Dictionary containing the posterior CME parameters extracted plus the log_weights
    """
    # Get the number of ensemble members from ds
    n_members = len(ds["ens_no"].values)

    # Prior (first row) and posterior (last row) of each variable; ds may hold several runs
    #  (run, obs, member), in which case run runNo is taken
    names = ["t_init", "v", "width", "lon", "lat", "thick"]
    # (ds[k][0, :] does not fail on several runs, so the run dimension is checked, not caught)
    if "run" in ds.dims:
        run_values = list(np.ravel(ds["run"].values))
        ds = ds.isel(run=run_values.index(runNo) if runNo in run_values else 0)
    prior = {k: ds[k].values[0, :] for k in names}
    post = {k: ds[k].values[-1, :] for k in names}
    log_weights_in = ds["log_weight"].values[-1, :]
    (
        prior_t_init_in,
        prior_v_in,
        prior_width_in,
        prior_lon_in,
        prior_lat_in,
        prior_thick_in,
    ) = (prior[k] for k in names)
    (
        post_t_init_in,
        post_v_in,
        post_width_in,
        post_lon_in,
        post_lat_in,
        post_thick_in,
    ) = (post[k] for k in names)

    sec_to_cme_prior = [float(prior_t_init_in[i]) * u.s for i in range(n_members)]
    v_prior = [prior_v_in[i] * u.km / u.s for i in range(n_members)]
    width_prior = [prior_width_in[i] * u.deg for i in range(n_members)]
    lon_prior = [prior_lon_in[i] * u.deg for i in range(n_members)]
    lat_prior = [prior_lat_in[i] * u.deg for i in range(n_members)]
    thick_prior = [prior_thick_in[i] * u.solRad for i in range(n_members)]

    sec_to_cme_post = [float(post_t_init_in[i]) * u.s for i in range(n_members)]
    v_post = [post_v_in[i] * u.km / u.s for i in range(n_members)]
    width_post = [post_width_in[i] * u.deg for i in range(n_members)]
    lon_post = [post_lon_in[i] * u.deg for i in range(n_members)]
    lat_post = [post_lat_in[i] * u.deg for i in range(n_members)]
    thick_post = [post_thick_in[i] * u.solRad for i in range(n_members)]
    log_weights_post = [log_weights_in[i] for i in range(n_members)]
    weights_post = np.array([np.exp(log_weights_post[i]) for i in range(n_members)])

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


def get_true_cme_arrival_time(
    ds,
    use_model,
    true_cme_t_init,
    true_cme_speed,
    true_cme_width,
    true_cme_lat,
    true_cme_lon,
    true_cme_thick,
    cme_hit_object="EARTH",
    sim_time=5.0 * u.day,
):
    """
    Arrival time and speed at cme_hit_object of the true CME of a twin experiment, from a model set up
    as for the ensemble's arrival times (make_arrival_model: the run's ambient solar wind, longitudes
    300-420 deg and, for compress_surf3d, the SURF3d latitude range), so the two are comparable
    :param ds: The run's cme_pars.nc (several runs concatenated along "run": the first is used)
    :param use_model: "surf", "compress_surf", "compress_surf3d" or "huxt"
    :param true_cme_t_init: Time the true CME is at cme_init_rad
    :param true_cme_speed, true_cme_width, true_cme_lat, true_cme_lon, true_cme_thick: True CME (km/s,
        deg, deg, deg, solar radii)
    :param cme_hit_object: Body to compute the arrival at
    :param sim_time: Model run length (a CME arriving later counts as a miss)
    :return: (arrival datetime or NaN, arrival speed km/s or NaN)
    """
    if use_model not in ["surf", "compress_surf", "compress_surf3d", "huxt"]:
        sys.exit(
            "Unknown use_model name, expected 'surf', 'compress_surf', 'compress_surf3d' or 'huxt'"
        )
    ds_run = ds.isel(run=0) if "run" in ds.dims else ds
    model, start_time = make_arrival_model(
        ds_run, use_model, sim_time=sim_time, dt_scale=20
    )
    print(f"Model start: {start_time}")

    # Generate CME object
    v_true = float(true_cme_speed) * u.km / u.s
    # Launch at r_min, not at cme_init_rad (see get_transit_time_to_r_min)
    sec_to_cme_true = float(
        (true_cme_t_init - start_time).total_seconds()
    ) * u.s + get_transit_time_to_r_min(ds_run, v_true)
    width_true = float(true_cme_width) * u.deg
    lon_true = float(true_cme_lon) * u.deg
    lat_true = float(true_cme_lat) * u.deg
    thick_true = float(true_cme_thick) * u.solRad

    if use_model in ["surf", "compress_surf", "compress_surf3d"]:
        cme_object: S.ConeCME = S.ConeCME(
            t_launch=sec_to_cme_true,
            v=v_true,
            width=width_true,
            longitude=lon_true,
            latitude=lat_true,
            thickness=thick_true,
            cme_fixed_duration=True,
            fixed_duration=12 * 3600 * u.s,
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
            fixed_duration=12 * 3600 * u.s,
        )

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
    cme_arrival_prior, cme_arr_prior_speed, prior_count_hit, prior_stats = (
        get_arrival_time_and_speed(model, use_model, cme_hit_object)
    )

    # Make posterior CME arrival times
    model.solve([post_cme])
    cme_arrival_post, cme_arr_post_speed, post_count_hit, post_stats = (
        get_arrival_time_and_speed(model, use_model, cme_hit_object)
    )

    print(
        f"cme_arrival_prior={cme_arrival_prior},\ncme_arr_prior_speed={cme_arr_prior_speed},\n"
        f" prior_count_hit={prior_count_hit},\n prior_stats={prior_stats},\n,"
        f"cme_arrival_post={cme_arrival_post},\n cme_arr_post_speed={cme_arr_post_speed},\n"
        f" post_count_hit={post_count_hit},\n post_stats={post_stats}"
    )
    return (
        cme_arrival_prior,
        cme_arr_prior_speed,
        prior_count_hit,
        prior_stats,
        cme_arrival_post,
        cme_arr_post_speed,
        post_count_hit,
        post_stats,
    )


def get_cme_arrival_time(
    ds,
    use_model,
    runNo=0,
    real_arrival_time=None,
    cme_hit_object="EARTH",
    base_file_path=None,
    parallel_ens=False,
    sim_time_days=4.0,
    event=None,
):
    """
    Prior and posterior CME arrival times at cme_hit_object of run runNo: statistics, a histogram of
    the arrival-time errors and cme_arrival_stats.json in the run folder. The arrival times are read
    from the run's arr_obs.nc (stage 0 = prior, last stage = posterior); compute_arrival_times_per_obs
    computes those two stages only if they are not there yet, so the model is not rerun for them.
    :param ds: Not used (the run is read from base_file_path/run_<runNo>); kept for existing calls
    :param use_model: Model used in the run
    :param runNo: Run number
    :param real_arrival_time: Observed (or true) arrival time, for the errors
    :param cme_hit_object: Body for the arrivals
    :param base_file_path: Directory containing the run_<runNo> output folders
    :param parallel_ens: Solve the members in parallel where arrivals must be computed
    :param sim_time_days: Model run length (days) of the arrival runs, as for
        compute_arrival_times_per_obs (CMEs arriving later count as misses)
    :param event: Event or twin experiment of the run (see compute_profiles_and_arrivals_per_obs)
    """
    if base_file_path is None:
        raise ValueError(
            "get_cme_arrival_time needs base_file_path (the folder of the run_NNN folders)"
        )
    run_dir = os.path.join(base_file_path, f"run_{runNo:03d}")
    par_ds = xr.load_dataset(os.path.join(run_dir, "cme_pars.nc"))
    n_members = par_ds.sizes["n_ens"]
    final_stage = par_ds.sizes["n_obs"] - 1

    arr_file = compute_arrival_times_per_obs(
        base_file_path,
        runNo=runNo,
        use_model=use_model,
        stages=[0, final_stage],
        cme_hit_object=cme_hit_object,
        sim_time_days=sim_time_days,
        parallel=parallel_ens,
        event=event,
    )
    if arr_file is None:
        print(f"get_cme_arrival_time: no arrival times for run {runNo}")
        return None
    with xr.load_dataset(arr_file) as arr_ds:
        arr_times = arr_ds["arrival_time"].values
        arr_speeds = np.asarray(arr_ds["arrival_speed"].values, dtype=float)

    def stage_arrivals(stage):
        times = [
            (None if np.isnat(t) else pd.Timestamp(t).to_pydatetime())
            for t in arr_times[stage]
        ]
        return times, list(arr_speeds[stage])

    cme_arrival_prior, cme_arr_prior_speed = stage_arrivals(0)
    cme_arrival_post, cme_arr_post_speed = stage_arrivals(final_stage)
    # Posterior weights (normalised by renorm_weights_post below)
    log_w_post = np.asarray(par_ds["log_weight"].values[final_stage], dtype=float)
    weights_post = np.exp(
        np.where(np.isfinite(log_w_post), log_w_post, -np.inf) - np.nanmax(log_w_post)
    )
    prior_hit_rate = np.mean([t is not None for t in cme_arrival_prior])
    post_hit_rate = np.mean([t is not None for t in cme_arrival_post])

    print(f"Cme_arrival_prior = {cme_arrival_prior}")
    print(f"Cme_arr_prior_speed = {cme_arr_prior_speed}")
    print(f"cme_arrival_post = {cme_arrival_post}")
    print(f"cme_arr_post_speed = {cme_arr_post_speed}")

    base_time = datetime.datetime(2000, 1, 1, 0, 0, 0)
    seconds_arrival_prior = np.array(
        [
            (
                (cme_arrival_prior[i] - base_time).total_seconds()
                if isinstance(cme_arrival_prior[i], datetime.datetime)
                else np.nan
            )
            for i in range(n_members)
        ]
    )
    seconds_arrival_post = np.array(
        [
            (
                (cme_arrival_post[i] - base_time).total_seconds()
                if isinstance(cme_arrival_post[i], datetime.datetime)
                else np.nan
            )
            for i in range(n_members)
        ]
    )
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

        mean_prior_arrival = base_time + datetime.timedelta(seconds=mean_prior_arr_sec)
        median_prior_arrival = base_time + datetime.timedelta(
            seconds=median_prior_arr_sec
        )
        sd_prior_arr_speed = np.nanstd(cme_arr_prior_speed)

        print(
            f"\nMean_prior_arrival = {mean_prior_arrival} +/- {sd_prior_arr_hr} hours"
        )
        print(
            f"Mean_prior_speed = {mean_prior_arr_speed} +/- {sd_prior_arr_speed} km/s"
        )
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

        mean_post_arrival = base_time + datetime.timedelta(seconds=mean_post_arr_sec)
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
            weighted_mean_post_arr_sec = np.average(
                seconds_arrival_post[ind_req], weights=renorm_weights_post
            )
            weighted_mean_post_arrival = base_time + datetime.timedelta(
                seconds=weighted_mean_post_arr_sec
            )

            weighted_median_post_arr_sec = weighted_median(
                seconds_arrival_post[ind_req], weights=renorm_weights_post
            )

            weighted_median_post_arrival = base_time + datetime.timedelta(
                seconds=weighted_median_post_arr_sec
            )

            weighted_mean_post_arr_speed = np.average(
                cme_arr_post_speed[ind_req], weights=renorm_weights_post
            )
            weighted_median_post_arr_speed = weighted_median(
                cme_arr_post_speed[ind_req], weights=renorm_weights_post
            )
            # Calculate (biased) weighted posterior variance
            weighted_var_post_arr_sec_biased = np.average(
                (seconds_arrival_post[ind_req] - weighted_mean_post_arr_sec) ** 2,
                weights=renorm_weights_post,
            )
            weighted_var_post_arr_speed_biased = np.average(
                (cme_arr_post_speed[ind_req] - weighted_mean_post_arr_speed) ** 2,
                weights=renorm_weights_post,
            )

            # Transform to unbiased variance estimator. For normalised (reliability) weights the
            #  correction is 1 / (1 - sum(w^2)); n / (n - 1) is only right for equal weights.
            sum_w2 = np.sum(renorm_weights_post**2)
            b_factor = 1.0 / (1.0 - sum_w2) if sum_w2 < 1.0 else np.nan
            weighted_var_post_arr_sec_unbiased = (
                b_factor * weighted_var_post_arr_sec_biased
            )
            weighted_var_post_arr_speed_unbiased = (
                b_factor * weighted_var_post_arr_speed_biased
            )

            # Calculate weighted standard deviation
            weighted_sd_post_arr_sec = np.sqrt(weighted_var_post_arr_sec_unbiased)
            weighted_sd_post_arr_hr = weighted_sd_post_arr_sec / 3600.0
            weighted_sd_post_arr_speed = np.sqrt(weighted_var_post_arr_speed_unbiased)
            ##################################################################

            print(
                f"\nWeighted_mean_post_arrival = {weighted_mean_post_arrival} +/- {weighted_sd_post_arr_hr} hours"
            )
            print(
                f"Weighted_mean_post_speed = {weighted_mean_post_arr_speed} +/- {weighted_sd_post_arr_speed} km/s"
            )
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
            for pri_arr in cme_arrival_prior
            if isinstance(pri_arr, datetime.datetime)
        ]
        post_arrival_error_hours = [
            ((post_arr - real_arrival_time).total_seconds()) / 3600.0
            for post_arr in cme_arrival_post
            if isinstance(post_arr, datetime.datetime)
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
            print(
                f"weights_post_plot - renorm_weights_post = {weights_post_plot-renorm_weights_post}"
            )

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
        print(
            f"\nPrior mean_arrival_time_error = {mean_prior_arrival_error_hours} hours"
        )
        print(f"Post mean_arrival_time_error = {mean_post_arrival_error_hours} hours")
        print(
            f"\nPrior Median_arrival_time_error = {median_prior_arrival_error_hours} hours"
        )
        print(
            f"Post Median_arrival_time_error = {median_post_arrival_error_hours} hours"
        )
        print(
            f"\nWeighted Mean_arrival_time_error = {weighted_mean_post_arrival_error_hours} hours"
        )
        print(
            f"Weighted Median_arrival_time_error = {weighted_median_post_arrival_error_hours} hours"
        )

        max_y_axis = 10 * np.ceil(n_members / 40)
        fig, ax = plt.subplots(
            figsize=(16, 12), nrows=2, ncols=1, sharex=True, layout="constrained"
        )
        ax[0].hist(prior_arrival_error_hours, bins=np.arange(-10, 20.5, 1))
        ax[0].set_title(f"Prior CME Arrival Error at {cme_hit_object}", fontsize=24)
        ax[0].set_ylim([0, max_y_axis])
        ax[0].tick_params(axis="both", which="major", labelsize=16)
        ax[0].set_ylabel("No. of particles", fontsize=20)

        ax[1].hist(post_arrival_error_hours, bins=np.arange(-10, 20.5, 1))
        ax[1].set_title(f"Posterior CME Arrival Error at {cme_hit_object}", fontsize=24)
        ax[1].set_ylim([0, max_y_axis])
        ax[1].set_ylabel("No. of particles", fontsize=20)
        ax[1].tick_params(axis="both", which="major", labelsize=16)
        ax[-1].set_xlabel("Arrival time error (hours)", fontsize=20)

        if base_file_path is None:
            plt.show()
        else:
            out_dir_path = os.path.join(base_file_path, f"run_{runNo:03d}")
            filePath = os.path.join(out_dir_path, "histogram_arrival_time_error.png")
            safe_savefig(plt.gcf(), filePath)

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
        print(
            f"plot_elongation_profiles: {elon_file} not found (run predates elongation output?)"
        )
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
        figsize=(12, 8),
        nrows=2,
        ncols=1,
        sharex=True,
        sharey=True,
        layout="constrained",
    )

    panels = [
        (
            ax[0],
            elon_prior,
            prior_median,
            col_prior,
            "Prior",
            "Prior member",
            "Prior median",
        ),
        (
            ax[1],
            elon_post,
            post_wmedian,
            col_post,
            "Posterior",
            "Posterior member",
            "Posterior weighted median",
        ),
    ]
    for axi, elon, summary, col, panel_name, member_label, summary_label in panels:
        # Individual members: thin, translucent lines (one legend entry for all of them)
        for m in range(n_ens):
            axi.plot(
                obs_time,
                elon[:, m],
                color=col,
                linewidth=0.8,
                alpha=0.25,
                label=member_label if m == 0 else None,
                zorder=1,
            )
        # Ensemble summary: 2 px line with markers
        axi.plot(
            obs_time,
            summary,
            color=col,
            linewidth=2.0,
            marker="o",
            markersize=4,
            label=summary_label,
            zorder=2,
        )
        # Observations: dark markers (hollow where the observation was not assimilated)
        if obs_sd is not None:
            axi.errorbar(
                obs_time[assimilated],
                observation[assimilated],
                yerr=obs_sd,
                fmt="none",
                ecolor=col_obs,
                elinewidth=1.0,
                capsize=2,
                zorder=3,
            )
        axi.plot(
            obs_time[assimilated],
            observation[assimilated],
            linestyle="none",
            marker="o",
            markersize=7,
            markerfacecolor=col_obs,
            markeredgecolor="white",
            markeredgewidth=1.0,
            label="Observation (assimilated)",
            zorder=4,
        )
        if np.any(~assimilated):
            axi.plot(
                obs_time[~assimilated],
                observation[~assimilated],
                linestyle="none",
                marker="o",
                markersize=7,
                markerfacecolor="none",
                markeredgecolor=col_obs,
                markeredgewidth=1.0,
                label="Observation (not assimilated)",
                zorder=4,
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


def plot_bias_evolution(
    base_file_path,
    runNo=-1,
    fig_title=None,
    craft=None,
    band_quantiles=(0.05, 0.95),
    save_fig=True,
):
    """
    How the observation-bias coefficient beta (obs_bias.py) evolves through the assimilation window.

    Reads <run dir>/cme_pars.nc (bias and log_weight of every member for the prior and after each
    observation, and the bias settings in its attributes) and <run dir>/ens_elongations.nc
    (observation times and elongations, which observations were assimilated and which caused
    resampling, and the ESS of the updated weights). Panels:
      (a) beta of every member (dot area ~ weight), weighted median and band, the prior mean +/- sd
          and, in twin experiments, the injected bias;
      (b) only for the "linear"/"quadratic" predictors: the correction beta * p(y) the bias makes at
          each observed elongation y (deg), i.e. beta converted to elongation;
      (c) weighted sd of beta relative to the prior sd, and the ESS (fraction of members) of the
          updated weights at each observation.
    Only the prior and the ensembles after assimilated observations are shown (an observation that
    was not assimilated leaves the ensemble unchanged). Dotted vertical lines mark resampling steps.

    :param base_file_path: Directory containing the run_<runNo> output folders
    :param runNo: Run number
    :param fig_title: Figure title
    :param craft: Spacecraft whose observations were assimilated (for the file name)
    :param band_quantiles: Weighted quantiles of the shaded band
    :param save_fig: Save to <run dir>/bias_evolution[_<craft>_assim].png (else show)
    :return: (fig, axes), or (None, None) if the run has no estimated bias term
    """
    import matplotlib.dates as mdates
    from obs_bias import bias_predictor, bias_units

    run_dir = os.path.join(base_file_path, f"run_{runNo:03d}")
    par_file = os.path.join(run_dir, "cme_pars.nc")
    elon_file = os.path.join(run_dir, "ens_elongations.nc")
    for f in [par_file, elon_file]:
        if not os.path.isfile(f):
            print(f"plot_bias_evolution: {f} not found")
            return None, None

    with xr.load_dataset(par_file) as p_ds:
        if "bias" not in p_ds:
            print(
                f"plot_bias_evolution: {par_file} has no bias variable (run without a bias term)"
            )
            return None, None
        beta_all = p_ds["bias"].values.astype(
            float
        )  # (n_obs + 1, n_ens): prior, after obs k
        lw_all = p_ds["log_weight"].values.astype(float)
        attrs = dict(p_ds.attrs)
    with xr.load_dataset(elon_file) as e_ds:
        obs_time = pd.to_datetime(e_ds["obs_time"].values)
        obs = e_ds["observation"].values.astype(float)
        assimilated = e_ds["assimilated"].values.astype(bool)
        resampled = (
            e_ds["resampled"].values.astype(bool)
            if "resampled" in e_ds
            else np.zeros_like(assimilated)
        )
        ess_upd = (
            e_ds["ess_updated"].values.astype(float)
            if "ess_updated" in e_ds
            else np.full(len(obs), np.nan)
        )

    mode = str(attrs.get("bias_mode", "estimate"))
    if mode != "estimate":
        print(
            f"plot_bias_evolution: bias mode is '{mode}', so beta is not estimated; nothing to plot"
        )
        return None, None
    predictor = str(attrs.get("bias_predictor", "constant"))
    e_ref = attrs.get("bias_e_ref_deg", np.nan)
    e_scale = attrs.get("bias_e_scale_deg", np.nan)
    pred_kw = dict(predictor=predictor)
    if predictor != "constant":
        pred_kw.update(e_ref=float(e_ref), e_scale=float(e_scale))
    units = bias_units(**pred_kw)
    prior_mean = float(attrs.get("bias_prior_mean", np.nan))
    prior_sd = float(attrs.get("bias_prior_sd", np.nan))
    true_bias = float(attrs.get("true_bias", np.nan))

    n_obs = min(len(obs_time), beta_all.shape[0] - 1)
    n_ens = beta_all.shape[1]
    k_assim = np.flatnonzero(assimilated[:n_obs])
    if k_assim.size == 0:
        print("plot_bias_evolution: no observation was assimilated")
        return None, None

    # Stages shown: prior (row 0), then the ensemble after each assimilated observation k (row k + 1),
    #  the prior placed one observation spacing before the first assimilated observation
    t_assim = obs_time[k_assim]
    dt = (
        np.median(np.diff(t_assim.values))
        if k_assim.size > 1
        else np.timedelta64(1, "h")
    )
    x_t = np.concatenate([[t_assim.values[0] - dt], t_assim.values])
    rows = np.concatenate([[0], k_assim + 1])
    beta = beta_all[rows, :]
    lw = lw_all[rows, :]
    w = np.exp(
        lw - np.nanmax(np.where(np.isfinite(lw), lw, -np.inf), axis=1, keepdims=True)
    )
    w = np.where(np.isfinite(w), w, 0.0)
    w = w / np.sum(w, axis=1, keepdims=True)

    q = np.array(
        [
            weighted_quantiles(
                beta[i], w[i], [band_quantiles[0], 0.5, band_quantiles[1]]
            )
            for i in range(len(rows))
        ]
    )
    mean = np.sum(w * beta, axis=1)
    sd = np.sqrt(np.sum(w * (beta - mean[:, None]) ** 2, axis=1))

    show_corr = predictor != "constant"
    n_panels = 3 if show_corr else 2
    fig, axes = plt.subplots(
        n_panels,
        1,
        figsize=(11, 3.4 * n_panels + 1.0),
        sharex=True,
        layout="constrained",
        height_ratios=[1.3] + [1.0] * (n_panels - 1),
    )
    col_ens, col_med, col_prior, col_true = "#2a78d6", "#1f2430", "#9ca3af", "#c2410c"

    def decorate(ax):
        ax.axvspan(t_assim.values[0], t_assim.values[-1], color="#f3f5f8", zorder=0)
        for k in k_assim[resampled[k_assim]]:
            ax.axvline(obs_time.values[k], color="0.55", lw=0.8, ls=":", zorder=1)
        ax.grid(True, color="#e5e7eb", lw=0.6)
        ax.set_axisbelow(True)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    # (a) beta
    ax = axes[0]
    decorate(ax)
    if np.isfinite(prior_mean) and np.isfinite(prior_sd):
        ax.axhspan(
            prior_mean - prior_sd,
            prior_mean + prior_sd,
            color=col_prior,
            alpha=0.18,
            lw=0,
            label=f"prior mean ± sd ({prior_mean:g} ± {prior_sd:g})",
        )
        ax.axhline(prior_mean, color=col_prior, lw=1.0)
    for i in range(len(rows)):
        ax.scatter(
            np.repeat(x_t[i], n_ens),
            beta[i],
            s=8 + 220 * w[i],
            color=col_ens,
            alpha=0.35,
            lw=0,
            label="members (area ∝ weight)" if i == 0 else None,
        )
    ax.fill_between(
        x_t,
        q[:, 0],
        q[:, 2],
        color=col_ens,
        alpha=0.15,
        lw=0,
        label=f"weighted {int(band_quantiles[0] * 100)}–{int(band_quantiles[1] * 100)}% band",
    )
    ax.plot(
        x_t, q[:, 1], color=col_med, lw=2, marker="o", ms=3.5, label="weighted median"
    )
    if np.isfinite(true_bias):
        ax.axhline(
            true_bias,
            color=col_true,
            lw=1.6,
            ls="--",
            label=f"injected bias ({true_bias:g})",
        )
    ax.axhline(0, color="0.3", lw=0.6)
    ax.set_ylabel(f"β ({units})", fontsize=12)
    ax.set_title(
        f"(a) Bias coefficient β, {predictor} predictor", loc="left", fontsize=12
    )
    ax.legend(fontsize=9, frameon=False, loc="best", ncol=2)
    ax.annotate(
        "prior",
        (x_t[0], q[0, 1]),
        xytext=(0, 10),
        textcoords="offset points",
        ha="center",
        fontsize=9,
        color="0.35",
    )

    # (b) correction at each observed elongation
    if show_corr:
        ax = axes[1]
        decorate(ax)
        y_at = np.concatenate(
            [[obs[k_assim[0]]], obs[k_assim]]
        )  # prior: at the first observation
        p_at = bias_predictor(y_at, **pred_kw)
        corr = beta * p_at[:, None]
        qc = np.array(
            [
                weighted_quantiles(
                    corr[i], w[i], [band_quantiles[0], 0.5, band_quantiles[1]]
                )
                for i in range(len(rows))
            ]
        )
        ax.fill_between(
            x_t,
            qc[:, 0],
            qc[:, 2],
            color=col_ens,
            alpha=0.15,
            lw=0,
            label="weighted band",
        )
        ax.plot(
            x_t,
            qc[:, 1],
            color=col_med,
            lw=2,
            marker="o",
            ms=3.5,
            label="weighted median",
        )
        if np.isfinite(prior_mean) and np.isfinite(prior_sd):
            ax.fill_between(
                x_t,
                (prior_mean - prior_sd) * p_at,
                (prior_mean + prior_sd) * p_at,
                color=col_prior,
                alpha=0.18,
                lw=0,
                label="prior mean ± sd",
            )
        if np.isfinite(true_bias):
            ax.plot(
                x_t,
                true_bias * p_at,
                color=col_true,
                lw=1.6,
                ls="--",
                label="injected bias",
            )
        ax.axhline(0, color="0.3", lw=0.6)
        ax.set_ylabel("β·p(y) (deg)", fontsize=12)
        ax.set_title(
            "(b) Correction the bias term makes at each observed elongation y",
            loc="left",
            fontsize=12,
        )
        ax.legend(fontsize=9, frameon=False, loc="best")

    # (c) spread and ESS
    ax = axes[-1]
    decorate(ax)
    if np.isfinite(prior_sd) and prior_sd > 0:
        ax.plot(
            x_t,
            sd / prior_sd,
            color=col_med,
            lw=2,
            marker="o",
            ms=3.5,
            label="weighted sd of β / prior sd",
        )
        ax.axhline(1, color=col_prior, lw=1.0)
        ax.set_ylabel("sd / prior sd", fontsize=12)
    else:
        ax.plot(
            x_t, sd, color=col_med, lw=2, marker="o", ms=3.5, label="weighted sd of β"
        )
        ax.set_ylabel(f"sd ({units})", fontsize=12)
    ax.set_ylim(bottom=0)
    ax2 = ax.twinx()
    ax2.plot(
        t_assim.values,
        ess_upd[k_assim] / n_ens,
        color="#eb6834",
        lw=1.4,
        marker="s",
        ms=3,
        label="ESS / N of the updated weights",
    )
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("ESS / N", color="#eb6834", fontsize=12)
    ax2.spines["top"].set_visible(False)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.plot([], [], color="0.55", lw=0.8, ls=":", label="resampling step")
    h1, l1 = ax.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=9, frameon=False, loc="lower left")
    ax.set_title(
        f"({'c' if show_corr else 'b'}) Spread of β and effective sample size",
        loc="left",
        fontsize=12,
    )
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H:%M"))
    ax.set_xlabel("Observation time (UT); shaded: assimilation window", fontsize=12)

    fig.suptitle(
        fig_title if fig_title else f"Observation-bias term, run {runNo}", fontsize=15
    )
    if save_fig:
        name = "bias_evolution" + (f"_{craft}_assim" if craft is not None else "")
        out_path = safe_savefig(fig, os.path.join(run_dir, name + ".png"), dpi=150)
        print(f"Saved bias evolution plot to {out_path}")
        plt.close(fig)
    else:
        plt.show()
    return fig, axes


def experiment_events_file(base_file_path, run_no):
    """
    The copy of events.yaml that data_assimilation_routine.py saved in an experiment folder (the
    folder above run_NNN): events_run_NNN.yaml if events.yaml was edited before run NNN, otherwise
    events.yaml; None if the experiment has no copy (runs before 5 Nov 2026)
    """
    for name in (f"events_run_{run_no:03d}.yaml", "events.yaml"):
        path = os.path.join(base_file_path, name)
        if os.path.isfile(path):
            return path
    return None


def read_twin_truth(ds, base_file_path, fallback, fallback_from_run=False):
    """
    The truth CME of the twin run being analysed, which may have been made with other settings than
    the current events.yaml:
    - the true_* attributes of cme_pars.nc (runs since 5 Nov 2026), otherwise
    - the truth_<v>_<width>_<lon>_<lat>_<thick> folder in base_file_path; the launch time is not in
      the folder name, so it is taken from `fallback` relative to the model start, i.e. the
      run's model_init_time + (launch time - model start of the twin's entry in events.yaml)
    - otherwise `fallback`
    :param ds: The run's cme_pars.nc (with model_init_time)
    :param base_file_path: Folder of the run_NNN folders
    :param fallback: Truth from events.yaml (get_twin_cme), plus "model_start"
    :param fallback_from_run: The fallback comes from the experiment's own copy of events.yaml
        (experiment_events_file), so it is used before the folder name
    :return: (truth: dict with t_init, speed, width, lon, lat, thick; where it came from)
    """
    attrs = ds.attrs
    if "true_v" in attrs:
        truth = {
            "t_init": datetime.datetime.fromisoformat(str(attrs["true_t_init"])),
            "speed": float(attrs["true_v"]),
            "width": float(attrs["true_width"]),
            "lon": float(attrs["true_lon"]),
            "lat": float(attrs["true_lat"]),
            "thick": float(attrs["true_thick"]),
        }
        return truth, "cme_pars.nc"
    if fallback_from_run:
        return {
            k: fallback[k] for k in ("t_init", "speed", "width", "lon", "lat", "thick")
        }, "the experiment's copy of events.yaml"

    for part in reversed(
        os.path.normpath(base_file_path).replace("/", os.sep).split(os.sep)
    ):
        if not part.startswith("truth_"):
            continue
        try:
            speed, width, lon, lat, thick = (
                float(x) for x in part[len("truth_") :].split("_")[:5]
            )
        except ValueError:
            break
        model_init_time = pd.to_datetime(ds["model_init_time"].values).to_pydatetime()
        truth = {
            "t_init": model_init_time + (fallback["t_init"] - fallback["model_start"]),
            "speed": speed,
            "width": width,
            "lon": lon,
            "lat": lat,
            "thick": thick,
        }
        return truth, f"folder name {part} (launch time from events.yaml)"

    return {
        k: fallback[k] for k in ("t_init", "speed", "width", "lon", "lat", "thick")
    }, "events.yaml"


def load_profile_bias(run_dir):
    """
    The observation-bias term of a run, for adding to modelled elongation profiles.

    The filter compares the observations with the bias-corrected elongation h' = h + beta * p(h)
    (obs_bias.py), applied in ObservationOperator.make_obs_op. The full elongation profiles
    (calculate_elongation_profiles, compute_elongation_profiles_per_obs) come straight from the tracked
    CME flank and do not include it, so it is added when they are plotted (no model reruns needed).
    :param run_dir: Run folder containing cme_pars.nc
    :return: (beta, predictor kwargs): beta (n_obs + 1, n_ens), row 0 = prior and row k + 1 = after
        observation k; or (None, None) if the run has no bias term or beta is zero for every member
    """
    from obs_bias import (
        bias_predictor,
    )  # noqa: F401  (checked here so a missing module fails early)

    par_file = os.path.join(run_dir, "cme_pars.nc")
    if not os.path.isfile(par_file):
        return None, None
    with xr.load_dataset(par_file) as p_ds:
        if "bias" not in p_ds:
            return None, None
        beta = p_ds["bias"].values.astype(float)
        attrs = dict(p_ds.attrs)
    if str(attrs.get("bias_mode", "off")) == "off" or not np.any(
        np.nan_to_num(beta) != 0
    ):
        return None, None
    predictor = str(attrs.get("bias_predictor", "constant"))
    pred_kw = dict(predictor=predictor)
    if predictor != "constant":
        pred_kw.update(
            e_ref=float(attrs.get("bias_e_ref_deg")),
            e_scale=float(attrs.get("bias_e_scale_deg")),
        )
    return beta, pred_kw


def apply_profile_bias(elon, beta_row, pred_kw):
    """
    Add each member's bias term to its elongation profile: h' = h + beta * p(h), for h > 0 only
    (profiles are 0 before the CME is launched, and NaN where it is not seen; both are left alone).
    :param elon: (n_ens, n_profile) modelled elongations (deg)
    :param beta_row: (n_ens) bias coefficients of the same ensemble
    :param pred_kw: Predictor settings from load_profile_bias
    :return: (n_ens, n_profile) bias-corrected elongations
    """
    from obs_bias import bias_predictor

    elon = np.asarray(elon, dtype=float)
    beta_row = np.nan_to_num(np.asarray(beta_row, dtype=float))[:, None]
    with np.errstate(invalid="ignore"):
        corr = beta_row * bias_predictor(elon, **pred_kw)
        return np.where(np.isfinite(elon) & (elon > 0), elon + corr, elon)


def backfill_da_profiles(
    base_file_path,
    runNo=-1,
    use_model=None,
    event=None,
    cme_hit_object="EARTH",
    sim_time_days=4.0,
    compute_missing=True,
):
    """
    Fill in the prior and/or posterior elongation profiles missing from a data assimilation run's
    <run dir>/ens_elongations.nc (the run could not compute them: a member's model run or tracking
    raised; runs since 10 Nov 2026 give such a member a NaN profile instead), so that
    plot_full_elongation_profiles has both panels, without rerunning the data assimilation.
    The profiles of the ensemble at the missing stage (prior = stage 0, posterior = the ensemble
    after the last assimilated observation) are taken from <run dir>/elon_prof_obs.nc
    (compute_profiles_and_arrivals_per_obs), computing that stage first if it is not there yet
    (compute_missing; one ensemble of model runs, set up as the run was made). They are written
    into ens_elongations.nc as {prior,posterior}_profile_{elon,extrapolated,log_weight}, with the
    attribute {stage}_profile_source = "osse_ana backfill"; the arrivals of these members are in
    arr_obs.nc (not added to ens_elongations.nc). A stage that cannot be filled is left out, with a
    message (and a traceback of any member that failed again).
    :param base_file_path: Directory containing the run_<runNo> output folders
    :param runNo: Run number
    :param use_model: Model used in the run (None = from ens_elongations.nc)
    :param event: Event of the run (see run_routine; None = the run's own record)
    :param cme_hit_object: Body for the arrivals computed alongside the profiles
    :param sim_time_days: Model run length in days (as the data assimilation's profile runs)
    :param compute_missing: Run the models for a stage not yet in elon_prof_obs.nc
    :return: List of the stages ("prior", "posterior") added to ens_elongations.nc
    """
    run_dir = os.path.join(base_file_path, f"run_{runNo:03d}")
    elon_file = os.path.join(run_dir, "ens_elongations.nc")
    par_file = os.path.join(run_dir, "cme_pars.nc")
    if not (os.path.isfile(elon_file) and os.path.isfile(par_file)):
        return []
    elon_ds = xr.load_dataset(elon_file)
    if "profile_time" not in elon_ds.coords:
        # The run did not compute profiles at all (compute_elongation_profiles off, or older code)
        return []
    assimilated = np.flatnonzero(elon_ds["assimilated"].values.astype(bool))
    stage_no = {
        "prior": 0,
        "posterior": int(assimilated[-1]) + 1 if assimilated.size else 0,
    }
    missing = [st for st in stage_no if f"{st}_profile_elon" not in elon_ds]
    if not missing:
        return []
    print(
        f"backfill_da_profiles: run {runNo}: the {' and '.join(missing)} elongation profiles are "
        f"missing from {elon_file}"
    )

    def stage_profiles():
        prof_file = read_path(run_dir, "profiles_per_obs")
        if not os.path.isfile(prof_file):
            return None
        return xr.load_dataset(prof_file)

    prof_ds = stage_profiles()
    to_compute = [
        stage_no[st]
        for st in missing
        if prof_ds is None
        or stage_no[st] >= prof_ds.sizes.get("stage", 0)
        or not bool(prof_ds["computed"].values[stage_no[st]])
    ]
    if to_compute and compute_missing:
        print(f"backfill_da_profiles: computing stage(s) {sorted(set(to_compute))}")
        try:
            compute_profiles_and_arrivals_per_obs(
                base_file_path,
                runNo=runNo,
                use_model=use_model,
                stages=sorted(set(to_compute)),
                cme_hit_object=cme_hit_object,
                sim_time_days=sim_time_days,
                want_profiles=True,
                want_arrivals=True,
                event=event,
            )
        except Exception as err:
            print(
                f"backfill_da_profiles: could not compute the profiles: {err}\n"
                f"{traceback.format_exc()}"
            )
        prof_ds = stage_profiles()
    if prof_ds is None:
        print("backfill_da_profiles: no per-observation profiles to fill in from")
        return []

    prof_times = pd.to_datetime(prof_ds["profile_time"].values)
    da_times = pd.to_datetime(elon_ds["profile_time"].values)
    if len(prof_times) != len(da_times) or np.any(
        np.abs((prof_times - da_times).total_seconds()) > 1.0
    ):
        print(
            "backfill_da_profiles: the per-observation profiles are on other times than "
            f"{elon_file}'s; not filled in"
        )
        return []

    added = []
    for st in missing:
        k = stage_no[st]
        if k >= prof_ds.sizes["stage"] or not bool(prof_ds["computed"].values[k]):
            print(f"backfill_da_profiles: no {st} profiles (stage {k}) to fill in")
            continue
        elon_ds[f"{st}_profile_elon"] = (
            ["n_ens", "n_profile"],
            np.asarray(prof_ds["elon_profile"].values[k], dtype=float),
        )
        elon_ds[f"{st}_profile_extrapolated"] = (
            ["n_ens", "n_profile"],
            np.asarray(prof_ds["extrapolated"].values[k]).astype(np.int8),
        )
        elon_ds[f"{st}_profile_log_weight"] = (
            ["n_ens"],
            np.asarray(prof_ds["log_weight"].values[k], dtype=float),
        )
        elon_ds.attrs[f"{st}_profile_stage"] = int(k)
        elon_ds.attrs[f"{st}_profile_source"] = (
            "osse_ana backfill (elon_prof_obs.nc stage " f"{k})"
        )
        n_nan = int(
            np.sum(np.all(~np.isfinite(elon_ds[f"{st}_profile_elon"].values), axis=1))
        )
        print(
            f"backfill_da_profiles: added the {st} profiles (stage {k}"
            + (f"; {n_nan} member(s) without any elongation" if n_nan else "")
            + ")"
        )
        added.append(st)

    if added:
        # Write to a temporary file and replace, so an interrupted write leaves the original
        tmp_file = elon_file + ".tmp"
        elon_ds.to_netcdf(tmp_file, mode="w")
        os.replace(tmp_file, elon_file)
    return added


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
    apply_bias=True,
    show_raw=True,
):
    """
    Plot the full (3-day) modelled CME flank elongation profiles of the prior and posterior
    ensembles against the observations.
    With apply_bias (default), the observation-bias term of each member is added, so the profiles
    are what the filter compared with the observations (see load_profile_bias); with show_raw the
    weighted median without it is also drawn (dashed). Runs without a bias term are unchanged.

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
    :param apply_bias: Add the members' observation-bias term to the profiles (if the run has one)
    :param show_raw: Also draw the weighted median without the bias term (only when it was added)
    :return: (fig, ax), or (None, None) if the file or the profiles do not exist
    """
    import matplotlib.dates as mdates

    run_dir = os.path.join(base_file_path, f"run_{runNo:03d}")
    elon_file = os.path.join(run_dir, "ens_elongations.nc")
    if not os.path.isfile(elon_file):
        print(f"plot_full_elongation_profiles: {elon_file} not found")
        return None, None
    beta_all, bias_kw = load_profile_bias(run_dir) if apply_bias else (None, None)

    with xr.load_dataset(elon_file) as elon_ds:
        if "profile_time" not in elon_ds.coords:
            print(
                f"plot_full_elongation_profiles: no elongation profiles in {elon_file} "
                f"(run predates profile output or compute_elongation_profiles was off)"
            )
            return None, None
        profile_time = pd.to_datetime(elon_ds["profile_time"].values)
        obs_time = pd.to_datetime(elon_ds["obs_time"].values)
        observation = elon_ds["observation"].values
        assimilated = elon_ds["assimilated"].values.astype(bool)
        stages = {}
        for stage in ["prior", "posterior"]:
            if f"{stage}_profile_elon" in elon_ds:
                stages[stage] = dict(
                    elon=elon_ds[f"{stage}_profile_elon"].values.astype(
                        float
                    ),  # (n_ens, n_profile)
                    extrap=elon_ds[f"{stage}_profile_extrapolated"].values.astype(bool),
                    log_weight=elon_ds[f"{stage}_profile_log_weight"].values.astype(
                        float
                    ),
                )

    if not stages:
        print(
            f"plot_full_elongation_profiles: no prior/posterior profiles in {elon_file}"
        )
        return None, None

    # Observation-bias term: the prior profile is of the prior ensemble (cme_pars.nc row 0) and the
    #  posterior profile of the final ensemble (last row)
    if beta_all is not None:
        for stage, row in [("prior", 0), ("posterior", -1)]:
            if stage in stages:
                stages[stage]["elon_raw"] = stages[stage]["elon"].copy()
                stages[stage]["elon"] = apply_profile_bias(
                    stages[stage]["elon"], beta_all[row], bias_kw
                )

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
        figsize=(12, 4.5 * len(panel_info)),
        nrows=len(panel_info),
        ncols=1,
        sharex=True,
        sharey=True,
        layout="constrained",
        squeeze=False,
    )
    ax = ax[:, 0]

    lo_q, hi_q = band_quantiles
    for axi, (stage, col, panel_name) in zip(ax, panel_info):
        elon = stages[stage]["elon"].copy()
        extrap = stages[stage]["extrap"]
        log_w = stages[stage]["log_weight"]
        n_ens = elon.shape[0]

        if not include_extrapolated:
            elon[extrap] = np.nan
        elon_model = np.where(extrap, np.nan, elon)  # model-tracked part only
        elon_extrap = np.where(extrap, elon, np.nan)  # extrapolated part only

        # Assimilation window
        if shade_window and np.any(assimilated):
            axi.axvspan(
                obs_time[assimilated].min(),
                obs_time[assimilated].max(),
                color=col_window,
                zorder=0,
                label="Assimilation window",
            )

        # Members: model-tracked solid, extrapolated dashed (one legend entry each)
        for m in range(n_ens):
            axi.plot(
                profile_time,
                elon_model[m, :],
                color=col,
                linewidth=0.8,
                alpha=0.25,
                label=f"{stage.capitalize()} member" if m == 0 else None,
                zorder=1,
            )
            if include_extrapolated and np.any(np.isfinite(elon_extrap[m, :])):
                axi.plot(
                    profile_time,
                    elon_extrap[m, :],
                    color=col,
                    linewidth=0.8,
                    alpha=0.25,
                    linestyle=(0, (3, 2)),
                    label="Extrapolated (before tracked)" if m == 0 else None,
                    zorder=1,
                )

        # Weighted median and band, counting members not yet in / already out of the field of view as
        #  below / above it (see _weighted_profile_summary)
        band_lo, med, band_hi, _ = _weighted_profile_summary(
            elon,
            log_w,
            min_weight_fraction=min_weight_fraction,
            band_quantiles=band_quantiles,
        )
        axi.fill_between(
            profile_time,
            band_lo,
            band_hi,
            color=col,
            alpha=0.18,
            linewidth=0,
            label=f"Weighted {int(round(100 * lo_q))}-{int(round(100 * hi_q))}% range",
            zorder=2,
        )
        axi.plot(
            profile_time,
            med,
            color=col,
            linewidth=2.0,
            label="Weighted median"
            + (" (model + bias term)" if beta_all is not None else ""),
            zorder=3,
        )
        if (beta_all is not None) and show_raw:
            raw = stages[stage]["elon_raw"].copy()
            if not include_extrapolated:
                raw[extrap] = np.nan
            _, med_raw, _, _ = _weighted_profile_summary(
                raw,
                log_w,
                min_weight_fraction=min_weight_fraction,
                band_quantiles=band_quantiles,
            )
            axi.plot(
                profile_time,
                med_raw,
                color=col,
                linewidth=1.4,
                linestyle=(0, (4, 2)),
                alpha=0.8,
                label="Weighted median without bias term",
                zorder=3,
            )

        # Observations
        if obs_sd is not None:
            axi.errorbar(
                obs_time,
                observation,
                yerr=obs_sd,
                fmt="none",
                ecolor=col_obs,
                elinewidth=1.0,
                capsize=2,
                zorder=4,
            )
        axi.plot(
            obs_time[assimilated],
            observation[assimilated],
            linestyle="none",
            marker="o",
            markersize=6,
            markerfacecolor=col_obs,
            markeredgecolor="white",
            markeredgewidth=1.0,
            label="Observation (assimilated)",
            zorder=5,
        )
        if np.any(~assimilated):
            axi.plot(
                obs_time[~assimilated],
                observation[~assimilated],
                linestyle="none",
                marker="o",
                markersize=6,
                markerfacecolor="none",
                markeredgecolor=col_obs,
                markeredgewidth=1.0,
                label="Observation (not assimilated)",
                zorder=5,
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
    vals = [
        observation[in_obs] + (obs_sd or 0.0),
        observation[in_obs] - (obs_sd or 0.0),
    ]
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
            out_path = os.path.join(
                run_dir, f"elongation_profiles_full_{craft}_assim.png"
            )
        out_path = safe_savefig(fig, out_path, dpi=150)
        print(f"Saved full elongation profile plot to {out_path}")
        plt.close(fig)
    else:
        plt.show()

    return fig, ax


def _stage_list(stages, n_stage, assimilated):
    """
    Stages to compute: "assimilated" (the prior and the ensemble after each assimilated observation),
    "all", or a list of stage numbers
    """
    if isinstance(stages, (list, tuple, np.ndarray)):
        return sorted({int(s) for s in stages if 0 <= int(s) < n_stage})
    if stages == "all":
        return list(range(n_stage))
    return [0] + [
        k + 1 for k in range(min(len(assimilated), n_stage - 1)) if assimilated[k]
    ]


def _separate_arrivals(
    par_ds, use_model, pars, members, cme_hit_object, sim_time_days, model, pool
):
    """
    Arrival times (s after the model start) and speeds of some members of an ensemble from separate
    model runs (make_arrival_model), for members whose observation-operator run gave none
    :param pars: (6, n_ens) parameters of the ensemble: t_init (s), v, width, lon, lat, thick
    :param members: Indices of the members to run
    :return: dict member -> (arrival seconds after the model start or NaN, speed km/s or NaN)
    """
    model_obj, start_time = model
    v = pars[1] * u.km / u.s
    t_launch = (pars[0] * u.s) + get_transit_time_to_r_min(par_ds, v)
    cme_cls = H.ConeCME if use_model in ["huxt"] else S.ConeCME
    cmes = {
        i: cme_cls(
            t_launch=t_launch[i],
            v=v[i],
            width=pars[2][i] * u.deg,
            longitude=pars[3][i] * u.deg,
            latitude=pars[4][i] * u.deg,
            thickness=pars[5][i] * u.solRad,
            cme_fixed_duration=True,
            fixed_duration=12 * 3600 * u.s,
        )
        for i in members
    }
    if pool is not None:
        jobs = {
            i: pool.apply_async(
                _arrival_for_cme, (model_obj, use_model, c, cme_hit_object)
            )
            for i, c in cmes.items()
        }
        results = {i: j.get() for i, j in jobs.items()}
    else:
        results = {
            i: _arrival_for_cme(model_obj, use_model, c, cme_hit_object)
            for i, c in cmes.items()
        }
    return {
        i: (
            ((t - start_time).total_seconds(), s) if t is not None else (np.nan, np.nan)
        )
        for i, (t, s) in results.items()
    }


def run_routine(par_ds, elon_ds, base_file_path, run_no, event=None, use_model=None):
    """
    A RunDataAssimilationRoutine set up as the run in base_file_path/run_<run_no> was made, for
    model runs of its ensembles (elongation profiles, arrivals), independent of the event and
    settings da_surf_setup.py is set to now. From the run's own record where it has one:
      - the event (event, else cme_pars.nc's ssw_event or twin_experiment), read from the copy of
        events.yaml saved with the experiment (experiment_events_file) when there is one
      - the model start (cme_pars.nc model_init_time), the number of members (n_ens) and the model
        (use_model, else the run's)
      - the observation times and values (ens_elongations.nc), so no observation file is read and no
        synthetic truth run is made
      - the model grid and observer (cme_pars.nc model_* and obs_* attributes, runs since 8 Nov 2026)
    Anything the run does not record (older runs: the model grid; and the model's fixed constants) is
    taken from da_surf_setup.Setup.
    :param par_ds: The run's cme_pars.nc
    :param elon_ds: The run's ens_elongations.nc, or None
    :param base_file_path: Directory containing the run_<run_no> folders
    :param run_no: Run number
    :param event: Event or twin experiment the run was made for (None = from the run's record, else
        da_surf_setup.py's event, with a note)
    :param use_model: Model (None = from the run's record, else da_surf_setup.py's)
    :return: RunDataAssimilationRoutine
    """
    from data_assimilation_routine import RunDataAssimilationRoutine
    from da_surf_setup import Setup, use_events_file

    attrs = dict(par_ds.attrs)
    if event is None:
        event = attrs.get("ssw_event") or attrs.get("twin_experiment")
    if not event:
        # Older runs that do not record their event: da_surf_setup.py's event
        event = Setup.get_ssw_event(object.__new__(Setup))
        print(
            f"Note: run {run_no} in {base_file_path} does not record its event; using {event!r} "
            f"(da_surf_setup.py). Give the event (osse_ana.py --event) if it was another."
        )
    event = str(event)
    events_copy = experiment_events_file(base_file_path, run_no)
    if use_model is None:
        use_model = attrs.get("use_model") or (
            elon_ds.attrs.get("use_model") if elon_ds is not None else None
        )
    n_ens = int(par_ds.sizes["n_ens"])
    model_init_time = pd.to_datetime(
        np.ravel(par_ds["model_init_time"].values)[0]
    ).to_pydatetime()
    obs_times = (
        [t.to_pydatetime() for t in pd.to_datetime(elon_ds["obs_time"].values)]
        if elon_ds is not None and "obs_time" in elon_ds
        else None
    )
    observations = (
        [float(x) for x in np.asarray(elon_ds["observation"].values, dtype=float)]
        if elon_ds is not None and "observation" in elon_ds
        else None
    )

    def recorded(key):
        """A recorded attribute, or None if absent or NaN"""
        value = attrs.get(key)
        if value is None:
            return None
        if np.ndim(value) == 0 and not np.isfinite(float(value)):
            return None
        return value

    with use_events_file(events_copy):
        twin: bool = bool(attrs.get("twin_experiment")) or is_twin_experiment(event)

    class RunSetup(Setup):
        def get_ssw_event(self):
            return event

        def get_use_synthetic_obs(self):
            return twin

        def get_surf_init_time(self):
            return model_init_time

        def get_use_model(self):
            return str(use_model) if use_model else super().get_use_model()

        def initialise_da_parameters(self):
            d = super().initialise_da_parameters()
            d.update(n_members=n_ens, run_start=-1, n_runs=1)
            return d

        def get_twin_latitude_range(self):
            lat_min, lat_max = recorded("model_lat_min_deg"), recorded(
                "model_lat_max_deg"
            )
            if lat_min is not None and lat_max is not None:
                return float(lat_min) * u.deg, float(lat_max) * u.deg
            return super().get_twin_latitude_range()

        def initialise_surf_parameters(self):
            d = super().initialise_surf_parameters()
            for key, attr, unit in (
                ("n_lat_surf", "model_n_lat_surf", None),
                ("lon_start", "model_lon_start_deg", u.deg),
                ("lon_stop", "model_lon_stop_deg", u.deg),
                ("r_min", "model_r_min_rs", u.solRad),
                ("dt_scale", "model_dt_scale", None),
            ):
                value = recorded(attr)
                if value is not None:
                    d[key] = (
                        float(value) * unit if unit is not None else type(d[key])(value)
                    )
            return d

        def initialise_observation_parameters(self):
            d = super().initialise_observation_parameters()
            if obs_times is not None:
                d["n_obs"] = len(obs_times)
                d["obs_times"] = list(obs_times)
            for key, attr, unit in (
                ("obs_radius", "obs_radius_au", u.AU),
                ("obs_lon", "obs_lon_deg", u.deg),
                ("obs_lat", "obs_lat_deg", u.deg),
            ):
                values = attrs.get(attr)
                if values is not None:
                    values = [float(x) * unit for x in np.atleast_1d(values)]
                    d[key] = values if len(values) > 1 else values[0]
            return d

    class RunRoutine(RunDataAssimilationRoutine):
        def get_observations(self):
            # The run's own observations (no observation file read, no synthetic truth run)
            if observations is None or obs_times is None:
                return super().get_observations()
            if self.use_synthetic_obs:
                return list(observations)
            obs_lon = (
                list(self.obs_lon)
                if isinstance(self.obs_lon, list)
                else [self.obs_lon] * len(observations)
            )
            return list(obs_times), obs_lon, list(observations), len(observations)

    with use_events_file(events_copy):
        return RunRoutine(setup_class=RunSetup)


def compute_profiles_and_arrivals_per_obs(
    base_file_path,
    runNo=-1,
    use_model=None,
    stages="assimilated",
    cme_hit_object="EARTH",
    sim_time_days=4.0,
    want_profiles=True,
    want_arrivals=True,
    parallel=True,
    overwrite=False,
    event=None,
):
    """
    Full elongation profiles and/or CME arrival times at cme_hit_object of every member, for the
    ensemble at each observation (stage 0 = prior, stage k+1 = after observation k), rebuilt from
    <run dir>/cme_pars.nc. Each distinct ensemble is run ONCE: the observation operator's model run
    (set up as in the data assimilation, RunDataAssimilationRoutine.calculate_elongation_profiles)
    gives the profile and, from the same solved model, the arrival.

    Model runs are avoided where possible:
      - stages already in the output files are kept (unless overwrite)
      - an ensemble identical to one already computed (the filter only reweighted, or the
        observation was not assimilated) is copied (reused_from = that stage)
      - the prior and posterior ensembles' profiles and arrivals saved by the data assimilation run
        in ens_elongations.nc are used (reused_from = -2); its arrivals only if their body and run
        length match cme_hit_object and sim_time_days
    The model runs are set up as the run was made (run_routine: its event, model start, members,
    model, observations and recorded grid), whatever da_surf_setup.py is set to now. The arrivals
    come from separate runs (make_arrival_model) for members whose run does not reach the body (a
    compress_surf3d latitude grid that misses it), and for every member if the run cannot be set up.

    Output: <run dir>/elon_prof_obs.nc (want_profiles) with
        elon_profile (n_stage, n_ens, n_profile): elongation (deg), NaN where not visible/not computed
        extrapolated (n_stage, n_ens, n_profile): 1 where extrapolated before the first tracked point
        log_weight (n_stage, n_ens), computed (n_stage), reused_from (n_stage), stage_obs_no (n_stage)
        coords: stage, ens_no, profile_time, stage_obs_time
    and <run dir>/arr_obs.nc (want_arrivals) with
        arrival_time (n_stage, n_ens): arrival time (NaT if the CME does not arrive within the run)
        arrival_speed (n_stage, n_ens): arrival speed (km/s)
        log_weight, computed, reused_from, stage_obs_no as above; coords: stage, ens_no, stage_obs_time
    (reused_from: -1 = run for this stage, -2 = from the data assimilation run, k = copied from stage k)

    :param base_file_path: Directory containing the run_<runNo> output folders
    :param runNo: Run number
    :param use_model: Model used in the run (None = from ens_elongations.nc)
    :param stages: "assimilated", "all" or a list of stage numbers
    :param cme_hit_object: Body for the arrivals
    :param sim_time_days: Model run length in days from the model start (CMEs arriving later miss)
    :param want_profiles: Compute the profiles
    :param want_arrivals: Compute the arrivals
    :param parallel: Solve members in parallel
    :param overwrite: Recompute every requested stage, ignoring existing output files
    :param event: Event or twin experiment of events.yaml the run was made for (see run_routine;
        None = the run's own record: cme_pars.nc ssw_event or twin_experiment)
    :return: (profile file or None, arrival file or None)
    """
    from init_sir import initialise_cme_parameter_ensemble_dict

    run_dir = os.path.join(base_file_path, f"run_{runNo:03d}")
    par_file = os.path.join(run_dir, "cme_pars.nc")
    elon_file = os.path.join(run_dir, "ens_elongations.nc")
    if not os.path.isfile(par_file):
        print(f"compute_profiles_and_arrivals_per_obs: {par_file} not found")
        return None, None
    par_ds = xr.load_dataset(par_file)
    n_stage = par_ds.sizes["n_obs"]  # prior + one row per observation
    n_ens = par_ds.sizes["n_ens"]
    elon_ds = xr.load_dataset(elon_file) if os.path.isfile(elon_file) else None
    if elon_ds is not None:
        assimilated = elon_ds["assimilated"].values.astype(bool)
        obs_time = pd.to_datetime(elon_ds["obs_time"].values)
    else:
        # Older runs: every observation stage, no observation times
        assimilated = np.ones(n_stage - 1, dtype=bool)
        obs_time = pd.to_datetime([pd.NaT] * (n_stage - 1))
    if want_profiles and elon_ds is None:
        print(
            f"compute_profiles_and_arrivals_per_obs: {elon_file} not found; no profiles"
        )
        want_profiles = False
    if use_model is None:
        use_model = (
            str(elon_ds.attrs.get("use_model", "compress_surf3d"))
            if elon_ds is not None
            else "compress_surf3d"
        )
    stage_list = _stage_list(stages, n_stage, assimilated)
    model_init_time = pd.to_datetime(
        np.ravel(par_ds["model_init_time"].values)[0]
    ).to_pydatetime()
    log_weight = np.asarray(par_ds["log_weight"].values, dtype=float)
    par_names = ["t_init", "v", "width", "lon", "lat", "thick"]

    def stage_pars(st):
        return np.array(
            [np.asarray(par_ds[p].values[st], dtype=float) for p in par_names]
        )

    keys = {st: np.round(stage_pars(st), 6).tobytes() for st in range(n_stage)}

    prof_out = write_path(run_dir, "profiles_per_obs")
    arr_out = write_path(run_dir, "arrivals_per_obs")
    # Check now, before the model runs, that the results can be written (Windows path limit)
    if want_profiles:
        check_path_length(prof_out, what="per-observation elongation profile file")
    if want_arrivals:
        check_path_length(arr_out, what="per-observation arrival-time file")

    # Results so far: existing output files
    prof = None  # dict(elon, extrap, computed, reused_from, times)
    prof_read = read_path(run_dir, "profiles_per_obs")
    if want_profiles and os.path.isfile(prof_read) and not overwrite:
        with xr.load_dataset(prof_read) as p_ds:
            prof = dict(
                elon=np.asarray(p_ds["elon_profile"].values, dtype=float),
                extrap=np.asarray(p_ds["extrapolated"].values).astype(np.int8),
                computed=np.asarray(p_ds["computed"].values).astype(np.int8),
                reused_from=np.asarray(p_ds["reused_from"].values).astype(int),
                times=pd.to_datetime(p_ds["profile_time"].values)
                .to_pydatetime()
                .tolist(),
            )
    arr = dict(
        sec=np.full((n_stage, n_ens), np.nan),
        speed=np.full((n_stage, n_ens), np.nan),
        computed=np.zeros(n_stage, dtype=np.int8),
        reused_from=np.full(n_stage, -1, dtype=int),
    )
    arr_read = read_path(run_dir, "arrivals_per_obs")
    if want_arrivals and os.path.isfile(arr_read) and not overwrite:
        with xr.load_dataset(arr_read) as a_ds:
            same_target = str(
                a_ds.attrs.get("cme_hit_object", cme_hit_object)
            ).upper() == str(cme_hit_object).upper() and np.isclose(
                float(a_ds.attrs.get("sim_time_days", sim_time_days)),
                float(sim_time_days),
            )
            if same_target and a_ds.sizes.get("stage") == n_stage:
                times = a_ds["arrival_time"].values
                arr["sec"] = np.where(
                    np.isnat(times),
                    np.nan,
                    (times - np.datetime64(model_init_time)) / np.timedelta64(1, "s"),
                ).astype(float)
                arr["speed"] = np.asarray(a_ds["arrival_speed"].values, dtype=float)
                arr["computed"] = np.asarray(a_ds["computed"].values).astype(np.int8)
                arr["reused_from"] = np.asarray(a_ds["reused_from"].values).astype(int)
            else:
                print(
                    f"{arr_read} was computed for another body or run length; recomputing the arrivals"
                )

    def prof_done(st):
        return prof is not None and bool(prof["computed"][st])

    need_prof = [st for st in stage_list if want_profiles and not prof_done(st)]
    need_arr = [st for st in stage_list if want_arrivals and not arr["computed"][st]]
    if not need_prof and not need_arr:
        print(
            f"compute_profiles_and_arrivals_per_obs: run {runNo}: all requested stages already computed"
        )
        return (prof_read if want_profiles else None), (
            arr_read if want_arrivals else None
        )
    changed_prof = changed_arr = False

    def init_prof(times):
        nonlocal prof
        prof = dict(
            elon=np.full((n_stage, n_ens, len(times)), np.nan),
            extrap=np.zeros((n_stage, n_ens, len(times)), dtype=np.int8),
            computed=np.zeros(n_stage, dtype=np.int8),
            reused_from=np.full(n_stage, -1, dtype=int),
            times=list(times),
        )

    def same_times(times):
        return (
            prof is not None
            and len(prof["times"]) == len(times)
            and all(
                abs((a - b).total_seconds()) < 1.0 for a, b in zip(prof["times"], times)
            )
        )

    # 1. Ensembles identical to a stage already computed
    for st in list(need_prof):
        src = next(
            (
                s
                for s in range(n_stage)
                if s != st and keys[s] == keys[st] and prof_done(s)
            ),
            None,
        )
        if src is not None:
            prof["elon"][st], prof["extrap"][st] = (
                prof["elon"][src],
                prof["extrap"][src],
            )
            prof["computed"][st], prof["reused_from"][st] = 1, src
            need_prof.remove(st)
            changed_prof = True
    for st in list(need_arr):
        src = next(
            (
                s
                for s in range(n_stage)
                if s != st and keys[s] == keys[st] and arr["computed"][s]
            ),
            None,
        )
        if src is not None:
            arr["sec"][st], arr["speed"][st] = arr["sec"][src], arr["speed"][src]
            arr["computed"][st], arr["reused_from"][st] = 1, src
            need_arr.remove(st)
            changed_arr = True

    # 2. The prior and posterior ensembles run by the data assimilation (ens_elongations.nc)
    if elon_ds is not None:
        last_assim = np.flatnonzero(assimilated)
        da_stage = {
            "prior": int(elon_ds.attrs.get("prior_profile_stage", 0)),
            "posterior": int(
                elon_ds.attrs.get(
                    "posterior_profile_stage",
                    int(last_assim[-1]) + 1 if last_assim.size else 0,
                )
            ),
        }
        arrivals_match = str(
            elon_ds.attrs.get("profile_arrival_object", "")
        ).upper() == str(cme_hit_object).upper() and np.isclose(
            float(elon_ds.attrs.get("profile_sim_time_days", np.nan)),
            float(sim_time_days),
        )
        for name, st_da in da_stage.items():
            if f"{name}_profile_elon" not in elon_ds or not 0 <= st_da < n_stage:
                continue
            same = [s for s in set(need_prof) | set(need_arr) if keys[s] == keys[st_da]]
            if not same:
                continue
            da_times = (
                pd.to_datetime(elon_ds["profile_time"].values).to_pydatetime().tolist()
            )
            if prof is None and want_profiles:
                init_prof(da_times)
            for st in same:
                if st in need_prof and same_times(da_times):
                    prof["elon"][st] = np.asarray(
                        elon_ds[f"{name}_profile_elon"].values, dtype=float
                    )
                    prof["extrap"][st] = np.asarray(
                        elon_ds[f"{name}_profile_extrapolated"].values
                    ).astype(np.int8)
                    prof["computed"][st], prof["reused_from"][st] = 1, -2
                    need_prof.remove(st)
                    changed_prof = True
                if (
                    st in need_arr
                    and arrivals_match
                    and f"{name}_profile_arrival_time" in elon_ds
                    and np.all(elon_ds[f"{name}_profile_arrival_available"].values == 1)
                ):
                    times = elon_ds[f"{name}_profile_arrival_time"].values
                    arr["sec"][st] = np.where(
                        np.isnat(times),
                        np.nan,
                        (times - np.datetime64(model_init_time))
                        / np.timedelta64(1, "s"),
                    ).astype(float)
                    arr["speed"][st] = np.asarray(
                        elon_ds[f"{name}_profile_arrival_speed"].values, dtype=float
                    )
                    arr["computed"][st], arr["reused_from"][st] = 1, -2
                    need_arr.remove(st)
                    changed_arr = True
            if same:
                print(
                    f"Stage(s) {sorted(same)}: from the data assimilation run's own {name} runs"
                )

    # 3. Run the remaining ensembles, one model run per member for both outputs
    groups: dict[bytes, list[int]] = {}
    for st in sorted(set(need_prof) | set(need_arr)):
        groups.setdefault(keys[st], []).append(st)
    n_obsop_runs = n_separate_runs = 0
    run_da = None
    setup_matches = False
    arrival_model = None
    pool = None
    try:
        if groups:
            # Set up as the run was made, not as da_surf_setup.py is set now
            try:
                run_da = run_routine(
                    par_ds,
                    elon_ds,
                    base_file_path,
                    runNo,
                    event=event,
                    use_model=use_model,
                )
            except Exception as err:
                if need_prof:
                    raise
                print(
                    f"Could not set up the run's model runs ({err}); the arrivals come from "
                    f"separate model runs"
                )
            setup_matches = run_da is not None and (
                run_da.n_members == n_ens
                and abs((model_init_time - run_da.surf_init_time).total_seconds())
                <= 1.0
                and str(run_da.use_model).lower() == str(use_model).lower()
            )
            # (run_routine sets these from the run, so they match unless the set-up failed)
            if need_prof and not setup_matches:
                raise ValueError(
                    f"Could not set up run {runNo} as it was made (members {run_da.n_members} vs "
                    f"{n_ens}, model start {run_da.surf_init_time} vs {model_init_time}, model "
                    f"{run_da.use_model} vs {use_model})"
                )
            if need_arr and not setup_matches:
                print(
                    "The run's set-up could not be reproduced (members, model start or model): the "
                    "arrivals come from separate model runs"
                )
        for key, sts in groups.items():
            rep = sts[0]
            pars = stage_pars(rep)
            do_prof = any(st in need_prof for st in sts)
            arr_sts = [st for st in sts if st in need_arr]
            do_arr_obsop = bool(arr_sts) and setup_matches
            sec_row = np.full(n_ens, np.nan)
            speed_row = np.full(n_ens, np.nan)
            missing = list(range(n_ens)) if arr_sts else []
            if do_prof or do_arr_obsop:
                cme_par_dict = initialise_cme_parameter_ensemble_dict(
                    n_ensemble=n_ens, surf_init_time=run_da.surf_init_time
                )
                cme_par_dict["t_init"] = [
                    model_init_time + datetime.timedelta(seconds=float(s))
                    for s in pars[0]
                ]
                cme_par_dict["v"] = list(pars[1]) * u.km / u.s
                cme_par_dict["width"] = list(pars[2]) * u.deg
                cme_par_dict["lon"] = list(pars[3]) * u.deg
                cme_par_dict["lat"] = list(pars[4]) * u.deg
                cme_par_dict["thick"] = list(pars[5]) * u.solRad
                # The profiles are the tracked CME flank, without the bias term (added when plotting),
                #  so they do not depend on the bias coefficients (and the stage keys ignore them)
                if "bias" in par_ds:
                    cme_par_dict["bias"] = [
                        float(b)
                        for b in np.asarray(par_ds["bias"].values[rep], dtype=float)
                    ]
                lw = np.where(np.isfinite(log_weight[rep]), log_weight[rep], -1e31)
                cme_par_dict["log_weight"] = list(lw)
                cme_par_dict["weight"] = list(np.exp(lw - np.max(lw)))
                label = (
                    "prior"
                    if rep == 0
                    else f"after observation {rep - 1} ({obs_time[rep - 1]})"
                )
                what = " and ".join(
                    w
                    for w, on in (("profiles", do_prof), ("arrivals", do_arr_obsop))
                    if on
                )
                print(f"Stage(s) {sts}: computing {what} for the ensemble {label}")
                res = run_da.calculate_elongation_profiles(
                    cme_par_dict,
                    obs_index=max(rep - 1, 0),
                    arrival_object=cme_hit_object if do_arr_obsop else None,
                    model_sim_time=sim_time_days * u.day if do_arr_obsop else None,
                )
                n_obsop_runs += n_ens
                if do_prof:
                    times = run_da.get_profile_times()
                    if prof is None:
                        init_prof(times)
                    elif not same_times(times):
                        raise ValueError(
                            f"The profile times of {prof_read} differ from the current setup's; "
                            f"recompute with overwrite=True"
                        )
                    for st in sts:
                        if st in need_prof:
                            prof["elon"][st] = res["elon"]
                            prof["extrap"][st] = np.asarray(res["extrapolated"]).astype(
                                np.int8
                            )
                            prof["computed"][st] = 1
                            prof["reused_from"][st] = -1 if st == rep else rep
                    changed_prof = True
                if do_arr_obsop:
                    available = np.asarray(res["arrival_available"], dtype=bool)
                    for i in np.flatnonzero(available):
                        t = res["arrival_time"][i]
                        if isinstance(t, datetime.datetime):
                            sec_row[i] = (t - model_init_time).total_seconds()
                            speed_row[i] = res["arrival_speed"][i]
                    missing = [int(i) for i in np.flatnonzero(~available)]
            if arr_sts and missing:
                if arrival_model is None:
                    arrival_model = make_arrival_model(
                        par_ds, use_model, sim_time=sim_time_days * u.day
                    )
                    if parallel:
                        pool = Pool(processes=max(1, min(available_cpus() - 1, n_ens)))
                print(
                    f"Stage(s) {arr_sts}: separate arrival runs for {len(missing)} member(s)"
                    + (
                        ""
                        if not do_arr_obsop
                        else " (body outside the model's latitude grid)"
                    )
                )
                for i, (sec, spd) in _separate_arrivals(
                    par_ds,
                    use_model,
                    pars,
                    missing,
                    cme_hit_object,
                    sim_time_days,
                    arrival_model,
                    pool,
                ).items():
                    sec_row[i], speed_row[i] = sec, spd
                n_separate_runs += len(missing)
            for st in arr_sts:
                arr["sec"][st], arr["speed"][st] = sec_row, speed_row
                arr["computed"][st] = 1
                arr["reused_from"][st] = -1 if st == rep else rep
                changed_arr = True
    finally:
        if run_da is not None:
            run_da.worker_pool.shutdown()
        if pool is not None:
            pool.close()
            pool.join()

    stage_obs_time = np.array(
        [np.datetime64("NaT")] + [np.datetime64(t) for t in obs_time[: n_stage - 1]],
        dtype="datetime64[ns]",
    )
    common = dict(
        log_weight=(["stage", "ens_no"], log_weight),
        stage_obs_no=(["stage"], np.arange(n_stage) - 1),
    )
    prof_file = prof_read if (want_profiles and os.path.isfile(prof_read)) else None
    if want_profiles and prof is not None and changed_prof:
        out_ds = xr.Dataset(
            data_vars=dict(
                elon_profile=(["stage", "ens_no", "n_profile"], prof["elon"]),
                extrapolated=(["stage", "ens_no", "n_profile"], prof["extrap"]),
                computed=(["stage"], prof["computed"]),
                reused_from=(["stage"], prof["reused_from"]),
                **common,
            ),
            coords=dict(
                stage=("stage", np.arange(n_stage)),
                ens_no=("ens_no", np.arange(n_ens)),
                profile_time=(
                    "n_profile",
                    np.array(prof["times"], dtype="datetime64[ns]"),
                ),
                stage_obs_time=("stage", stage_obs_time),
            ),
            attrs=dict(
                description="Full elongation profiles of the ensemble at each observation; stage 0 = prior, "
                "stage k+1 = ensemble after observation k",
                elongation_units="deg",
            ),
        )
        out_ds.to_netcdf(prof_out, mode="w")
        prof_file = prof_out
    arr_file = arr_read if (want_arrivals and os.path.isfile(arr_read)) else None
    if want_arrivals and changed_arr:
        arrival_time = np.array(
            [
                [
                    (
                        np.datetime64(
                            model_init_time + datetime.timedelta(seconds=float(s))
                        )
                        if np.isfinite(s)
                        else np.datetime64("NaT")
                    )
                    for s in row
                ]
                for row in arr["sec"]
            ],
            dtype="datetime64[ns]",
        )
        out_ds = xr.Dataset(
            data_vars=dict(
                arrival_time=(["stage", "ens_no"], arrival_time),
                arrival_speed=(["stage", "ens_no"], arr["speed"]),
                computed=(["stage"], arr["computed"]),
                reused_from=(["stage"], arr["reused_from"]),
                **common,
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
                model_init_time=str(model_init_time),
            ),
        )
        out_ds.to_netcdf(arr_out, mode="w")
        arr_file = arr_out
    print(
        f"compute_profiles_and_arrivals_per_obs: run {runNo}: {n_obsop_runs} observation-operator model "
        f"runs, {n_separate_runs} separate arrival runs"
    )
    return prof_file, arr_file


def compute_elongation_profiles_per_obs(
    base_file_path,
    runNo=-1,
    stages="assimilated",
    overwrite=False,
    use_model=None,
    cme_hit_object="EARTH",
    sim_time_days=4.0,
    with_arrivals=True,
    event=None,
):
    """
    Full (3-day) modelled CME flank elongation profile of every member for the ensemble at each
    observation, in <run dir>/elon_prof_obs.nc: see compute_profiles_and_arrivals_per_obs, which also
    computes the arrival times (arr_obs.nc) from the same model runs when with_arrivals is True, so
    compute_arrival_times_per_obs then has nothing left to run.
    :return: Path of the profile file, or None if the inputs are missing
    """
    prof_file, _ = compute_profiles_and_arrivals_per_obs(
        base_file_path,
        runNo=runNo,
        use_model=use_model,
        stages=stages,
        cme_hit_object=cme_hit_object,
        sim_time_days=sim_time_days,
        want_profiles=True,
        want_arrivals=with_arrivals,
        overwrite=overwrite,
        event=event,
    )
    return prof_file


def compute_arrival_times_per_obs(
    base_file_path,
    runNo=-1,
    use_model="compress_surf3d",
    stages="assimilated",
    cme_hit_object="EARTH",
    sim_time_days=4.0,
    parallel=True,
    overwrite=False,
    event=None,
):
    """
    CME arrival time and speed at cme_hit_object of every member for the ensemble at each observation,
    in <run dir>/arr_obs.nc: see compute_profiles_and_arrivals_per_obs (only stages not yet computed
    are run).
    :return: Path of the arrival file, or None if the inputs are missing
    """
    _, arr_file = compute_profiles_and_arrivals_per_obs(
        base_file_path,
        runNo=runNo,
        use_model=use_model,
        stages=stages,
        cme_hit_object=cme_hit_object,
        sim_time_days=sim_time_days,
        want_profiles=False,
        want_arrivals=True,
        parallel=parallel,
        overwrite=overwrite,
        event=event,
    )
    return arr_file


def _weighted_profile_summary(
    elon, log_w, min_weight_fraction=0.5, band_quantiles=(0.05, 0.95)
):
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
            elon_cens[m, : idx[0]] = -np.inf
            elon_cens[m, idx[-1] + 1 :] = np.inf
    lo_q, hi_q = band_quantiles
    med, b_lo, b_hi = (np.full(n_prof, np.nan) for _ in range(3))
    for k in range(n_prof):
        known = ~np.isnan(elon_cens[:, k])
        if np.sum(w[known]) >= min_weight_fraction:
            b_lo[k], med[k], b_hi[k] = weighted_quantiles(
                elon_cens[:, k], w, [lo_q, 0.5, hi_q]
            )
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
    apply_bias=True,
    show_raw=True,
):
    """
    Plot, for the prior and for the ensemble after each observation, every member's full elongation
    profile with the weighted median and band, against the observations. One figure per stage, saved in
    <run dir>/prof_obs/ (formerly elongation_profiles_per_obs/). Observations already assimilated at that stage are filled,
    later ones hollow, and the observation just assimilated is circled and marked with a vertical line.
    Reads elon_prof_obs.nc, or ens_elongation_profiles_per_obs.nc from older runs (see compute_elongation_profiles_per_obs).

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
    :param apply_bias: Add each member's observation-bias term to its profile (if the run has one), so
        the profiles are what the filter compared with the observations (see load_profile_bias). The
        profile file itself holds the model elongations without it.
    :param show_raw: Also draw the weighted median without the bias term (only when it was added)
    :return: List of saved file paths
    """
    import matplotlib.dates as mdates

    run_dir = os.path.join(base_file_path, f"run_{runNo:03d}")
    prof_file = read_path(run_dir, "profiles_per_obs")
    elon_file = os.path.join(run_dir, "ens_elongations.nc")
    if not (os.path.isfile(prof_file) and os.path.isfile(elon_file)):
        print(
            f"plot_elongation_profiles_per_obs: {prof_file} or {elon_file} not found "
            f"(run compute_elongation_profiles_per_obs first)"
        )
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
        resampled = (
            e_ds["resampled"].values.astype(bool) if "resampled" in e_ds else None
        )

    # Observation-bias term: stage st of the profile file is the ensemble of cme_pars.nc row st
    beta_all, bias_kw = load_profile_bias(run_dir) if apply_bias else (None, None)
    elon_raw_all = elon_all.copy()
    if beta_all is not None:
        for st in np.flatnonzero(computed):
            if st < beta_all.shape[0]:
                elon_all[st] = apply_profile_bias(elon_all[st], beta_all[st], bias_kw)

    col_ens = "#2a78d6"
    col_obs = "#1a1a19"
    col_now = "#eb6834"
    col_grid = "#d9d8d2"

    # Common axes for all stages
    if focus_on_window and np.any(assimilated):
        pad = pd.Timedelta(hours=window_pad_hours)
        t_lo, t_hi = (
            obs_time[assimilated].min() - pad,
            obs_time[assimilated].max() + pad,
        )
    else:
        t_lo, t_hi = profile_time.min(), profile_time.max()
    in_prof = (profile_time >= t_lo) & (profile_time <= t_hi)
    in_obs = (obs_time >= t_lo) & (obs_time <= t_hi) & np.isfinite(observation)
    vals = [observation[in_obs]]
    for st in np.flatnonzero(computed):
        e = np.where(extrap_all[st] & ~include_extrapolated, np.nan, elon_all[st])[
            :, in_prof
        ]
        vals.append(e[np.isfinite(e) & (e > 0)])
    all_vals = np.concatenate(vals) if vals else np.array([])
    pad_e = (obs_sd or 0.0) + 0.05 * max(
        np.ptp(all_vals) if all_vals.size else 1.0, 1.0
    )
    y_lim = (
        (max(0.0, np.min(all_vals) - pad_e), np.max(all_vals) + pad_e)
        if all_vals.size
        else (0, 35)
    )

    out_dir = write_path(run_dir, "profiles_per_obs_dir")
    check_path_length(
        out_dir, what="per-observation profile plot folder", allowance=32, error=False
    )
    os.makedirs(out_dir, exist_ok=True)
    saved = []
    for st in np.flatnonzero(computed):
        k_now = st - 1  # observation just assimilated (-1 for the prior)
        elon = elon_all[st].copy()
        if not include_extrapolated:
            elon[extrap_all[st]] = np.nan
        b_lo, med, b_hi, w = _weighted_profile_summary(
            elon,
            lw_all[st],
            min_weight_fraction=min_weight_fraction,
            band_quantiles=band_quantiles,
        )

        fig, ax = plt.subplots(figsize=(12, 6), layout="constrained")
        w_rel = w / np.max(w)
        for m in range(elon.shape[0]):
            alpha = 0.08 + 0.5 * w_rel[m] if weight_member_lines else 0.25
            ax.plot(
                profile_time,
                elon[m],
                color=col_ens,
                lw=0.8,
                alpha=alpha,
                label=(
                    "Member (opacity = weight)"
                    if (m == 0 and weight_member_lines)
                    else ("Member" if m == 0 else None)
                ),
                zorder=1,
            )
        ax.fill_between(
            profile_time,
            b_lo,
            b_hi,
            color=col_ens,
            alpha=0.18,
            lw=0,
            label=f"Weighted {int(round(100 * band_quantiles[0]))}-"
            f"{int(round(100 * band_quantiles[1]))}% range",
            zorder=2,
        )
        ax.plot(
            profile_time,
            med,
            color=col_ens,
            lw=2.0,
            label="Weighted median"
            + (" (model + bias term)" if beta_all is not None else ""),
            zorder=3,
        )
        if (beta_all is not None) and show_raw:
            raw = elon_raw_all[st].copy()
            if not include_extrapolated:
                raw[extrap_all[st]] = np.nan
            _, med_raw, _, _ = _weighted_profile_summary(
                raw,
                lw_all[st],
                min_weight_fraction=min_weight_fraction,
                band_quantiles=band_quantiles,
            )
            ax.plot(
                profile_time,
                med_raw,
                color=col_ens,
                lw=1.4,
                ls=(0, (4, 2)),
                alpha=0.8,
                label="Weighted median without bias term",
                zorder=3,
            )

        done = assimilated & (np.arange(len(observation)) <= k_now)
        later = ~done & np.isfinite(observation)
        if obs_sd is not None:
            ax.errorbar(
                obs_time[done],
                observation[done],
                yerr=obs_sd,
                fmt="none",
                ecolor=col_obs,
                elinewidth=1.0,
                capsize=2,
                zorder=4,
            )
        ax.plot(
            obs_time[done],
            observation[done],
            ls="none",
            marker="o",
            ms=6,
            mfc=col_obs,
            mec="white",
            mew=1.0,
            label="Observation (assimilated so far)",
            zorder=5,
        )
        ax.plot(
            obs_time[later],
            observation[later],
            ls="none",
            marker="o",
            ms=6,
            mfc="none",
            mec="#8c8b85",
            mew=1.0,
            label="Observation (later / not assimilated)",
            zorder=5,
        )
        if k_now >= 0:
            ax.axvline(obs_time[k_now], color=col_now, lw=1.2, ls=(0, (4, 2)), zorder=0)
            ax.plot(
                obs_time[k_now],
                observation[k_now],
                ls="none",
                marker="o",
                ms=13,
                mfc="none",
                mec=col_now,
                mew=2.0,
                label="Observation just assimilated",
                zorder=6,
            )

        ess = 1.0 / np.sum(w**2)
        if k_now < 0:
            title = "Prior ensemble"
        else:
            how = ""
            if resampled is not None:
                how = ", resampled" if resampled[k_now] else ", weights updated only"
            title = (
                f"After observation {k_now} ({obs_time[k_now]:%d %b %H:%M}, "
                f"{observation[k_now]:.2f}°{how})"
            )
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


def make_arrival_model(
    ds,
    use_model,
    sim_time=4.0 * u.day,
    dt_scale=20,
    latitude_min=-15 * u.deg,
    latitude_max=15 * u.deg,
):
    """
    Build the model used to compute CME arrival times, with the same set-up as get_cme_arrival_time
    (longitudes 300-420 deg, SURF3d latitudes -15 to +15 deg, dt_scale 20) but a configurable run
    length, from a single run's cme_pars.nc dataset.
    :param ds: Dataset loaded from one run's cme_pars.nc
    :param use_model: "surf", "compress_surf", "compress_surf3d" or "huxt"
    :param sim_time: Model run length (CMEs arriving after the end of the run count as misses)
    :param dt_scale: Model output time step scale
    :param latitude_min, latitude_max: SURF3d latitude range (compress_surf3d)
    :return: (model, model start datetime)
    """
    start_time = pd.to_datetime(
        np.ravel(ds["model_init_time"].values)[0]
    ).to_pydatetime()
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
            v_boundary=v_bound_in,
            cr_num=cr_num,
            cr_lon_init=ert.lon_c.to(u.deg),
            latitude=ert.lat.to(u.deg),
            lon_start=lon_start.to(u.rad),
            lon_stop=lon_end.to(u.rad),
            simtime=sim_time,
            dt_scale=dt_scale,
            r_min=r_min_in,
            solver="huxt" if use_model == "surf" else "hydro",
        )
    elif use_model in ["compress_surf3d"]:
        model = S.SURF3d(
            v_map=v_bound_in,
            v_map_long=ds["vr_longs"].values * u.deg,
            v_map_lat=ds["vr_lats"].values * u.deg,
            cr_num=cr_num,
            cr_lon_init=ert.lon_c.to(u.deg),
            latitude_min=latitude_min,
            latitude_max=latitude_max,
            lon_start=lon_start.to(u.rad),
            lon_stop=lon_end.to(u.rad),
            simtime=sim_time,
            dt_scale=dt_scale,
            r_min=r_min_in,
            solver="hydro",
        )
    else:
        model = H.HUXt(
            v_boundary=v_bound_in,
            cr_num=cr_num,
            cr_lon_init=ert.lon_c.to(u.deg),
            latitude=ert.lat.to(u.deg),
            lon_start=lon_start.to(u.rad),
            lon_stop=lon_end.to(u.rad),
            simtime=sim_time,
            dt_scale=dt_scale,
            r_min=r_min_in,
        )
    return model, start_time


def _arrival_for_cme(model, use_model, cme, cme_hit_object="EARTH"):
    """
    Solve the model for one CME and return its arrival at cme_hit_object
    (worker function for compute_arrival_times_per_obs)
    :return: (arrival datetime or None, arrival speed in km/s or NaN)
    """
    model.solve([cme])
    arr_time, arr_speed, hit, _ = get_arrival_time_and_speed(
        model, use_model, cme_hit_object
    )
    if not hit:
        return None, np.nan
    return arr_time, float(arr_speed)


def _animation_cme(par_ds, use_model, stage, member):
    """
    The cone CME to animate: an ensemble member at a stage, the ensemble's weighted mean, or the twin
    experiment's truth
    :param par_ds: The run's cme_pars.nc
    :param use_model: Model
    :param stage: Stage (0 = prior, k = after the k-th observation)
    :param member: Member index, "mean" (weighted mean of the parameters at the stage; longitude as
        a circular mean) or "truth"
    :return: (cone CME object, label for titles and file names, parameter values)
    """
    cme_cls = H.ConeCME if use_model in ["huxt"] else S.ConeCME
    model_init_time = pd.to_datetime(
        np.ravel(par_ds["model_init_time"].values)[0]
    ).to_pydatetime()
    par_names = ["t_init", "v", "width", "lon", "lat", "thick"]
    if member == "truth":
        attrs = par_ds.attrs
        if "true_v" not in attrs:
            raise ValueError(
                "member='truth' needs a twin run whose cme_pars.nc records its truth (true_* "
                "attributes, runs since 5 Nov 2026)"
            )
        t_init_s = (
            datetime.datetime.fromisoformat(str(attrs["true_t_init"])) - model_init_time
        ).total_seconds()
        values = [
            t_init_s,
            float(attrs["true_v"]),
            float(attrs["true_width"]),
            float(attrs["true_lon"]),
            float(attrs["true_lat"]),
            float(attrs["true_thick"]),
        ]
        label = "truth"
    elif member == "mean":
        lw = np.asarray(par_ds["log_weight"].values[stage], dtype=float)
        w = np.exp(lw - np.nanmax(lw))
        w = np.where(np.isfinite(w), w, 0.0)
        w = w / w.sum()
        values = []
        for p in par_names:
            x = np.asarray(par_ds[p].values[stage], dtype=float)
            if p == "lon":
                rad = np.deg2rad(x)
                values.append(
                    float(
                        np.rad2deg(
                            np.arctan2(np.sum(w * np.sin(rad)), np.sum(w * np.cos(rad)))
                        )
                    )
                )
            else:
                values.append(float(np.sum(w * x)))
        label = f"stage{stage:02d}_mean"
    else:
        values = [
            float(np.asarray(par_ds[p].values[stage], dtype=float)[member])
            for p in par_names
        ]
        label = f"stage{stage:02d}_member{member:03d}"
    v = values[1] * u.km / u.s
    cme = cme_cls(
        t_launch=values[0] * u.s + get_transit_time_to_r_min(par_ds, v),
        v=v,
        width=values[2] * u.deg,
        longitude=values[3] * u.deg,
        latitude=values[4] * u.deg,
        thickness=values[5] * u.solRad,
        cme_fixed_duration=True,
        fixed_duration=12 * 3600 * u.s,
    )
    return cme, label, tuple(np.round(values, 6))


def animate_surf_solution(
    base_file_path,
    runNo=-1,
    stage="final",
    member="best",
    kind="auto",
    use_model=None,
    sim_time_days=4.0,
    duration=10,
    fps=20,
    lon=0.0 * u.deg,
    lat=None,
    plot_rmax=None,
    dt_scale=20,
    out_dir=None,
):
    """
    Animate the model solar wind solution with the CME of ensemble members, the ensemble mean or the
    twin truth, using the animation functions of SURF (surf.surf_analysis.animate, animate_with_ts
    and animate_3d) or, for HUXt, huxt_analysis.animate. Each CME is solved again in the run's
    ambient solar wind (its cme_pars.nc), with the arrival-time set-up (make_arrival_model:
    longitudes 300-420 deg and, for compress_surf3d, the run's latitude range when recorded, else
    -15 to 15 deg). The model is solved ONCE per distinct CME: all the kinds of animation of that CME
    are made from the same solution, and a stage whose CME is the same as another's (e.g. the prior
    and final means when no observation changed the ensemble) reuses that solution.
    :param base_file_path: Directory containing the run_<runNo> folders
    :param runNo: Run number
    :param stage: "prior" (0), "final" (after the last observation), a stage number, or a list
    :param member: "best" (largest weight at the stage), "mean" (weighted mean of the parameters at
        the stage), "truth" (twin runs), a member index, or a list of these
    :param kind: "slice" (one latitude: animate, compressible solutions shown as speed, density and
        temperature), "with_ts" (one latitude with the time series at Earth: animate_with_ts),
        "3d" (compress_surf3d: radial-latitude and radial-longitude slices, animate_3d), "auto"
        (3d for compress_surf3d when ffmpeg is available, else slice), or a list of these
    :param use_model: Model (None = the run's, from cme_pars.nc or ens_elongations.nc)
    :param sim_time_days: Model run length (days from the model start)
    :param duration: Animation length (s)
    :param fps: Frames per second
    :param lon: Longitude of the radial-latitude slice ("3d")
    :param lat: Latitude of the slice ("slice", "with_ts": the nearest SURF3d latitude) and of the
        radial-longitude slice ("3d"); None = Earth's latitude at the model start
    :param plot_rmax: Outer radius of the plots (solar radii, no units; None = the model's)
    :param dt_scale: Model output time step scale (more frames for smaller values)
    :param out_dir: Folder for the animations (None = <run dir>/animations)
    :return: List of the files written
    """
    from matplotlib.animation import writers
    import surf.surf_analysis as SA

    run_dir = os.path.join(base_file_path, f"run_{runNo:03d}")
    par_file = os.path.join(run_dir, "cme_pars.nc")
    if not os.path.isfile(par_file):
        print(f"animate_surf_solution: {par_file} not found")
        return []
    par_ds = xr.load_dataset(par_file)
    n_stage = par_ds.sizes["n_obs"]
    n_ens = par_ds.sizes["n_ens"]
    elon_file = os.path.join(run_dir, "ens_elongations.nc")
    if use_model is None:
        use_model = str(par_ds.attrs.get("use_model", ""))
        if not use_model and os.path.isfile(elon_file):
            with xr.load_dataset(elon_file) as e_ds:
                use_model = str(e_ds.attrs.get("use_model", ""))
        use_model = use_model or "compress_surf3d"
    ffmpeg = writers.is_available("ffmpeg")

    # Stages
    stages = []
    for st in stage if isinstance(stage, (list, tuple)) else [stage]:
        st = 0 if st == "prior" else (n_stage - 1 if st == "final" else int(st))
        if not 0 <= st < n_stage:
            raise ValueError(f"stage {st} not in 0..{n_stage - 1}")
        if st not in stages:
            stages.append(st)

    # Kinds of animation
    kinds = []
    for k in kind if isinstance(kind, (list, tuple)) else [kind]:
        if k == "auto":
            k = "3d" if use_model == "compress_surf3d" and ffmpeg else "slice"
        if k not in ("slice", "with_ts", "3d"):
            raise ValueError(f"kind {k!r}: expected auto, slice, with_ts or 3d")
        if k == "3d" and use_model != "compress_surf3d":
            raise ValueError("kind='3d' needs compress_surf3d")
        if k == "3d" and not ffmpeg:
            print(
                "animate_surf_solution: SURF's animate_3d needs ffmpeg; animating a slice "
                "instead"
            )
            k = "slice"
        if k not in kinds:
            kinds.append(k)

    # Members
    members = member if isinstance(member, (list, tuple)) else [member]
    for m in members:
        if m not in ("best", "mean", "truth") and not (
            isinstance(m, (int, np.integer)) and 0 <= m < n_ens
        ):
            raise ValueError(
                f"member {m!r}: expected 'best', 'mean', 'truth' or a member number "
                f"0..{n_ens - 1}"
            )

    # The run's SURF3d latitude range when it records one
    lat_min = par_ds.attrs.get("model_lat_min_deg")
    lat_max = par_ds.attrs.get("model_lat_max_deg")
    lat_range = (
        (float(lat_min) * u.deg, float(lat_max) * u.deg)
        if lat_min is not None
        and lat_max is not None
        and np.isfinite(float(lat_min))
        and np.isfinite(float(lat_max))
        else (-15 * u.deg, 15 * u.deg)
    )
    out_dir = out_dir or os.path.join(run_dir, "animations")
    os.makedirs(out_dir, exist_ok=True)
    suffix = ".mp4" if ffmpeg else ".gif"

    # The CMEs to animate (the truth once, whatever the stage)
    jobs = []
    for st in stages:
        log_weight = np.asarray(par_ds["log_weight"].values[st], dtype=float)
        for m in members:
            if m == "truth" and any(j[1] == "truth" for j in jobs):
                continue
            m_st = int(np.nanargmax(log_weight)) if m == "best" else m
            jobs.append((st, m_st))

    written = []
    solved = {}  # CME parameters -> solved model (one model run per distinct CME)
    for st, m in jobs:
        cme, label, key = _animation_cme(par_ds, use_model, st, m)
        if key in solved:
            model, start_time = solved[key]
            print(
                f"Animating run {runNo} {label}: same CME as an earlier one, no new model run"
            )
        else:
            model, start_time = make_arrival_model(
                par_ds,
                use_model,
                sim_time=sim_time_days * u.day,
                dt_scale=dt_scale,
                latitude_min=lat_range[0],
                latitude_max=lat_range[1],
            )
            print(f"Animating run {runNo} {label} ({use_model}): solving the model")
            model.solve([cme])
            solved[key] = (model, start_time)
        lat_plot = lat
        if lat_plot is None:
            lat_plot = S.Observer("EARTH", Time(start_time, format="datetime")).lat.to(
                u.deg
            )
        tag = f"run{runNo}_{label}"
        if use_model == "huxt":
            import huxt.huxt_analysis as HA

            HA.animate(model, tag)
            print("HUXt saves its animation in its own figures folder")
            continue
        # One latitude: the nearest SURF3d slice, or the 2-D model itself
        if use_model == "compress_surf3d":
            k_lat = int(np.argmin(np.abs(model.lat.to(u.deg) - lat_plot)))
            slice_model = model.SURFlat[k_lat]
        else:
            slice_model = model
        for kd in kinds:
            out_path = os.path.join(out_dir, f"{use_model}_{label}_{kd}{suffix}")
            check_path_length(out_path, what="animation")
            if kd == "3d":
                SA.animate_3d(
                    model,
                    lon=lon,
                    lat=lat_plot,
                    tag=tag,
                    duration=duration,
                    fps=fps,
                    outputfilepath=out_path,
                )
                written.append(out_path)
            elif kd == "with_ts":
                path = SA.animate_with_ts(
                    slice_model,
                    tag=tag,
                    duration=duration,
                    fps=fps,
                    outputfilepath=out_path,
                    plot_rmax=plot_rmax,
                )
                written.append(str(path))
            else:
                path = SA.animate(
                    slice_model,
                    tag,
                    duration=duration,
                    fps=fps,
                    outputfilepath=out_path,
                    plot_rmax=plot_rmax,
                )
                written.append(str(path))
    print(
        f"animate_surf_solution: {len(written)} animation(s), {len(solved)} model run(s) for run "
        f"{runNo}"
    )
    return written


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
    arr_obs.nc, see compute_arrival_times_per_obs) into one figure:
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
    in_file = read_path(run_dir, "arrivals_per_obs")
    if not os.path.isfile(in_file):
        print(
            f"plot_arrival_time_histograms_per_obs: {in_file} not found "
            f"(run compute_arrival_times_per_obs first)"
        )
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
    if isinstance(
        real_arrival_time, (datetime.datetime, np.datetime64, pd.Timestamp)
    ) and not pd.isna(real_arrival_time):
        real_h = (
            pd.Timestamp(real_arrival_time) - pd.Timestamp(t_ref)
        ).total_seconds() / 3600.0
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
        q = (
            weighted_quantiles(
                hours[st][ok], w[ok], [band_quantiles[0], 0.5, band_quantiles[1]]
            )
            if np.any(ok)
            else np.full(3, np.nan)
        )
        return hist, q, frac_arrive

    hists, quants, fracs = {}, {}, {}
    for st in stages:
        hists[st], quants[st], fracs[st] = stage_hist(st)
        label = "prior" if st == 0 else f"after obs {st - 1}"
        med = to_time(quants[st][1]) if np.isfinite(quants[st][1]) else "none arrive"
        print(
            f"Stage {st:3d} ({label}): weighted median arrival {med}, "
            f"arriving weight {100 * fracs[st]:.0f}%"
        )

    col_prior = "#8c8b85"
    col_post = "#eb6834"
    col_real = "#1a1a19"
    centres = 0.5 * (edges[:-1] + edges[1:])
    fig, ax = plt.subplots(
        2,
        1,
        figsize=(13, 11),
        layout="constrained",
        gridspec_kw=dict(height_ratios=[1, 1.3]),
    )

    # Top: overlaid histograms
    axt = ax[0]
    s_first, s_last = stages[0], stages[-1]
    mids = stages[1:-1]
    if mids.size > n_overlay:
        mids = mids[np.round(np.linspace(0, mids.size - 1, n_overlay)).astype(int)]
    cmap = plt.get_cmap("Blues")
    for j, st in enumerate(mids):
        c = cmap(0.35 + 0.6 * j / max(1, mids.size - 1))
        when = (
            ""
            if pd.isna(stage_obs_time[st])
            else f" ({stage_obs_time[st]:%d %b %H:%M})"
        )
        axt.step(
            to_time(centres),
            hists[st],
            where="mid",
            color=c,
            lw=1.4,
            label=f"After obs {st - 1}{when}",
        )
    axt.fill_between(
        to_time(centres),
        hists[s_first],
        step="mid",
        color=col_prior,
        alpha=0.35,
        label="Prior" if s_first == 0 else f"Stage {s_first}",
    )
    axt.fill_between(
        to_time(centres),
        hists[s_last],
        step="mid",
        color=col_post,
        alpha=0.45,
        label=f"Final (after obs {s_last - 1})" if s_last > 0 else "Final",
    )
    if real_h is not None:
        axt.axvline(
            to_time(real_h),
            color=col_real,
            lw=2,
            ls=(0, (4, 2)),
            label="Observed arrival",
        )
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
    pcm = axb.pcolormesh(
        x_edges, y_edges, mat, cmap="Blues", shading="flat", vmin=0, vmax=np.nanmax(mat)
    )
    fig.colorbar(pcm, ax=axb, label=f"Probability per {bin_hours:g} h bin", pad=0.01)
    xs = np.arange(stages.size)
    q = np.array([quants[st] for st in stages])
    axb.plot(
        xs,
        to_time(q[:, 1]),
        color=col_post,
        lw=2,
        marker="o",
        ms=3,
        label="Weighted median",
    )
    axb.plot(
        xs,
        to_time(q[:, 0]),
        color=col_post,
        lw=1,
        ls=(0, (3, 2)),
        label=f"Weighted {int(round(100 * band_quantiles[0]))}-{int(round(100 * band_quantiles[1]))}%",
    )
    axb.plot(xs, to_time(q[:, 2]), color=col_post, lw=1, ls=(0, (3, 2)))
    if real_h is not None:
        axb.axhline(
            to_time(real_h),
            color=col_real,
            lw=2,
            ls=(0, (4, 2)),
            label="Observed arrival",
        )
    labels = [
        (
            "Prior"
            if st == 0
            else (
                f"{st - 1}\n{stage_obs_time[st]:%H:%M}"
                if not pd.isna(stage_obs_time[st])
                else f"{st - 1}"
            )
        )
        for st in stages
    ]
    step = max(1, int(np.ceil(stages.size / 25)))
    axb.set_xticks(xs[::step])
    axb.set_xticklabels(labels[::step], fontsize=8.5)
    axb.set_xlabel("Ensemble after observation (number, time UT)", fontsize=13)
    axb.set_ylabel("Arrival time (UT)", fontsize=13)
    axb.yaxis.set_major_formatter(mdates.DateFormatter("%d %b %H:%M"))
    axb.set_title(
        "Arrival-time distribution of the ensemble at every observation",
        loc="right",
        fontsize=14,
    )
    axb.legend(fontsize=10, frameon=False, loc="upper right")

    # Fraction of weight that arrives within the run, where some does not
    if any(fracs[st] < 0.999 for st in stages):
        trans = axb.get_xaxis_transform()
        for j, st in enumerate(stages):
            if fracs[st] < 0.999:
                axb.text(
                    xs[j],
                    0.985,
                    f"{100 * fracs[st]:.0f}%",
                    transform=trans,
                    fontsize=7,
                    ha="center",
                    va="top",
                    color="#b03030",
                )
        axb.text(
            0.0,
            1.0,
            "red %: weight of members arriving within the model run",
            transform=axb.transAxes,
            fontsize=8,
            color="#b03030",
            ha="left",
            va="bottom",
        )

    fig.suptitle(
        fig_title if fig_title else f"CME arrival times, run {runNo}", fontsize=16
    )
    if save_fig:
        name = write_path(run_dir, "arrival_hist_per_obs") + (
            f"_{craft}_assim" if craft is not None else ""
        )
        out_path = name + ".png"
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
            print(
                f"compute_rank_pit: run {run_no} skipped ({elon_file} or {par_file} not found)"
            )
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
    return dict(
        pit=pit,
        observation=obs_all,
        obs_time=obs_time,
        assimilated=assim_all,
        n_members=n_ens,
        run_nos=[r[0] for r in records],
        n_skipped=n_skipped,
    )


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

    res = compute_rank_pit(
        base_file_path, run_nos, obs_sd=obs_sd, include_obs_error=include_obs_error
    )
    if res is None:
        print(
            "plot_rank_histograms_per_obs: no runs with ens_elongations.nc and cme_pars.nc"
        )
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

    out_dir = (
        os.path.join(base_file_path, f"run_{res['run_nos'][0]:03d}")
        if len(res["run_nos"]) == 1
        else base_file_path
    )
    suffix = f"_{craft}_assim" if craft is not None else ""
    rank_label = f"Rank (members below observation, 0-{n_mem})" + (
        ", with obs. error" if include_obs_error else ""
    )
    col_bar = "#2a78d6"
    col_exp = "#1a1a19"
    col_grid = "#d9d8d2"
    ranks_axis = np.arange(n_mem + 1)
    expected = 1.0 / (n_mem + 1)

    def norm_counts(vals):
        counts = np.bincount(vals.astype(int), minlength=n_mem + 1)[: n_mem + 1].astype(
            float
        )
        return counts / counts.sum() if counts.sum() > 0 else counts

    # Ranks to a csv for reference
    rows = []
    for j, run_no in enumerate(res["run_nos"]):
        for k in range(n_obs):
            rows.append(
                dict(
                    run=run_no,
                    obs_no=k,
                    obs_time=obs_time[k],
                    observation=res["observation"][j, k],
                    assimilated=bool(res["assimilated"][j, k]),
                    pit=pit[j, k],
                    rank=rank[j, k],
                )
            )
    pd.DataFrame(rows).to_csv(
        os.path.join(out_dir, "rank_pit_per_obs.csv"), index=False
    )

    # 1. One rank histogram per observation time
    n_rows = int(np.ceil(ranked.size / n_cols))
    fig, axs = plt.subplots(
        n_rows,
        n_cols,
        figsize=(2.6 * n_cols, 2.2 * n_rows + 1.2),
        sharex=True,
        sharey=True,
        layout="constrained",
        squeeze=False,
    )
    for i, k in enumerate(ranked):
        ax = axs.flat[i]
        vals = rank[:, k][np.isfinite(rank[:, k])]
        ax.bar(ranks_axis, norm_counts(vals), width=0.9, color=col_bar)
        ax.axhline(expected, color=col_exp, lw=1, ls=(0, (4, 2)))
        when = "" if pd.isna(obs_time[k]) else f" {obs_time[k]:%d %b %H:%M}"
        y_txt = np.nanmean(res["observation"][:, k])
        ax.set_title(
            f"obs {k}{when}\n{y_txt:.2f}°, n={vals.size}", fontsize=9, loc="left"
        )
        ax.grid(True, axis="y", color=col_grid, lw=0.5)
        ax.set_axisbelow(True)
    for ax in axs.flat[ranked.size :]:
        ax.set_visible(False)
    for ax in axs[-1, :]:
        ax.set_xlabel("Rank", fontsize=9)
        ax.set_xlim(-0.6, n_mem + 0.6)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    for ax in axs[:, 0]:
        ax.set_ylabel("Normalised count", fontsize=9)
    runs_txt = f"{len(res['run_nos'])} run" + ("s" if len(res["run_nos"]) > 1 else "")
    fig.suptitle(
        (fig_title + ": " if fig_title else "")
        + f"rank histogram at each observation time ({runs_txt}, {n_mem} members; "
        f"dashed = reliable, 1/{n_mem + 1})",
        fontsize=13,
    )
    safe_savefig(
        fig, os.path.join(out_dir, f"rank_histograms_per_obs{suffix}.png"), dpi=130
    )
    plt.close(fig)

    # 2. Pooled rank histogram, and rank against observation time
    fig, ax = plt.subplots(
        1,
        2,
        figsize=(14, 5),
        layout="constrained",
        gridspec_kw=dict(width_ratios=[1, 1.6]),
    )
    vals = rank[np.isfinite(rank)]
    ax[0].bar(ranks_axis, norm_counts(vals), width=0.9, color=col_bar)
    ax[0].axhline(
        expected,
        color=col_exp,
        lw=1.2,
        ls=(0, (4, 2)),
        label=f"Reliable ensemble (1/{n_mem + 1})",
    )
    ax[0].set_xlabel(rank_label, fontsize=11)
    ax[0].set_ylabel("Normalised count", fontsize=11)
    ax[0].set_xlim(-0.6, n_mem + 0.6)
    ax[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    ax[0].set_title(
        f"Pooled over {np.isfinite(rank).any(axis=0).sum()} observations, {runs_txt} "
        f"(n = {vals.size})",
        loc="left",
        fontsize=11,
    )
    ax[0].legend(frameon=False, fontsize=9)
    ax[0].grid(True, axis="y", color=col_grid, lw=0.5)
    ax[0].set_axisbelow(True)

    t_ok = not obs_time[ranked].isna().any()
    x_all = obs_time if t_ok else np.arange(n_obs)
    ax[1].axhspan(
        0.05 * (n_mem + 1) - 0.5,
        0.95 * (n_mem + 1) - 0.5,
        color="#ecebe6",
        zorder=0,
        label="Central 90% of a reliable ensemble",
    )
    ax[1].axhline(n_mem / 2, color=col_grid, lw=1, zorder=0)
    cmap = plt.get_cmap("tab10")
    for j, run_no in enumerate(res["run_nos"]):
        ok = np.isfinite(rank[j])
        ax[1].plot(
            np.asarray(x_all)[ok],
            rank[j, ok],
            marker="o",
            ms=4,
            lw=0.8,
            color=col_bar if n_runs == 1 else cmap(j % 10),
            alpha=0.9 if n_runs <= 10 else 0.4,
            label=f"run {run_no}" if n_runs <= 10 else None,
        )
    ax[1].set_ylim(-0.5, n_mem + 0.5)
    ax[1].yaxis.set_major_locator(MaxNLocator(integer=True))
    ax[1].set_ylabel(rank_label, fontsize=11)
    ax[1].set_xlabel(
        "Observation time (UT)" if t_ok else "Observation number", fontsize=11
    )
    if t_ok:
        ax[1].xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H:%M"))
    ax[1].set_title(
        f"Rank of each observation (0: below all members, {n_mem}: above all members)",
        loc="left",
        fontsize=11,
    )
    ax[1].legend(frameon=False, fontsize=9, loc="best")
    ax[1].grid(True, color=col_grid, lw=0.5)
    ax[1].set_axisbelow(True)
    if res["n_skipped"]:
        ax[1].text(
            0.0,
            -0.18,
            f"{res['n_skipped']} observation(s) not ranked: some members had no elongation",
            transform=ax[1].transAxes,
            fontsize=8,
            color="#b03030",
        )
    fig.suptitle(
        (fig_title + ": " if fig_title else "")
        + "rank histogram of the forecast ensemble",
        fontsize=13,
    )
    safe_savefig(
        fig, os.path.join(out_dir, f"rank_histogram_summary{suffix}.png"), dpi=140
    )
    plt.close(fig)
    print(f"Saved rank histograms to {out_dir}")
    return res


class SkipEvent(Exception):
    """An event that cannot be analysed (e.g. no results folder); main() reports it and goes on"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Command-line options of osse_ana.py (any left out keep the values set in main())"""
    parser = argparse.ArgumentParser(
        description="Analyse data assimilation runs of one or more events or twin experiments "
        "(events.yaml), one after another. Settings not given here are set in main()."
    )
    parser.add_argument(
        "--event",
        nargs="+",
        metavar="EVENT",
        help="Events or twin experiments to analyse, in order, e.g. 09A 12A 13B "
        "(default: event_list in main())",
    )
    parser.add_argument(
        "--img", nargs="+", help="Image types, e.g. diff (default: img_list in main())"
    )
    parser.add_argument(
        "--start-run", type=int, help="First run number (default: start_run in main())"
    )
    parser.add_argument(
        "--n-runs", type=int, help="Number of runs (default: nRuns in main())"
    )
    parser.add_argument(
        "--animate",
        action="store_true",
        help="Make the SURF animations of each run (animate_surf_solution), even if "
        "make_surf_animation is off in main()",
    )
    parser.add_argument(
        "--animate-member",
        nargs="+",
        metavar="MEMBER",
        help="Members to animate: mean, best, truth (twin runs) or member numbers "
        "(default: animation_member in main())",
    )
    parser.add_argument(
        "--animate-stage",
        nargs="+",
        help="Stages to animate: prior, final or numbers (default: animation_stage in main())",
    )
    parser.add_argument(
        "--animate-kind",
        nargs="+",
        choices=["auto", "slice", "with_ts", "3d"],
        help="Animations: auto, slice, with_ts and/or 3d (default: animation_kind in main())",
    )
    parser.add_argument(
        "--no-animate",
        action="store_true",
        help="Do not make the SURF animations (make_surf_animation in main())",
    )
    args = parser.parse_args(argv)
    if args.n_runs is not None and args.n_runs < 1:
        parser.error("--n-runs must be 1 or more")
    return args


def main(argv: list[str] | None = None):
    args = parse_args(argv)
    load_environment()
    nRuns = 1
    start_run = -1
    t_initTruth = None
    vTruth = None
    widthTruth = None
    lonTruth = None
    latTruth = None

    obs_dir = os.getenv("OBS_DIR")
    # indep_cov\truth_20080101 - 0000_495_37.4_0_0_0\prior_20080101 - 0100_495_37.4_0_0_0\nEns - 5_8_300.0 deg_0.0 deg

    # Events and twin experiments of events.yaml to analyse, one after another (or give them on the
    #  command line: python osse_ana.py --event 09A 12A 13B). Each event's spacecraft comes from its
    #  code (09A -> STEREO-A, 13B -> STEREO-B, ssw_ -> STEREO-A, twin experiments -> twin).
    event_list = ["14B"]
    """
        "09A",
        "03A",
        "10A",
        "04B",
        "06B",
        "08B",
        "13B",
        "14B",
        "15B",
    ]  """  # ["twin"]
    img_list = ["diff"]  # ["norm", "diff"]
    par_type = "donki"
    use_model = "compress_surf3d"
    # Compute and plot the ensemble's full elongation profiles at every observation (one extra
    #  ensemble of model runs per distinct ensemble, i.e. per resampling step)
    make_profiles_per_obs = True
    # Compute every member's CME arrival time for the ensemble at every observation and plot the
    #  distributions as histograms. The model runs for arrival_sim_time_days from its start time; CMEs
    #  arriving later count as misses (4 days covers arrivals up to ~4 days after the model start).
    make_arrivals_per_obs = True
    arrival_sim_time_days = 4.0
    # Fill in the prior/posterior elongation profiles a data assimilation run could not compute
    #  (backfill_da_profiles: from elon_prof_obs.nc, computing that ensemble first if needed)
    backfill_missing_profiles = True
    # Rank (Talagrand) histograms of the observations within the forecast ensemble
    make_rank_histograms = False
    # Evolution of the observation-bias coefficient through the assimilation window (runs with an
    #  estimated bias term only)
    make_bias_plot = True
    # Animations of the SURF solution (animate_surf_solution, using SURF's own animate_3d and
    #  animate_with_ts): by default the ensemble-mean CME of the prior and of the final (posterior)
    #  ensemble, each as a 3-D slice movie and as a slice with the time series at Earth. One model
    #  run per distinct CME (2 per run here, 1 if the ensemble never changed); both kinds of movie
    #  come from the same run. Stages: "prior", "final" or numbers; members: "mean", "best",
    #  "truth" (twin runs) or numbers; kinds: "3d", "with_ts", "slice" or "auto".
    make_surf_animation = False
    animation_stage = ["prior", "final"]
    animation_member = "mean"
    animation_kind = ["3d", "with_ts"]
    animation_duration = 10
    animation_fps = 20
    n_ens = 50
    delta_aux_pf = 0.98  # initialise_da_parameters: delta_aux_pf
    # Twin experiments: folder with the run_NNN folders of the run to analyse (a folder, used when
    #  event_list holds one twin experiment, or {twin experiment: folder} for several), e.g.
    #  r"C:\Users\ss905122\PycharmProjects\SIR_HUXt\output\comp_surf\bias_cl3_WSA_v\ens_50_uncorr\model_COMPRESS_SURF3D\truth_495.0_37.4_0.0_0.0_0.0\ob_est_lin_0.00_2.00_prior_525.0_37.4_0.0_0.0_0.0_0.98"
    #  The truth plotted is the run's own (read_twin_truth), not necessarily the current events.yaml.
    #  None = the folder a run made with the current settings would be in: OUT_BASE_DIR (.env), the
    #  twin's entry in events.yaml and the da_surf_setup.Setup settings below
    twin_results_dir = os.path.join(
        "C:\\",
        "Users",
        "ss905122",
        "PycharmProjects",
        "SIR_HUXt",
        "output",
        "comp_surf",
        "bias_WSA_v",
        f"ens_{n_ens}_uncorr",
        f"model_{use_model.upper()}",
        "truth_495.0_37.4_0.0_0.0_0.0",
        f"ob_est_lin_0.00_2.00_fgs_ons_prior_495.0_37.4_0.0_0.0_0.0_{delta_aux_pf}",
    )

    # Observation-bias term (get_bias_settings: mode, predictor_type, prior_mean, prior_sd);
    #  None for runs with bias mode "off"
    twin_bias_folder = bias_dir_name("estimate", "linear", 0.0, 2.0)
    # Bias injected into the synthetic observations (get_bias_settings: true_bias)
    twin_true_bias = 0.0

    # Settings given on the command line
    if args.event:
        event_list = list(args.event)
    if args.img:
        img_list = list(args.img)
    if args.start_run is not None:
        start_run = args.start_run
    if args.n_runs is not None:
        nRuns = args.n_runs
    if args.animate:
        make_surf_animation = True
    if args.no_animate:
        make_surf_animation = False
    if args.animate_member:
        animation_member = [
            int(m) if m.lstrip("-").isdigit() else m for m in args.animate_member
        ]
        animation_member = (
            animation_member[0] if len(animation_member) == 1 else animation_member
        )
    if args.animate_stage:
        animation_stage = [int(x) if x.isdigit() else x for x in args.animate_stage]
    if args.animate_kind:
        animation_kind = list(args.animate_kind)

    known_events = load_events()
    unknown = [e for e in event_list if e not in known_events]
    if unknown:
        sys.exit(
            f"Unknown event(s) {unknown}: add them to {events_file()} "
            f"(events there: {', '.join(known_events)})"
        )
    twin_events = [e for e in event_list if is_twin_experiment(e)]
    if isinstance(twin_results_dir, str) and len(twin_events) > 1:
        print(
            f"Note: twin_results_dir is one folder but there are {len(twin_events)} twin experiments "
            f"({', '.join(twin_events)}); their folders are built from OUT_BASE_DIR instead. Give "
            f"twin_results_dir as {{twin experiment: folder}} to set each one."
        )

    all_comb_list = [(x, z) for x in event_list for z in img_list]
    assert use_model in ["surf", "compress_surf", "compress_surf3d", "huxt"]

    done: list[str] = []
    skipped: list[tuple[str, str]] = []
    for event, img in all_comb_list:
        # Observing spacecraft, from the event code
        craft = event_craft(event)
        craft_name = {"sta": "STEREO-A", "stb": "STEREO-B"}.get(craft, "twin")
        print(f"\nevent: {event}, craft: {craft}, img: {img}")
        # (no truth carried over from a previous event)
        t_initTruth = vTruth = widthTruth = lonTruth = latTruth = None
        try:

            if is_twin_experiment(event):
                # Truth and first guess of the twin experiment in events.yaml; the run folders are named
                #  from them as data_assimilation_routine.py names them
                truth = get_twin_cme(event, "truth")
                fg = get_twin_cme(event, "first_guess")
                true_cme_t_init = truth["t_init"]
                true_cme_speed = truth["speed"]
                true_cme_width = truth["width"]
                true_cme_lon = truth["lon"]
                true_cme_lat = truth["lat"]
                true_cme_thick = truth["thick"]
                truth_dir = (
                    f"truth_{true_cme_speed}_{true_cme_width}_{true_cme_lon}_{true_cme_lat}"
                    f"_{true_cme_thick}"
                )
                prior_dir = (
                    f"prior_{fg['speed']:.1f}_{fg['width']:.1f}_{fg['lon']:.1f}_{fg['lat']:.1f}"
                    f"_{fg['thick']:.1f}"
                )
                # Synthetic observations: the run folder is named from the truth
                obs_dir_name = truth_dir

                vTruth = true_cme_speed
                widthTruth = true_cme_width
                lonTruth = true_cme_lon

            else:
                # The event's values from events.yaml (see da_surf_setup.load_events)
                event_info: dict = get_event(event)
                if "arrival_time" not in event_info:
                    raise SkipEvent("no arrival_time in events.yaml")
                real_arrival_time = event_info["arrival_time"]
                real_arrival_speed = event_info.get("arrival_speed", 0)
                cme_hit_object = "EARTH"
                fig_title = event_info.get("title", "CME-{event}").format(
                    event=event, craft_name=craft_name
                )

                # Run folder of the prior: given in events.yaml for events whose runs use older folder
                #  names, otherwise named from the first guess as data_assimilation_routine.py names it
                use_other: bool = par_type != "donki"
                prior_dir = event_info.get(
                    "prior_dir_other" if use_other else "prior_dir"
                )
                if prior_dir is None:
                    fg = (
                        event_info.get("first_guess_other", event_info["first_guess"])
                        if use_other
                        else event_info["first_guess"]
                    )
                    prior_dir = (
                        f"prior_{float(fg['speed']):.1f}_{float(fg['width']):.1f}"
                        f"_{float(fg['lon']):.1f}_{float(fg['lat']):.1f}_{float(fg['thick']):.1f}"
                    )

                if f"obs_times_{craft}" in event_info:
                    # Observation times given in events.yaml (the ssw events)
                    obs_times = event_info[f"obs_times_{craft}"]
                    obs_dir_name = f"obs_{craft}_{event}_{img}"
                else:
                    # Observation times from the event's observation file
                    obs_file_path = os.path.join(obs_dir, event_info["obs_file"])
                    obs_class = Observations(
                        use_model=use_model,
                        use_synthetic_obs=False,
                        obs_filenames=obs_file_path,
                    )
                    obs_times, obs_lon, obs_lat, obs = obs_class.l5_read_obs_from_file()
                    # Run folder of the observations, as data_assimilation_routine.py names it
                    #  (obs_<event's spacecraft>_<event>_<image type>)
                    obs_dir_name = f"obs_{craft}_{event}_{img}"

            # Folder given for this twin experiment (twin_results_dir), if any
            twin_dir_given = (
                twin_results_dir.get(event)
                if isinstance(twin_results_dir, dict)
                else (twin_results_dir if len(twin_events) == 1 else None)
            )
            if is_twin_experiment(event) and twin_dir_given is not None:
                baseFilePath = twin_dir_given
            elif is_twin_experiment(event):
                # Twin results, in the folders data_assimilation_routine.make_output_dir makes:
                #  OUT_BASE_DIR/ens_<n>_uncorr/model_<MODEL>/truth_...[_tb<true bias>]/[<bias>_]prior_..._<delta>
                twin_obs_dir = obs_dir_name + (
                    true_bias_suffix(twin_true_bias) if twin_true_bias != 0 else ""
                )
                # Twins whose runs sample the first guess and/or the observation noise (events.yaml)
                twin_entry = get_event(event)
                sampling_tag = twin_sampling_dir_name(
                    bool(twin_entry.get("sample_first_guess", False)),
                    bool(twin_entry.get("new_obs_noise_per_run", False)),
                )
                twin_prior_dir = (
                    f"{sampling_tag}_{prior_dir}" if sampling_tag else prior_dir
                )
                twin_prior_dir = (
                    f"{twin_bias_folder}_{twin_prior_dir}"
                    if twin_bias_folder
                    else twin_prior_dir
                )
                baseFilePath = os.path.join(
                    os.getenv("OUT_BASE_DIR"),
                    f"ens_{n_ens}_uncorr",
                    f"model_{use_model.upper()}",
                    twin_obs_dir,
                    f"{twin_prior_dir}_{delta_aux_pf}",
                )
            else:
                baseFilePath = os.path.join(
                    "C:\\",
                    "Users",
                    "ss905122",
                    "PycharmProjects",
                    "SIR_HUXt",
                    "output",
                    "comp_surf",
                    "bias_WSA_v",
                    f"ens_{n_ens}_mo_cone",
                    f"model_{use_model.upper()}",
                    obs_dir_name,
                    f"ob_est_lin_0.00_2.00_{prior_dir}_0.98",  # "New folder",
                )

            # The observation-bias folder may be written with its short name (ob_est_const_0.00_2.00, used
            #  by runs since 31 Oct 2026) or the long name of older runs (obsbias_estimate_constant_prior0.00_2.00):
            #  whichever exists is used. Leave it out for runs without a bias term.
            baseFilePath = resolve_output_path(baseFilePath)
            print(f"baseFilePath = {baseFilePath}")
            if not os.path.isdir(baseFilePath):
                raise SkipEvent(f"results folder not found: {baseFilePath}")
            missing_runs = [
                r
                for r in range(start_run, start_run + nRuns)
                if not os.path.isfile(
                    os.path.join(baseFilePath, f"run_{r:03d}", "cme_pars.nc")
                )
            ]
            if missing_runs:
                raise SkipEvent(
                    f"no cme_pars.nc for run(s) {missing_runs} in {baseFilePath}"
                )
            # Warn if files written into the run folders would exceed the Windows path limit
            if path_too_long(
                os.path.join(baseFilePath, f"run_{start_run:03d}"), allowance=46
            ):
                print(
                    f"Warning: the run folders under {baseFilePath} leave fewer than 46 characters for file "
                    f"names before the Windows path limit ({PathLimits.MAX_PATH}); some figures or netCDF files may not "
                    f"be saved. Shorten the folder names (e.g. the bias folder to its ob_... name) or OUT_BASE_DIR."
                )

            # Read in all nc files and concatenate them into a single xarray object
            for ir, runNo in enumerate(range(start_run, start_run + nRuns)):
                print(runNo)

                filePath = os.path.join(baseFilePath, f"run_{runNo:03d}", "cme_pars.nc")
                with xr.load_dataset(filePath) as dsTemp:
                    dsTemp.expand_dims(dim={"run": 1})
                    dsTemp["run"] = (("run",), [runNo])

                    if ir == 0:
                        ds = dsTemp.copy()
                    else:
                        ds = xr.concat([ds, dsTemp], "run")

            if is_twin_experiment(event):
                # Truth of the run being analysed (it may have been made with other settings than the
                #  current events.yaml)
                # The experiment's copy of events.yaml, saved by the DA run (runs since 5 Nov 2026)
                events_copy = experiment_events_file(baseFilePath, start_run)
                twin_name = str(ds.attrs.get("twin_experiment", event))
                if events_copy is not None and is_twin_experiment(
                    twin_name, events_copy
                ):
                    print(f"Twin parameters from {events_copy} (entry {twin_name!r})")
                    fallback = dict(
                        get_twin_cme(twin_name, "truth", events_copy),
                        model_start=get_event(twin_name, events_copy)["model_start"],
                    )
                else:
                    fallback = dict(truth, model_start=get_event(event)["model_start"])
                run_truth, truth_source = read_twin_truth(
                    ds,
                    baseFilePath,
                    fallback,
                    fallback_from_run=events_copy is not None
                    and is_twin_experiment(twin_name, events_copy),
                )
                print(f"Truth CME (from {truth_source}): {run_truth}")
                if truth_source != "events.yaml" and any(
                    not np.isclose(float(run_truth[k]), float(truth[k]))
                    for k in ("speed", "width", "lon", "lat", "thick")
                ):
                    print(
                        f"Note: the run's truth differs from the truth of {event!r} in events.yaml"
                        f" ({truth}); the run's truth is used"
                    )
                true_cme_t_init = run_truth["t_init"]
                true_cme_speed = run_truth["speed"]
                true_cme_width = run_truth["width"]
                true_cme_lon = run_truth["lon"]
                true_cme_lat = run_truth["lat"]
                true_cme_thick = run_truth["thick"]
                vTruth = true_cme_speed
                widthTruth = true_cme_width
                lonTruth = true_cme_lon

            print(ds)

            def stage_mean(name):
                """Mean over the members (and over the runs, when ds holds several) per stage"""
                values = np.asarray(ds[name].values, dtype=float)
                return (
                    values.mean(axis=(0, 2))
                    if values.ndim == 3
                    else values.mean(axis=1)
                )

            n_runs_in_ds = int(ds.sizes["run"]) if "run" in ds["t_init"].dims else 1
            if n_runs_in_ds > 1:
                print(f"(means over the members of all {n_runs_in_ds} runs)")
            model_init = pd.to_datetime(
                np.ravel(ds["model_init_time"].values)[0]
            ).to_pydatetime()
            cme_t_init_mean = [
                model_init + datetime.timedelta(seconds=float(i))
                for i in stage_mean("t_init")
            ]
            cme_speed_mean = stage_mean("v")
            cme_width_mean = stage_mean("width")
            cme_lon_mean = stage_mean("lon")
            cme_lat_mean = stage_mean("lat")

            print("=======================================================")
            print(
                f"Mean CME launch time = Prior: {cme_t_init_mean[0]} - Post: {cme_t_init_mean[-1]}"
            )
            print(
                f"Mean CME speed = Prior: {cme_speed_mean[0]:.2f} - Post: {cme_speed_mean[-1]:.2f}"
            )
            print(
                f"Mean CME width = Prior: {cme_width_mean[0]:.2f} - Post: {cme_width_mean[-1]:.2f}"
            )
            print(
                f"Mean CME lon = Prior: {cme_lon_mean[0]:.2f} - Post: {cme_lon_mean[-1]:.2f}"
            )
            print(
                f"Mean CME lat = Prior: {cme_lat_mean[0]:.2f} - Post: {cme_lat_mean[-1]:.2f}"
            )
            print("=========================================================")

            if is_twin_experiment(event):
                real_arrival_time, real_arrival_speed = get_true_cme_arrival_time(
                    ds=ds,
                    use_model=use_model,
                    true_cme_t_init=true_cme_t_init,
                    true_cme_speed=true_cme_speed,
                    true_cme_width=true_cme_width,
                    true_cme_lat=true_cme_lat,
                    true_cme_lon=true_cme_lon,
                    true_cme_thick=true_cme_thick,
                    cme_hit_object="EARTH",
                )
            print(f"real_arrival_time = {real_arrival_time}")
            print(f"real_arrival_speed = {real_arrival_speed}")

            # Prior and posterior arrival times of each run (read from its arr_obs.nc; only those two
            #  stages are computed here if missing, with the members in parallel)
            for ir, runNo in enumerate(range(start_run, start_run + nRuns)):
                get_cme_arrival_time(
                    ds=ds,
                    runNo=runNo,
                    use_model=use_model,
                    real_arrival_time=real_arrival_time,
                    cme_hit_object="EARTH",
                    base_file_path=baseFilePath,
                    parallel_ens=True,
                    sim_time_days=arrival_sim_time_days,
                    event=event,
                )

            n_obs = len(ds["obs_no"][:])

            for ir, runNo in enumerate(range(start_run, start_run + nRuns)):
                if is_twin_experiment(event):
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
                    yTruth=t_initTruth,
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
                    yTruth=latTruth,
                    runNo=runNo,
                    parUnits=r"$^\circ$",
                    n_obs=n_obs,
                    base_file_path=baseFilePath,
                    craft=craft,
                )

                # Observation-bias coefficient through the assimilation window
                if make_bias_plot:
                    plot_bias_evolution(
                        base_file_path=baseFilePath,
                        runNo=runNo,
                        fig_title=fig_title,
                        craft=craft,
                    )

                # Animation of the SURF solution with a member's CME (SURF's animation functions)
                if make_surf_animation:
                    animate_surf_solution(
                        base_file_path=baseFilePath,
                        runNo=runNo,
                        stage=animation_stage,
                        member=animation_member,
                        kind=animation_kind,
                        use_model=use_model,
                        sim_time_days=arrival_sim_time_days,
                        duration=animation_duration,
                        fps=animation_fps,
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
                #  (first filling in any the data assimilation run could not compute)
                if backfill_missing_profiles:
                    backfill_da_profiles(
                        base_file_path=baseFilePath,
                        runNo=runNo,
                        use_model=use_model,
                        event=event,
                        cme_hit_object="EARTH",
                        sim_time_days=arrival_sim_time_days,
                    )
                plot_full_elongation_profiles(
                    base_file_path=baseFilePath,
                    runNo=runNo,
                    obs_sd=np.sqrt(0.5),
                    fig_title=fig_title,
                    craft=craft,
                )

                # Full elongation profiles of the ensemble at every observation (one model run per member
                #  of each distinct ensemble not yet computed; cached in elon_prof_obs.nc)
                if make_profiles_per_obs:
                    # (the same model runs also give the arrival times for arr_obs.nc when
                    #  make_arrivals_per_obs is set, so compute_arrival_times_per_obs below has
                    #  nothing left to run)
                    compute_elongation_profiles_per_obs(
                        base_file_path=baseFilePath,
                        runNo=runNo,
                        use_model=use_model,
                        cme_hit_object="EARTH",
                        sim_time_days=arrival_sim_time_days,
                        with_arrivals=make_arrivals_per_obs,
                        event=event,
                    )
                    plot_elongation_profiles_per_obs(
                        base_file_path=baseFilePath,
                        runNo=runNo,
                        obs_sd=np.sqrt(0.5),
                        fig_title=fig_title,
                        craft=craft,
                    )

                # CME arrival times of the ensemble at every observation, collated into histograms
                #  (one ensemble of model runs per distinct ensemble; cached in arr_obs.nc)
                if make_arrivals_per_obs:
                    compute_arrival_times_per_obs(
                        base_file_path=baseFilePath,
                        runNo=runNo,
                        use_model=use_model,
                        cme_hit_object="EARTH",
                        sim_time_days=arrival_sim_time_days,
                        event=event,
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
        except SkipEvent as reason:
            print(f"Skipping {event} ({img}): {reason}")
            skipped.append((f"{event} ({img})", str(reason)))
            continue
        except Exception as err:  # (one failing event does not stop the others)
            traceback.print_exc()
            print(f"Skipping {event} ({img}) after an error: {err!r}")
            skipped.append((f"{event} ({img})", f"error: {err!r}"))
            continue
        done.append(f"{event} ({img})")

    print("\n=======================================================")
    print(f"Analysed: {', '.join(done) if done else 'none'}")
    for name, reason in skipped:
        print(f"Skipped: {name}: {reason}")

    return None


if __name__ == "__main__":
    main()
