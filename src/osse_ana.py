import glob
import os
import sys
import json

from IPython.core.pylabtools import figsize
from astropy.time import Time
import astropy.units as u
import h5py
import matplotlib.pyplot as plt
import datetime
import matplotlib as mpl
import numpy as np

import huxt.huxt as H
import xarray as xr
import scipy.stats as st

import sunpy.coordinates.sun as sn
import seaborn as sns
import pandas as pd
import holoviews as hv
import colorcet as cc
from colorcet.plotting import sine_combs
from sunpy.coordinates.sun import orientation


#hv.extension('matplotlib')


def plot_par_values_over_single_run(
    ds, xVarName, yVarName, yTruth, xLabel, parNameLegend, runNo=0, parUnits='', n_obs=24, base_file_path=None
):
    fig, ax = plt.subplots(1, 1)
    xPlot = ds[xVarName][:].values

    yTrue = yTruth
    try:
        ds[yVarName][runNo, :, :].values
    except:
        yPlot = ds[yVarName][:, :].values
    else:
        yPlot = ds[yVarName][runNo, :, :].values
    yMean = np.median(yPlot, axis=1)

    nEns = len(ds["ens_no"].values)
    for ir in range(nEns):
        if ir == 0:
            ax.plot(xPlot, yPlot[:, ir], color='pink', alpha=0.5, label=f"{parNameLegend}")
        else:
            ax.plot(xPlot, yPlot[:, ir], color='pink', alpha=0.5)
    ax.plot(xPlot, yMean, color='red', label=f"Median {parNameLegend} over model runs")
    ax.plot([xPlot.min(), xPlot.max()], [yTrue, yTrue], color='k', linestyle='dashed', label=f"True {parNameLegend}")
    ax.legend()
    ax.set_title(f"Run_no = {runNo}")
    ax.set_xlim(0, n_obs)
    ax.set_xlabel(f"{xLabel}")
    ax.set_ylabel(f"{parNameLegend} ({parUnits})")
    if base_file_path is None:
        plt.show()
    else:
        filePath = os.path.join(
            baseFilePath, f"run_{runNo:03d}", f"{parNameLegend}_vs_obs_no.png"
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


def get_cme_arrival_time(ds, runNo=0, real_arrival_time=None, base_file_path=None):
    print(ds["model_init_time"].values)
    start_time = datetime.datetime.strptime(str(ds["model_init_time"].values), '%Y-%m-%dT%H:%M:%S.000000000')
    print(start_time)
    start_time_astro = Time(start_time, format='datetime')
    cr_num: int = np.trunc(sn.carrington_rotation_number(start_time))
    ert = H.Observer("EARTH", start_time_astro)
    lon_start = 300 * u.deg
    lon_end = 420 * u.deg

    # Initialise list
    cme_arrival_prior = []
    cme_arrival_post = []

    n_members = len(ds["ens_no"].values)

    model: HUXt = H.HUXt(
        v_boundary=ds["ambient_vr"].values * u.km / u.s,
        cr_num=cr_num,
        cr_lon_init=ert.lon_c.to(u.deg),
        latitude=ert.lat.to(u.deg),
        lon_start=lon_start.to(u.rad),
        lon_stop=lon_end.to(u.rad),
        simtime=5 * u.day,
        dt_scale=20,
        r_min=ds["r_min"].values * u.solRad,
        #        accel_limit=accel_limit
    )

    # Generate CME object
    # print(f"cme_launch_time = {cme_launch_time}")
    sec_to_cme_prior = [
        float(ds["t_init"][0, i].values) * u.s for i in range(n_members)
    ]
    v_prior = [
        ds["v"][0, i].values * u.km / u.s for i in range(n_members)
    ]
    width_prior = [
        ds["width"][0, i].values * u.deg for i in range(n_members)
    ]
    lon_prior = [
        ds["lon"][0, i].values * u.deg for i in range(n_members)
    ]
    lat_prior = [
        ds["lat"][0, i].values * u.deg for i in range(n_members)
    ]
    thick_prior = [
        ds["thick"][0, i].values * u.solRad for i in range(n_members)
    ]


    sec_to_cme_post = [
        float(ds["t_init"][-1, i].values) * u.s for i in range(n_members)
    ]
    v_post = [
        ds["v"][-1, i].values * u.km / u.s for i in range(n_members)
    ]
    width_post = [
        ds["width"][-1, i].values * u.deg for i in range(n_members)
    ]
    lon_post = [
        ds["lon"][-1, i].values * u.deg for i in range(n_members)
    ]
    lat_post = [
        ds["lat"][-1, i].values * u.deg for i in range(n_members)
    ]
    thick_post = [
        ds["thick"][-1, i].values * u.solRad for i in range(n_members)
    ]
    log_weights_post = [
        ds["log_weight"][-1, i].values for i in range(n_members)
    ]
    weights_post = np.array([
        np.exp(log_weights_post[i]) for i in range(n_members)
    ])

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

    prior_count_hit = 0
    post_count_hit = 0
    prior_stat_dict = {}
    post_stat_dict = {}

    for i in range(n_members):
        # Make prior CME arrival times
        model.solve([prior_cme_objects[i]])
        cme = model.cmes[0]
        prior_stats = cme.compute_arrival_at_body('EARTH')
        #print(f"prior_stats_{i}={stats}")
        prior_stat_dict[f"{i}"] = prior_stats

        if prior_stats['hit']:
            cme_arrival_prior.append(prior_stats['t_arrive'].datetime)
            prior_count_hit = prior_count_hit + 1
        else:
            cme_arrival_prior.append(np.nan)

        # Make posterior CME arrival times
        model.solve([post_cme_objects[i]])
        cme = model.cmes[0]
        post_stats = cme.compute_arrival_at_body('EARTH')
        post_stat_dict[f"{i}"] = post_stats

        #print(f"post_stats_{i} = {stats}")
        if post_stats['hit']:
            cme_arrival_post.append(post_stats['t_arrive'].datetime)
            post_count_hit = post_count_hit + 1
        else:
            cme_arrival_post.append(np.nan)

    prior_hit_rate = prior_count_hit / n_members
    post_hit_rate = post_count_hit / n_members


    print(f"Cme_arrival_prior = {cme_arrival_prior}")
    print(f"cme_arrival_post = {cme_arrival_post}")

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

    print(f"prior_CME_hit_rate = {prior_hit_rate * 100}%")
    if len(seconds_arrival_prior) > 0:
        avg_prior_arr_sec = np.nanmean(seconds_arrival_prior)
        avg_prior_arrival = base_time + datetime.timedelta(
            seconds=avg_prior_arr_sec
        )
        print(f"avg_prior_arrival = {avg_prior_arrival}")
    else:
        avg_prior_arr_sec = np.nan

    print(f"\nPost CME_hit_rate = {post_hit_rate * 100}%")
    if len(seconds_arrival_post) > 0:
        ind_req = ~np.isnan(seconds_arrival_post)
        weighted_avg_post_arr_sec = np.average(seconds_arrival_post[ind_req], weights=weights_post[ind_req])
        weighted_avg_post_arrival = base_time + datetime.timedelta(
            seconds=weighted_avg_post_arr_sec
        )
        print(f"weighted_avg_post_arrival = {weighted_avg_post_arrival}")

        avg_post_arr_sec = np.nanmean(seconds_arrival_post)
        avg_post_arrival = base_time + datetime.timedelta(
            seconds=avg_post_arr_sec
        )
        print(f"avg_post_arrival = {avg_post_arrival}")
    else:
        weighted_avg_post_arr_sec = np.nan
        avg_post_arr_sec = np.nan

    if real_arrival_time is not None:
        prior_arrival_error_hours = [
            ((pri_arr - real_arrival_time).total_seconds()) / 3600.0
            for pri_arr in cme_arrival_prior if isinstance(pri_arr, datetime.datetime)
        ]
        post_arrival_error_hours = [
            ((post_arr - real_arrival_time).total_seconds()) / 3600.0
            for post_arr in cme_arrival_post if isinstance(post_arr, datetime.datetime)
        ]
        weights_post_plot = [
            w for i, w in enumerate(weights_post) if isinstance(cme_arrival_post[i], datetime.datetime)
        ]
        print(np.max(weights_post_plot))
        fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True)
        ax[0].hist(prior_arrival_error_hours)
        ax[0].set_title("Prior CME Arrival Error at Earth")
        ax[0].set_ylim([0, 15])

        ax[0].set_ylabel("Count")
        ax[1].hist(post_arrival_error_hours)
        ax[1].set_title("Unweighted posterior CME Arrival Error at Earth")
        ax[1].set_ylim([0, 15])

        ax[2].hist(post_arrival_error_hours, weights=weights_post_plot)
        ax[2].set_title("Weighted posterior CME Arrival Error at Earth")
        ax[2].set_ylim([0, 1.5])
        ax[2].set_xlabel("Arrival time error (hours)")
        if base_file_path is None:
            plt.show()
        else:
            out_dir_path = os.path.join(base_file_path, f"run_{runNo:03d}")
            filePath = os.path.join(
                out_dir_path, "histogram_arrival_time_error.png"
            )
            plt.savefig(filePath)

            out_dict = {
                "prior_cme_arrival_s_from_20000101": list(seconds_arrival_prior),
                "post_cme_arrival_s_from_20000101": list(seconds_arrival_post),
                "prior_cme_hit_rate": prior_hit_rate,
                "post_cme_hit_rate": post_hit_rate,
                "avg_prior_arrival": float(avg_prior_arr_sec),
                "avg_post_arrival": float(avg_post_arr_sec),
                "weighted_avg_post_arrival": float(weighted_avg_post_arr_sec)
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
        plotPrior=False, cmap=cc.m_gouldian,
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
    if np.isnan(cbarMin + cbarMax):
        im = ax.hexbin(xValues, yValues, gridsize=30, cmap=cmap)
    else:
        im = ax.hexbin(xValues, yValues, gridsize=30, cmap=cmap, vmin=cbarMin, vmax=cbarMax)

    ax.plot(
        [xMin, xMax], [yTruth, yTruth],
        color='r', linestyle='dashed', label=f"True {yVarName}"
    )
    ax.plot(
        [xTruth, xTruth], [yMin, yMax],
        color='r', linestyle='dashed', label=f"True {yVarName}"
    )
    plt.colorbar(im, label='count', ax=ax, orientation='vertical')

    # Set limits if provided
    if not np.isnan(xMin + xMax):
        ax.set_xlim(xMin, xMax)

    if not np.isnan(yMin + yMax):
        ax.set_ylim(yMin, yMax)

    ax.set_xlabel(f"{xLabel}")
    ax.set_ylabel(f"{yLabel}")
    if plotPrior:
        ax.set_title(f'Histogram of prior {xVarName} against {yVarName}')
    else:
        ax.set_title(f'Histogram of posterior {xVarName} against {yVarName}')

    return None


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
    vTruth = 495
    widthTruth = 37.4
    lonTruth = 0

    #indep_cov\truth_20080101 - 0000_495_37.4_0_0_0\prior_20080101 - 0100_495_37.4_0_0_0\nEns - 5_8_300.0 deg_0.0 deg

    # baseFilePath = os.path.join(
    #     "C:\\", "Users", "ss905122", "PycharmProjects", "SIR_HUXt", "output2",
    #     "mo_cone",# "New folder",
    #     "truth_495.0_37.4_0.0_0.0_0.0", "prior_470_37.0_-4_0_0","0.98",
    # )
    event_list = ["ssw_007", "ssw_008", "ssw_009", "ssw_012"]
    craft_list = ["sta", "stb"]
    img_list = ["norm", "diff"]
    all_comb_list = [
        (x, y, z) for x in event_list for y in craft_list for z in img_list
    ]

    for (event, craft, img) in [("ssw_012", "stb", "norm")]:#all_comb_list:
        print(f"\nevent: {event}, craft: {craft}, img: {img}")
        #event = "ssw_007"
        #craft = "stb"
        #img = "diff"

        if event == "ssw_007":
            prior_dir = "prior_1010_66_-30_0_0"
            real_arrival_time = datetime.datetime(2012, 9, 3, 11, 23, 0)

        elif event == "ssw_008":
            prior_dir = "prior_872_110_20_4_0"
            real_arrival_time = datetime.datetime(2012, 9, 30, 22, 13, 0)
        elif event == "ssw_009":
            prior_dir = "prior_698_84_9_-24_0"
            real_arrival_time = datetime.datetime(2012, 10, 8, 4, 31, 0)
        elif event == "ssw_012":
            prior_dir = "prior_664_94_22_20_0"
            real_arrival_time = datetime.datetime(2012, 11, 23, 21, 12, 0)
        else:
            sys.exit(f"Unknown event {event}.")
        obs_dir_name = f"obs_{craft}_{event}_{img}"
        baseFilePath = os.path.join(
            "C:\\", "Users", "ss905122", "PycharmProjects", "SIR_HUXt", "output2", "MAS_v",
            "ens_50", "mo_cone", obs_dir_name, # "New folder",
            prior_dir, "0.98",
        )

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
        for ir, runNo in enumerate(range(start_run, start_run + nRuns)):
            get_cme_arrival_time(ds, runNo, real_arrival_time=real_arrival_time, base_file_path=baseFilePath)

        n_obs = len(ds["obs_no"][:])
        # for i in range(n_obs):
        #     plot_sample_cov(
        #         ds, 0, i, vars_req=["t_init", "v", "width", "lon", "lat"]
        #     )
        # # plot_sample_cov(
        #     ds, "all", -1, vars_req=["v", "lon", "width"]
        # )

        for ir, runNo in enumerate(range(nRuns)):

            plot_par_values_over_single_run(
                ds,
                "obs_no",
                "v",
                vTruth,
                "Observation number",
                "CME speed",
                runNo=runNo,
                parUnits="km/s",
                n_obs=n_obs
            )
            plot_par_values_over_single_run(
                ds,
                "obs_no",
                "width",
                widthTruth,
                "Observation number",
                "CME Width",
                runNo=runNo,
                parUnits="$^\circ$",
                n_obs=n_obs
            )
            plot_par_values_over_single_run(
                ds,
                "obs_no",
                "lon",
                lonTruth,
                "Observation number",
                "CME Longitude",
                runNo=runNo,
                parUnits="$^\circ$",
                n_obs=n_obs
            )
    sys.exit()

    colours_for_plots = sns.color_palette(cc.glasbey, n_colors=nRuns)
    plot_par_values_over_mult_runs(
        ds,
        "obs_no",
        "v",
        vTruth,
        "Observation number",
        "CME speed",
        parUnits="km/s"
    )
    plot_par_values_over_mult_runs(
        ds,
        "obs_no",
        "width",
        widthTruth,
        "Observation number",
        "CME Width",
        parUnits="$^\circ$"
    )
    plot_par_values_over_mult_runs(
        ds,
        "obs_no",
        "lon",
        lonTruth,
        "Observation number",
        "CME Longitude",
        parUnits="$^\circ$"
    )

    fig, axes = plt.subplots(figsize=(15, 10), nrows=2, ncols=3)
    axes = axes.flatten()
    xNames = ["v", "v", "width", "v", "v", "width"]
    yNames = ["width", "lon", "lon", "width", "lon", "lon"]

    xTruth = [vTruth, vTruth, widthTruth, vTruth, vTruth, widthTruth]
    yTruth = [widthTruth, lonTruth, lonTruth, widthTruth, lonTruth, lonTruth]

    xLab = ["CME speed (km/s)", "CME speed (km/s)", "CME Width (deg)", "CME speed (km/s)","CME speed (km/s)", "CME width (deg)"]
    yLab = ["CME Width (deg)", "CME Longitude (deg)", "CME Longitude (deg)", "CME Width (deg)", "CME Longitude (deg)", "CME Longitude (deg)"]

    xMin = [300, 300, 25, 300, 300, 25]
    yMin = [25, -20, -20, 25, -20, -20]

    xMax = [700, 700, 60, 700, 700, 60]
    yMax = [55, 20, 20, 55, 20, 20]

    """xMin = [380, 380, 24, 380, 380, 24]
    yMin = [24, -14, -14, 24, -14, -14]

    xMax = [630, 630, 52, 630, 630, 52]
    yMax = [52, 14, 14, 52, 14, 14]"""

    cbarMin = 0
    cbarMax = 50

    for i in range(6):
        if i < 3:
            plot_par_histograms_over_mult_runs(
                ds, axes[i], xNames[i], yNames[i],
                xTruth[i], yTruth[i], xLab[i], yLab[i],
                plotPrior=True, cmap=cc.m_gouldian, cbarMin=cbarMin, cbarMax=cbarMax,
                xMin=xMin[i], xMax=xMax[i], yMin=yMin[i], yMax=yMax[i]
            )
        else:
            plot_par_histograms_over_mult_runs(
                ds, axes[i], xNames[i], yNames[i],
                xTruth[i], yTruth[i], xLab[i], yLab[i],
                plotPrior=False, cmap=cc.m_gouldian, cbarMin=cbarMin, cbarMax=cbarMax,
                xMin=xMin[i], xMax=xMax[i], yMin=yMin[i], yMax=yMax[i]
            )
    plt.show()

    return None

if __name__ == "__main__":
    main()