"""
osse_plots.py

Plots of data-assimilation output (the ``cme_pars.nc`` datasets written by data_assimilation_routine):
    plot_par_values_over_single_run    : every member's parameter value vs observation number for one run
    plot_par_values_over_mult_runs     : median over members, for several runs
    plot_par_histograms_over_mult_runs : 2D hexbin histogram of two parameters (prior or posterior)
    plot_sample_cov                    : correlation heat map of the ensemble at one observation
"""
import os

import colorcet as cc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr

from osse_stats import weighted_median


def plot_par_values_over_single_run(
        ds: xr.Dataset,
        xVarName: str,
        yVarName: str,
        xLabel: str,
        fig_title: str,
        parNameLegend: str,
        yTruth: float | None = None,
        obs_times: list | None = None,
        tick_cadence: int | None = None,
        craft: str | None = None,
        runNo: int = 0,
        parUnits: str = '',
        n_obs: int = 24,
        base_file_path: str | None = None
) -> None:
    """
    Plot every ensemble member's value of one CME parameter against observation number for one run,
    with the ensemble median and mean (and the truth, if given).
    :param ds: DA output dataset (optionally with a leading "run" dimension)
    :param xVarName: x coordinate (usually "obs_no")
    :param yVarName: Parameter to plot, e.g. "v"
    :param xLabel, fig_title, parNameLegend, parUnits: Axis label, title, legend name and units
    :param yTruth: True value (drawn as a dashed line if not None)
    :param obs_times: Observation datetimes, used to label every tick_cadence-th tick
    :param tick_cadence: Tick spacing (in observations) when obs_times is given
    :param craft: Added to the file name when given
    :param runNo: Run to plot (used to index ds if it has a run dimension, and in the output path)
    :param n_obs: x-axis upper limit
    :param base_file_path: If None show the figure, otherwise save it to <base>/run_NNN/<name>.png
    """
    if (obs_times is not None) and (tick_cadence is not None):
        assert tick_cadence > 0

    fig, ax = plt.subplots(
        figsize=(10, 6), nrows=1, ncols=1, layout="constrained"
    )
    xPlot = ds[xVarName][:].values

    yTrue = yTruth
    # Datasets with a run dimension are indexed by runNo; single-run datasets are used as they are
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
    # (the weighted median / mean are computed above but not currently plotted)
    ax.plot(xPlot, y_median, color='red', label=f"Median {parNameLegend}")
    ax.plot(xPlot, y_mean, color='blue', label=f"Mean {parNameLegend}")
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
        plt.close(fig)
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
        plt.close(fig)

    return None


def plot_par_values_over_mult_runs(
    ds: xr.Dataset,
    xVarName: str,
    yVarName: str,
    yTruth: float,
    xLabel: str,
    parNameLegend: str,
    parUnits: str = ''
) -> None:
    """
    Plot the ensemble-median value of a parameter against observation number for every run, plus the
    median over runs and the truth.
    :param ds: DA output datasets concatenated along a "run" dimension
    :param xVarName, yVarName: Coordinate / parameter names
    :param yTruth: True value of the parameter
    :param xLabel, parNameLegend, parUnits: Axis label, legend name and units
    """
    fig, ax = plt.subplots(1, 1)
    xPlot = ds[xVarName][:].values

    yTrue = yTruth
    yPlot = np.median(ds[yVarName], axis=2)

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
    plt.close(fig)

    return None


def plot_par_histograms_over_mult_runs(
    ds: xr.Dataset, ax: plt.Axes, xVarName: str, yVarName: str, xTruth: float, yTruth: float,
        xLabel: str, yLabel: str,
        plotPrior: bool = False, cmap=cc.m_gouldian, grid_size: int = 10,
        xMin: float = np.nan, xMax: float = np.nan, yMin: float = np.nan, yMax: float = np.nan,
        cbarMin: float = np.nan, cbarMax: float = np.nan
):
    """
    2D hexbin histogram of two parameters over all runs and members, for the prior (obs index 0)
    or the posterior (last obs index), with the truth marked when axis limits are given.
    :param ds: DA output datasets concatenated along a "run" dimension
    :param ax: Axes to draw on
    :param xVarName, yVarName: Parameters on the x / y axes
    :param xTruth, yTruth: True values (drawn as dashed lines)
    :param xLabel, yLabel: Axis labels
    :param plotPrior: Plot the prior instead of the posterior
    :param cmap, grid_size: Colour map and hexbin grid size
    :param xMin, xMax, yMin, yMax: Axis limits (NaN = automatic)
    :param cbarMin, cbarMax: Colour-bar limits (NaN = automatic)
    :return: im: The hexbin collection (for a colour bar)
    """
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
        ds: xr.Dataset, run_no: int | str, obs_no: int, vars_req: list[str] = ["v", "lon", "width"]
) -> None:
    """
    Heat map of the correlation matrix of the ensemble at one observation index.
    :param ds: DA output datasets concatenated along a "run" dimension
    :param run_no: Run to use, or "all" to pool the members of every run
    :param obs_no: Observation index (0 = prior)
    :param vars_req: Parameters to include
    """
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

    tick_labels = [
        i for i in df.columns
    ]

    fig, ax = plt.subplots(1, 1)
    sns.heatmap(
        df.corr(), annot=True, cmap=plt.cm.RdBu_r,
        vmin=-1, vmax=1, ax=ax,
        xticklabels=tick_labels, yticklabels=tick_labels
    )
    ax.set_title(f"Correlation matrix for obs_no = {obs_no}, run_no = {run_no}")
    plt.show()
    plt.close(fig)

    return None
