import glob
import os

from IPython.core.pylabtools import figsize
from astropy.time import Time
import astropy.units as u
import h5py
import matplotlib.pyplot as plt

import matplotlib as mpl
import numpy as np

import xarray as xr
import scipy.stats as st

import sunpy.coordinates.sun as sn
import seaborn as sns
import holoviews as hv
import colorcet as cc
from colorcet.plotting import sine_combs
from sunpy.coordinates.sun import orientation


#hv.extension('matplotlib')


def plot_par_values_over_single_run(
    ds, xVarName, yVarName, yTruth, xLabel, parNameLegend, runNo=0, parUnits=''
):
    fig, ax = plt.subplots(1, 1)
    xPlot = ds[xVarName][:].values

    yTrue = yTruth
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
    ax.set_xlabel(f"{xLabel}")
    ax.set_ylabel(f"{parNameLegend} ({parUnits})")
    plt.show()

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
        im = ax.hexbin(xValues, yValues, gridsize=15, cmap=cmap)
    else:
        im = ax.hexbin(xValues, yValues, gridsize=15, cmap=cmap, vmin=cbarMin, vmax=cbarMax)

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


def main():
    nRuns = 100
    vTruth = 495
    widthTruth = 37.4
    lonTruth = 0

    #indep_cov\truth_20080101 - 0000_495_37.4_0_0_0\prior_20080101 - 0100_495_37.4_0_0_0\nEns - 5_8_300.0 deg_0.0 deg
    baseFilePath = os.path.join(
        "C:\\", "Users", "ss905122", "PycharmProjects", "SIR_HUXt", "output", "figures", "indep_cov",
        "truth_20080101-0000_495_37.4_0_0_0", "prior_20080101-0100_495_37.4_0_0_0", "nEns-50_8_300.0 deg_0.0 deg"
    )

    # Read in all nc files and concatenate them into a single xarray object
    for runNo in range(nRuns):
        print(runNo)
        filePath = os.path.join(
            baseFilePath, f"runNo_{runNo:04d}", "cme_pars.nc"
        )
        with xr.load_dataset(filePath) as dsTemp:
            dsTemp.expand_dims(dim={"run": 1})
            dsTemp["run"] = (("run",), [runNo])

            if runNo == 0:
                ds = dsTemp.copy()
            else:
                ds = xr.concat([ds, dsTemp], "run")

    print(ds)

    """for ir in range(nRuns):
        plot_par_values_over_single_run(
            ds,
            "obs_no",
            "v",
            vTruth,
            "Observation number",
            "CME speed",
            runNo=ir,
            parUnits="km/s"
        )
        plot_par_values_over_single_run(
            ds,
            "obs_no",
            "width",
            widthTruth,
            "Observation number",
            "CME Width",
            runNo=ir,
            parUnits="$^\circ$"
        )
        plot_par_values_over_single_run(
            ds,
            "obs_no",
            "lon",
            lonTruth,
            "Observation number",
            "CME Longitude",
            runNo=ir,
            parUnits="$^\circ$"
        )
"""
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

    xMin = [440, 440, 32, 440, 440, 30]
    yMin = [32, -9, -9, 30, -9, -9]

    xMax = [520, 520, 42.5, 520, 520, 42.5]
    yMax = [42, 1, 1, 45, 2, 2]

    xMin = [380, 380, 24, 380, 380, 24]
    yMin = [24, -14, -14, 24, -14, -14]

    xMax = [630, 630, 52, 630, 630, 52]
    yMax = [52, 14, 14, 52, 14, 14]

    cbarMin = 0
    cbarMax = 150

    for i in range(6):
        if i < 3:
            plot_par_histograms_over_mult_runs(
                ds, axes[i], xNames[i], yNames[i],
                xTruth[i], yTruth[i], xLab[i], yLab[i],
                plotPrior=True, cmap=cc.m_gouldian,# cbarMin=cbarMin, cbarMax=cbarMax,
                xMin=xMin[i], xMax=xMax[i], yMin=yMin[i], yMax=yMax[i]
            )
        else:
            plot_par_histograms_over_mult_runs(
                ds, axes[i], xNames[i], yNames[i],
                xTruth[i], yTruth[i], xLab[i], yLab[i],
                plotPrior=False, cmap=cc.m_gouldian,# cbarMin=cbarMin, cbarMax=cbarMax,
                xMin=xMin[i], xMax=xMax[i], yMin=yMin[i], yMax=yMax[i]
            )
    plt.show()

    return None

if __name__ == "__main__":
    main()