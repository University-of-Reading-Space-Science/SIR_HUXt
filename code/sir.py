#@package sir
# This module will take inputs from HUXt and perform an SIR to
#  generate an updated set of weights and particles
import numpy as np
import datetime
import os
import sys
import pandas as pd
import xarray as xr
import json

import huxt.huxt as H
import huxt.huxt_analysis as HA
from scipy.special.cython_special import log_wright_bessel

import sir_huxt_mono_obs as shmo
import sunpy.coordinates.sun as sn
import astropy.units as u
from astropy.time import Time
import matplotlib.pyplot as plt
import seaborn as sns
import colorcet as cc

def setup_huxt(
        start_datetime=datetime.datetime(2008, 1, 1, 0, 0, 0),
        vr_in=np.zeros(128) + 400 * u.km / u.s,
        lon_start=290*u.deg,
        lon_stop=380*u.deg,
        sim_time=2*u.day,
        dt_scale=20,
        r_min=30 * u.solRad
):
    """
    Initialise HUXt with some predetermined boundary/initial conditions
    Here a uniform 400km/s wind is used, and HUXt time is set to 2008-01-01T00:00:00.
    :param start_datetime: Initial datetime of HUXt simulation
    :param vr_in: Initial radial solar wind speed in km/s
    :param lon_start: Initial longitude of HUXt simulation in degrees
    :param lon_stop: Final longitude of HUXt simulation in degrees
    :param sim_time: HUXt simulation time in seconds
    :param dt_scale: Scalar specifying cadence of output timesteps

    :return model: HUXt model object with required ambient wind conditions at required
                    longitudes and latitude
    """
    assert(type(start_datetime) == datetime.datetime)

    start_time = Time(start_datetime, scale='utc')
    cr_num = np.fix(sn.carrington_rotation_number(start_time))
    ert = H.Observer('EARTH', start_time)

    #vr_in = np.asarray(vr_in)
    #print(sim_time)
    # Set up HUXt for a sim_time-day simulation, outputting every dt_scale
    model = H.HUXt(
        v_boundary=vr_in,
        cr_num=cr_num,
        cr_lon_init=ert.lon_c.to(u.deg),
        latitude=ert.lat.to(u.deg),
        lon_start=lon_start.to(u.rad),
        lon_stop=lon_stop.to(u.rad),
        simtime=sim_time,
        dt_scale=dt_scale,
        r_min=r_min,
        accel_limit=False
    )

    # model1d = H.HUXt(v_boundary=vr_in, cr_num=cr_num, cr_lon_init=ert.lon_c, latitude=ert.lat.to(u.deg),
    #                  lon_out=0 * u.deg, simtime=5 * u.day, dt_scale=4)

    return model


def initialise_cme_parameter_ensemble_arrays(
        n_ensemble,
        huxt_init_time=datetime.datetime(2008, 1, 1, 0, 0, 0)
):
    """
    Function to initialise empty arrays for storing the CME parameters for each ensemble member at each analysis step
    :param n_ensemble: The number of ensemble members in the SIR analysis
    :return parameter_arrays: A dictionary of parameter keys and an empty array for storing each ensemble member value
    """
    keys = [
        't_init', 'v', 'width', 'lon', 'lat', 'thick',
        'huxt_init_time', 't_transit', 'v_hit',
        'likelihood', 'weight'
    ]
    if n_ensemble > 1:
        parameter_arrays = {k:np.zeros(n_ensemble) for k in keys}
        parameter_arrays['n_members'] = n_ensemble
        parameter_arrays['huxt_init_time'] = huxt_init_time

        #Initialise the weights in log-space
        parameter_arrays['weight'] = -np.log(n_ensemble) * np.ones(n_ensemble)
        #print(list(parameter_arrays.keys()))
    else:
        parameter_arrays = {k: 0 for k in keys}
        parameter_arrays['n_members'] = 1
        parameter_arrays['huxt_init_time'] = huxt_init_time

    return parameter_arrays


def zscore(x):
    """
    Compute z-scores of a parameter
    :param x: An array of values
    :return x_z: An array of Z scores of x
    :return x_avg: The mean of x
    :return x_std: The standard of deviation of x
    """

    x_avg = np.mean(x)
    x_std = np.std(x)
    x_z = (x - x_avg) / x_std

    return x_z, x_avg, x_std


def inv_zscore(x_z, x_avg, x_std):
    """
    Compute the inverse zscore transform to go from a zscore back to a state variable.
    :param x_z: An array of Z scores of x
    :param x_avg: The mean of x
    :param x_std: The standard of deviation of x
    :return x: An array of values
    """

    x = x_std * x_z + x_avg

    return x

def cme_par_to_z_state_vector(cme_par_array, par_req):
    ## Function to convert the CME parameters into a state vector
    #  @param cme_par_array Array that contains all CME parameters required
    #  @param par_req List containing the names of all the required parameters from cme_par_array

    # Extract the ensemble of required parameters
    nPar = len(par_req)

    # Extract weights of particles
    weights = cme_par_array['weight']

    # If there are non-finite weights, set weight to zero (i.e. discard particle)
    weightCond = ~np.isfinite(weights)
    if np.sum(weightCond) > 0:
        weights[weightCond] = 0

    #Extract number of ensemble members from length of weights array and initialise variables
    nEns = len(weights)

    state_ens = np.zeros((nEns, nPar))
    z_ens = np.zeros((nEns, nPar))
    x_mean = np.zeros(nPar)
    x_std = np.zeros(nPar)

    # Extract all required parameters, put them in an ensemble matrix, then calculate the zscores for output
    for i, ip in enumerate(par_req):
        par_temp = cme_par_array[ip]
        if ip == 'lon':
            lonCond = par_temp > 180
            par_temp[lonCond] = par_temp[lonCond] - 360

        state_ens[:, i] = cme_par_array[ip]

        #Convert to zscores
        z_ens[:, i], x_mean[i], x_std[i] = zscore(state_ens[:, i])

    return z_ens, x_mean, x_std


def z_state_vector_to_cme_par(z_ens, x_mean, x_std, weights, par_req, cme_par_array):
    ## Function to convert the state vector back into the required CME parameters
    #  @param z_ens (nEns, nPar)-array containing the state ensemble converted to z-scores
    #  @param x_mean (nPar)-array containing the mean of the ensemble parameters
    #  @param x_std (nPar)-array containing the standard deviation of the ensemble parameters
    #  @param par_req List containing the names of all the required parameters from cme_par_array
    #  @param cme_par_array Array that contains all CME parameters required
    #  @return cme_par_array Updated parameter dictionary

    # Extract the ensemble of required parameters
    nEns, nPar = np.shape(z_ens)

    # Put weights of particles into cme_par_array
    cme_par_array['weight'] = weights

    # Extract all required parameters, put them in an ensemble matrix, then calculate the zscores for output
    for i, ip in enumerate(par_req):
        par_temp = inv_zscore(z_ens[:, i], x_mean[i], x_std[i])

        if ip == 'lon':
            lonCond = par_temp < 0
            par_temp[lonCond] = par_temp[lonCond] + 360

        cme_par_array[ip] = par_temp

    return cme_par_array


def add_units_from_cme_par(
        cme_par_array,
        par_names=['t_init', 'v', 'width', 'lon', 'lat', 'thick']
):
    """
    Routine to remove units from the CME parameters
    :param cme_par_array: Dictionary containing CME parameters with following keys:
        ['t_init', 'v', 'width', 'lon', 'lat', 'thick', 'weight']
    :return: CME parameters in appropriate units but with astropy units removed
    """
    nPar = len(par_names)
    nEns = len(cme_par_array['weight'])

    cme_pars = np.zeros((nEns, nPar))

    # Standardise the units and remove the astropy units
    cme_pars[:, 0] = [
        cme_par_array['huxt_init_time'] + datetime.timedelta(seconds=cme_par_array['t_init'][i])
        for i in range(nEns)
    ]
    #print(cme_par_array['width'])
    cme_par_array['v'] = cme_par_array['v'] * u.km / u.s
    cme_par_array['width'] = cme_par_array['width'] * u.deg

    cme_par_array['lon'] = cme_par_array['lon'] * u.deg
    lonCond = cme_par_array['lon'] < 0
    cme_par_array['lon'][lonCond, 3] = cme_par_array['lon'][lonCond, 3] + 360

    cme_par_array['lat'] = cme_par_array['lat'].to(u.deg).value
    cme_par_array['thick'] = cme_par_array['thick'].to(u.solRad).value

    return cme_par_array


def remove_units_from_cme_par(
        cme_par_array,
        par_names=['t_init', 'v', 'width', 'lon', 'lat', 'thick']
):
    """
    Routine to remove units from the CME parameters
    :param cme_par_array: Dictionary containing CME parameters with following keys:
        ['t_init', 'v', 'width', 'lon', 'lat', 'thick', 'weight']
    :return: CME parameters in appropriate units but with astropy units removed
    """
    nPar = len(par_names)
    nEns = len(cme_par_array['weight'])

    cme_pars = np.zeros((nEns, nPar))

    # Standardise the units and remove the astropy units
    cme_pars[:, 0] = [
        (cme_par_array['t_init'][i] - cme_par_array['huxt_init_time']).total_seconds()
        for i in range(nEns)
    ]
    #print(cme_par_array['width'])
    cme_pars[:, 1] = cme_par_array['v'].to(u.km / u.s).value
    cme_pars[:, 2] = cme_par_array['width'].to(u.deg).value

    cme_pars[:, 3] = cme_par_array['lon'].to(u.deg).value
    lonCond = cme_pars[:, 3] > 180
    cme_pars[lonCond, 3] = cme_pars[lonCond, 3] - 360

    cme_pars[:, 4] = cme_par_array['lat'].to(u.deg).value
    cme_pars[:, 5] = cme_par_array['thick'].to(u.solRad).value

    return cme_pars


def cme_par_to_state_vector(cme_pars, par_req):
    ## Function to convert the CME parameters into a state vector
    #  @param cme_pars Array that contains all CME parameters required
    #  @param par_req List containing the names of all the required parameters from cme_par_array

    # Extract the ensemble of required parameters
    nPar = len(par_req)

    """# Extract weights of particles
    #weights = cme_par_array['weight']

    # If there are non-finite weights, set weight to zero (i.e. discard particle)
    weightCond = ~np.isfinite(weights)
    if np.sum(weightCond) > 0:
        weights[weightCond] = 0
"""
    #Extract number of ensemble members from length of weights array and initialise variables
    nEns = len(cme_pars[:, 0])

    state_ens = np.zeros((nEns, nPar))

    # Extract all required parameters, put them in an ensemble matrix
    for i, ip in enumerate(par_req):
        #par_temp = cme_par_array[ip]
        """if ip == 'v':
            par_temp = par_temp#.to(u.km/u.s).value
        elif ip == 'lon':
            lonCond = par_temp > 180 # * u.deg
            par_temp[lonCond] = par_temp[lonCond] - 360 #.to(u.deg).value - 360
        elif ip == 'lat':
            par_temp = par_temp#.to(u.deg).value
        elif ip == 'width':
            par_temp = par_temp#.to(u.deg).value
        elif ip == 'thick':
            par_temp = par_temp#.to(u.solRad).value
        elif ip == 't_init':
            par_temp = (
                cme_par_array['t_init'] - cme_par_array['huxt_init_time']
            ).total_seconds() # * u.s"""
        if ip == 't_init':
            indReq = 0
        elif ip == 'v':
            indReq = 1
        elif ip == 'width':
            indReq = 2
        elif ip == 'lon':
            indReq = 3
        elif ip == 'lat':
            indReq = 4
        elif ip == 'thick':
            indReq = 5
        else:
            #print("Invalid parameter requested. Exiting.")
            sys.exit()

        state_ens[:, i] = cme_pars[:, indReq]

    return state_ens


def get_particle_weights(cme_par_array):
    ## Function to convert the CME parameters into a state vector
    #  @param cme_pars Dictionary that contains all CME parameters required, including weights

    # Extract weights of particles
    weights = cme_par_array['weight']

    # If there are non-finite weights, set weight to zero (i.e. discard particle)
    weightCond = ~np.isfinite(weights)
    if np.sum(weightCond) > 0:
        weights[weightCond] = 0

    return weights


def state_vector_to_cme_par(state_ens, weights, par_req, cme_par_array):
    ## Function to convert the state vector back into the required CME parameters
    #  @param z_ens (nEns, nPar)-array containing the state ensemble converted to z-scores
    #  @param x_mean (nPar)-array containing the mean of the ensemble parameters
    #  @param x_std (nPar)-array containing the standard deviation of the ensemble parameters
    #  @param par_req List containing the names of all the required parameters from cme_par_array
    #  @param cme_par_array Array that contains all CME parameters required
    #  @return cme_par_array Updated parameter dictionary

    # Extract the ensemble of required parameters
    nEns, nPar = np.shape(state_ens)

    # Put weights of particles into cme_par_array
    cme_par_array['weight'] = weights

    # Extract all required parameters, put them in an ensemble matrix, then calculate the zscores for output
    for i, ip in enumerate(par_req):
        par_temp = state_ens[:, i]

        if ip == 't_init':
            par_temp = (
                    cme_par_array['huxt_init_time']
                    + datetime.timedelta(seconds=par_temp)
            )
        elif ip == 'v':
            par_temp = par_temp * u.km/u.s
        elif ip == 'width':
            par_temp = par_temp * u.deg
        elif ip == 'lon':
            par_temp = par_temp * u.deg
            lonCond = par_temp < 0 * u.deg
            par_temp[lonCond] = par_temp[lonCond] + (360  * u.deg)
        elif ip == 'lat':
            par_temp = par_temp * u.deg
        elif ip == 'thick':
            par_temp = par_temp * u.solRad

        cme_par_array[ip] = par_temp

    return cme_par_array


# Systematic resampling routine for parameters
def systematic_resampling(weights, rng=None):
    ## Definition to perform systematic resampling
    #  @param weights (n_Ens) array contain each particle's weight
    #  @param rng Seed for random generator
    #  @return resampInd Indices of resampled particles
    if rng is None:
        rng = np.random.default_rng()

    # Calculate number of ensemble members required and initialise output
    #  variable to hold resampled particles indices
    nEns = len(weights)
    resampInd = np.zeros(nEns, dtype=int)

    # Define interval for particles to resample
    positions = (rng.random() + np.arange(nEns)) / nEns

    # Calculate cumulative sum for all weights
    cumSum = np.cumsum(weights)

    i, j = 0, 0
    while i < nEns:
        if positions[i] < cumSum[j]:
            resampInd[i] = j
            i += 1
        else:
            j += 1

    return resampInd


def shrink_par(pars, log_weights=None, delta=0.98):
    ## Liu-West shrinkage of the parameters towards the weighted mean
    #  @param pars (nEns, nPar) array containing parameters for each particle
    #  @param weights (nEns) array containing the weights of the particles
    #  @param delta  Discount factor in (0, 1] that determines the shrinkage factor, a=((3 * delta) - 1)/(2 * delta)
    #   typically between 0.95-0.99.
    #   Larger delta (e.g., 0.99) gives less jitter and more shrinkage, suitable when posterior is well-identified.
    #   Smaller delta (e.g., 0.95–0.97) adds more diversity when particle degeneracy is a risk.
    #  @return shrunk_pars (nEns, nPar) array containing parameters shrunk to mean

    #pars = pars
    nEns, nPar = pars.shape
    #print((pars))

    if log_weights is None:
        # If no weights are provided, calculate an arithmetic mean
        meanPar = np.mean(pars, axis=0)
    else:
        # If weights are provided, calculate weighted mean

        # Transform from log-space and normalise weights
        weights = np.asarray(np.exp(log_weights))
        weights = weights / weights.sum()

        # Compute weighted mean parameter
        meanPar = np.average(pars, axis=0, weights=weights)

    # Calculate shrinkage factor required
    a = ((3 * delta) - 1) / (2 * delta)

    print(f"meanPar = {meanPar}")
    #print(f"a = {a}")
    #print(f"1-a = {1 - a}")

    # Shrink towards the mean
    shrunk_pars = (a * pars) + ((1 - a) * meanPar)

    return shrunk_pars


def obs_op(
        huxtObject, cme_par, obs_lon, obs_time_in_jd, r_min=30 * u.solRad, cme_init_rad=12 * u.solRad
):
    """
    obs_op: The purpose of this definition is to perform the observation operator
              function that maps from the cme_parameters to observation space (in this
              case, the CME's flank position)
    :param huxtObject: HUXt object that contains the ambient solar wind that the cme will be propagated through
    :param cme_par: Dictionary containing CME parameters with following keys:
    #  ['t_init', 'v', 'width', 'lon', 'lat', 'thick']
    :param huxt_start_time: Initial time of huxt object
    :param obs_timein_jd: Observation time that CME needs to be run to

    :return: hx: CME flank estimated by model
    """

    # Extract CME parameters from list
    cme_speed = cme_par[1] * u.km/u.s

    dist_to_inner_rad = (r_min.to(u.solRad) - cme_init_rad.to(u.solRad)).to(u.km)
    cme_transit_time = (dist_to_inner_rad / cme_speed).to(u.s)
    cme_launch_time = cme_par[0] * u.s + cme_transit_time
    cme_width = cme_par[2] * u.deg
    cme_lon = cme_par[3] * u.deg
    cme_lat = cme_par[4] * u.deg
    cme_thickness = cme_par[5] * u.solRad
    #print(cme_lat)

    # Generate CME object
    cme = H.ConeCME(
        t_launch=cme_launch_time,
        longitude=cme_lon,
        latitude=cme_lat,
        width=cme_width,
        v=cme_speed,
        thickness=cme_thickness#,
#        cme_fixed_duration=True,
#        fixed_duration=12 * 60 * 60 * u.s
    )
    #print(cme.coords.items())
    # Run CME through HUXt
    huxtObject.solve([cme])
    cme_member = huxtObject.cmes[0]
    # Plot this out
    t_interest = obs_time_in_jd
    #fig, ax = HA.plot(huxtObject, t_interest)
    #plt.show()

    # Calculate CME flank
    obsObject = shmo.Observer(huxtObject, cme_member, obs_lon)
    cme_flank = obsObject.compute_flank_profile(cme_member)

    # Get the CME elongation at the required observation time
    indReq = np.argmin(abs(cme_flank['time'].values - obs_time_in_jd))
    hx = cme_flank['el'].values[indReq]

    return hx


def likelihood_function_gaussian(
        obs,
        obs_cov,
        huxtObject,
        cme_par,
        obs_lon,
        obs_time_in_jd,
        r_min=30 * u.solRad,
        cme_init_rad=12 * u.solRad
):
    """
    likelihood_function_gaussian: The purpose of this definition is to calculate the
      likelihood function
    :param obs: Observation of the CME flank
    :param obs_cov: Observation error covariance matrix of the CME flank
    :param huxtObject: HUXt object that contains the ambient solar wind that the cme will be propagated through
    :param cme_par: CME parameters with following keys:
      ['t_init', 'v', 'width', 'lon', 'lat', 'thick']
    :param obs_lon: Observation longitude

    :return: likelihood: The likelihood function
    """

    # Calculate the log likelihood function
    loglik = log_likelihood_function_gaussian(
        obs, obs_cov, huxtObject, cme_par, obs_lon, obs_time_in_jd, r_min=r_min, cme_init_rad=cme_init_rad
    )

    # Calculate the likelihood function by taking the exponent
    likelihood = np.exp(loglik)

    return likelihood


def likelihood_function(
        obs, obs_cov, huxtObject, cme_par, obs_lon, obs_time_in_jd, r_min=30 * u.solRad, cme_init_rad=12 * u.solRad
):
    """
    likelihood_function: The purpose of this definition is to calculate the likelihood function with no assumptions
    :param obs: Observation of the CME flank
    :param obs_cov: Observation error covariance matrix of the CME flank
    :param huxtObject: HUXt object that contains the ambient solar wind that the cme will be propagated through
    :param cme_par: CME parameters with following keys:
      ['t_init', 'v', 'width', 'lon', 'lat', 'thick']
    :param obs_lon: Observation longitude

    :return: likelihood: The likelihood function
    """

    # Call the required likelihood function to get the likelihood
    likelihood = likelihood_function_gaussian(obs, obs_cov, huxtObject, cme_par, obs_lon, obs_time_in_jd, r_min=r_min, cme_init_rad=cme_init_rad)

    return likelihood


def log_likelihood_function_gaussian(
        obs, obs_cov, huxtObject, cme_par, obs_lon, obs_time_in_jd, r_min=30 * u.solRad, cme_init_rad=12 * u.solRad
):
    """
    log_likelihood_function_gaussian: The purpose of this definition is to calculate the
      logarithm of the likelihood function
    :param obs: Observation of the CME flank
    :param obs_cov: Observation error covariance matrix of the CME flank
    :param huxtObject: HUXt object that contains the ambient solar wind that the cme will be propagated through
    :param cme_par: cme parameters with following keys:
       ['t_init', 'v', 'width', 'lon', 'lat', 'thick']
    :param obs_lon: Observation longitude

    :return: loglik: The logarithm of the likelihood function
    """

    # Calculate the observation operator (what the model thinks the observation should be)
    hx = obs_op(huxtObject, cme_par, obs_lon, obs_time_in_jd, r_min=r_min, cme_init_rad=cme_init_rad)
    """print(f"obs={obs}")
    print("hx: ", hx)"""

    # Calculate the innovation (difference between the observations and the observation operator)
    innov = obs - hx
    #print(innov)
    # Calculate the inverse of the observation error covariance and calculate the logarithm
    #  of the likelihood function
    #print(np.ndim(obs_cov))
    if np.ndim(obs_cov) == 0:
        obs_cov_1 = 1.0 / obs_cov

        loglik = -obs_cov_1 * innov * innov

        #print("loglik: ", loglik)
        return loglik
    else:
        obs_cov_1 = np.linalg.pinv(obs_cov)

        loglik = -np.transpose(innov).dot(
            obs_cov_1.dot(innov)
        )

        #print("loglik: ", loglik)
        return loglik


def log_likelihood_function(obs, obs_cov, huxtObject, cme_par, obs_lon, obs_time_in_jd, r_min=30*u.solRad, cme_init_rad=12*u.solRad):
    """
    log_likelihood_function: The purpose of this definition is to calculate the logarithm of the likelihood
    :param obs: Observation of the CME flank
    :param obs_cov: Observation error covariance matrix of the CME flank
    :param huxtObject: HUXt object that contains the ambient solar wind that the cme will be propagated through
    :param cme_par: CME parameters with following keys:
      ['t_init', 'v', 'width', 'lon', 'lat', 'thick']
    :param obs_lon: Observation longitude

    :return: log_likelihood: The logarithm of the likelihood function
    """

    # Call the likelihood function and then taken its logarithm
    log_likelihood = log_likelihood_function_gaussian(
        obs, obs_cov, huxtObject, cme_par, obs_lon, obs_time_in_jd, r_min=r_min, cme_init_rad=cme_init_rad
    )
    #log_likelihood = np.log(likelihood)

    return log_likelihood


def jacob_log(x_list):
    """
    Function to calculate the Jacobian logarithm, given by the sum:
       Jacob_log = log( sum_{j=1}^{N}[exp{x_list_j}] )
    :param x_list: List of exponents to be used in Jacobian logarithm

    :return: jacob_log: Value of Jacobian logarithm
    """

    # Initialise Jacobian logarithm as first exponent
    jacob_log = x_list[0]

    for i in range(1, len(x_list)):
        # Calculate the components
        term1 = np.max([jacob_log, x_list[i]])

        t2exp = -np.abs(x_list[i] - jacob_log)
        term2 = np.log(1 + np.exp(t2exp))

        jacob_log = term1 + term2

    return jacob_log


def make_synthetic_obs(
        true_cme_par_array, obs_lon, obs_cov, obs_times_in_datetime,
        huxt_init_time=datetime.datetime(2008, 1, 1, 0, 0, 0),
        vr_in=np.zeros(128) + 400 * u.km / u.s,
        lon_start=290*u.deg,
        lon_stop=380*u.deg,
        sim_time=5*u.day,
        dt_scale=20,
        r_min=30 * u.solRad,
        cme_init_rad=12 * u.solRad
):
    """
    Function to create synthetic observations
    :param true_cme_par_array:
    :param obs_lon:
    :param obs_cov:
    :param obs_times_in_datetime:
    :param huxt_init_time:
    :param vr_in:
    :param lon_start:
    :param lon_stop:
    :param time_tolerance:
    :param dt_scale:
    :return:
    """

    initAstroTime = Time(huxt_init_time, format='datetime', scale='utc')
    obs_time_in_jd = [
        Time(obs_t_i, format='datetime', scale='utc').jd
        for obs_t_i in obs_times_in_datetime
    ]
    #sim_time = (obs_astroTime - initAstroTime).to(u.day) + time_tolerance
    #obs_time_in_jd = obs_astroTime.jd  # - initAstroTime.jd

    # Initialise HUXt model object for each ensemble member
    model = setup_huxt(
        start_datetime=huxt_init_time,
        vr_in=vr_in,
        lon_start=lon_start,
        lon_stop=lon_stop,
        sim_time=sim_time,
        dt_scale=dt_scale,
        r_min=r_min
    )

    # Extract CME parameters from list
    # Calculate time taken to go from cme's initial radius to huxt inner boundary
    cme_speed = true_cme_par_array['v'].to(u.km / u.s)

    dist_to_inner_rad = (r_min.to(u.solRad) - cme_init_rad.to(u.solRad)).to(u.km)
    cme_transit_time = dist_to_inner_rad / cme_speed
    cme_launch_time = (
        true_cme_par_array['t_init'] - true_cme_par_array['huxt_init_time']
    ).total_seconds() * u.s + cme_transit_time
    print(true_cme_par_array['t_init'] + datetime.timedelta(seconds=cme_launch_time.value))
    cme_width = true_cme_par_array['width'].to(u.deg)
    cme_lon = true_cme_par_array['lon'].to(u.deg)
    if cme_lon > (180 * u.deg):
        cme_lon = cme_lon - (360 * u.deg)

    cme_lat = true_cme_par_array['lat'].to(u.deg)
    cme_thickness = true_cme_par_array['thick'].to(u.solRad)

    # Generate CME object
    cme = H.ConeCME(
        t_launch=cme_launch_time,
        longitude=cme_lon,
        latitude=cme_lat,
        width=cme_width,
        v=cme_speed,
        thickness=cme_thickness#,
        # cme_fixed_duration=True,
        # fixed_duration=12 * 60 * 60 * u.s
    )
    #print(cme.coords.items())

    # Run CME through HUXt
    model.solve([cme])
    cme_member = model.cmes[0]

    # Calculate CME flank
    obsObject = shmo.Observer(model, cme_member, obs_lon)
    cme_flank = obsObject.compute_flank_profile(cme_member)

    synth_obs = []
    for it, obs_t in enumerate(obs_time_in_jd):
        # Get the CME elongation at the required observation time
        indReq = np.argmin(abs(cme_flank['time'].values - obs_t))
        obs_pert = 0 #np.random.normal(loc=0, scale=obs_cov)
        synth_obs.append(cme_flank['el'].values[indReq] + obs_pert)

        # Plot this out
        #print(obs_times_in_datetime[it])
        t_interest = (obs_times_in_datetime[it] - huxt_init_time).total_seconds() * u.s
        #print(t_interest.value)
        fig, ax = HA.plot(model, t_interest)
        ax.set_title(f"Synth obs CME at {obs_times_in_datetime[it]}")
        plt.show()
    return synth_obs


def calc_eff_sample_size(log_weights):
    """
    Definition to calculate the effective sample size using
    log(N_eff) = 2*Jacob_log(log_weights) - Jacob_log(2*log_weights)
    :param weights: Weights of particles
    :return: n_eff_ens
    """

    log_n_eff = 2 * jacob_log(log_weights) - jacob_log(2 * log_weights)
    #print(f"log(N_eff) = ", log_n_eff)
    n_eff_ens = np.exp(log_n_eff)

    return n_eff_ens


def auxPf(
        cme_par_array, obs, obs_cov, obs_lon, obs_time, inflFact,
        fixed_ambient=True, pars_in_state=['v', 'lon', 'width'],
        huxt_init_time=datetime.datetime(2008, 1, 1, 0, 0, 0),
        vr_in=np.zeros(128) + 400 * u.km / u.s,
        lon_start=290*u.deg,
        lon_stop=380*u.deg,
        time_tolerance=0.5*u.day,
        dt_scale=20,
        delta=0.98,
        r_min=30*u.solRad,
        cme_init_rad=12 * u.solRad,
        rng=None
):
    """
    Function to run the auxillary particle filter
    :param pars:
    :param obs:
    :param obs_cov:
    :param obs_lon:
    :param obs_time:
    :param log_weights: Unnormalised logarithm of the weights
    :param fixed_ambient:
    :param pars_in_state:
    :param hux_init_time:
    :param vr_in:
    :param lon_start:
    :param lon_stop:
    :param time_tolerance:
    :param dt_scale:
    :param delta:
    :param rng:
    :return:
    """

    if rng is None:
        rng = np.random.default_rng()

    #pars, x_mean, x_std = cme_par_to_state_vector(cme_par_array, pars_in_state)
    #print(cme_par_array['lon'])
    parsNoUnits = remove_units_from_cme_par(cme_par_array)
    log_weights = get_particle_weights(cme_par_array)
    pars = cme_par_to_state_vector(parsNoUnits, pars_in_state)
    #print(pars)
    #pars = np.asarray(pars)   # Array containing all parameters as required

    obs = np.asarray(obs)
    initAstroTime = Time(huxt_init_time, format='datetime', scale='utc')
    obs_astroTime = Time(obs_time, format='datetime', scale='utc')
    sim_time = (obs_astroTime - initAstroTime).to(u.day)  + time_tolerance
    obs_time_in_jd = obs_astroTime.jd# - initAstroTime.jd

    nEns, nPar = pars.shape

    # Shrink the parameters to the mean
    shrunk_pars = shrink_par(
        pars=pars,
        log_weights=log_weights,
        delta=delta
    )
    #print(f"shrunk_pars={shrunk_pars}")
    # Calculate the covariance of the ensemble of parameters
    cov_pars = np.cov(pars, rowvar=False)

    if cov_pars.ndim == 0:
        cov_pars = np.array([[cov_pars]])
    print(cov_pars)
    #print(f"3*delta - 1= {3 * delta - 1}")
    #print(f"2 * delta = {2 * delta}")

    # Calculate covariance scaling factor h, from the delta quantity input into function
    h2 = 1 - ( ((3 * delta) - 1) / (2 * delta) ) ** 2
    #print(h2)

    stoch_weight_cov = h2 * inflationFact * cov_pars
    #print(f"stoch_weight_cov: {stoch_weight_cov}")
    #print(np.shape(cov_pars))

    # Initialise unnormalised probability list
    log_likelihood_list = []
    log_unnorm_prob = []
    for j in range(nEns):
        if fixed_ambient:
            if j == 0:
                # Initialise HUXt model object for each ensemble member
                model = setup_huxt(
                    start_datetime=huxt_init_time,
                    vr_in=vr_in,
                    lon_start=lon_start,
                    lon_stop=lon_stop,
                    sim_time=sim_time,
                    dt_scale=dt_scale,
                    r_min=r_min
                )
        else:
            # Initialise HUXt model object for each ensemble member
            model = setup_huxt(
                start_datetime=huxt_init_time,
                vr_in=vr_in,
                lon_start=lon_start,
                lon_stop=lon_stop,
                sim_time=sim_time,
                dt_scale=dt_scale,
                r_min=r_min
            )
        # Calculate the log-likelihood for this ensemble member
        cmeReqPar = ['t_init', 'v', 'width', 'lon', 'lat', 'thick']
        par_arr_j = []

        for ip, parVal in enumerate(cmeReqPar):
            if parVal in pars_in_state:
                # Get index of parVal in pars_in_state
                indReq = pars_in_state.index(parVal)
                par_arr_j.append(shrunk_pars[j, indReq])
            else:
                par_arr_j.append(parsNoUnits[j, ip])

            # if parVal == 't_init':
            #     if type(par_arr_j[ip]) is datetime.datetime:
            #         par_arr_j[ip] = (
            #             par_arr_j[ip] - cme_par_array['huxt_init_time']
            #         ).total_seconds()

        log_likelihood_ens = log_likelihood_function(
            obs,
            obs_cov,
            model,
            par_arr_j,
            obs_lon,
            obs_time_in_jd,
            r_min=r_min,
            cme_init_rad=cme_init_rad
        )

        # Store log-likelihood ensemble in list
        log_likelihood_list.append(log_likelihood_ens)

        #print("log_weights_j: ", log_weights[j])
        #print("log_likelihood_ens: ", log_likelihood_ens)

        # Calculate the unnormalised logarithm of the probability of choosing this ensemble member
        log_unnorm_prob.append(log_weights[j] + log_likelihood_ens)

    # Calculate the normalisation factors for log_unnorm_probability and log_likelihood
    norm_factor_lik = jacob_log(log_likelihood_list)
    norm_factor_prob = jacob_log(log_unnorm_prob)

    # Normalise the logarithm of the probabilities
    log_likelihood_normed = log_likelihood_list - norm_factor_lik
    log_norm_prob = log_unnorm_prob - norm_factor_prob

    cme_poste_par = np.zeros((n_ens, len(cmeReqPar)))
    #print("Log unnormalized probability", log_unnorm_prob)
    #print("Log norm probability: ", log_norm_prob)

    # Generate the logCDF with the Jacobian logarithm
    logCDF = np.zeros(nEns)
    for j in range(nEns):
        if j==0:
            logCDF[j] = log_norm_prob[j]
        else:
            term1 = np.max([logCDF[j-1], log_norm_prob[j]])
            t2exp = -np.abs(logCDF[j-1] - log_norm_prob[j])
            term2 = np.log(1 + np.exp(t2exp))

            logCDF[j] = term1 + term2

    #print(f'logCDF: {logCDF}')
    # Stochastic resampling step
    resampPar = np.zeros_like(pars)
    resampWeights = np.zeros_like(log_weights)
    for j in range(nEns):
        # Draw random value between 0 and 1 (can't be equal to 0 as we will take the logarithm)
        rand_value = 0
        while rand_value == 0:
            rand_value = rng.uniform(0, 1)

        log_rand_value = np.log(rand_value)

        #print(f'log_rand_value: {log_rand_value}')
        # Find the index i, such that logCDF[i-1] < log(rand_value) \leq logCDF[i]
        if log_rand_value <= logCDF[0]:
            indReq = 0
        elif log_rand_value > logCDF[-1]:
            indReq = nEns - 1
        else:
            indReq = [ n for n, i in enumerate(logCDF) if i > log_rand_value ][0]

        #print(f'indReq: {indReq}')
        #print(f"shrunk_pars[indReq, :]: {shrunk_pars[indReq, :]}")
#        sys.exit()
        resampPar[j, :] = (
            np.random.multivariate_normal(
                mean=shrunk_pars[indReq, :], cov=stoch_weight_cov
            )
        )

        # Calculate the log-likelihood for this ensemble member
        cmeReqPar = ['t_init', 'v', 'width', 'lon', 'lat', 'thick']
        par_arr_j = []

        for ip, parVal in enumerate(cmeReqPar):
            if parVal in pars_in_state:
                # Get index of parVal in pars_in_state
                indReq = pars_in_state.index(parVal)
                par_arr_j.append(resampPar[j, indReq])
            else:
                par_arr_j.append(parsNoUnits[j, ip])

        cme_poste_par[j, :] = par_arr_j
        # Get the likelihood of the resampled parameters
        resamp_log_likelihood = log_likelihood_function(
            obs,
            obs_cov,
            model,
            par_arr_j,
            obs_lon,
            obs_time_in_jd,
            r_min=r_min,
            cme_init_rad=cme_init_rad
        )
        resampWeights[j] = resamp_log_likelihood - log_likelihood_normed[j]

    resampWeightNorm = jacob_log(resampWeights)
    resampWeights = resampWeights - resampWeightNorm

    print(f"n_eff_sample_size: {calc_eff_sample_size(resampWeights)}")
    #print(f"resampWeights: {resampWeights}")

    cme_par_array = state_vector_to_cme_par(resampPar, resampWeights, pars_in_state, cme_par_array)

    return resampPar, resampWeights, cme_par_array


#
def generateCovInitCMEPar(file_path, vars_req):
    """
    Generate the covariance matrix for the initial parameters using Blair's CME list
    provided in file_path
    :param file_path: File path to CME list to estimate covariance from
    :param vars_req: List of variables to estimate covariance of
        Accepted inputd = ['v', 'width', 'lon', 'lat', 'thick']
    :return: initCovPar: Covariance matrix for the initial parameters
    """

    # Open and read in CME list
    dfCMEpar = pd.read_csv(file_path)

    # Rename column headers to be consistent with SIR-HUXt and then delete columns that aren't in vars_req
    dfCMEpar.rename({'V': 'v', 'Ang_rad': 'width'}, axis=1, inplace=True)
    dfCMEpar.drop(
        columns=[col for col in dfCMEpar if col not in vars_req],
        inplace=True
    )

    # Calculate covariance matrix from dfCMEpar
    initCovPar = dfCMEpar.cov(numeric_only=True)
    print(initCovPar)

    return initCovPar


#####################################################################
# TEST DEFINITIONS
#####################################################################
def test_cme_par_to_state(nEns, seed=np.nan):
    ## Function to test whether cme_par_to_state_vector and state_vector to cme_par are actually inverses
    #   of one another

    if ~np.isnan(seed):
        np.random.seed(seed)

    keys = ['t_init', 'v', 'width', 'lon', 'lat', 'thick', 't_transit', 'v_hit', 'likelihood', 'weight']
    init_par_arr = {k: np.random.rand(nEns) for k in keys}
    init_par_arr['n_members'] = nEns

    sum_weight = init_par_arr['weight'].sum()
    sum_lik = init_par_arr['likelihood'].sum()

    init_par_arr['weight'] = [init_par_arr['weight'][i] / sum_weight for i in range(nEns)]
    init_par_arr['likelihood'] = [init_par_arr['likelihood'][i] / sum_lik for i in range(nEns)]

    par_req = ['v', 'width', 'lon', 'lat', 'thick']

    z_ens, x_mean, x_std = cme_par_to_state_vector(init_par_arr, par_req)

    final_par_arr = {k: np.zeros(nEns) for k in keys}
    final_par_arr['n_members'] = nEns

    weights = init_par_arr['weight']

    shrunk_pars = shrink_par(z_ens, weights, delta=0.98)

    final_par_arr = state_vector_to_cme_par(z_ens, x_mean, x_std, weights, par_req, final_par_arr)

    print("Differences between initial and final arrays for required parameters:")
    print(" (should be zero/machine precision) \n\n")
    for i, ip in enumerate(par_req):
        print(f"{ip}: {final_par_arr[ip] - init_par_arr[ip]}")

    return


if __name__ == "__main__":
    plt.close('all')

    outBaseDir = os.path.join(
        "C:\\", "Users", "ss905122", "PycharmProjects", "SIR_HUXt", "output", "figures", "highSpread"
    )
    if not os.path.isdir(outBaseDir):
        os.makedirs(outBaseDir)
    file_path = os.path.join(
        "C:\\", "Users", "ss905122", "PycharmProjects", "SIR_HUXt", "blairCMElistSingle.csv"
    )
    vars_req = ['v', 'width', 'lon']
    initCMEparCov = generateCovInitCMEPar(file_path, vars_req)

    huxt_init_time = datetime.datetime(2008, 1, 1, 0, 0, 0)
    n_ens = 25

    cme_par_array = initialise_cme_parameter_ensemble_arrays(n_ens, huxt_init_time)
    true_cme_par_array = initialise_cme_parameter_ensemble_arrays(1, huxt_init_time)

    # Initialise true CME parameters
    true_cme_t_init = datetime.datetime(2008, 1, 1, 0, 0, 0)
    true_cme_speed = 495
    true_cme_width = 37.4
    true_cme_lon = 0
    true_cme_lat = 0
    true_cme_thick = 0

    true_cme_par_array['t_init'] = true_cme_t_init
    true_t_init_str = true_cme_par_array['t_init'].strftime("%Y%m%d-%H%M")

    true_cme_par_array['v'] = true_cme_speed * u.km / u.s
    true_cme_par_array['width'] = true_cme_width * u.deg
    true_cme_par_array['lon'] = true_cme_lon * u.deg
    true_cme_par_array['lat'] = true_cme_lat * u.deg
    true_cme_par_array['thick'] = true_cme_thick * u.solRad

    outputObsDir = os.path.join(
        outBaseDir,
        f"truth_{true_t_init_str}_{true_cme_speed}_{true_cme_width}_{true_cme_lon}_{true_cme_lat}_{true_cme_thick}"
    )
    cme_init_rad = 12 * u.solRad
    r_min = 30 * u.solRad
    """np.array([
        datetime.datetime(2008, 1, 1, 1, 0, 0),
        datetime.datetime(2008, 1, 1, 1, 0, 0),
        datetime.datetime(2008, 1, 1, 1, 0, 0),
#        datetime.datetime(2008, 1, 1, 1, 0, 0),
        datetime.datetime(2008, 1, 1, 1, 0, 0)
    ]))"""
    """cme_par_array['v'] = np.array([500, 600, 400, 550]) * u.km / u.s
    print(np.mean(cme_par_array['v']))
    cme_par_array['width'] = np.array([60, 60, 60, 60]) * u.deg
    cme_par_array['lon'] = np.array([5, 5, 5, 5]) * u.deg
    cme_par_array['lat'] = np.array([5, 5, 5, 5]) * u.deg
    cme_par_array['thick'] = np.array([5, 5, 5, 5]) * u.solRad"""
    #pars_req = ['v', 'lon', 'width']
    #pars, weights = cme_par_to_state_vector(cme_par_array, pars_req)
    #obs = [23]
    obs_cov = 0.3# * np.eye(len(obs))

    nObs = 8
    obs_lon = 300 * u.deg
    obs_lat = 0 * u.deg
    obs_times = [
        datetime.datetime(2008, 1, 1, 9, 0, 0)
        + datetime.timedelta(hours=3 * i) for i in range(1, 1 + nObs)
    ]
    synth_obs = make_synthetic_obs(
        true_cme_par_array, obs_lon, obs_cov, obs_times,
        huxt_init_time=datetime.datetime(2008, 1, 1, 0, 0, 0),
        vr_in=np.zeros(128) + 400 * u.km / u.s,
        lon_start=290*u.deg,
        lon_stop=430*u.deg,
        sim_time=3*u.day,
        dt_scale=1,
        r_min=r_min,
        cme_init_rad=cme_init_rad
    )
    print(synth_obs)

    #######################################################################
    outTruthDir = os.path.join(
        outBaseDir,
        f"truth_{true_t_init_str}_{true_cme_speed}_{true_cme_width}_{true_cme_lon}_{true_cme_lat}_{true_cme_thick}"
    )
    if not os.path.isdir(outTruthDir):
        os.makedirs(outTruthDir)

    fig, ax = plt.subplots(1, 1)
    ax.plot(obs_times, synth_obs, '-', marker='^', label='Obs')
    plt.savefig(os.path.join(outputObsDir, 'synth_obs.png'))

    #######################################################################
    # Initialise CME parameters
    mean_cme_t_init = datetime.datetime(2008, 1, 1, 1, 0, 0)
    mean_cme_t_init_str = mean_cme_t_init.strftime("%Y%m%d-%H%M")
    mean_cme_speed = 477
    mean_cme_width = 37
    mean_cme_lon = -4
    mean_cme_lat = 0
    mean_cme_thick = 0

    n_runs = 100

    for runNo in range(n_runs):
        cme_par_array['t_init'] = np.array(
            [mean_cme_t_init for _ in range(n_ens)]
        )
        cme_par_array['v'] = np.array(
            [
                np.random.uniform(low=0.7 * mean_cme_speed, high=1.3 * mean_cme_speed) for _ in range(n_ens)
                # mean_cme_speed + np.random.normal(loc=0, scale=50, size=None)) for _ in range(n_ens)
            ]
        ) * u.km / u.s
        cme_par_array['width'] = np.array(
            [
                np.random.uniform(low=mean_cme_width - 15, high=mean_cme_width + 15) for _ in range(n_ens)
                # mean_cme_width + np.random.normal(loc=0, scale=5, size=None) for _ in range(n_ens)
            ]
        ) * u.deg
        cme_par_array['lon'] = np.array(
            [
                np.random.uniform(low=mean_cme_lon - 15, high=mean_cme_lon + 15) for _ in range(n_ens)
                # mean_cme_lon + np.random.normal(loc=0, scale=5, size=None) for _ in range(n_ens)
            ]
        ) * u.deg
        # cme_par_array['lon'] = np.array(
        #     [mean_cme_lon
        #      for _ in range(n_ens)]
        # ) * u.deg
        cme_par_array['lat'] = np.array(
            [mean_cme_lat for _ in range(n_ens)]
        ) * u.deg
        cme_par_array['thick'] = np.array(
            [mean_cme_thick for _ in range(n_ens)]
        ) * u.solRad

        outputDir = os.path.join(
            outBaseDir,
            f"truth_{true_t_init_str}_{true_cme_speed}_{true_cme_width}_{true_cme_lon}_{true_cme_lat}_{true_cme_thick}",
            f"prior_{mean_cme_t_init_str}_{mean_cme_speed}_{mean_cme_width}_{mean_cme_lon}_{mean_cme_lat}_{mean_cme_thick}",
            f"nEns-{n_ens}_{nObs}_{obs_lon}_{obs_lat}",
            f"run_{runNo:03d}"
        )
        if not os.path.isdir(outputDir):
            os.makedirs(outputDir)

        cme_v_values = np.zeros((n_ens, len(synth_obs)))
        cme_width_values = np.zeros((n_ens, len(synth_obs)))

        cme_saved_pars = np.zeros((len(synth_obs) + 1, n_ens, 6))
        cme_elon = np.zeros((len(synth_obs) + 1, n_ens, 37))
        cme_times = np.zeros((len(synth_obs) + 1, n_ens, 37))

        # Save prior parameters in an array
        cme_saved_pars[0, :, 0] = [
            (cme_par_array['t_init'][i] - cme_par_array['huxt_init_time']).total_seconds()
            for i in range(n_ens)
        ]
        # print(cme_par_array['width'])
        cme_saved_pars[0, :, 1] = cme_par_array['v'].to(u.km / u.s).value
        cme_saved_pars[0, :, 2] = cme_par_array['width'].to(u.deg).value

        cme_saved_pars[0, :, 3] = cme_par_array['lon'].to(u.deg).value
        lonCond = cme_saved_pars[0, :, 3] > 180
        cme_saved_pars[0, lonCond, 3] = cme_saved_pars[0, lonCond, 3] - 360

        cme_saved_pars[0, :, 4] = cme_par_array['lat'].to(u.deg).value
        cme_saved_pars[0, :, 5] = cme_par_array['thick'].to(u.solRad).value
        ###############################################################################

        for yi, y_obs in enumerate(synth_obs):
            print(yi, y_obs)
            inflatAt15Rs = 4
            inflTerm1 = ((1 - inflatAt15Rs) / 15.0) * y_obs
            inflTerm2 = (2 * inflatAt15Rs) - 1
            inflationFact = inflTerm1 + inflTerm2
            #-(7.0 * y_obs / 15.0) + 15.0

            inflExp = -(yi / 5.0) + 6
            #inflationFact = 2 ** inflExp
            print(f"inflation factor: {inflationFact}")
            resampPar, resampWeights, cme_par_array = auxPf(
                cme_par_array, y_obs, obs_cov, obs_lon, obs_times[yi], inflFact=inflationFact,# log_weights,
                fixed_ambient=True, pars_in_state=['v', 'width', 'lon'], #'lon', 'width'],
                huxt_init_time=huxt_init_time,
                vr_in=np.zeros(128) + 400 * u.km / u.s,
                lon_start=290 * u.deg,
                lon_stop=430 * u.deg,
                time_tolerance=0.5 * u.day,
                dt_scale=20,
                delta=0.9,
                r_min=r_min,
                cme_init_rad = cme_init_rad,
                rng=None
            )
            print(f"cme_par_array: {cme_par_array}")
            print(f"mean resamp par: {np.mean(resampPar)}")
            #cme_v_values[yi, :] = resampPar
            #print(f"sum(resampWeights): {sum(np.exp(resampWeights))}")

            # Standardise the units and remove the astropy units
            cme_saved_pars[yi + 1, :, 0] = [
                (cme_par_array['t_init'][i] - cme_par_array['huxt_init_time']).total_seconds()
                for i in range(n_ens)
            ]
            # print(cme_par_array['width'])
            cme_saved_pars[yi + 1, :, 1] = cme_par_array['v'].to(u.km / u.s).value
            cme_saved_pars[yi + 1, :, 2] = cme_par_array['width'].to(u.deg).value

            cme_saved_pars[yi + 1, :, 3] = cme_par_array['lon'].to(u.deg).value
            lonCond = cme_saved_pars[yi + 1, :, 3] > 180
            cme_saved_pars[yi + 1, lonCond, 3] = cme_saved_pars[yi + 1, lonCond, 3] - 360

            cme_saved_pars[yi + 1, :, 4] = cme_par_array['lat'].to(u.deg).value
            cme_saved_pars[yi + 1, :, 5] = cme_par_array['thick'].to(u.solRad).value

            cme_v_values[:, yi] = cme_par_array['v'].to(u.km / u.s).value
            cme_width_values[:, yi] = cme_par_array['width'].to(u.deg).value
            #print(cme_par_array)

        # Save cme_parameters into a .nc file
        # Make an xarray object
        cme_par_ds = xr.Dataset(
            data_vars=dict(
                model_init_time=huxt_init_time,
                cme_init_rad=cme_init_rad,
                r_min=r_min,
                ambient_vr=(["n_lon"], np.zeros(128) + 400 * u.km / u.s),
                t_init=(["n_obs", "n_ens"], cme_saved_pars[:, :, 0]),
                v=(["n_obs", "n_ens"], cme_saved_pars[:, :, 1]),
                width=(["n_obs", "n_ens"], cme_saved_pars[:, :, 2]),
                lon=(["n_obs", "n_ens"], cme_saved_pars[:, :, 3]),
                lat=(["n_obs", "n_ens"], cme_saved_pars[:, :, 4]),
                thick=(["n_obs", "n_ens"], cme_saved_pars[:, :, 5])
            ),
            coords=dict(
                obs_no=("n_obs", range(len(synth_obs) + 1)),
                ens_no=("n_ens", range(n_ens)),
                huxt_lon=("n_lon", (2 * np.pi / 128.) * np.arange(128))
            ),
        )
        print(cme_par_ds)
        outParFile = os.path.join(outputDir, "cme_pars.nc")
        cme_par_ds.to_netcdf(outParFile)
        #
        # for yi, y_obs in enumerate(synth_obs):
        #     # Calculate the elongation profile for the current observation for the next three days
        #     # Initialise HUXt model object for each ensemble member
        #     model3 = setup_huxt(
        #         start_datetime=huxt_init_time,
        #         vr_in=np.zeros(128) + 400 * u.km / u.s,
        #         lon_start=290 * u.deg,
        #         lon_stop=430 * u.deg,
        #         sim_time=3 * u.day,
        #         dt_scale=20,
        #         r_min=30 * u.solRad
        #     )
        #
        #     for m in range(n_ens):
        #         # Extract CME parameters from list
        #         cme_speed = cme_saved_pars[yi, m, 1] * u.km / u.s
        #
        #         dist_to_inner_rad = (r_min.to(u.solRad) - cme_init_rad.to(u.solRad)).to(u.km)
        #         cme_transit_time = dist_to_inner_rad / cme_speed
        #         cme_launch_time = (
        #             cme_saved_pars[yi, m, 0] * u.s + cme_transit_time
        #         )
        #         cme_width = cme_saved_pars[yi, m, 2] * u.deg
        #
        #         cme_lon = cme_saved_pars[yi, m, 3] * u.deg
        #         cme_lat = cme_saved_pars[yi, m, 4] * u.deg
        #         cme_thickness = cme_saved_pars[yi, m, 5] * u.solRad
        #
        #         # Generate CME object
        #         cme = H.ConeCME(
        #             t_launch=cme_launch_time,
        #             longitude=cme_lon,
        #             latitude=cme_lat,
        #             width=cme_width,
        #             v=cme_speed,
        #             thickness=cme_thickness
        #         )
        #         # print(cme.coords.items())
        #
        #         # Run CME through HUXt
        #         model3.solve([cme])
        #         cme_member = model3.cmes[0]
        #
        #         # # Plot this out
        #         # t_interest = (obs_times[yi] - huxt_init_time).total_seconds() * u.s
        #         # fig, ax = HA.plot(model3, t_interest)
        #         # ax.set_title(f"CME at {obs_times[yi]} for ensemble member {m}")
        #         # plt.show()
        #
        #         # Calculate CME flank
        #         obsObject = shmo.Observer(model3, cme_member, obs_lon)
        #         #print(f"len={len(obsObject.compute_flank_profile(cme_member)['el'].values)}")
        #         #sys.exit()
        #         cme_times[yi, m, :] = obsObject.compute_flank_profile(cme_member)['time'].values
        #         cme_elon[yi, m, :] = obsObject.compute_flank_profile(cme_member)['el'].values
        #     #print(cme_times[yi, 0, :])
        #     #print([Time(cme_times[yi, 0, k], format='jd').to_datetime() for k in range(37)])
        #     # fig, ax = plt.subplots(1, 1)
        #     # ax.plot(obs_times, synth_obs, '-', marker='^', label='Obs')
        #     # ax.plot()
        #     # plt.show()
        #
        # for yi, y_obs in enumerate(synth_obs):
        #     fig, ax = plt.subplots(1, 1)
        #     for m in range(n_ens):
        #         plot_cme_times=[
        #             Time(cme_times[yi, m, k], format='jd').to_datetime() for k in range(37)
        #         ]
        #         ax.plot(plot_cme_times, cme_elon[yi, m, :], color='salmon')
        #     ax.plot(obs_times, synth_obs, '-', marker='^', label='Obs', color='b')
        #     ax.plot(obs_times[yi], synth_obs[yi], '-', marker='^', label='Current obs assim.', color='c')
        #     ax.legend()
        #     ax.set_xlim((obs_times[0] - datetime.timedelta(hours=1), obs_times[-1] + datetime.timedelta(hours=1)))
        #     ax.set_ylim((0, 30))
        #     plt.savefig(os.path.join(outputDir, f"obsNo{yi}_elonEns.png"))
        #
        # colours_for_hist = sns.color_palette(cc.glasbey, n_colors=len(synth_obs))
        # fig, ax = plt.subplots(1, 1)
        # for yi, y_obs in enumerate(synth_obs):
        #     ax.hist(
        #         cme_v_values[:, yi], bins=20, color=colours_for_hist[yi], weights=resampWeights, density=True, alpha=0.7, label=f"Obs_no: {yi + 1}"
        #     )
        #     ax.legend(fontsize='x-small')
        # plt.savefig(os.path.join(outputDir, f"cme_speed_hist.png"))
        #
        # fig, ax = plt.subplots(1, 1)
        # for yi, y_obs in enumerate(synth_obs):
        #     ax.hist(
        #         cme_width_values[:, yi], bins=20, color=colours_for_hist[yi], weights=resampWeights, density=True, alpha=0.7, label=f"Obs_no: {yi + 1}"
        #     )
        #     ax.legend()
        # plt.savefig(os.path.join(outputDir, f"cme_width_hist.png"))