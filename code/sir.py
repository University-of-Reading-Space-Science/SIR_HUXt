#@package sir
# This module will take inputs from HUXt and perform an SIR to
#  generate an updated set of weights and particles
import numpy as np
import datetime

import huxt.huxt as H
import sir_huxt_mono_obs as shmo

def setup_huxt(
        start_datetime=datetime.datetime(2008, 1, 1, 0, 0, 0),
        vr_in=np.zeros(128) + 400 * u.km / u.s,
        lon_start=290*u.deg,
        lon_stop=380*u.deg,
        sim_time=2*u.day,
        dt_scale=20
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
    np.assert(type(start_datetime) == datetime.datetime)

    start_time = Time(start_datetime, scale='utc')
    cr_num = np.fix(sn.carrington_rotation_number(start_time))
    ert = H.Observer('EARTH', start_time)

    vr_in = np.asarray(vr_in)

    # Set up HUXt for a sim_time-day simulation, outputting every dt_scale
    model = H.HUXt(v_boundary=vr_in, cr_num=cr_num, cr_lon_init=ert.lon_c, latitude=ert.lat.to(u.deg),
                   lon_start=lon_start, lon_stop=lon_stop, simtime=sim_time, dt_scale=dt_scale)

    # model1d = H.HUXt(v_boundary=vr_in, cr_num=cr_num, cr_lon_init=ert.lon_c, latitude=ert.lat.to(u.deg),
    #                  lon_out=0 * u.deg, simtime=5 * u.day, dt_scale=4)

    return model


def initialise_cme_parameter_ensemble_arrays(n_ensemble):
    """
    Function to initialise empty arrays for storing the CME parameters for each ensemble member at each analysis step
    :param n_ensemble: The number of ensemble members in the SIR analysis
    :return parameter_arrays: A dictionary of parameter keys and an empty array for storing each ensemble member value
    """
    keys = ['t_init', 'v', 'width', 'lon', 'lat', 'thick', 't_transit', 'v_hit', 'likelihood', 'weight']
    parameter_arrays = {k:np.zeros(n_ensemble) for k in keys}
    parameter_arrays['n_members'] = n_ensemble

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

def cme_par_to_state_vector(cme_par_array, par_req):
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


def state_vector_to_cme_par(z_ens, x_mean, x_std, weights, par_req, cme_par_array):
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


def shrink_par(pars, weights, delta=0.98):
    ## Liu-West shrinkage of the parameters towards the weighted mean
    #  @param pars (nEns, nPar) array containing parameters for each particle
    #  @param weights (nEns) array containing the weights of the particles
    #  @param delta  Discount factor in (0, 1] that determines the shrinkage factor, a=((3 * delta) - 1)/(2 * delta)
    #   typically between 0.95-0.99.
    #   Larger delta (e.g., 0.99) gives less jitter and more shrinkage, suitable when posterior is well-identified.
    #   Smaller delta (e.g., 0.95–0.97) adds more diversity when particle degeneracy is a risk.
    #  @return shrunk_pars (nEns, nPar) array containing parameters shrunk to mean

    pars = np.asarray(pars)
    weights = np.asarray(weights)
    nEns, nPar = pars.shape
    print(np.shape(pars))

    # Normalise weights
    weights = weights / weights.sum()

    # Compute weighted mean parameter
    meanPar = np.average(pars, axis=0, weights=weights)

    # Calculate shrinkage factor required
    a = ((3 * delta) - 1) / (2 * delta)

    # Shrink towards the mean
    shrunk_pars = (a * pars) - ((1 - a) * meanPar)

    return shrunk_pars


def obs_op(
        huxtObject, cme_par, obs_lon
):
    """
    obs_op: The purpose of this definition is to perform the observation operator
              function that maps from the cme_parameters to observation space (in this
              case, the CME's flank position)
    :param huxtObject: HUXt object that contains the ambient solar wind that the cme will be propagated through
    :param cme_par: Dictionary containing CME parameters with following keys:
    #  ['t_init', 'v', 'width', 'lon', 'lat', 'thick']
    :param huxt_start_time: Initial time of huxt object
    :param obs_time: Observation time that CME needs to be run to

    :return: hx: CME flank estimated by model
    """

    # Extract CME parameters from list
    cme_launch_time = cme_par['t_init']
    cme_speed = cme_par['v']
    cme_lat = cme_par['lat']
    cme_lon = cme_par['lon']
    cme_width = cme_par['width']
    cme_thickness = cme_par['thick']

    # Generate CME object
    cme = huxtObject.ConeCME(
        t_launch=cme_launch_time,
        longitude=cme_lon,
        latitude=cme_lat,
        width=cme_width,
        v=cme_speed,
        thickness=cme_thickness
    )

    # Run CME through HUXt
    model.solve([cme])

    # Calculate CME flank
    obsObject = shmo.Observer(huxtObject, cme, obs_lon)

    hx = obsObject.compute_synthetic_obs(cme)

    return hx


def log_likelihood_function_gaussian(
        obs, obs_cov, huxtObject, cme_par, obs_lon
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
    hx = obs_op(huxtObject, cme_par, obs_lon)

    # Calculate the innovation (difference between the observations and the observation operator)
    innov = obs - hx

    # Calculate the inverse of the observation error covariance and calculate the logarithm
    #  of the likelihood function
    if np.shape(obs_cov)[0] > 1:
        obs_cov_1 = np.linalg.pinv(obs_cov)

        loglik = np.transpose(innov).dot( obs_cov_1.dot(innov) )
    else:
        obs_cov_1 = 1.0 / obs_cov

        loglik = obs_cov_1 * innov * innov

    return loglik


def likelihood_function_gaussian(obs, obs_cov, huxtObject, cme_par, obs_lon):
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
        obs, obs_cov, huxtObject, cme_par, obs_lon
    )

    # Calculate the likelihood function by taking the exponent
    likelihood = np.exp(loglik)

    return likelihood


def likelihood_function(
        obs, obs_cov, huxtObject, cme_par, obs_lon
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
    likelihood = likelihood_function_gaussian(obs, obs_cov, huxtObject, cme_par, obs_lon)

    return likelihood

def auxPf(
        pars, obs, obs_cov, obs_lon, obs_time, weights,
        fixed_ambient=True, pars_in_state=['v', 'lon', 'width'],
        hux_init_time=datetime.datetime(2008, 1, 1, 0, 0, 0),
        vr_in=np.zeros(128) + 400 * u.km / u.s,
        lon_start=290*u.deg,
        lon_stop=380*u.deg,
        time_tolerance=0.5*u.day,
        dt_scale=20,
        delta=0.98, rng=None
):

    pars = np.asarray(pars)   # Array containing all parameters as required
    obs = np.asarray(obs)
    sim_time = obs_time + time_tolerance
    nEns, nPar = pars.shape

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
                    dt_scale=dt_scale
                )
        else:
            # Initialise HUXt model object for each ensemble member
            model = setup_huxt(
                start_datetime=huxt_init_time,
                vr_in=vr_in,
                lon_start=lon_start,
                lon_stop=lon_stop,
                sim_time=sim_time,
                dt_scale=dt_scale
            )

        # Calculate the likelihood for this ensemble member
        likelihood_ens = likelihood_function(
            obs,
            obs_cov,
            model,
            pars,
            obs_lon
        )






    return None

def liu_west_resample_pars(pars, weights, delta=0.98, rng=None):
    ## Liu-West resampling for parameters
    #  @param pars (nEns, nPar) array containing parameters to update
    #  @param weights (n_Ens) array contain each particle's weight
    #  @param delta Discount factor in (0, 1] that determines the shrinkage factor, a=((3 * delta) - 1)/(2 * delta)
    #   typically between 0.95-0.99.
    #   Larger delta (e.g., 0.99) gives less jitter and more shrinkage, suitable when posterior is well-identified.
    #   Smaller delta (e.g., 0.95–0.97) adds more diversity when particle degeneracy is a risk.
    #  @param rng Seed for random generator
    #  @return new_pars Resampled particles

    pars = np.asarray(pars)
    weights = np.asarray(weights)
    nEns, nPar = pars.shape

    #Normalise weights
    weights = weights / weights.sum()

    # Compute weighted mean parameter
    meanPar = np.average(pars, axis=0, weights=weights)

    # Estimate particle covariance from the particles and the weights
    pertPar = pars - meanPar
    particleCov = (pertPar.T * weights) @ pertPar

    # Calculate shrinkage factors and variance adjustment required
    a = ((3 * delta) - 1) / (2 * delta)
    h2 = 1.0 - (a ** 2)

    if rng is None:
        rng = np.random.default_rng()

    # Resample from particle distribution
    resampInd = systematic_resampling(weights, rng)
    pars_resampled = pars[resampInd, :]

    # Shrink particles to the mean parameter
    shrunkParticles = a * pars_resampled + (1 - a) * meanPar

    # Perform Cholesky decomposition to get square-root of particle covariance matrix
    #  then draw random samples from a standard normal and transform to normal distributed sample
    #  with mean 0 and covariance particleCov (Transformation is Y=0+sqrt(particleCov)X
    mp = 1e-12
    # Add mp*I onto particleCov to ensure machine precision doesn't affect positive-definiteness
    L = np.linalg.cholesky(particleCov + (mp * np.eye(nPar)))
    noises = rng.normal(size=(nEns, nPar)) @ L.T

    # Calculate the jitter term to perturb the particles away from one another
    jitter = np.sqrt(h2) * noises

    # Combine the shrunkParticles with the jitter to get the newly resampled particles with target mean and variance
    new_pars = shrunkParticles + jitter

    return new_pars

# --- General APF for N-Dimensional State ---
def apf_nd_state(y, f, h, nParticles=1000, state_dim=3, param_dim=2,
                 delta=0.98, priors=None, q=0.05, r=0.2,
                 x0_mean=None, x0_std=None, rng=None):
    """
    Auxiliary Particle Filter for N-Dimensional State and Parameter Estimation.

    Parameters
    ----------
    y : array (T,)
        Observations.
    f : callable
        State transition function: f(x, theta) -> next state (shape: state_dim).
    h : callable
        Observation function: h(x) -> predicted observation.
    nParticles : int
        Number of particles.
    state_dim : int
        Dimension of state vector.
    param_dim : int
        Dimension of parameter vector.
    delta : float
        Liu–West discount factor.
    priors : dict or None
        Parameter priors: {'mean': array(param_dim), 'std': array(param_dim)}.
    q : float
        Process noise variance (applied to each state dimension).
    r : float
        Observation noise variance.
    x0_mean, x0_std : array-like
        Initial state mean and std (length = state_dim).
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    x_mean : (T, state_dim)
        Filtered state means.
    theta_mean : (T, param_dim)
        Filtered parameter means.
    """
    if rng is None:
        rng = np.random.default_rng()
    T = len(y)
    if x0_mean is None:
        x0_mean = np.zeros(state_dim)
    if x0_std is None:
        x0_std = np.ones(state_dim)

    # Initialize state particles
    x = rng.normal(loc=x0_mean, scale=x0_std, size=(nParticles, state_dim))

    # Initialize parameter particles
    if priors is None:
        theta = rng.normal(loc=np.ones(param_dim), scale=0.5, size=(nParticles, param_dim))
    else:
        theta = rng.normal(loc=priors['mean'], scale=priors['std'], size=(nParticles, param_dim))

    weights = np.full(nParticles, 1.0 / nParticles)
    x_mean = np.zeros((T, state_dim))
    theta_mean = np.zeros((T, param_dim))

    for t in range(T):
        # Predict next state for auxiliary weights
        x_pred = np.array([f(x[i], theta[i]) for i in range(nParticles)])
        y_pred = np.array([h(x_pred[i]) for i in range(nParticles)])

        # Auxiliary weights
        aux_loglik = -0.5 * ((y[t] - y_pred)**2 / (r + q) + np.log(2*np.pi*(r+q)))
        aux_w = np.exp(aux_loglik - np.max(aux_loglik))
        aux_w /= aux_w.sum()

        # Resample based on auxiliary weights
        idx = systematic_resampling(aux_w, rng)
        x = x[idx]
        theta = theta[idx]

        # Propagate states with process noise
        x_next = np.array([f(x[i], theta[i]) for i in range(nParticles)])
        x = x_next + rng.normal(scale=np.sqrt(q), size=(nParticles, state_dim))

        # Compute observation likelihood
        y_pred = np.array([h(x[i]) for i in range(nParticles)])
        innov = y[t] - y_pred
        loglik = -0.5 * (innov**2 / r + np.log(2*np.pi*r))
        logw = loglik - np.max(loglik)
        w = np.exp(logw)
        w /= w.sum()

        # Compute means
        x_mean[t] = np.sum(w[:, None] * x, axis=0)
        theta_mean[t] = np.average(theta, axis=0, weights=w)

        # Liu–West resample parameters
        theta = liu_west_resample_thetas(theta, w, delta=delta, rng=rng)

        weights = np.full(nParticles, 1.0 / nParticles)

    return x_mean, theta_mean



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


def test_PF_out():
    # --- Example Usage ---
    rng = np.random.default_rng(123)
    T = 50
    state_dim = 4
    param_dim = 2
    theta_true = np.array([0.9, 0.6])

    # Define transition and observation functions
    def f(x, theta):
        # Example nonlinear dynamics for N-D state
        x_next = np.zeros_like(x)
        x_next[0] = theta[0] * x[0] + 0.5 * x[1]
        x_next[1] = theta[1] * x[1] + 0.3 * x[2]
        x_next[2] = 0.8 * x[2] + np.sin(x[0])
        x_next[3] = 0.5 * x[3] + np.cos(x[1])

        return x_next

    def h(x):
        hx = x.sum()
        return hx  # observation = sum of state components

    # Simulate data
    x_true = np.zeros((T, state_dim))
    y = np.zeros(T)
    x_true[0] = np.array([0.5, -0.3, 0.2, 0.1])
    q, r = 0.05, 0.1
    for t in range(T):
        y[t] = h(x_true[t]) + rng.normal(scale=np.sqrt(r))
        if t < T - 1:
            x_true[t + 1] = f(x_true[t], theta_true) + rng.normal(scale=np.sqrt(q), size=state_dim)

    # Run APF
    priors = {'mean': np.array([1.0, 0.5]), 'std': np.array([0.3, 0.3])}
    x_mean, theta_mean = apf_nd_state(y, f, h, nParticles=2000, state_dim=state_dim,
                                      param_dim=param_dim, delta=0.98, priors=priors,
                                      q=q, r=r, x0_mean=np.zeros(state_dim),
                                      x0_std=np.ones(state_dim), rng=rng)

    print("True theta:", theta_true)
    print("Estimated theta (final mean):", theta_mean[-1])
    rmse_state = np.sqrt(np.mean((x_mean - x_true) ** 2))
    print("State RMSE:", rmse_state)

    return

if __name__ == "__main__":

    test_cme_par_to_state(nEns=10, seed=123)