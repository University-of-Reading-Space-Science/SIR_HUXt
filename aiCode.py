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
