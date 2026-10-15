"""
sir_likelihood.py

Gaussian likelihood of an observation given the model-predicted observation (the observation
operator output, ``hx``). Used by the particle filter to weight each ensemble member.

For a single observation y with error variance R:
    log L = -0.5 * (y - Hx)^2 / R
and for a vector of observations with error covariance matrix R:
    log L = -0.5 * (y - Hx)^T R^{-1} (y - Hx)
(both up to an additive constant, which cancels when the weights are normalised).
"""
import sys

import numpy as np
import numpy.typing as npt


def _to_float_or_array(x: float | list[float] | npt.NDArray[float]) -> float | npt.NDArray[float]:
    """
    Helper to standardise the inputs of LikelihoodFunction.
    Single-element inputs (python/numpy scalars, 0-d arrays, length-1 lists/arrays, 1x1 matrices)
    are returned as a python float; anything larger is returned as a float numpy array.
    """
    arr = np.asarray(x, dtype=float)
    if arr.size == 1:
        return float(arr.reshape(-1)[0])
    return arr


class LikelihoodFunction:
    """
    Gaussian likelihood of one (possibly multi-element) observation for one ensemble member.

    Example::

        log_l = LikelihoodFunction(obs=y, obs_cov=R, hx=hx_member).make_log_likelihood()
    """
    def __init__(
            self,
            obs: float | list[float] | npt.NDArray[float],
            obs_cov: list[float] | npt.NDArray[float] | float,
            hx: list[float] | npt.NDArray[float] | float
    ) -> None:
        """
        Standardise and validate the inputs.
        :param obs: Observation of the CME flank
        :param obs_cov: Observation error covariance matrix of the CME flank
            (for a single observation this is the error *variance*)
        :param hx: Observation operator (what model thinks the observation should be),
           should be precomputed previous to calculating the likelihood function
        """
        self.obs: float | npt.NDArray[float] = _to_float_or_array(obs)
        self.obs_cov: float | npt.NDArray[float] = _to_float_or_array(obs_cov)
        self.hx: float | npt.NDArray[float] = _to_float_or_array(hx)

        # If vectors/matrices are involved, check they are properly specified
        if isinstance(self.obs, np.ndarray) or isinstance(self.hx, np.ndarray):
            n_obs = np.size(self.obs)

            # Assert that the obs, hx and obs_cov are all of consistent shapes
            assert np.shape(self.hx) == np.shape(self.obs), (
                f"np.shape(hx) = {np.shape(self.hx)} != np.shape(obs) = {np.shape(self.obs)}"
            )

            if np.ndim(self.obs_cov) == 1:
                # Interpret a vector as the diagonal of the covariance matrix
                self.obs_cov = np.diag(self.obs_cov)
            elif np.ndim(self.obs_cov) == 0:
                self.obs_cov = self.obs_cov * np.eye(n_obs)

            assert np.shape(self.obs_cov) == (n_obs, n_obs), (
                f"np.shape(obs_cov) = {np.shape(self.obs_cov)}, expected {(n_obs, n_obs)}"
            )

            # Assert that the observation error covariance matrix is symmetric (to a tolerance of 1e-8),
            #  has non-negative diagonal elements and is invertible
            assert np.all(np.abs(self.obs_cov - np.transpose(self.obs_cov)) < 1e-8)
            assert np.all(np.diag(self.obs_cov) >= 0)
            assert np.linalg.det(self.obs_cov) > 0
        else:
            assert self.obs_cov > 0, f"obs_cov = {self.obs_cov} must be positive"


    def log_likelihood_gaussian(self) -> float:
        """
        log_likelihood_function_gaussian: The purpose of this definition is to calculate the
          logarithm of the likelihood function using a Gaussian distribution (up to an additive constant)
              log L = -0.5 * (y - Hx)^T R^{-1} (y - Hx)
        :return: log_likelihood: The logarithm of the likelihood function
            (nan if the observation operator is nan, e.g. CME not tracked)
        """

        # Calculate the innovation (difference between the observations and the observation operator)
        innov: npt.NDArray[float] | float = self.obs - self.hx

        if isinstance(innov, float):
            # 1D-case
            log_likelihood: float = -0.5 * innov * innov / self.obs_cov
        elif isinstance(innov, np.ndarray):
            # Dimension of observation is greater than 1
            obs_cov_1: npt.NDArray[float] = np.linalg.inv(self.obs_cov)
            log_likelihood: float = float(-0.5 * innov.dot(obs_cov_1.dot(innov)))
        else:
            log_likelihood = np.nan
            sys.exit(f"type(innov) = {type(innov)} is an unsupported variable type, should be float or np.ndarray.")

        return log_likelihood


    def likelihood_gaussian(self) -> float:
        """
        likelihood_function_gaussian: The purpose of this definition is to calculate the
          likelihood function using a Gaussian distribution

        :return: likelihood: The likelihood function
        """

        # Calculate the log likelihood function
        log_likelihood: float = self.log_likelihood_gaussian()

        # Calculate the likelihood function by taking the exponent
        likelihood: float = np.exp(log_likelihood)

        return likelihood


    def make_log_likelihood(self) -> float:
        """
        make_log_likelihood: The purpose of this definition is to calculate the logarithm
         of the likelihood function
        :return: log_likelihood: The logarithm of the likelihood function
        """
        log_likelihood: float = self.log_likelihood_gaussian()

        return log_likelihood


    def make_likelihood(self) -> float:
        """
        make_likelihood: The purpose of this definition is to calculate the likelihood function
        :return: likelihood: The likelihood function
        """
        likelihood: float = self.likelihood_gaussian()

        return likelihood
