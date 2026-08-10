import numpy as np
import numpy.typing as npt
import sys

class LikelihoodFunction:
    def __init__(
            self,
            obs: list[float] | npt.NDArray[float] | float,
            obs_cov: list[float] | npt.NDArray[float] | float,
            hx: list[float] | npt.NDArray[float] | float
    ):
        """
        Class to calculate the likelihood function and store functions
         related to different likelihood functions
        :param obs: Observation of the CME flank
        :param obs_cov: Observation error covariance matrix of the CME flank
        :param hx: Observation operator (what model thinks the observation should be),
           should be precomputed previous to calculating the likelihood function
        """
        # If a list is provided to the class, convert it to an array
        if isinstance(obs, list):
            if len(obs) == 1:
                try:
                    obs[0].item()
                except:
                    self.obs: float = obs[0]
                else:
                    self.obs: float = obs[0].item()
            else:
                self.obs: npt.NDArray[float] = np.array(obs)
        elif (isinstance(obs, np.ndarray)) and (np.ndim(obs) == 0):
            print(f"obs = {obs}: {type(obs)}, np.ndim = {np.ndim(obs)}")
            self.obs: float = obs.item()
        elif isinstance(obs, np.float64):
            self.obs: float = float(obs) #.item()
        else:
            self.obs: npt.NDArray[float] | float = obs

        if isinstance(obs_cov, list):
            if len(obs_cov) == 1:
                try:
                    obs_cov[0].item()
                except:
                    self.obs_cov: float = obs_cov[0]
                else:
                    self.obs_cov: float = obs_cov[0].item()
            else:
                self.obs_cov: npt.NDArray[float] = np.array(obs_cov)
        elif (isinstance(obs_cov, np.ndarray)) and (np.ndim(obs_cov) == 0):
            self.obs_cov: float = obs_cov.item().astype(float)
        elif isinstance(obs_cov, np.float64):
            self.obs_cov: float = obs_cov.item()
        else:
            self.obs_cov: npt.NDArray[float] | float = obs_cov

        if isinstance(hx, list):
            if len(hx) == 1:
                try:
                    hx[0].item()
                except:
                    self.hx: float = hx[0]
                else:
                    self.hx: float = hx[0].item()
            else:
                self.hx: npt.NDArray[float] = np.array(hx)
        elif (isinstance(hx, np.ndarray)) and (np.ndim(hx) == 0):
            self.hx: float = hx.item()
        elif isinstance(hx, np.float64):
            self.hx: float = hx.item()
        else:
            self.hx: npt.NDArray[float] | float = hx

        # Assert that the input variables are all the same type
        assert (
            (type(self.hx) == type(self.obs_cov) == type(self.obs)),
            f"type(self.hx) = {self.hx}, type(self.obs_cov) = {type(self.obs_cov)}, type(self.obs) = {type(self.obs)}"
        )


        # If scalars and matrices are involved, check they are properly specified
        if type(hx) is npt.NDArray[float]:
            # Assert that the obs, hx and obs_cov are all of consistent shapes
            assert np.shape(self.hx) == np.shape(self.obs)
            assert np.shape(self.obs_cov) == (np.len(self.obs), np.len(self.obs))

            # Assert that the observation error covariance matrix is symmetric (to a tolerance of 1e-8, invertible
            # and the diagonal elements are all positive (if obs_cov is symmetrical,
            # it should be invertible too)
            assert np.linalg.det(obs_cov) > 0
            assert np.all((obs_cov - np.transpose(obs_cov)) < 1e-8)
            assert (obs_cov[i, i] >= 0 for i in range(len(self.obs)))

            # If we only have 1 dimension arrays, set hx, obs and obs_cov to floats
            if np.ndim(self.hx) == 1:
                #print(f"hx={self.hx}")
                self.hx: float = hx[0]

            if np.ndim(self.obs) == 1:
                #print(f"obs={self.obs}")
                self.obs: float = obs[0]

            if np.ndim(self.obs_cov) == 1:
                #print(f"obs_cov={self.obs_cov}")
                self.obs_cov: float = obs_cov[0]


    def log_likelihood_gaussian(self) -> float:
        """
        log_likelihood_function_gaussian: The purpose of this definition is to calculate the
          logarithm of the likelihood function using a Gaussian distribution
        :return: log_likelihood: The logarithm of the likelihood function
        """

        # Calculate the innovation (difference between the observations and the observation operator)
        innov: npt.NDArray[float] | float = self.obs - self.hx

        # Calculate the inverse of the observation error covariance and calculate the logarithm
        #  of the likelihood function
        if np.ndim(innov) == 0:
            innov = float(innov)
        elif np.ndim(innov) == 1:
            innov = float(innov[0])

        if np.ndim(self.obs_cov) == 0:
            # 1D-case
            obs_cov_1: float = float(1.0 / self.obs_cov)

        elif np.ndim(self.obs_cov) == 1:
            obs_cov_1: float = 1.0 / float(self.obs_cov.item())

        else:
            # Dimension of observation is greater than 1
            obs_cov_1: npt.NDArray[float] = np.linalg.pinv(self.obs_cov)

        if isinstance(innov, float):
            assert isinstance(obs_cov_1, float)

            log_likelihood: float = -obs_cov_1 * innov * innov
        elif isinstance(innov, np.ndarray):
            assert(isinstance(obs_cov_1, np.ndarray))
            log_likelihood: float = -np.transpose(innov).dot(obs_cov_1.dot(innov))
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
