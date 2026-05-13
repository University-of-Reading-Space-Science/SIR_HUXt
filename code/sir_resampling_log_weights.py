import numpy as np
import bisect
from typing import Generator


def jacobian_log(exponents: list[float]) -> float:
    """
    Function to calculate the Jacobian logarithm of the exponents, given by the sum:
        Jacob_log = log( sum_{j=1}^{N}[exp{exponents_j}] )
    Used for normalising the log_weights
    :return: Jacobian logarithm of exponents
    """

    # Initialise the Jacobian logarithm as the first exponent
    jacob_log: float = exponents[0]

    lenExp: int = len(exponents)

    # Update the Jacobian logarithm based upon the algorithm in Gentner (2008) Algorithm 1
    for i in range(1, lenExp):
        # Calculate the components
        term1: float = np.max([jacob_log, exponents[i]])

        t2exp: float = -np.abs(exponents[i] - jacob_log)
        term2: float = np.log(1 + np.exp(t2exp))

        jacob_log: float = term1 + term2

    return jacob_log


class ResampleParsLogWeights:
    def __init__(
            self,
            log_weights: list[float],
            n_ensemble: int = None,
            rng: Generator = None
    ):
        self.log_weights: list[float] = log_weights

        if n_ensemble is None:
            self.n_ensemble: int = len(log_weights)
        else:
            assert n_ensemble > 0
            assert n_ensemble == len(log_weights)
            self.n_ensemble: int = n_ensemble

        if rng is None:
            self.rng: Generator = np.random.default_rng()
        else:
            self.rng: Generator = rng


    def norm_log_weights(self) -> list[float]:
        """
        Function to normalise the logarithmic weights so that
        :return: Normalised weights
        """
        return [self.log_weights[i] - jacobian_log(self.log_weights) for i in range(self.n_ensemble)]


    def make_cdf_log_weights(self) -> list[float]:
        """
        Function to make the cumulative distribution function of the weights
        :return: Cumulative distribution function of the weights
        """
        # Get the normalised weights
        normalised_weights: list[float] = self.norm_log_weights()

        cdf_weights: list[float] = [
            jacobian_log(normalised_weights[:(i + 1)])# / np.sum(normalised_weights)
            for i in range(0, self.n_ensemble)
        ]

        # Round weights to nearest billionth to avoid machine precision errors
        cdf_weights: list[float] = [
            float(np.round(x, 9)) for x in cdf_weights
        ]
        return cdf_weights


    # Systematic resampling routine for parameters
    def systematic_resampling_log_weights(self, init_position: float = None):
        """
        Function to systematically resample the particles based upon the weights of each particle
        :param init_position: Initial position for the resampling (to be specified for testing purposes)
        :return: Indices of particles to resample
        """
        # The default behaviour for init_position is to use a random number between 0 and 1
        if init_position is None:
            init_position: float = self.rng.uniform(low=0, high=1) / self.n_ensemble

        assert (init_position >= 0) and (init_position <= (1.0 / self.n_ensemble))
        # Calculate number of ensemble members required and initialise output
        #  variable to hold resampled particles indices
        # resamp_ind = np.zeros(self.n_ensemble, dtype=int)

        # Define interval for particles to resample and round to nearest billionth to remove machine precision errors
        positions: list[float] = [
            init_position + (k / self.n_ensemble)
            for k in range(self.n_ensemble)
        ]
        positions: list[float] = [
            -np.inf if x == 0 else float(np.round(np.log(x), 9))
            for x in positions
        ]
        # (rng.random() + np.arange(self.n_ensemble)) / self.n_ensemble

        # Calculate CDF all weights
        cdf_val: list[float] = self.make_cdf_log_weights()
        print(f"pos={positions}")
        resamp_ind: list[int] = [
            bisect.bisect_left(cdf_val, x) for x in positions
        ]

        return resamp_ind