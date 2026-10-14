import numpy as np
import bisect
from typing import Generator


def jacobian_log(exponents: list[float]) -> float:
    """
    Function to calculate the Jacobian logarithm of the exponents, given by the sum:
        Jacob_log = log( sum_{j=1}^{N}[exp{exponents_j}] )
    Used for normalising the log_weights.
    Computed with the numerically stable log-sum-exp trick (equivalent to the recursion in
    Gentner (2008) Algorithm 1). NaN entries are ignored (previously a NaN in the first entry,
    or two consecutive -inf entries, turned the whole result into NaN).
    :return: Jacobian logarithm of exponents
    """
    x = np.asarray(exponents, dtype=float).ravel()
    x = x[~np.isnan(x)]

    if x.size == 0:
        return np.nan

    x_max = np.max(x)
    if not np.isfinite(x_max):
        # All -inf (=> log(0) = -inf) or at least one +inf (=> +inf)
        return float(x_max)

    jacob_log: float = float(x_max + np.log(np.sum(np.exp(x - x_max))))

    return jacob_log


class ResampleParsLogWeights:
    def __init__(
            self,
            log_weights: list[float],
            n_ensemble: int = None,
            rng: Generator = None
    ):
        self.log_weights: list[float] = log_weights

        # Remove nan or infinite values from log_weights
        # (previous condition `~np.isnan(x) or np.isfinite(x)` let +/-inf through)
        self.log_weights: list[float] = [
            float(x) if np.isfinite(x) else -1e31
            for x in log_weights
        ]

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
        log_norm_const: float = jacobian_log(self.log_weights)
        return [self.log_weights[i] - log_norm_const for i in range(self.n_ensemble)]


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

        # Calculate CDF all weights
        cdf_val: list[float] = self.make_cdf_log_weights()
        # Clip to the last particle in case rounding leaves cdf_val[-1] marginally below a position
        resamp_ind: list[int] = [
            min(bisect.bisect_left(cdf_val, x), self.n_ensemble - 1) for x in positions
        ]

        return resamp_ind