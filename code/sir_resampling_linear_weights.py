import numpy as np
import bisect
from typing import Generator

class ResampleParsLinearWeights:
    def __init__(
            self,
            weights: list[float],
            n_ensemble: int = None,
            rng: Generator = None
    ):
        self.weights: list[float] = weights

        if n_ensemble is None:
            self.n_ensemble: int = len(weights)
        else:
            assert n_ensemble > 0
            assert n_ensemble == len(weights)
            self.n_ensemble: int = n_ensemble

        if rng is None:
            self.rng: Generator = np.random.default_rng()
        else:
            self.rng: Generator = rng

    def norm_weights(self) -> list[float]:
        """
        Function to normalise the weights so that they all sum to 1
        :return: Normalised weights
        """
        return self.weights / np.sum(self.weights)

    def make_cdf(self) -> list[float]:
        """
        Function to make the cumulative distribution function of the weights
        :return: Cumulative distribution function of the weights
        """
        # Get the normalised weights
        normalised_weights: list[float] = self.norm_weights()

        cdf_weights: list[float] = [
            np.sum(normalised_weights[:(i + 1)]) / np.sum(normalised_weights)
            for i in range(0, self.n_ensemble)
        ]

        # Round weights to nearest billionth to avoid machine precision errors
        cdf_weights: list[float] = [
            float(np.round(x, 9)) for x in cdf_weights
        ]

        return cdf_weights

    # Systematic resampling routine for parameters
    def systematic_resampling(self, init_position: float = None):
        """
        Function to systematically resample the particles based upon the weights of each particle
        :param init_position: Initial position for the resampling (to be specified for testing purposes)
        :return: Indices of particles to resample
        """
        # The default behaviour for init_position is to use a random number between 0 and 1
        if init_position is None:
            init_position = self.rng.uniform(low=0, high=1) / self.n_ensemble

        assert (init_position >= 0) and (init_position <= (1.0 / self.n_ensemble))
        # Calculate number of ensemble members required and initialise output
        #  variable to hold resampled particles indices
        # resamp_ind = np.zeros(self.n_ensemble, dtype=int)

        # Define interval for particles to resample and round to nearest billionth to remove machine precision errors
        positions: list[float] = [
            init_position + (k / self.n_ensemble) for k in range(self.n_ensemble)
        ]
        positions: list[float] = [
            float(np.round(x, 9)) for x in positions
        ]
        # (rng.random() + np.arange(self.n_ensemble)) / self.n_ensemble

        # Calculate CDF all weights
        cdf_val: list[float] = self.make_cdf()
        print(f"pos={positions}")
        resamp_ind: list[int] = [
            bisect.bisect_left(cdf_val, x) for x in positions
        ]

        return resamp_ind

# def shrink_par(pars, log_weights=None, delta=0.98):
#     ## Liu-West shrinkage of the parameters towards the weighted mean
#     #  @param pars (nEns, nPar) array containing parameters for each particle
#     #  @param weights (nEns) array containing the weights of the particles
#     #  @param delta  Discount factor in (0, 1] that determines the shrinkage factor, a=((3 * delta) - 1)/(2 * delta)
#     #   typically between 0.95-0.99.
#     #   Larger delta (e.g., 0.99) gives less jitter and more shrinkage, suitable when posterior is well-identified.
#     #   Smaller delta (e.g., 0.95–0.97) adds more diversity when particle degeneracy is a risk.
#     #  @return shrunk_pars (nEns, nPar) array containing parameters shrunk to mean
#
#     # pars = pars
#     nEns, nPar = pars.shape
#     # print((pars))
#
#     if log_weights is None:
#         # If no weights are provided, calculate an arithmetic mean
#         meanPar = np.mean(pars, axis=0)
#     else:
#         # If weights are provided, calculate weighted mean
#
#         # Transform from log-space and normalise weights
#         weights = np.asarray(np.exp(log_weights))
#         weights = weights / weights.sum()
#
#         # Compute weighted mean parameter
#         meanPar = np.average(pars, axis=0, weights=weights)
#
#     # Calculate shrinkage factor required
#     a = ((3 * delta) - 1) / (2 * delta)
#
#     print(f"meanPar = {meanPar}")
#     # print(f"a = {a}")
#     # print(f"1-a = {1 - a}")
#
#     # Shrink towards the mean
#     shrunk_pars = (a * pars) + ((1 - a) * meanPar)
#
#     return shrunk_pars