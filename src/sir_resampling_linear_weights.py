"""
sir_resampling_linear_weights.py

Systematic resampling of particles with linear (not logarithmic) weights.
See ``sir_resampling_log_weights.py`` for the log-space version used by the Aux-PF.
"""
import bisect

import numpy as np
import numpy.typing as npt
from numpy.random import Generator


class ResampleParsLinearWeights:
    """
    Systematic resampling of an ensemble from its (unnormalised) linear weights.

    Example::

        indices = ResampleParsLinearWeights(weights, rng=rng).systematic_resampling()
    """
    def __init__(
            self,
            weights: list[float],
            n_ensemble: int | None = None,
            rng: Generator | None = None
    ) -> None:
        """
        :param weights: Particle weights (need not be normalised)
        :param n_ensemble: Number of ensemble members (defaults to len(weights))
        :param rng: Random number generator used to draw the systematic-resampling offset
        """
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


    def norm_weights(self) -> npt.NDArray[float]:
        """
        Function to normalise the weights so that they all sum to 1
        :return: Normalised weights
        """
        # np.asarray so that a python list of weights works too (list / float raises TypeError)
        weights = np.asarray(self.weights, dtype=float)
        return weights / np.sum(weights)


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
    def systematic_resampling(self, init_position: float | None = None) -> list[int]:
        """
        Systematically resample the particles: N evenly spaced positions u_k = u_0 + k/N,
        u_0 ~ U(0, 1/N), each select the first particle whose CDF value is >= u_k
        :param init_position: Initial position for the resampling (to be specified for testing purposes)
        :return: Indices of particles to resample
        """
        # The default behaviour for init_position is to use a random number between 0 and 1/N
        if init_position is None:
            init_position = self.rng.uniform(low=0, high=1) / self.n_ensemble

        assert (init_position >= 0) and (init_position <= (1.0 / self.n_ensemble))
        # Define interval for particles to resample and round to nearest billionth to remove machine precision errors
        positions: list[float] = [
            init_position + (k / self.n_ensemble) for k in range(self.n_ensemble)
        ]
        positions: list[float] = [
            float(np.round(x, 9)) for x in positions
        ]

        # Calculate the CDF of all weights
        cdf_val: list[float] = self.make_cdf()
        resamp_ind: list[int] = [
            min(bisect.bisect_left(cdf_val, x), self.n_ensemble - 1) for x in positions
        ]

        return resamp_ind
