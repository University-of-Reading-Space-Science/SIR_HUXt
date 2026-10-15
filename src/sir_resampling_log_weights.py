"""
sir_resampling_log_weights.py

Systematic resampling of particles whose weights are stored as *logarithms*.

Working in log-space avoids underflow when likelihoods are tiny (e.g. members far from the
observation). The cumulative distribution function (CDF) of the weights and the systematic
resampling positions are therefore both expressed as logarithms before the search.

See ``sir_resampling_linear_weights.py`` for the equivalent routine with linear weights.
"""
import bisect

import numpy as np
import numpy.typing as npt
from numpy.random import Generator


def jacobian_log(exponents: list[float] | npt.NDArray[float]) -> float:
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
    """
    Systematic resampling of an ensemble from its (unnormalised) log-weights.

    Non-finite log-weights are treated as (effectively) zero weight. If *every* weight is invalid the
    weights are reset to uniform.

    Example::

        indices = ResampleParsLogWeights(log_weights, rng=rng).systematic_resampling_log_weights()
    """
    def __init__(
            self,
            log_weights: list[float],
            n_ensemble: int | None = None,
            rng: Generator | None = None
    ) -> None:
        """
        :param log_weights: Logarithm of the (unnormalised) particle weights
        :param n_ensemble: Number of ensemble members (defaults to len(log_weights))
        :param rng: Random number generator used to draw the systematic-resampling offset
        """
        self.log_weights: list[float] = log_weights

        # Replace nan or infinite values in log_weights with a very small log-weight (~zero weight)
        # (previous condition `~np.isnan(x) or np.isfinite(x)` let +/-inf through)
        self.log_weights: list[float] = [
            float(x) if np.isfinite(x) else -1e31
            for x in log_weights
        ]

        # If every particle failed (non-finite, or already set to the -1e31 sentinel upstream, e.g. no
        #  ensemble member's CME was tracked at the observation time) there is no information in the
        #  weights, so fall back to uniform weights. Previously -1e31 + log(N) == -1e31 in floating point,
        #  so the normalised weights were all exp(0) = 1 and every particle was resampled onto member 0.
        if all(x <= -1e30 for x in self.log_weights):
            self.log_weights = [0.0 for _ in self.log_weights]

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
        Normalise the log-weights so that the corresponding linear weights sum to one,
        i.e. subtract log(sum_j exp(log_w_j)).
        :return: Normalised log-weights
        """
        log_norm_const: float = jacobian_log(self.log_weights)
        return [self.log_weights[i] - log_norm_const for i in range(self.n_ensemble)]


    def make_cdf_log_weights(self) -> list[float]:
        """
        Logarithm of the cumulative distribution function (CDF) of the normalised weights:
            log_cdf_i = log( sum_{j<=i} w_j )
        :return: Log-CDF of the weights (last entry ~0, i.e. CDF ~1)
        """
        # Get the normalised weights
        normalised_weights: list[float] = self.norm_log_weights()

        cdf_weights: list[float] = [
            jacobian_log(normalised_weights[:(i + 1)])
            for i in range(0, self.n_ensemble)
        ]

        # Round weights to nearest billionth to avoid machine precision errors
        cdf_weights: list[float] = [
            float(np.round(x, 9)) for x in cdf_weights
        ]

        return cdf_weights


    # Systematic resampling routine for parameters
    def systematic_resampling_log_weights(self, init_position: float | None = None) -> list[int]:
        """
        Systematically resample the particles: N evenly spaced positions
            u_k = u_0 + k/N,  u_0 ~ U(0, 1/N),  k = 0..N-1
        are compared against the CDF of the weights (both in log-space), and each position
        selects the first particle whose CDF value is >= u_k.
        :param init_position: Initial position u_0 (specify for testing; random by default)
        :return: Indices of the particles to resample (one per new ensemble member)
        """
        # The default behaviour for init_position is to use a random number between 0 and 1/N
        if init_position is None:
            init_position: float = self.rng.uniform(low=0, high=1) / self.n_ensemble

        assert (init_position >= 0) and (init_position <= (1.0 / self.n_ensemble))
        # Define interval for particles to resample and round to nearest billionth to remove machine precision errors
        positions: list[float] = [
            init_position + (k / self.n_ensemble)
            for k in range(self.n_ensemble)
        ]
        positions: list[float] = [
            -np.inf if x == 0 else float(np.round(np.log(x), 9))
            for x in positions
        ]

        # Calculate the (log-)CDF of all weights
        cdf_val: list[float] = self.make_cdf_log_weights()
        # Clip to the last particle in case rounding leaves cdf_val[-1] marginally below a position
        resamp_ind: list[int] = [
            min(bisect.bisect_left(cdf_val, x), self.n_ensemble - 1) for x in positions
        ]

        return resamp_ind