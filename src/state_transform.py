"""
Handling of correlated CME parameters in the auxiliary particle filter (see aux_pf.py).

The particles carry parameter correlations automatically (each particle is a joint draw), but the
Liu-West step (shrink each particle towards the weighted mean, then jitter it with a Gaussian of
covariance h^2 * Sigma_ens) works badly when the posterior lies on a narrow, curved ridge in the
natural parameters and Sigma_ens is estimated from ~20 particles, many of them duplicates. This
module changes the coordinates in which the Liu-West step is done, and regularises Sigma_ens:

  1. Reparametrisation ("reparam"):
       "none"     the state as it is (t_init in s, v in km/s, ...)
       "log_v"    v -> ln v
       "tref_logv" v -> ln v and t_init -> t_ref = t_init + (r_ref - cme_init_rad) / v, the time the
                  CME nose reaches the reference radius r_ref ("t_ref_rs", solar radii; None = the
                  model's inner boundary r_min, i.e. the time the observation operator starts the
                  CME at). The elongations fix the time the CME reaches the distances where it is
                  observed, i.e. they constrain t_init + (r_obs - r0) / v, which is a curved ridge
                  in (t_init, v) or (t_init, ln v) but a nearly straight line t_ref ~ const in
                  (t_ref, ln v) when r_ref is close to r_obs. The posterior correlation of t_ref
                  and ln v is smallest for r_ref near the front distance at the early-to-middle
                  observations (~50-60 Rs for HI elongations 8-30 deg in the 09A twin geometry;
                  see the notes). (If t_init is not in the state, this is the same as "log_v".)
  2. Whitening ("whiten"): z = L^-1 (u - mu), with mu and L L^T the mean and (regularised)
     covariance of the PRIOR ensemble in the reparametrised variables u. Fixed for the whole run.
     It puts all parameters on a common, unit-free scale and makes "independent in z" mean
     "correlated like the prior" in the natural variables.
     NB: the Gaussian jitter with the full ensemble covariance is affine invariant, so whitening
     on its own does not change the jitter; it matters through the shrinkage and floor below.
  3. Covariance shrinkage ("shrinkage_alpha"): the jitter covariance is
        Sigma = (1 - alpha) Sigma_ens + alpha diag(Sigma_ens)
     in the working coordinates (z if whitened). With whitening, this pulls the noisy sample
     correlations towards the prior's correlation structure while keeping the ensemble's
     variances (no inflation). alpha = 0 keeps the plain weighted covariance.
  4. Jitter floor ("min_jitter_sd"): a minimum jitter standard deviation on each working axis,
     in units of the prior sd when whitened (otherwise in the parameter's own units), so the
     jitter never collapses to zero along an axis after repeated resampling. 0 switches it off.
  5. Parameters carried from the prior ("carry_from_prior"): parameters the observations cannot
     constrain (e.g. width, lat; and lon near the blind line phi + eps = 90 deg) are taken out of
     the state vector used by the Liu-West step. Each particle keeps its own prior value, inherited
     through resampling with the rest of its parameters (aux_pf copies the parent's whole
     cme_par_array row), so their prior correlation with the state parameters is kept.

All of this is off unless settings["enabled"] is True (see da_surf_setup.Setup.get_correlation_settings).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt


def default_correlation_settings() -> dict:
    """
    The default correlation settings (a new dictionary on every call): correlation handling off
    """
    return {
        "enabled": False,
        "reparam": "tref_logv",  # "none" | "log_v" | "tref_logv"
        "t_ref_rs": 50.0,  # reference radius for t_ref (solar radii); None = r_min
        "whiten": True,
        "prior_cov_shrinkage": 0.1,  # regularises the 20-member prior covariance used for whitening
        "shrinkage_alpha": 0.2,  # Liu-West covariance shrinkage towards its diagonal
        "min_jitter_sd": 0.02,  # jitter sd floor (prior sds if whitened)
        "carry_from_prior": [],  # e.g. ["width", "lat"]
    }


def resolve_correlation_settings(settings: dict | None) -> dict:
    """
    Fill in missing correlation settings with default_correlation_settings() and check them
    :param settings: Dictionary from Setup.get_correlation_settings (or None for all defaults, i.e. off)
    :return: Complete settings dictionary
    """
    allowed_reparam: tuple[str, str, str] = ("none", "log_v", "tref_logv")
    out: dict[str, Any] = default_correlation_settings()
    if settings:
        out.update(settings)
    out["carry_from_prior"] = list(out.get("carry_from_prior") or [])
    if out["reparam"] not in allowed_reparam:
        raise ValueError(
            f"correlation reparam must be one of {allowed_reparam}, not {out['reparam']!r}"
        )
    for k in ("shrinkage_alpha", "prior_cov_shrinkage"):
        if not 0.0 <= float(out[k]) <= 1.0:
            raise ValueError(f"correlation {k} must be in [0, 1], not {out[k]}")
    if out["t_ref_rs"] is not None and float(out["t_ref_rs"]) <= 0:
        raise ValueError("correlation t_ref_rs must be > 0 (solar radii) or None")
    if float(out["min_jitter_sd"]) < 0:
        raise ValueError("correlation min_jitter_sd must be >= 0")
    return out


def filter_state_pars(pars_in_state: list[str], settings: dict) -> list[str]:
    """
    The parameters the Liu-West step works on: pars_in_state without those carried from the prior
    :param pars_in_state: All parameters in the state vector
    :param settings: Resolved correlation settings
    :return: List of parameters updated by the particle filter
    """
    if not settings.get("enabled", False):
        return list(pars_in_state)
    carried: set[str] = set(settings["carry_from_prior"])
    kept: list[str] = [p for p in pars_in_state if p not in carried]
    if not kept:
        raise ValueError(
            "carry_from_prior removes every parameter from the state vector"
        )
    return kept


def weighted_mean_cov(
    x: npt.NDArray[np.float64], log_weights: npt.ArrayLike | None = None
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Weighted mean and covariance of an (n_ens, n_par) ensemble, with weights exp(log_weights)
    :return: mean (n_par,), cov (n_par, n_par)
    """
    x = np.asarray(x, dtype=float)
    if log_weights is None:
        w: npt.NDArray[np.float64] = np.ones(x.shape[0])
    else:
        lw: npt.NDArray[np.float64] = np.asarray(log_weights, dtype=float)
        lw = np.where(np.isfinite(lw), lw, -np.inf)
        w = np.exp(lw - np.max(lw))
    w = w / w.sum()
    mean: npt.NDArray[np.float64] = w @ x
    cov: npt.NDArray[np.float64] = np.atleast_2d(np.cov(x, rowvar=False, aweights=w))
    return mean, cov


def weighted_corr(
    x: npt.NDArray[np.float64], log_weights: npt.ArrayLike | None = None
) -> npt.NDArray[np.float64]:
    """Weighted correlation matrix of an (n_ens, n_par) ensemble (NaN for a parameter with no spread)"""
    c: npt.NDArray[np.float64]
    _, c = weighted_mean_cov(x, log_weights)
    sd: npt.NDArray[np.float64] = np.sqrt(np.diag(c))
    with np.errstate(invalid="ignore", divide="ignore"):
        return c / np.outer(sd, sd)


class StateTransform:
    """
    Maps the filter's natural state vector (columns ordered as pars, units as in ToStateVector:
    t_init in s since surf_init_time, v in km/s, angles in deg, ...) to the working coordinates in
    which the Liu-West shrinkage and jitter are done, and back.
    """

    RS_KM: float = 695700.0  # solar radius (km)

    def __init__(
        self,
        pars: list[str],
        settings: dict,
        r_min_rs: float = 21.5,
        cme_init_rad_rs: float = 12.0,
        prior_state: npt.NDArray[np.float64] | None = None,
        prior_log_weights: npt.ArrayLike | None = None,
    ) -> None:
        """
        :param pars: Parameters in the state vector (after removing those carried from the prior)
        :param settings: Resolved correlation settings (resolve_correlation_settings)
        :param r_min_rs: Model inner boundary (solar radii); the reference radius if settings["t_ref_rs"] is None
        :param cme_init_rad_rs: Radius at which t_init is defined (solar radii)
        :param prior_state: (n_ens, n_par) prior ensemble in natural units, needed if whitening
        :param prior_log_weights: Log weights of the prior ensemble (None = equal)
        """
        self.pars: list[str] = list(pars)
        self.settings: dict[str, Any] = settings
        self.enabled: bool = bool(settings.get("enabled", False))
        self.reparam: str = settings["reparam"] if self.enabled else "none"
        r_ref: float | None = settings.get("t_ref_rs")
        self.r_ref_rs: float = float(r_min_rs) if r_ref is None else float(r_ref)
        self.dt_dist_km: float = (self.r_ref_rs - float(cme_init_rad_rs)) * self.RS_KM
        self.i_v: int | None = self.pars.index("v") if "v" in self.pars else None
        self.i_t: int | None = (
            self.pars.index("t_init") if "t_init" in self.pars else None
        )
        self.use_log_v: bool = (
            self.reparam in ("log_v", "tref_logv") and self.i_v is not None
        )
        self.use_tb: bool = (
            self.reparam == "tref_logv"
            and self.i_v is not None
            and self.i_t is not None
        )
        self.whiten: bool = self.enabled and bool(settings.get("whiten", False))
        self.mu: npt.NDArray[np.float64] = np.zeros(len(self.pars))
        self.L: npt.NDArray[np.float64] = np.eye(len(self.pars))
        self.L_inv: npt.NDArray[np.float64] = np.eye(len(self.pars))
        if self.whiten:
            if prior_state is None:
                raise ValueError(
                    "StateTransform: whitening needs the prior ensemble (prior_state)"
                )
            self.set_prior(prior_state, prior_log_weights)

    # ---------------------------------------------------------------- labels
    def working_labels(self) -> list[str]:
        """Names of the working coordinates (before whitening)"""
        lab: list[str] = list(self.pars)
        if self.use_log_v:
            lab[self.i_v] = "ln_v"
        if self.use_tb:
            lab[self.i_t] = f"t_ref({self.r_ref_rs:g}Rs)"
        return lab

    # ---------------------------------------------------------------- reparametrisation
    def _reparam(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        u: npt.NDArray[np.float64] = np.array(x, dtype=float, copy=True)
        if self.use_tb:
            u[:, self.i_t] = x[:, self.i_t] + self.dt_dist_km / x[:, self.i_v]
        if self.use_log_v:
            if np.any(x[:, self.i_v] <= 0):
                raise ValueError(
                    "StateTransform: CME speed <= 0 cannot be log-transformed"
                )
            u[:, self.i_v] = np.log(x[:, self.i_v])
        return u

    def _unreparam(self, u: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        x: npt.NDArray[np.float64] = np.array(u, dtype=float, copy=True)
        if self.use_log_v:
            x[:, self.i_v] = np.exp(u[:, self.i_v])
        if self.use_tb:
            x[:, self.i_t] = u[:, self.i_t] - self.dt_dist_km / x[:, self.i_v]
        return x

    # ---------------------------------------------------------------- whitening
    def set_prior(
        self,
        prior_state: npt.NDArray[np.float64],
        prior_log_weights: npt.ArrayLike | None = None,
    ) -> None:
        """
        Fix the whitening transform from the prior ensemble. The prior covariance (from only
        n_members particles) is shrunk towards its diagonal by settings["prior_cov_shrinkage"],
        and a parameter with no prior spread is given unit variance, so the Cholesky factor exists.
        """
        u: npt.NDArray[np.float64] = self._reparam(np.atleast_2d(prior_state))
        c: npt.NDArray[np.float64]
        mu: npt.NDArray[np.float64]
        mu, c = weighted_mean_cov(u, prior_log_weights)
        a: float = float(self.settings.get("prior_cov_shrinkage", 0.0))
        d: npt.NDArray[np.float64] = np.diag(c).copy()
        d[~(d > 0)] = 1.0
        c = (1 - a) * c + a * np.diag(d)
        c[np.diag_indices_from(c)] = np.where(np.diag(c) > 0, np.diag(c), 1.0)
        # tiny ridge in case the prior ensemble is rank deficient (n_members <= n_par)
        c = c + 1e-9 * np.diag(np.diag(c))
        self.mu = mu
        self.L = np.linalg.cholesky(c)
        self.L_inv = np.linalg.inv(self.L)

    # ---------------------------------------------------------------- public maps
    def forward(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Natural state (n_ens, n_par) -> working coordinates"""
        x = np.atleast_2d(np.asarray(x, dtype=float))
        if not self.enabled:
            return x.copy()
        u: npt.NDArray[np.float64] = self._reparam(x)
        if self.whiten:
            u = (u - self.mu) @ self.L_inv.T
        return u

    def inverse(self, z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Working coordinates (n_ens, n_par) -> natural state"""
        z = np.atleast_2d(np.asarray(z, dtype=float))
        if not self.enabled:
            return z.copy()
        u: npt.NDArray[np.float64] = z @ self.L.T + self.mu if self.whiten else z
        return self._unreparam(u)

    # ---------------------------------------------------------------- jitter covariance
    def regularise_cov(self, cov: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Shrink the (working-coordinate) ensemble covariance towards its diagonal:
          (1 - alpha) cov + alpha diag(cov)
        """
        if not self.enabled:
            return cov
        a: float = float(self.settings.get("shrinkage_alpha", 0.0))
        return (1 - a) * cov + a * np.diag(np.diag(cov))

    def floor_jitter_cov(
        self, scaled_cov: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """
        Raise the jitter variance on any working axis below min_jitter_sd^2 to min_jitter_sd^2
        (applied to the already-scaled jitter covariance h^2 Sigma)
        """
        if not self.enabled:
            return scaled_cov
        s: float = float(self.settings.get("min_jitter_sd", 0.0))
        if s <= 0:
            return scaled_cov
        out: npt.NDArray[np.float64] = np.array(scaled_cov, dtype=float, copy=True)
        d: npt.NDArray[np.float64] = np.diag(out)
        add: npt.NDArray[np.float64] = np.clip(s**2 - d, 0.0, None)
        out[np.diag_indices_from(out)] = d + add
        return out
