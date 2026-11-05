"""
obs_bias.py
Online estimation of a single model-bias term in the HI elongation observation operator.

The bias-corrected observation operator for ensemble member i is
    h'_i = h_i + beta_i * p(h_i)
where h_i is the member's modelled elongation (deg), beta_i its bias coefficient and p a predictor
computed from the member's own modelled elongation (as in variational bias correction, Dee 2005;
Auligne, McNally & Dee 2007). beta is part of the particle filter's state ("bias" in pars_in_state),
so it is shrunk, jittered and resampled with the CME parameters (Liu & West 2001).

The predictor is chosen to describe the curvature seen in the innovation statistics (the model is
close to the observations below ~15 deg and then runs increasingly ahead), which a change of CME
speed or launch time, giving an error that grows roughly linearly in time, cannot mimic:
    "quadratic": p = (max(h - E_REF, 0) / E_SCALE)^2
    "linear":    p =  max(h - E_REF, 0) / E_SCALE        (default, see BiasPredictor)
    "constant":  p = 1
For "quadratic" and "linear", beta is the bias in degrees at a modelled elongation of E_REF + E_SCALE
(10 deg with the defaults E_REF = 0 deg and E_SCALE = 10 deg), and the correction is zero at and below E_REF.
For "constant", beta is a fixed offset in degrees at every elongation, and E_REF and E_SCALE are not
used. A negative beta means the model runs ahead of the observed CME (the observations are at lower
elongation than the model).

The prior for beta is N(prior_mean, prior_sd^2): prior_mean = 0 by default, or derived from
innovation_statistics.py (see prior_mean_from_innovation_statistics): b(E_REF + E_SCALE) - b(E_REF)
of the fitted bias for "quadratic"/"linear", the mean innovation for "constant".
"""

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class BiasPredictor:
    """
    Bias predictor, shared by the members and, in twin experiments, the synthetic truth.
    The defaults are used wherever a predictor is not given; a run can choose another one with the
    "predictor", "e_ref_deg" and "e_scale_deg" keys of its bias settings (Setup.get_bias_settings).
    """

    type: str = "linear"
    e_ref_deg: float = 0.0
    e_scale_deg: float = 10.0

    ALLOWED_PREDICTORS: ClassVar[tuple[str, ...]] = ("quadratic", "linear", "constant")
    ALLOWED_MODES: ClassVar[tuple[str, ...]] = ("off", "fixed", "estimate")

    def kwargs(self) -> dict:
        """Keyword arguments for bias_predictor / bias_units / apply_bias"""
        return dict(predictor=self.type, e_ref=self.e_ref_deg, e_scale=self.e_scale_deg)


def bias_predictor(
    h: npt.ArrayLike,
    predictor: str | None = None,
    e_ref: float | None = None,
    e_scale: float | None = None,
) -> npt.NDArray[np.float64]:
    """
    Predictor p(h) of the bias term from modelled elongation(s) h (deg). NaN where h is NaN.
    :param h: Modelled elongation(s), deg
    :param predictor: "quadratic", "linear" or "constant" (default BiasPredictor.type)
    :param e_ref: Elongation above which the bias acts (default BiasPredictor.e_ref_deg)
    :param e_scale: Elongation scale (default BiasPredictor.e_scale_deg)
    :return: Predictor value(s), same shape as h
    """
    default: BiasPredictor = BiasPredictor()
    predictor = default.type if predictor is None else predictor
    e_ref = default.e_ref_deg if e_ref is None else e_ref
    e_scale = default.e_scale_deg if e_scale is None else e_scale
    h = np.asarray(h, dtype=float)
    if predictor == "constant":
        return np.where(np.isfinite(h), 1.0, np.nan)
    x: npt.NDArray[np.float64] = np.maximum(h - e_ref, 0.0) / e_scale
    if predictor == "linear":
        return x
    if predictor == "quadratic":
        return x**2
    raise ValueError(
        f"Unknown bias predictor {predictor}; expected one of {BiasPredictor.ALLOWED_PREDICTORS}"
    )


def bias_units(
    predictor: str | None = None,
    e_ref: float | None = None,
    e_scale: float | None = None,
) -> str:
    """Units of beta for labels: "deg" for "constant", otherwise "deg at <E_REF + E_SCALE> deg"."""
    default: BiasPredictor = BiasPredictor()
    predictor = default.type if predictor is None else predictor
    if predictor == "constant":
        return "deg"
    e_ref = default.e_ref_deg if e_ref is None else e_ref
    e_scale = default.e_scale_deg if e_scale is None else e_scale
    return f"deg at {e_ref + e_scale:g} deg"


def apply_bias(
    h: npt.ArrayLike, beta: npt.ArrayLike, **kwargs: Any
) -> npt.NDArray[np.float64]:
    """
    Bias-corrected elongations h + beta * p(h).
    :param h: (n_members, n_times) or (n_members,) modelled elongations, deg
    :param beta: (n_members,) bias coefficients, or a scalar
    :return: Array like h
    """
    h = np.asarray(h, dtype=float)
    beta = np.asarray(beta, dtype=float)
    if h.ndim == 2:
        beta = np.broadcast_to(np.atleast_1d(beta), (h.shape[0],))[:, None]
    else:
        beta = np.broadcast_to(beta, h.shape)
    return h + beta * bias_predictor(h, **kwargs)


def prior_mean_from_innovation_statistics(
    stats_file: str,
    source: str = "openloop",
    predictor: str | None = None,
    e_ref: float | None = None,
    e_scale: float | None = None,
) -> tuple[float, dict[str, Any]]:
    """
    Prior mean of beta from innovation_statistics.py, matched to the bias predictor:
    "quadratic" / "linear": beta is the bias at E_REF + E_SCALE relative to E_REF (the predictor is
        zero at E_REF and 1 at E_REF + E_SCALE), so the prior mean is the increase of the fitted bias
        polynomial b(e) between them:
            prior_mean = b(E_REF + E_SCALE) - b(E_REF)
        Any offset b(E_REF) itself is not represented by this single term. A warning is printed if
        E_REF + E_SCALE lies outside the elongation range the statistics cover (the polynomial is then
        extrapolated; prefer statistics from more or longer events).
    "constant": beta is an offset at all elongations, so the prior mean is the mean innovation
        (observation minus model) over all the innovations of the source; E_REF and E_SCALE are not used.
    NB innovation_statistics.py fits b against the OBSERVED elongation, here it is used as a function
    of modelled elongation; the two are close where the bias is small.
    :param stats_file: Path to innovation_statistics.json
    :param source: Innovation source to use ("openloop", "forecast" or "analysis"); "openloop" is the
        least affected by the assimilation and so the preferred estimate of model bias
    :param predictor: "quadratic", "linear" or "constant" (default BiasPredictor.type)
    :param e_ref: Predictor reference elongation (default BiasPredictor.e_ref_deg)
    :param e_scale: Predictor elongation scale (default BiasPredictor.e_scale_deg)
    :return: (prior_mean, info dict)
    """
    default: BiasPredictor = BiasPredictor()
    predictor = default.type if predictor is None else predictor
    if predictor not in BiasPredictor.ALLOWED_PREDICTORS:
        raise ValueError(
            f"Unknown bias predictor {predictor}; expected one of {BiasPredictor.ALLOWED_PREDICTORS}"
        )
    e_ref = default.e_ref_deg if e_ref is None else e_ref
    e_scale = default.e_scale_deg if e_scale is None else e_scale
    with open(stats_file) as f:
        stats: dict[str, Any] = json.load(f)
    st: dict[str, Any] = stats["sources"][source]

    # Elongation range covered by the statistics (bins with at least 3 values)
    bins: list[dict[str, Any]] = [
        b for b in st.get("by_elongation", []) if b.get("n", 0) >= 3
    ]
    e_range: list[float] = (
        [min(b["lo"] for b in bins), max(b["hi"] for b in bins)]
        if bins
        else [np.nan, np.nan]
    )

    if predictor == "constant":
        if "mean_innovation" in st:
            beta: float = float(st["mean_innovation"])
        elif bins:  # older statistics files: n-weighted mean of the bin means
            beta = float(
                sum(b["n"] * b["mean_innovation"] for b in bins)
                / sum(b["n"] for b in bins)
            )
        else:
            raise ValueError(
                f"No mean innovation for source '{source}' in {stats_file}"
            )
        info: dict[str, Any] = dict(
            stats_file=stats_file,
            source=source,
            predictor=predictor,
            data_e_range=e_range,
            mean_innovation=beta,
            extrapolated=False,
            n=st.get("n"),
            n_events=st.get("n_events"),
        )
        return beta, info

    if "bias_c0" not in st:
        raise ValueError(f"No bias estimate for source '{source}' in {stats_file}")
    coeffs: list[float] = st.get("bias_coeffs", [st["bias_c0"], st.get("bias_c1", 0.0)])
    bias_at: Callable[[float], float] = lambda e: float(
        np.polyval(np.asarray(coeffs, float)[::-1], e)
    )
    beta = bias_at(e_ref + e_scale) - bias_at(e_ref)
    extrapolated: bool = bool(bins) and not (
        e_range[0] <= e_ref and e_ref + e_scale <= e_range[1]
    )
    if extrapolated:
        print(
            f"Warning: the {source} statistics in {stats_file} cover {e_range[0]:.0f}-{e_range[1]:.0f} deg, "
            f"so the bias at {e_ref:.0f} and {e_ref + e_scale:.0f} deg is extrapolated; the prior mean "
            f"({beta:.2f} deg) may be unreliable."
        )
    info = dict(
        stats_file=stats_file,
        source=source,
        predictor=predictor,
        bias_coeffs=coeffs,
        data_e_range=e_range,
        bias_at_e_ref=bias_at(e_ref),
        bias_at_e_ref_plus_scale=bias_at(e_ref + e_scale),
        extrapolated=extrapolated,
        n=st.get("n"),
        n_events=st.get("n_events"),
    )
    return beta, info


def resolve_bias_settings(settings: dict[str, Any] | None) -> dict[str, Any]:
    """
    Fill in and check a bias-settings dict (from Setup.get_bias_settings), computing the prior mean
    from innovation statistics if requested.
    Keys:
        mode: "off" (no bias term; default), "fixed" (every member uses prior_mean, not estimated)
              or "estimate" (beta in the state vector, prior N(prior_mean, prior_sd^2))
        prior_source: "zero" (prior_mean = 0 unless prior_mean is given) or "innovation_statistics"
        prior_mean, prior_sd: prior of beta (deg at E_REF + E_SCALE; deg at all elongations for the
            "constant" predictor)
        innovation_statistics_file, innovation_statistics_source: used with prior_source
            "innovation_statistics"
        true_bias: bias injected into synthetic (twin) observations (deg); 0 for none
        predictor_type, e_ref_deg, e_scale_deg: the bias predictor (optional; defaults from
            BiasPredictor)
    :return: Completed settings dict, with the predictor as "predictor" (dict, as saved in
        cme_pars.nc) and "bias_predictor" (BiasPredictor)
    """
    default: BiasPredictor = BiasPredictor()
    s: dict[str, Any] = dict(
        mode="off",
        prior_source="zero",
        prior_mean=0.0,
        prior_sd=2.0,
        innovation_statistics_file=None,
        innovation_statistics_source="openloop",
        true_bias=0.0,
        predictor_type=default.type,
        e_ref_deg=default.e_ref_deg,
        e_scale_deg=default.e_scale_deg,
    )
    s.update(settings or {})
    if s["mode"] not in BiasPredictor.ALLOWED_MODES:
        raise ValueError(f"bias mode {s['mode']} not in {BiasPredictor.ALLOWED_MODES}")
    if s["predictor_type"] not in BiasPredictor.ALLOWED_PREDICTORS:
        raise ValueError(
            f"Unknown bias predictor {s['predictor_type']}; expected one of "
            f"{BiasPredictor.ALLOWED_PREDICTORS}"
        )
    predictor: BiasPredictor = BiasPredictor(
        type=s["predictor_type"],
        e_ref_deg=float(s["e_ref_deg"]),
        e_scale_deg=float(s["e_scale_deg"]),
    )
    s["bias_predictor"] = predictor
    if s["prior_source"] == "innovation_statistics":
        if not s["innovation_statistics_file"]:
            raise ValueError(
                "prior_source 'innovation_statistics' needs innovation_statistics_file"
            )
        s["prior_mean"], s["prior_info"] = prior_mean_from_innovation_statistics(
            s["innovation_statistics_file"],
            s["innovation_statistics_source"],
            **predictor.kwargs(),
        )
        print(
            f"Bias prior mean from {s['innovation_statistics_file']} ({s['innovation_statistics_source']}): "
            f"{s['prior_mean']:.3f} deg"
        )
    elif s["prior_source"] != "zero":
        raise ValueError("prior_source must be 'zero' or 'innovation_statistics'")
    # E_REF and E_SCALE have no effect on the "constant" predictor, so record them as NaN for it
    uses_scale: bool = predictor.type != "constant"
    s["predictor"] = dict(
        type=predictor.type,
        e_ref_deg=predictor.e_ref_deg if uses_scale else np.nan,
        e_scale_deg=predictor.e_scale_deg if uses_scale else np.nan,
    )
    return s
