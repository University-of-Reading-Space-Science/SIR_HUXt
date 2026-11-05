"""
Reuse of model runs within a data assimilation run (PFSettings.reuse_model_runs).

Each ensemble of model runs is solved from the model start, and the observation operator builds the
whole synthetic elongation track of each member before reading off the observation time. When the
particles have not changed since they were last run (after an observation that only updated the
weights, or after resampling, when the next observation sees the resampled particles again), their
elongation at the next observation is already in that earlier run. ElongationStore keeps, for each
member's CME parameters, the elongation (without the observation-bias term) at every observation that
has been evaluated from its runs, so that the filter can read them instead of solving the model again.

The stored values are those the observation operator would return if the model were run again for
that observation: the same model set-up and run length, and (see ObservationOperator's
extra_obs_times) an observer at the same position.
"""

from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import Any

import numpy as np
import numpy.typing as npt

from cme_par_ens import CmeParEns

# CME parameters that feed the model, and the units their values are compared in
_MEMBER_KEYS: tuple[tuple[str, str | None], ...] = (
    ("t_init", None),
    ("v", "km/s"),
    ("width", "deg"),
    ("lon", "deg"),
    ("lat", "deg"),
    ("thick", "solRad"),
)


def _as_key_value(x: Any, unit: str | None) -> Any:
    """A parameter value as a hashable number (a datetime is kept as it is)"""
    if isinstance(x, datetime.datetime):
        return x
    if unit is not None and hasattr(x, "to"):
        return float(x.to(unit).value)
    try:
        return float(x.value) if hasattr(x, "value") else float(x)
    except (TypeError, ValueError):
        return x  # e.g. a numpy datetime64


def _member_values(cme_par_dict: CmeParEns, key: str, n_members: int) -> list:
    """cme_par_dict[key] as one value per member (a single value is used for every member)"""
    values = cme_par_dict[key]
    try:
        out: list = list(values)
    except TypeError:
        out = [values]
    if len(out) == 1 and n_members > 1:
        out = out * n_members
    return out


def member_keys(cme_par_dict: CmeParEns) -> list[tuple]:
    """
    One key per member: the CME parameters that go into the model (not the weights or the bias
    coefficient, which do not change the model run)
    """
    n: int = int(cme_par_dict["n_members"])
    columns: list[list] = [
        [_as_key_value(x, unit) for x in _member_values(cme_par_dict, key, n)]
        for key, unit in _MEMBER_KEYS
    ]
    return [tuple(col[i] for col in columns) for i in range(n)]


class ElongationStore:
    """
    Elongations (deg, without the observation-bias term) of model runs already made, by member CME
    parameters and observation index. Used for one data assimilation run (one model set-up).
    """

    def __init__(self) -> None:
        self._store: dict[tuple, dict[int, float]] = {}

    def __len__(self) -> int:
        return len(self._store)

    def lookup(
        self, cme_par_dict: CmeParEns, obs_index: int
    ) -> npt.NDArray[np.float64] | None:
        """
        Every member's stored elongation at observation obs_index, or None unless all members have one
        :param cme_par_dict: Ensemble
        :param obs_index: Observation index
        :return: (n_members,) elongations, or None
        """
        out: list[float] = []
        for key in member_keys(cme_par_dict):
            values: dict[int, float] | None = self._store.get(key)
            if values is None or obs_index not in values:
                return None
            out.append(values[obs_index])
        return np.asarray(out, dtype=float)

    def add(
        self,
        cme_par_dict: CmeParEns,
        elongations: Mapping[int, Any],
    ) -> None:
        """
        Store the elongations of an ensemble's model runs
        :param cme_par_dict: Ensemble that was run
        :param elongations: {observation index: (n_members,) elongations without the bias term}
        """
        keys: list[tuple] = member_keys(cme_par_dict)
        for obs_index, values in elongations.items():
            values = np.asarray(values, dtype=float)
            for key, value in zip(keys, values):
                self._store.setdefault(key, {})[int(obs_index)] = float(value)
