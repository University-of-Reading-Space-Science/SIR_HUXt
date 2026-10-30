"""
Module containing functions that define the cme_par_dict structure and the associated indices required
for the cme_par_array and functions to map between the indices and keys of the array and dictionary
respectively.
"""


def required_dict_keys() -> set[str]:
    """
    Function to define the dictionary keys required for cme_par_dict
    :return: A Set of keys required for cme_par_dict
    """
    return {"t_init", "v", "width", "lon", "lat", "thick"}


def state_par_keys() -> set[str]:
    """
    Keys that can be in the particle filter's state vector (pars_in_state): the CME parameters of
    required_dict_keys plus the optional observation-bias coefficient "bias" (see obs_bias.py).
    "bias" is not a required key, so dictionaries without it still work (it is then taken as 0).
    :return: Set of allowed state-vector keys
    """
    return set(cme_par_key_to_index().keys())


def cme_par_keys_ordered() -> list[str]:
    """
    The six CME parameters (required_dict_keys) in cme_par_array column order, WITHOUT the optional
    observation-bias coefficient. Use this for anything describing the CME itself, e.g. the prior
    CME-parameter covariance matrices of make_prior_covariance_mat.py (always 6 x 6).
    :return: List of CME parameter keys
    """
    return [k for k in cme_par_key_to_index() if k in required_dict_keys()]


def cme_par_key_to_index() -> dict[str, int]:
    """
    Function to define the parameter index in the cme_par_array for the associated
    dictionary keys in cme_par_dict
    :return: A dictionary mapping dictionary keys of cme_par_dict to the
     parameter index of cme_par_array
    """
    return {"t_init": 0, "v": 1, "width": 2, "lon": 3, "lat": 4, "thick": 5, "bias": 6}


def cme_par_get_indices(cme_par_dict_key: str) -> int:
    """
    Function to define the parameter index in the cme_par_array arrays
     for specific dictionary keys in cme_par_dict
    :param cme_par_dict_key: Key name from cme_par_dict we want the index for.
        Must be in required_dictionary_fields
    :return indReq: Parameter index of cme_par_array
    """
    assert cme_par_dict_key in cme_par_key_to_index()

    cme_par_key_dict: dict[str, int] = cme_par_key_to_index()

    if cme_par_dict_key in cme_par_key_dict.keys():
        indReq: int = cme_par_key_dict[cme_par_dict_key]
    else:
        raise KeyError("Invalid dictionary key input into cme_par_get_indices.")

    # Assert that the index is one of the cme_par_array columns
    assert indReq in set(cme_par_key_to_index().values())

    return indReq


def cme_par_get_keys(cme_par_array_index: int) -> str:
    """
    Function to retrieve the appropriate cme_par_dict key based on the parameter
      index of cme_par_array
    :param cme_par_array_index: Parameter index of cme_par_array.
      Must be one of the column indices of cme_par_key_to_index (0-6).
    :return keyReq: Key name from cme_par_dict_key corresponding to
      cme_par_array parameter index
    """
    assert cme_par_array_index in set(cme_par_key_to_index().values())

    cme_par_key_dict: dict[str, int] = cme_par_key_to_index()

    keyReq: str = next(
        (key for key, value in cme_par_key_dict.items() if value == cme_par_array_index), None
    )

    if keyReq is None:
        raise IndexError(f"cme_par_index {cme_par_array_index} not found in cme_par_key_dict.")

    return keyReq
