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


def cme_par_key_to_index() -> dict[str, int]:
    """
    Function to define the parameter index in the cme_par_array for the associated
    dictionary keys in cme_par_dict
    :return: A dictionary mapping dictionary keys of cme_par_dict to the
     parameter index of cme_par_array
    """
    return {"t_init": 0, "v": 1, "width": 2, "lon": 3, "lat": 4, "thick": 5}


def cme_par_get_indices(cme_par_dict_key: str) -> int:
    """
    Function to define the parameter index in the cme_par_array arrays
     for specific dictionary keys in cme_par_dict
    :param cme_par_dict_key: Key name from cme_par_dict we want the index for.
        Must be in required_dictionary_fields
    :return indReq: Parameter index of cme_par_array
    """
    assert cme_par_dict_key in required_dict_keys()

    cme_par_key_dict: dict[str, int] = cme_par_key_to_index()

    if cme_par_dict_key in cme_par_key_dict.keys():
        indReq: int = cme_par_key_dict[cme_par_dict_key]
    else:
        raise KeyError("Invalid dictionary key input into cme_par_get_indices.")

    # Assert that there are only 6 possible indices that can be output
    assert indReq in {0, 1, 2, 3, 4, 5}

    return indReq


def cme_par_get_keys(cme_par_array_index: int) -> str:
    """
    Function to retrieve the appropriate cme_par_dict key based on the parameter
      index of cme_par_array
    :param cme_par_array_index: Parameter index of cme_par_array.
      Must be integer between 0 and 5 (inclusive).
    :return keyReq: Key name from cme_par_dict_key corresponding to
      cme_par_array parameter index
    """
    assert cme_par_array_index in {0, 1, 2, 3, 4, 5}

    cme_par_key_dict: dict[str, int] = cme_par_key_to_index()

    keyReq: str = next(
        (key for key, value in cme_par_key_dict.items() if value == cme_par_array_index), None
    )

    if keyReq is None:
        raise IndexError(f"cme_par_index {cme_par_index} not found in cme_par_key_dict.")

    return keyReq
