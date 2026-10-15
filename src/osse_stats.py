"""
osse_stats.py

Small statistics helpers for analysing the ensemble output: weighted median and NaN-aware
mean / median / standard deviation that return NaN (without warnings) for all-NaN input.
"""
import numpy as np
import numpy.typing as npt


def weighted_median(
        values: npt.NDArray[float] | list[float],
        weights: npt.NDArray[float] | list[float],
        axis: int = 0
) -> float | npt.NDArray[float]:
    """
    Compute the weighted median along a specified axis.

    Parameters
    ----------
    values : ndarray
        Input data.
    weights : ndarray
        Weights, same shape as values.
    axis : int, optional
        Axis along which to compute the weighted median.

    Returns
    -------
    ndarray
        Weighted median with the specified axis removed (a float for 1D input):
        the first sorted value whose cumulative weight reaches half the total weight.
    """
    values = np.asarray(values)
    weights = np.asarray(weights)

    if values.shape != weights.shape:
        raise ValueError("values and weights must have the same shape")

    if np.any(weights < 0):
        raise ValueError("weights must be non-negative")

    # Sort values and corresponding weights along the axis
    sort_idx = np.argsort(values, axis=axis)
    sorted_values = np.take_along_axis(values, sort_idx, axis=axis)
    sorted_weights = np.take_along_axis(weights, sort_idx, axis=axis)

    # Cumulative weights
    cum_weights = np.cumsum(sorted_weights, axis=axis)

    # Half total weight
    cutoff = np.sum(sorted_weights, axis=axis, keepdims=True) / 2

    # First index where cumulative weight exceeds cutoff
    median_idx = np.argmax(cum_weights >= cutoff, axis=axis)

    # Extract weighted median values
    result = np.take_along_axis(
        sorted_values,
        np.expand_dims(median_idx, axis=axis),
        axis=axis,
    )

    result = np.squeeze(result, axis=axis)
    if np.ndim(result) == 0:
        result = float(result)

    return result


def numpy_nan_mean(
        arr: npt.NDArray | list,
        axis: int=0
) -> float | npt.NDArray[float]:
    """
    Helper function to ensure that all-nan slices in nanmean are appropriately accounted for
    :param arr: array or list to calculate mean for
    :param axis: Axis to reduce over
    :return: mean_arr: Mean of array or list
    """
    if np.isnan(arr).all():
        mean_arr = np.nan
    else:
        mean_arr = np.nanmean(arr, axis=axis)

    return mean_arr


def numpy_nan_median(
        arr: npt.NDArray | list,
        axis: int=0
) -> float | npt.NDArray[float]:
    """
    Helper function to ensure that all-nan slices in nanmedian are appropriately accounted for
    :param arr: array or list to calculate the median for
    :param axis: Axis to reduce over
    :return: median_arr: Median of array or list
    """
    if np.isnan(arr).all():
        median_arr = np.nan
    else:
        median_arr = np.nanmedian(arr, axis=axis)

    return median_arr


def numpy_nan_std(
        arr: npt.NDArray | list,
        axis: int=0
) -> float | npt.NDArray[float]:
    """
    Helper function to ensure that all-nan slices in nanstd are appropriately accounted for
    :param arr: array or list to calculate the standard deviation for
    :param axis: Axis to reduce over
    :return: std_arr: Standard deviation of array or list
    """
    if np.isnan(arr).all():
        std_arr = np.nan
    else:
        std_arr = np.nanstd(arr, axis=axis)

    return std_arr
