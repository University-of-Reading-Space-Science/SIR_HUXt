"""
prior_cov_plots.py

Plots of prior CME-parameter covariance / correlation matrices (heat maps) and of the CME-parameter
samples they were estimated from (pair plots). Only used when ``plot_cme_cov=True``.
(seaborn is imported inside the functions: importing it at module level added ~2 s to every start-up.)
"""
import matplotlib.pyplot as plt
import numpy.typing as npt
import pandas as pd

from cme_par_dict_structure import cme_par_get_indices, required_dict_keys, cme_par_get_keys


def plot_sample_cov(
        df_cov: npt.NDArray[float] | pd.DataFrame,
        df_corr: npt.NDArray[float] | pd.DataFrame,
        cov_name: str,
        use_log_v: bool = False,
) -> None:
    """
    Heat maps of a 6x6 CME-parameter covariance matrix and the corresponding correlation matrix.
    :param df_cov: Covariance matrix (cme_par_array order)
    :param df_corr: Correlation matrix (cme_par_array order)
    :param cov_name: Name of the prior, used in the titles
    :param use_log_v: Label the speed axis "log(v)"
    """
    import seaborn as sns
    fig, ax = plt.subplots(1, 1)
    tick_labels = [
        cme_par_get_keys(i) for i in range(len(required_dict_keys()))
    ]
    if use_log_v:
        tick_labels[cme_par_get_indices("v")] = "log(v)"

    sns.heatmap(
        df_cov, annot=True, fmt=".3g", cmap=plt.cm.RdBu_r,
        vmin=-1, vmax=1, ax=ax,
        xticklabels=tick_labels, yticklabels=tick_labels
    )
    ax.set_title(f"Covariance matrix for {cov_name}")
    plt.show()
    plt.close(fig)

    fig, ax = plt.subplots(1, 1)
    sns.heatmap(
        df_corr, annot=True, fmt=".4g", cmap=plt.cm.RdBu_r,
        vmin=-1, vmax=1, ax=ax,
        xticklabels=tick_labels, yticklabels=tick_labels
    )
    ax.set_title(f"Correlation matrix for {cov_name}")
    plt.show()
    plt.close(fig)

    return None


def plot_pair_plot(
        df_cme_par: pd.DataFrame,
        cov_name: str,
        use_log_v: bool = False,
) -> None:
    """
    Seaborn pair plot (with KDE contours below the diagonal) of the CME-parameter samples a prior
    was estimated from.
    :param df_cme_par: DataFrame with one column per CME parameter
    :param cov_name, use_log_v: Unused (kept for symmetry with plot_sample_cov)
    """
    import seaborn as sns
    im = sns.pairplot(df_cme_par)
    im.map_lower(sns.kdeplot, levels=10, color="k")
    plt.show()
    plt.close(im.figure)

    return None
