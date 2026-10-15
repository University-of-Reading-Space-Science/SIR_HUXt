"""
make_prior_covariance_mat.py

Prior distributions for the CME parameters, used to draw the initial ensemble.

The implementation is split by prior type; this module re-exports everything so existing code
(``import make_prior_covariance_mat as mp_cov``) keeps working:

    prior_uncorrelated.py : uncorrelated Gaussian / uniform priors      (cme_cov_type "uncorr")
    prior_blair.py        : covariance from Blair's CME list
    prior_donki.py        : covariance from the DONKI cone-CME catalogue   (cme_cov_type "donki")
    prior_mo_cone.py      : covariance from the Met Office cone-CME files  (cme_cov_type "mo_cone")
    prior_cov_plots.py    : heat-map / pair plots of the above

Every ``make_*_samples`` function returns a (6, n_ens) array whose rows follow the cme_par_array
column order (t_init, v, width, lon, lat, thick). Running this file draws example samples.
"""
import datetime
import os

import numpy as np

# Re-exported for backwards compatibility (these functions were previously defined here)
from prior_cov_plots import plot_sample_cov, plot_pair_plot  # noqa: F401
from prior_uncorrelated import (  # noqa: F401
    uncorrelated_cov_mat, make_uncorrelated_samples, make_uncorr_uniform_samples,
)
from prior_blair import make_cov_blair, make_blair_samples  # noqa: F401
from prior_donki import make_donki_cov, make_donki_samples  # noqa: F401
from prior_mo_cone import (  # noqa: F401
    filter_first_mo_cone_cme, filter_final_mo_cone_cme, filter_mo_cone_cme_1month,
    make_mo_cone_file_list, make_mo_cone_cme_cov, make_mo_cone_samples,
)


def main(
        uncorr_samp: bool=True,
        blair_samp: bool=True,
        donki_samp: bool=True,
        mo_cone_samp: bool=True,
) -> None:
    """
    Draw example samples from each prior (paths are for the original development machine).
    :param uncorr_samp, blair_samp, donki_samp, mo_cone_samp: Which priors to sample
    """
    n_ens: int = 3
    mean_cme_pars: list[float] = [0.1, 1000.2, 2000.3, 3000.4, 4000.5, 5000.6]
    vars_req = np.array(["t_init", "v", "width", "lon", "lat", "thick"])

    sd_t_init = 0
    sd_v = 50
    sd_width = 10.6
    sd_lon = 4
    sd_lat = 2.5
    sd_thick = 0

    use_log_v = True
    plot_cme_cov = True

    if use_log_v:
        sd_v = np.log(sd_v)


    if uncorr_samp:
        rng = np.random.default_rng(42)
        uncorr_samp = make_uncorrelated_samples(
            n_ens, mean_cme_pars, rng,
            sd_t_init=sd_t_init, sd_v=sd_v, sd_width=sd_width,
            sd_lon=sd_lon, sd_lat=sd_lat, sd_thick=sd_thick
        )


    if blair_samp:
        rng = np.random.default_rng(42)
        blair_file_path = os.path.join(
            "C:\\", "Users", "ss905122", "PycharmProjects", "SIR_HUXt", "blairCMElistSingle.csv"
        )
        blair_samples = make_blair_samples(
            n_ens=n_ens,
            mean_cme_pars=mean_cme_pars,
            blair_file_path=blair_file_path,
            rng=rng,
            vars_req=vars_req,
            sd_t_init=sd_t_init,
            sd_v=sd_v,
            sd_width=sd_width,
            sd_lon=sd_lon,
            sd_lat=sd_lat,
            sd_thick=sd_thick,
            use_log_v=use_log_v,
            plot_cme_cov=plot_cme_cov,
        )

    if donki_samp:
        start_time = datetime.datetime(2017, 1, 1, 0, 0, 0)
        end_time = datetime.datetime(2026, 2, 1, 0, 0, 0)
        rng = np.random.default_rng(42)
        most_acc_only = "true"
        catalog = "ALL"
        feature = "LE"

        donki_samples = make_donki_samples(
            n_ens=n_ens,
            mean_cme_pars=mean_cme_pars,
            rng=rng,
            start_time=start_time,
            end_time=end_time,
            scale_corr=True,
            vars_req=vars_req,
            sd_t_init=sd_t_init,
            sd_v=sd_v,
            sd_width=sd_width,
            sd_lon=sd_lon,
            sd_lat=sd_lat,
            sd_thick=sd_thick,
            use_log_v=use_log_v,
            plot_cme_cov=plot_cme_cov,
            most_acc_only = most_acc_only,
            catalog = catalog,
            feature = feature
        )

    if mo_cone_samp:
        mo_start_time = datetime.datetime(2017, 1, 1, 5, 0, 0)
        mo_end_time = datetime.datetime(2026, 2, 1, 0, 0, 0)
        rng = np.random.default_rng(42)
        mo_cone_cov_dir = os.path.join(
            "C:\\", "Users", "ss905122", "PycharmProjects", "SIR_HUXt", "moConeCMECov"
        )
        if not os.path.exists(mo_cone_cov_dir):
            os.makedirs(mo_cone_cov_dir)

        mo_cme_cone_file_dir = os.path.join(
            "C:\\", "Users", "ss905122", "PycharmProjects", "moswoc_cone"
        )
        make_mo_cone_samples(
            n_ens=n_ens,
            mean_cme_pars=mean_cme_pars,
            rng=rng,
            mo_cone_file_dir=mo_cme_cone_file_dir,
            start_time=mo_start_time,
            end_time=mo_end_time,
            overwrite_cov=False,
            mo_cone_cov_dir=mo_cone_cov_dir,
            scale_corr=True,
            vars_req=vars_req,
            sd_t_init=sd_t_init,
            sd_v=sd_v,
            sd_width=sd_width,
            sd_lon=sd_lon,
            sd_lat=sd_lat,
            sd_thick=sd_thick,
            use_log_v=use_log_v,
            plot_cme_cov=plot_cme_cov,
        )
    return None

if __name__ == "__main__":
    main(
        uncorr_samp=False,
        blair_samp=False,
        donki_samp=True,
        mo_cone_samp=True,
    )