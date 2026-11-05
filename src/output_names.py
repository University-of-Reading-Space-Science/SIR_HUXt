"""
Names of the output folders and files shared by data_assimilation_routine.py (which writes the runs)
and osse_ana.py (which analyses them), kept short because Windows limits a full path to 259
characters (MAX_PATH). The netCDF/HDF5 library cannot write or read a longer path even when Windows
long paths are enabled, and reports it as "PermissionError: [Errno 13] Permission denied".

Older runs used longer names (the legacy names in run_files and alternative_names). Reading
functions look for the new name first and fall back to the old one, so existing output folders still
work; new output is always written under the new names.
"""

from __future__ import annotations

import os
import re


class PathLimits:
    """Windows path-length limits"""

    # Longest full path Windows accepts without long-path support (260 including the terminating NUL)
    MAX_PATH: int = 259

    # Room to leave in a run folder for the files written into it, by the routine and by osse_ana.py
    #  (the longest is e.g. "stage_000_obs_000_sta_assim.png" inside prof_obs/, ~40 characters)
    RUN_DIR_FILE_ALLOWANCE: int = 45


# ---------------------------------------------------------------- folder names
class _Codes:
    """Abbreviations used in folder names"""

    BIAS_MODE: dict[str, str] = {"estimate": "est", "fixed": "fix"}
    BIAS_PREDICTOR: dict[str, str] = {
        "quadratic": "quad",
        "linear": "lin",
        "constant": "const",
    }
    REPARAM: dict[str, str] = {"none": "n", "log_v": "lv", "tref_logv": "tr"}
    PAR: dict[str, str] = {
        "t_init": "t",
        "v": "v",
        "width": "w",
        "lon": "lo",
        "lat": "la",
        "thick": "h",
        "bias": "b",
    }


def bias_dir_name(mode: str, predictor: str, prior_mean: float, prior_sd: float) -> str:
    """
    Folder for runs with an observation-bias term, e.g. "ob_est_const_0.00_2.00"
     (older runs used "obsbias_estimate_constant_prior0.00_2.00"; see alternative_names)
    """
    return (
        f"ob_{_Codes.BIAS_MODE.get(mode, mode)}_{_Codes.BIAS_PREDICTOR.get(predictor, predictor)}"
        f"_{float(prior_mean):.2f}_{float(prior_sd):.2f}"
    )


def true_bias_suffix(true_bias: float) -> str:
    """Suffix of a twin's observation folder for an injected bias, e.g. "_tb0.50" (older runs: "_truebias0.50")"""
    return f"_tb{float(true_bias):.2f}"


def twin_sampling_dir_name(
    sample_first_guess: bool, new_obs_noise_per_run: bool
) -> str:
    """
    Tag of the prior folder of a twin experiment whose runs draw their first guess from the prior
    around the truth ("fgs") and/or draw new observation noise ("ons"); "" for neither
    """
    parts: list[str] = (["fgs"] if sample_first_guess else []) + (
        ["ons"] if new_obs_noise_per_run else []
    )
    return "_".join(parts)


def corr_dir_name(settings: dict) -> str:
    """
    Folder for runs with correlation handling (state_transform.py), e.g. "cr_tr50w_a0.2_f0.02_kw-la"
     = reparam tref_logv at 50 Rs, whitened, shrinkage alpha 0.2, jitter floor 0.02, width and lat
     carried from the prior
    """
    code: str | None = _Codes.REPARAM.get(settings["reparam"], settings["reparam"])
    if settings["reparam"] == "tref_logv":
        code += (
            "b"
            if settings.get("t_ref_rs") is None
            else format(float(settings["t_ref_rs"]), "g")
        )
    if settings.get("whiten"):
        code += "w"
    name: str = (
        f"cr_{code}_a{float(settings['shrinkage_alpha']):g}_f{float(settings['min_jitter_sd']):g}"
    )
    if settings.get("carry_from_prior"):
        name += "_k" + "-".join(
            _Codes.PAR.get(p, p) for p in settings["carry_from_prior"]
        )
    return name


def alternative_names(component: str) -> list[str]:
    """
    The other spelling(s) of a folder name that changed (new <-> legacy), e.g.
    "ob_est_const_0.00_2.00" <-> "obsbias_estimate_constant_prior0.00_2.00",
    "obs_x_tb0.50" <-> "obs_x_truebias0.50". Empty for any other name.
    """
    mode_long: dict[str, str] = {v: k for k, v in _Codes.BIAS_MODE.items()}
    pred_long: dict[str, str] = {v: k for k, v in _Codes.BIAS_PREDICTOR.items()}
    out: list[str] = []
    m: re.Match[str] | None = re.match(
        r"^ob_([a-z]+)_([a-z]+)_(-?[0-9.]+)_(-?[0-9.]+)$", component
    )
    if m:
        out.append(
            f"obsbias_{mode_long.get(m[1], m[1])}_{pred_long.get(m[2], m[2])}_prior{m[3]}_{m[4]}"
        )
    m = re.match(r"^obsbias_([a-z]+)_([a-z]+)_prior(-?[0-9.]+)_(-?[0-9.]+)$", component)
    if m:
        out.append(
            f"ob_{_Codes.BIAS_MODE.get(m[1], m[1])}_{_Codes.BIAS_PREDICTOR.get(m[2], m[2])}_{m[3]}_{m[4]}"
        )
    m = re.match(r"^(.*)_tb(-?[0-9.]+)$", component)
    if m:
        out.append(f"{m[1]}_truebias{m[2]}")
    m = re.match(r"^(.*)_truebias(-?[0-9.]+)$", component)
    if m:
        out.append(f"{m[1]}_tb{m[2]}")
    return out


def resolve_output_path(path: str, verbose: bool = True) -> str:
    """
    Return `path` if it exists. Otherwise walk down it and, wherever a folder is missing but its other
    spelling (alternative_names: the short names of output_names.py or the long names of older runs)
    exists, use that. Lets a path be written with either the old or the new folder names.
    If nothing matches, the path is returned unchanged and (verbose) the deepest existing folder and
    its contents are printed, to show where the path goes wrong.
    """
    if os.path.isdir(path):
        return path
    drive: str
    rest: str
    drive, rest = os.path.splitdrive(os.path.normpath(path))
    parts: list[str] = [p for p in rest.replace("\\", "/").split("/") if p]
    cur: str = drive + (os.sep if rest.startswith(("/", "\\")) else "")
    for i, part in enumerate(parts):
        nxt: str = os.path.join(cur, part)
        if not os.path.isdir(nxt):
            for alt in alternative_names(part):
                if os.path.isdir(os.path.join(cur, alt)):
                    nxt = os.path.join(cur, alt)
                    if verbose:
                        print(f"resolve_output_path: using '{alt}' for '{part}'")
                    break
            else:
                if verbose:
                    try:
                        contents: list[str] = sorted(os.listdir(cur)) if cur else []
                    except OSError:
                        contents = []
                    print(
                        f"resolve_output_path: '{part}' not found in {cur}"
                        + (
                            f" (alternatives tried: {alternative_names(part)})"
                            if alternative_names(part)
                            else ""
                        )
                        + (f"; it contains: {contents}" if contents else "")
                    )
                return path
        cur = nxt
    return cur


# ---------------------------------------------------------------- file names in a run folder
def run_files() -> dict[str, tuple[str, str]]:
    """Names of the output files in a run folder, key: (name, legacy name)"""
    return {
        "profiles_per_obs": ("elon_prof_obs.nc", "ens_elongation_profiles_per_obs.nc"),
        "profiles_per_obs_dir": ("prof_obs", "elongation_profiles_per_obs"),
        "arrivals_per_obs": ("arr_obs.nc", "cme_arrival_per_obs.nc"),
        "arrival_hist_per_obs": (
            "arr_hist_obs",
            "arrival_time_histograms_per_obs",
        ),  # figure stem
    }


def write_path(run_dir: str, key: str) -> str:
    """Path to write output `key` to (always the new name)"""
    return os.path.join(run_dir, run_files()[key][0])


def read_path(run_dir: str, key: str) -> str:
    """Path to read output `key` from: the new name if it exists, else the legacy name if that exists, else the new name"""
    new: str
    old: str
    new, old = (os.path.join(run_dir, n) for n in run_files()[key])
    if not os.path.exists(new) and os.path.exists(old):
        return old
    return new


# ---------------------------------------------------------------- path length
def path_too_long(path: str, allowance: int = 0) -> bool:
    """True if os.path.abspath(path) plus `allowance` characters exceeds PathLimits.MAX_PATH"""
    return len(os.path.abspath(path)) + allowance > PathLimits.MAX_PATH


def check_path_length(
    path: str, what: str = "output file", allowance: int = 0, error: bool = True
) -> bool:
    """
    Check a path against the Windows limit before doing expensive work that ends by writing to it.
    Only enforced on Windows (os.name == "nt"); elsewhere it always passes.
    :param path: Path to check
    :param what: Description for the message
    :param allowance: Extra characters that will be added (e.g. file names inside a folder)
    :param error: Raise OSError if too long (else print a warning)
    :return: True if the path is short enough
    """
    if os.name != "nt" or not path_too_long(path, allowance):
        return True
    n: int = len(os.path.abspath(path))
    msg: str = (
        f"The {what} path is {n}{f' (+{allowance})' if allowance else ''} characters, over the Windows "
        f"limit of {PathLimits.MAX_PATH}: {path}. netCDF files cannot be written there (it fails as "
        f"'PermissionError: [Errno 13]'). Shorten OUT_BASE_DIR in .env, or rename/move the output folder."
    )
    if error:
        raise OSError(msg)
    print(f"Warning: {msg}")
    return False
