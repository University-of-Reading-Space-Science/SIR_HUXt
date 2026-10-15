"""
event_catalogue.py

Per-event configuration for the real-observation experiments (previously long if/elif chains in
da_surf_setup.py). Events are identified by ``ssw_event``:
    "ssw_007", "ssw_008", "ssw_009", "ssw_012" : Solar Stormwatch events (STEREO-A/B HI, 2012)
    "01B" ... "15B"                            : events observed from STEREO-A ("..A") or STEREO-B ("..B")

Tables:
    SURF_INIT_TIMES        : model start time for each event
    FIRST_GUESS_DONKI      : first-guess cone-CME parameters from the DONKI catalogue
    FIRST_GUESS_LUKE_PAPER : alternative first-guess parameters (as in luke_paper_fg_cme_par.py;
                             identical to DONKI except for the four Solar Stormwatch events)
    OBS_FILE_NAMES         : observation file for each event (in $OBS_DIR)

First-guess times are the time the CME reached 21.5 solar radii; speeds in km/s, angles in degrees
(width is the full angular width), thickness in solar radii.
"""
import datetime
from typing import NamedTuple


class FirstGuessCME(NamedTuple):
    """First-guess cone-CME parameters for one event."""
    t_21rs: datetime.datetime  # Time the CME reached 21.5 solar radii
    speed: float               # km/s
    width: float               # deg (full angular width)
    lon: float                 # deg
    lat: float                 # deg
    thick: float               # solar radii


# Solar Stormwatch events, all read from the same HDF5 classification file
SSW_EVENTS: tuple[str, ...] = ("ssw_007", "ssw_008", "ssw_009", "ssw_012")

# Model (SURF/HUXt) start time for each event
SURF_INIT_TIMES: dict[str, datetime.datetime] = {
    "ssw_007": datetime.datetime(2012, 8, 31, 0, 0, 0),
    "ssw_008": datetime.datetime(2012, 9, 27, 15, 0, 0),
    "ssw_009": datetime.datetime(2012, 10, 4, 20, 0, 0),
    "ssw_012": datetime.datetime(2012, 11, 20, 0, 0, 0),
    "01B": datetime.datetime(2010, 4, 8, 4, 0, 0),
    "02A": datetime.datetime(2020, 10, 26, 12, 0, 0),
    "03A": datetime.datetime(2021, 11, 1, 18, 0, 0),
    "04B": datetime.datetime(2010, 5, 24, 10, 0, 0),
    "05B": datetime.datetime(2009, 12, 14, 1, 0, 0),
    "06B": datetime.datetime(2011, 6, 14, 4, 0, 0),
    "07B": datetime.datetime(2010, 3, 19, 11, 0, 0),
    "08B": datetime.datetime(2010, 4, 3, 3, 0, 0),
    "09A": datetime.datetime(2021, 10, 28, 8, 0, 0),
    "10A": datetime.datetime(2020, 12, 7, 8, 0, 0),
    "11A": datetime.datetime(2020, 9, 30, 7, 0, 0),
    "12A": datetime.datetime(2021, 2, 10, 16, 0, 0),
    "13B": datetime.datetime(2011, 5, 24, 23, 0, 0),
    "14B": datetime.datetime(2011, 7, 11, 6, 0, 0),
    "15B": datetime.datetime(2011, 10, 28, 17, 0, 0),
}

# First-guess CME parameters from the DONKI catalogue
FIRST_GUESS_DONKI: dict[str, FirstGuessCME] = {
    # event: FirstGuessCME(time at 21.5 rS, speed, width, lon, lat, thick)
    "ssw_007": FirstGuessCME(datetime.datetime(2012, 8, 31, 22, 43, 0), 1498, 150, -63, -15, 0),
    "ssw_008": FirstGuessCME(datetime.datetime(2012, 9, 28, 1, 56, 0), 1160, 170, 30, 5, 0),
    "ssw_009": FirstGuessCME(datetime.datetime(2012, 10, 5, 9, 15, 0), 650, 94, 10, -28, 0),
    "ssw_012": FirstGuessCME(datetime.datetime(2012, 11, 20, 17, 32, 0), 725, 60, 90, 5, 0),
    "01B": FirstGuessCME(datetime.datetime(2010, 4, 8, 20, 34, 0), 258, 50, -74, 18, 0),
    "02A": FirstGuessCME(datetime.datetime(2020, 10, 26, 5, 57, 0), 159, 36, -15, -10, 0),
    "03A": FirstGuessCME(datetime.datetime(2021, 11, 2, 5, 46, 0), 1188, 102, 5, 18, 0),
    "04B": FirstGuessCME(datetime.datetime(2010, 5, 24, 20, 31, 0), 542, 76, 24, 1, 0),
    "05B": FirstGuessCME(datetime.datetime(2009, 12, 16, 9, 40, 0), 542, 48, -7, 10, 0),
    "06B": FirstGuessCME(datetime.datetime(2011, 6, 14, 14, 13, 0), 672, 70, -34, 1, 0),
    "07B": FirstGuessCME(datetime.datetime(2010, 3, 20, 3, 5, 0), 239, 66, 36, -4, 0),
    "08B": FirstGuessCME(datetime.datetime(2010, 4, 3, 14, 36, 0), 740, 68, 16, -21, 0),
    "09A": FirstGuessCME(datetime.datetime(2021, 10, 28, 18, 19, 0), 1368, 100, 2, -23, 0),
    "10A": FirstGuessCME(datetime.datetime(2020, 12, 7, 18, 44, 0), 1436, 82, 8, -18, 0),
    "11A": FirstGuessCME(datetime.datetime(2020, 10, 1, 8, 49, 0), 126, 32, 14, 12, 0),
    "12A": FirstGuessCME(datetime.datetime(2021, 2, 10, 23, 12, 0), 262, 26, -33, -3, 0),
    "13B": FirstGuessCME(datetime.datetime(2011, 5, 25, 17, 40, 0), 598, 52, 6, -10, 0),
    "14B": FirstGuessCME(datetime.datetime(2011, 7, 11, 17, 40, 0), 552, 42, 14, -11, 0),
    "15B": FirstGuessCME(datetime.datetime(2011, 10, 29, 8, 36, 0), 589, 78, -7, 24, 0),
}

# Alternative first-guess CME parameters (luke_paper_fg_cme_par.py)
FIRST_GUESS_LUKE_PAPER: dict[str, FirstGuessCME] = {
    # event: FirstGuessCME(time at 21.5 rS, speed, width, lon, lat, thick)
    "ssw_007": FirstGuessCME(datetime.datetime(2012, 8, 31, 22, 46, 0), 1010, 66, -30, 0, 0),
    "ssw_008": FirstGuessCME(datetime.datetime(2012, 9, 28, 3, 49, 0), 872, 110, 20, 4, 0),
    "ssw_009": FirstGuessCME(datetime.datetime(2012, 10, 5, 8, 47, 0), 698, 84, 9, -24, 0),
    "ssw_012": FirstGuessCME(datetime.datetime(2012, 11, 20, 17, 40, 0), 664, 94, 22, 20, 0),
    "01B": FirstGuessCME(datetime.datetime(2010, 4, 8, 20, 34, 0), 258, 50, -74, 18, 0),
    "02A": FirstGuessCME(datetime.datetime(2020, 10, 26, 5, 57, 0), 159, 36, -15, -10, 0),
    "03A": FirstGuessCME(datetime.datetime(2021, 11, 2, 5, 46, 0), 1188, 102, 5, 18, 0),
    "04B": FirstGuessCME(datetime.datetime(2010, 5, 24, 20, 31, 0), 542, 76, 24, 1, 0),
    "05B": FirstGuessCME(datetime.datetime(2009, 12, 16, 9, 40, 0), 542, 48, -7, 10, 0),
    "06B": FirstGuessCME(datetime.datetime(2011, 6, 14, 14, 13, 0), 672, 70, -34, 1, 0),
    "07B": FirstGuessCME(datetime.datetime(2010, 3, 20, 3, 5, 0), 239, 66, 36, -4, 0),
    "08B": FirstGuessCME(datetime.datetime(2010, 4, 3, 14, 36, 0), 740, 68, 16, -21, 0),
    "09A": FirstGuessCME(datetime.datetime(2021, 10, 28, 18, 19, 0), 1368, 100, 2, -23, 0),
    "10A": FirstGuessCME(datetime.datetime(2020, 12, 7, 18, 44, 0), 1436, 82, 8, -18, 0),
    "11A": FirstGuessCME(datetime.datetime(2020, 10, 1, 8, 49, 0), 126, 32, 14, 12, 0),
    "12A": FirstGuessCME(datetime.datetime(2021, 2, 10, 23, 12, 0), 262, 26, -33, -3, 0),
    "13B": FirstGuessCME(datetime.datetime(2011, 5, 25, 17, 40, 0), 598, 52, 6, -10, 0),
    "14B": FirstGuessCME(datetime.datetime(2011, 7, 11, 17, 40, 0), 552, 42, 14, -11, 0),
    "15B": FirstGuessCME(datetime.datetime(2011, 10, 29, 8, 36, 0), 589, 78, -7, 24, 0),
}

# Observation file for each event (relative to $OBS_DIR)
OBS_FILE_NAMES: dict[str, str] = {
    "ssw_007": "SSW_cme_classifications.hdf5",
    "ssw_008": "SSW_cme_classifications.hdf5",
    "ssw_009": "SSW_cme_classifications.hdf5",
    "ssw_012": "SSW_cme_classifications.hdf5",
    "01B": "CME_01B__20100407__PA265_good_hi2.dat",
    "02A": "CME_02A__20201025__PA272_good_hi2.dat",
    "03A": "CME_03A__20211101__PA268_good_hi2.dat",
    "04B": "CME_04B__20100523__PA271_good_hi2.dat",
    "05B": "CME_05B__20091215__PA266_good_hi2.dat",
    "06B": "CME_06B__20110614__PA270_good_hi2.dat",
    "07B": "CME_07B__20100318__PA264_good_hi2.dat",
    "08B": "CME_08B__20100402__PA265_good_hi2.dat",
    "09A": "CME_09A__20211027__PA268_good_hi2.dat",
    "10A": "CME_10A__20201206__PA266_good_hi2.dat",
    "11A": "CME_11A__20200930__PA277_good_hi2.dat",
    "12A": "CME_12A__20210210__PA263_good_hi2.dat",
    "13B": "CME_13B__20110524__PA271_good_hi2.dat",
    "14B": "CME_14B__20110710__PA274_good_hi2.dat",
    "15B": "CME_15B__20111027__PA276_good_hi2.dat",
}
