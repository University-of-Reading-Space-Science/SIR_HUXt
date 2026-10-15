def initialise_fg_cme_parameters():
    # Initialise CME parameters
    obs_par_dict = initialise_observation_parameters()
    if obs_par_dict["ssw_event"] == "ssw_007":
        cme_at_21rs: datetime.datetime = datetime.datetime(2012, 8, 31, 22, 43, 0)
        print(f"CR = {sn.carrington_rotation_number(cme_at_21rs)}")
        fg_cme_speed: float = 1498
        fg_cme_width: float = 150
        fg_cme_lon: float = -63
        fg_cme_lat: float = -15
        fg_cme_thick: float = 0

    elif obs_par_dict["ssw_event"] == "ssw_008":
        cme_at_21rs: datetime.datetime = datetime.datetime(2012, 9, 28, 1, 56, 0)

        fg_cme_speed: float = 1160
        fg_cme_width: float = 170
        fg_cme_lon: float = 30
        fg_cme_lat: float = 5
        fg_cme_thick: float = 0

    elif obs_par_dict["ssw_event"] == "ssw_009":
        cme_at_21rs: datetime.datetime = datetime.datetime(2012, 10, 5, 9, 15, 0)

        fg_cme_speed: float = 650
        fg_cme_width: float = 94
        fg_cme_lon: float = 10
        fg_cme_lat: float = -28
        fg_cme_thick: float = 0

    elif obs_par_dict["ssw_event"] == "ssw_012":
        cme_at_21rs: datetime.datetime = datetime.datetime(2012, 11, 20, 17, 32, 0)

        fg_cme_speed: float = 725
        fg_cme_width: float = 60
        fg_cme_lon: float = 90
        fg_cme_lat: float = 5
        fg_cme_thick: float = 0

    else:
        sys.exit("Unknown ssw_event name, expected ssw_event = 'ssw_007', 'ssw_008', 'ssw_009' or 'ssw_012'")

    # Time taken to get from cme_init_rad to r_min
    huxt_dict = initialise_huxt_parameters()
    huxt_cme_ir: Quantity[u.solRad] = huxt_dict["cme_init_rad"]
    huxt_r_min: Quantity[u.solRad] = huxt_dict["r_min"]

    dist_in_km: Quantity[u.km] = (huxt_r_min - huxt_cme_ir).to(u.km)
    seconds_to_r_min: float = dist_in_km.to(u.km).value / fg_cme_speed

    fg_cme_t_init: datetime.datetime = cme_at_21rs - datetime.timedelta(seconds=seconds_to_r_min)
    fg_cme_t_init_str: str = fg_cme_t_init.strftime("%Y%m%d-%H%M")

    sd_cme_t_init: float = 3600  # In seconds
    sd_cme_speed: float = 25  # In km/s
    sd_cme_width: float = 5  # In deg
    sd_cme_lon: float = 5  # In deg
    sd_cme_lat: float = 5  # In deg
    sd_cme_thick: float = 0  # In solRad

    fg_cme_par_dict = {
        "fg_cme_t_init": fg_cme_t_init,
        "fg_cme_speed": fg_cme_speed,
        "fg_cme_width": fg_cme_width,
        "fg_cme_lon": fg_cme_lon,
        "fg_cme_lat": fg_cme_lat,
        "fg_cme_thick": fg_cme_thick,
        "sd_cme_t_init": sd_cme_t_init,
        "sd_cme_speed": sd_cme_speed,
        "sd_cme_width": sd_cme_width,
        "sd_cme_lon": sd_cme_lon,
        "sd_cme_lat": sd_cme_lat,
        "sd_cme_thick": sd_cme_thick
    }

    return fg_cme_par_dict