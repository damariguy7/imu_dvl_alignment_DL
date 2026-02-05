def convert_lat_lon_to_meters(data):
    """
    Convert latitude/longitude from radians to meters (local ENU coordinates)
    Assumes data columns are: [time, lat_rad, lon_rad, alt_m]
    """
    # Extract lat, lon, alt
    lat_rad = data.iloc[:, 1].values
    lon_rad = data.iloc[:, 2].values
    alt_m = data.iloc[:, 3].values

    # Use first point as reference (origin)
    lat0 = lat_rad[0]
    lon0 = lon_rad[0]

    # Earth radius in meters
    R = 6378137.0  # WGS84 equatorial radius

    # Convert to local ENU coordinates (East, North, Up)
    # East (X) - longitude difference
    east = (lon_rad - lon0) * R * np.cos(lat0)

    # North (Y) - latitude difference
    north = (lat_rad - lat0) * R

    # Up (Z) - altitude (already in meters, but negative for depth)
    up = -alt_m  # Negative because we want depth below surface

    return east, north, up