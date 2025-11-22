"""
Area of Interest (AOI) calculations for aircraft monitoring.
"""

from haversine import inverse_haversine, Direction, haversine


def fetch_aoi(coordinates: tuple, distance: int) -> tuple:
    """
    Function to return the coordinates for the boundaries of an area of interest.
    
    Args:
        coordinates: Tuple of (latitude, longitude) for center point
        distance: Distance in kilometers from center to boundary
    
    Returns:
        Tuple of ((max_north, max_south), (max_east, max_west))
    """
    max_n = inverse_haversine(point=coordinates, distance=distance, direction=Direction.NORTH)[0]
    max_e = inverse_haversine(point=coordinates, distance=distance, direction=Direction.EAST)[1]
    max_w = inverse_haversine(point=coordinates, distance=distance, direction=Direction.WEST)[1]
    max_s = inverse_haversine(point=coordinates, distance=distance, direction=Direction.SOUTH)[0]

    return (max_n, max_s), (max_e, max_w)


def in_aoi(lat_bounds: tuple, lon_bounds: tuple, target_lat_lon: tuple) -> bool:
    """
    Check if the target coordinates are within the AOI.
    
    Args:
        lat_bounds: Tuple of (max_north, max_south)
        lon_bounds: Tuple of (max_east, max_west)
        target_lat_lon: Tuple of (target_lat, target_lon)
    
    Returns:
        True if target is within bounds, False otherwise
    """
    max_n = lat_bounds[0]
    max_s = lat_bounds[1]
    max_w = lon_bounds[1]
    max_e = lon_bounds[0]

    t_lat = target_lat_lon[0]
    t_lon = target_lat_lon[1]

    if max_n > t_lat > max_s:
        if max_e > t_lon > max_w:
            return True
        else:
            return False
    else:
        return False


def fetch_dist(point_a: tuple, point_b: tuple) -> float:
    """
    Calculate distance between two geographic points using haversine formula.
    
    Args:
        point_a: Tuple of (lat, lon) for first point
        point_b: Tuple of (lat, lon) for second point
    
    Returns:
        Distance in kilometers
    """
    return haversine(point1=point_a, point2=point_b)

