import numpy as np

def haversine(lon1, lat1, lon2, lat2):
    # Calculate the great circle distance between two points on the earth (specified in decimal degrees)
    # Returns distance in meters
    from math import radians, sin, cos, sqrt, atan2
    R = 6371000  # Radius of earth in meters
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c

def calculate_heading(lat1, lon1, lat2, lon2):
    # Returns heading in degrees
    from math import atan2, radians, degrees, sin, cos
    dLon = radians(lon2 - lon1)
    lat1 = radians(lat1)
    lat2 = radians(lat2)
    x = sin(dLon) * cos(lat2)
    y = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dLon)
    heading = atan2(x, y)
    heading = degrees(heading)
    return (heading + 360) % 360

def detect_corners(df, window=5, num_corners=14):
    lats = df['latitude'].astype(float).values
    lons = df['longitude'].astype(float).values
    headings = []
    for i in range(1, len(lats)):
        headings.append(calculate_heading(lats[i-1], lons[i-1], lats[i], lons[i]))
    headings = np.array([0] + headings)
    delta_heading = np.abs(np.diff(headings, n=window, prepend=[headings[0]]*window))
    delta_heading = np.where(delta_heading > 180, 360 - delta_heading, delta_heading)
    # Get indices of the top num_corners largest heading changes
    corner_indices = np.argpartition(-delta_heading, num_corners)[:num_corners]
    # Sort by track order
    corner_indices = np.sort(corner_indices)
    return corner_indices.tolist() 