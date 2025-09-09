import geopandas as gpd
from shapely.affinity import affine_transform

def main():
    # Reading
    filepath = "plot/selected_missions.geojson"
    linestring_trajectories = read_geojson(filepath)

    # Convert to waypoints
    drones = []
    for geom in linestring_trajectories:
        trajectory = linestring_to_waypoints(geom)
        drones.append(trajectory)

    # Print waypoints
    for drone in drones:
        print(len(drone))
        for point in drone:
            print(f"{point[0]}, {point[1]}, {point[2]}")
        print("\n")

def read_geojson(filepath):
    gdf = gpd.read_file(filepath)
    gdf = gdf.set_crs(epsg=4326, allow_override=True)
    transformed = gdf.to_crs(epsg=32637).geometry
    return transformed

def linestring_to_waypoints(linestring):
    waypoints = []
    for coord in linestring.coords:
        if len(coord) == 3:
            x, y, z = coord
        else:
            x, y = coord
            z = 0  # or any default value
        waypoint = (x, y, z)
        waypoints.append(waypoint)
    return waypoints

def project_to_bbox(geoms, src_bbox, tgt_bbox):
    """
    Projects geometries from src_bbox to tgt_bbox using affine transformation.
    src_bbox and tgt_bbox are (minx, miny, maxx, maxy) tuples.
    """
    src_minx, src_miny, src_maxx, src_maxy = src_bbox
    tgt_minx, tgt_miny, tgt_maxx, tgt_maxy = tgt_bbox

    # Calculate scale and translation
    scale_x = (tgt_maxx - tgt_minx) / (src_maxx - src_minx)
    scale_y = (tgt_maxy - tgt_miny) / (src_maxy - src_miny)
    trans_x = tgt_minx - src_minx * scale_x
    trans_y = tgt_miny - src_miny * scale_y

    # Affine matrix for shapely.affinity.affine_transform: [a, b, d, e, xoff, yoff]
    # [a, b, d, e, xoff, yoff] maps (x, y) to (a*x + b*y + xoff, d*x + e*y + yoff)
    matrix = [scale_x, 0, 0, scale_y, trans_x, trans_y]

    projected = [affine_transform(geom, matrix) for geom in geoms]
    return projected