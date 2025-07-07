import numpy as np
import geopandas as gp
import pandas as pd
import shapely
from shapely.geometry import box, Point, LineString
from typing import List, Tuple, Union, Optional

def make_traj(trajs):
    t = []
    for traj in trajs:
        point = [traj[0][0], traj[1][0], traj[2][0]]
        t.append(point)
    if len(t) == 1:
        return 0
    ls = traj_to_linestring(t)
    # ls = transform_meter_global([ls])
    return ls

def traj_to_linestring(traj: Union[List[List[float]], np.ndarray],
                      force_2d: bool = False) -> Optional[LineString]:
    """
    Convert a trajectory to a Shapely LineString.

    Args:
        traj: Trajectory as list of points or numpy array
              - For 2D: [[x1, y1], [x2, y2], ...]
              - For 3D: [[x1, y1, z1], [x2, y2, z2], ...]
        force_2d: If True, forces 2D output even for 3D input (ignores Z coordinate)

    Returns:
        LineString object or None if trajectory has <= 1 point
    """
    if len(traj) <= 1:
        return None

    # Convert to numpy array for easier handling
    traj_array = np.asarray(traj)

    # Determine if trajectory is 2D or 3D
    if traj_array.shape[1] == 2:
        # 2D trajectory
        points = [Point(point[0], point[1]) for point in traj_array]
    elif traj_array.shape[1] == 3:
        # 3D trajectory
        if force_2d:
            # Use only X and Y coordinates
            points = [Point(point[0], point[1]) for point in traj_array]
        else:
            # Use all three coordinates
            points = [Point(point[0], point[1], point[2]) for point in traj_array]
    else:
        raise ValueError(f"Trajectory must have 2 or 3 coordinates per point, got {traj_array.shape[1]}")

    return LineString(points)

def path_to_traj(path, ls):
    traj = []
    for i in range(len(path)):
        j = path[i]
        traj.append([ls[j][0], ls[j][1], 0])
    return traj

def transform_meter_global(geom):
    gdf = gp.GeoDataFrame(geometry=geom, crs="EPSG:20437")
    gdf.to_crs(epsg=4326, inplace=True)
    return gdf.geometry

def transform_global_meter(geom):
    gdf = gp.GeoDataFrame(geometry=geom, crs="EPSG:4326")
    gdf.to_crs(epsg=20437, inplace=True)
    return gdf.geometry



def waypoints_to_traj(waypoints):
    traj = []
    for waypoint in waypoints:
        if isinstance(waypoint, float):
            continue
        traj.append([waypoint["x"], waypoint["y"]])
    return traj

def list2state(values):
    keys = ['x', 'y', 'z', 'xdot', 'ydot', 'zdot']
    return dict(zip(keys, values))

def state2list(values):
    return [values["x"], values["y"], values["z"]]

def point_to_waypoint(self, p, z):
    waypoint = [p.x, p.y, z, 0, 0, 0]
    return self.list2state(waypoint)

def dist_squared(self, xi, xi_1):
    return (xi['x'] - xi_1['x'])**2 + (xi['y'] - xi_1['y'])**2 + (xi['z'] - xi_1['z'])**2


# def transform_coords(self):
#     # Converting to meters projection
#     self.houses.to_crs(epsg=20437, inplace=True)
#     self.apts.to_crs(epsg=20437, inplace=True)
#     self.restaurants.to_crs(epsg=20437, inplace=True)
#     self.fire_station.to_crs(epsg=20437, inplace=True)
