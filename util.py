import numpy as np
import geopandas as gp
import pandas as pd
import shapely
from shapely.geometry import box, Point, LineString
from typing import List, Tuple, Union, Optional

def make_traj(trajs):
    if len(trajs) == 1:
        return 0
    t = [[traj[0][0], traj[1][0], traj[2][0]] for traj in trajs]
    return traj_to_linestring(t)

def path_to_traj(path, ls):
    return [[ls[j][0], ls[j][1]] for j in path]

def transform_meter_global(geom):
    return gp.GeoDataFrame(geometry=geom, crs="EPSG:20437").to_crs(epsg=4326).geometry

def transform_global_meter(geom):
    return gp.GeoDataFrame(geometry=geom, crs="EPSG:4326").to_crs(epsg=20437).geometry

def waypoints_to_traj(waypoints):
    return [[w["x"], w["y"]] for w in waypoints if not isinstance(w, float)]

def list2state(values):
    return dict(zip(('x', 'y', 'z', 'xdot', 'ydot', 'zdot'), values))

def state2list(values):
    return [values["x"], values["y"], values["z"]]

def point_to_waypoint(p, z):
    return {'x': p.x, 'y': p.y, 'z': z, 'xdot': 0, 'ydot': 0, 'zdot': 0}

def dist_squared(xi, xi_1):
    dx = xi['x'] - xi_1['x']
    dy = xi['y'] - xi_1['y']
    dz = xi['z'] - xi_1['z']
    return dx*dx + dy*dy + dz*dz

def traj_to_linestring(traj: Union[List[List[float]], np.ndarray],
                      force_2d: bool = False) -> Optional[LineString]:
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
