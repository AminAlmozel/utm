import numpy as np
import geopandas as gp
import pandas as pd
import shapely
from shapely.geometry import box, Point, LineString

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

def traj_to_linestring(traj):
    if len(traj) <= 1:
        return
    points = []
    for i in range(len(traj)):
        point = Point(traj[i][0], traj[i][1], traj[i][2]) # 3D trajectory
        # point = Point(traj[i][1], traj[i][0]) #2D trajectory
        points.append(point)
    s_line = LineString(points)
    return s_line

def transform_meter_global(geom):
    gdf = gp.GeoDataFrame(geometry=geom, crs="EPSG:20437")
    gdf.to_crs(epsg=4326, inplace=True)
    return gdf.geometry

def transform_global_meter(geom):
    gdf = gp.GeoDataFrame(geometry=geom, crs="EPSG:4326")
    gdf.to_crs(epsg=20437, inplace=True)
    return gdf.geometry

def traj_to_linestring(traj):
    points = []
    for i in range(len(traj)):
        point = Point(traj[i][0], traj[i][1])
        points.append(point)
    s_line = LineString(points)
    return s_line

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
