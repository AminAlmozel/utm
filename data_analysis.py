# Importing standard libraries
from math import radians, cos, sin, asin, sqrt
import time
from datetime import datetime
import random
from multiprocessing import Pool

import numpy as np
import geopandas as gp
import pandas as pd
import shapely
from shapely.geometry import box, Point, LineString
import glob

from sim_io import myio as io

def main():
    last = 'simulations/last/'
    temp = io.read_log_pickle(last)
    output = []
    for drone in temp:
        trajs = drone['trajs']
        destinations = drone["mission"]["destination"]

        t = measure_total_time(trajs)
        l = measure_traj_length(trajs).length
        ol = measure_straight_distance(destinations).length
        p = (l - ol) / ol * 100
        # if drone["id"] == 23: # for debugging
        #     l_ls = measure_traj_length(trajs)
        #     ol_ls = measure_straight_distance(destinations)
        #     geom = [l_ls, ol_ls]
        #     geom = io.transform_meter_global(geom)
        #     io.write_geom(geom, "data_analysis", "red")

        print()
        print("Drone: ", drone['id'])
        print("Time elapsed: \t\t%.2fs" %(t))
        print("Distance traveled: \t%.2fm" %(l))
        print("Straight line distance: %.2fm" %(ol))
        print("Percentage increase: \t%.2f" %(p))

        output.append([drone["id"], l, ol, t])
    df = pd.DataFrame(output)
    df.to_csv("plot/simulation/read/stats.csv", index=False, header=False)


def measure_traj_length(trajs):
    t = []
    for traj in trajs:
        point = [traj[0][0], traj[1][0], traj[2][0]]
        t.append(point)
    if len(t) == 1:
        return 0
    ls = io.traj_to_linestring(t)
    return ls

def measure_straight_distance(destinations):
    t = []
    for dest in destinations:
        point = [dest["x"], dest["y"], dest["z"]]
        t.append(point)
    if len(t) == 1:
        return 0
    ls = io.traj_to_linestring(t)
    return ls

def measure_total_time(trajs):
    dt = 0.1
    return len(trajs) * dt

def measure_safe_distance(trajs, safe, nfz):
    pass

def measure_emergency_landing_response(trajs):
    pass

def measure_mid_air_collisions(trajs):
    pass

def write_geom(geom, name, color):
    s = gp.GeoDataFrame(crs=4326, geometry=geom)
    s["stroke"] = color
    s["marker-color"] = color
    s["fill"] = color
    s.to_file('plot/' + name + '.geojson', driver='GeoJSON')

main()