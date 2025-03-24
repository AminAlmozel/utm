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
    temp = io.read_log_dict()
    output = []
    for drone in temp:
        trajs = drone['trajs']
        destinations = drone["mission"]["destination"]

        t = measure_total_time(trajs)
        l = measure_traj_length(trajs)
        ol = measure_straight_distance(destinations)

        print()
        print("Drone: ", drone['id'])
        print("Time elapsed:", t)
        print("Distance traveled: ", l)
        print("Straight line distance: ", ol)

        output.append([drone["id"], l, t])
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
    return ls.length

def measure_straight_distance(destinations):
    t = []
    for dest in destinations:
        point = [dest["x"], dest["y"], dest["z"]]
        t.append(point)
    if len(t) == 1:
        return 0
    ls = io.traj_to_linestring(t)
    return ls.length

def measure_total_time(trajs):
    dt = 0.1
    return len(trajs) * dt
main()