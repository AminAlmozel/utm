# Importing standard libraries
from math import radians, cos, sin, asin, sqrt
import time
from datetime import datetime, timedelta
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
    last = "simulations/run_25-04-08-130710/"
    temp = io.read_log_pickle(last)
    safe = import_geojson("plot/" + last + "safe.geojson")
    nfz = import_geojson("plot/" + last + "new_nfz.geojson")
    safe = safe.difference(nfz)

    print(temp[0]['born'])
    ground_stop = temp[0]['born']
    output = []
    for drone in temp:
        trajs = drone['trajs']
        destinations = drone["mission"]["destination"]
        traj = make_traj(trajs)
        t = measure_total_time(trajs)
        l = measure_traj_length(traj)
        ol = measure_straight_distance(destinations).length
        p = (l - ol) / ol * 100
        perf = measure_safe_distance(traj, safe, nfz)
        response = measure_emergency_landing_response(trajs, drone['born'], ground_stop)
        # if drone["id"] == 23: # for debugging
        #     l_ls = measure_traj_length(trajs)
        #     ol_ls = measure_straight_distance(destinations)
        #     geom = [l_ls, ol_ls]
        #     geom = io.transform_meter_global(geom)
        #     io.write_geom(geom, "data_analysis", "red")

        print()
        print("Drone: ", drone['id'])
        print("Mission: \t\t%s" %(drone['mission']['type']))
        print("Time elapsed: \t\t%.2fs" %(t))
        print("Distance traveled: \t%.2fm" %(l))
        print("Straight line distance: %.2fm" %(ol))
        print("Percentage increase: \t%.2f%%" %(p))
        print("Safety performance:")
        print("Safe: \t\t\t%.2fm (%.2f%%)" %(perf[0], perf[0] * 100 / l))
        print("Unafe: \t\t\t%.2fm (%.2f%%)" %(perf[1], perf[1] * 100 / l))
        print("Outside: \t\t%.2fm (%.2f%%)" %(perf[2], perf[2] * 100 / l))
        print("Response time: \t\t%.2fs" %(response))

        output.append([drone["id"], l, ol, t])
    df = pd.DataFrame(output)
    df.to_csv("plot/simulation/read/stats.csv", index=False, header=False)

def import_geojson(name):
    file = open(name)
    safe = gp.read_file(file).geometry[0]
    return io.transform_global_meter([safe])[0]

def make_traj(trajs):
    t = []
    for traj in trajs:
        point = [traj[0][0], traj[1][0], traj[2][0]]
        t.append(point)
    if len(t) == 1:
        return 0
    ls = io.traj_to_linestring(t)
    # ls = io.transform_meter_global([ls])
    return ls

def measure_traj_length(traj):
    return traj.length

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

def measure_safe_distance(traj, safe, nfz):
    l = traj.length
    safe_dist = traj.intersection(safe).length
    unsafe = traj.intersection(nfz).length
    s = gp.GeoSeries([safe, nfz])
    combined_area = s.union_all()
    outside = traj.difference(combined_area).length
    return [safe_dist, outside, unsafe]

def measure_geofence_breaches(trajs):
    pass

def measure_link_loss(trajs):
    pass

def measure_emergency_landing_response(trajs, born, ground_stop):
    T = timedelta(seconds = measure_total_time(trajs))
    landed = born + T
    if ground_stop > landed:
        return 0
    else:
        return (landed - ground_stop).total_seconds()

def measure_mid_air_collisions(trajs):
    pass

def measure_near_collisions(trajs):
    pass

def measure_throughput(trajs):
    pass

def write_geom(geom, name, color):
    s = gp.GeoDataFrame(crs=4326, geometry=geom)
    s["stroke"] = color
    s["marker-color"] = color
    s["fill"] = color
    s.to_file('plot/' + name + '.geojson', driver='GeoJSON')

main()