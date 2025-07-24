# Importing standard libraries
from math import radians, cos, sin, asin, sqrt
import time
from datetime import datetime, timedelta
import random
from multiprocessing import Pool
import os

import numpy as np
import geopandas as gp
import pandas as pd
import shapely
from shapely.geometry import box, Point, LineString
from shapely.prepared import prep
import glob


from sim_io import myio as io
from util import *

class DroneData:
    __slots__ = ['id', 'birthday', 'iteration', 'traj', 'xi_1', 'alive', 'state',
                 'factor', 'battery', 'mission_type', 'mission_destination',
                 'mission_progress', 'mission_waypoints', 'mission_status']

    def __init__(self, id, birthday, iteration, state, mission_type, mission_status,
                 mission_destination, mission_waypoints):
        self.id = id
        self.birthday = birthday
        self.iteration = iteration
        self.traj = []
        self.xi_1 = []
        self.alive = 1  # Alive, 0 is dead
        self.state = state
        self.factor = 0
        self.battery = 100
        self.mission_type = mission_type
        self.mission_destination = mission_destination
        self.mission_progress = 0
        self.mission_waypoints = mission_waypoints
        self.mission_status = mission_status

def main():
    mission_folder = "full_runs/mission3/*/"
    filename = mission_folder + "/*.pkl"
    list_of_files = glob.glob(filename)
    print(list_of_files)
    longest_flight = 0
    for pickle_file in list_of_files:
        temp = io.read_pickle(pickle_file)
        print("Number of drones: ", len(temp))
        safe = io.load_geojson_files("env/landing/everything.geojson", concat=True)["geometry"].union_all()
        nfz = io.load_geojson_files("env/forbidden/kaustforbidden.geojson", concat=True)["geometry"].union_all()
        # nfz = prep(nfz)
        safe = safe.difference(nfz)
        # safe = prep(safe)

        ground_stop = 18000
        output = []

        for drone in temp:
            traj = drone.traj  # Changed from ['trajs'] to .traj
            traj = make_traj(traj)  # Ensure traj is a LineString
            if isinstance(traj, int) or traj is None:
                continue
            destinations = drone.mission_destination  # Changed from ["mission"]["destination"]
            t = measure_total_time(traj)
            l = measure_traj_length(traj)
            if l == 0:
                continue
            l_safe, l_unsafe, l_outside = measure_safe_distance(traj, safe, nfz)
            # response = measure_emergency_landing_response(traj, drone.birthday, ground_stop)  # Changed from ['born'] to .birthday
            if l > longest_flight:
                longest_flight = l
            print()
            print("Drone: ", drone.id)  # Changed from ['id'] to .id
            print("Mission: \t\t%s" %(drone.mission_type))  # Changed from ['mission']['type'] to .mission_type
            print("Time elapsed: \t\t%.2fs" %(t))
            print("Distance traveled: \t%.2fm" %(l))
            # print("Straight line distance: %.2fm" %(ol))
            # print("Percentage increase: \t%.2f%%" %(p))
            print("Safety performance:")
            print("Safe: \t\t\t%.2fm (%.2f%%)" %(l_safe, l_safe * 100 / l))
            print("Unsafe: \t\t%.2fm (%.2f%%)" %(l_unsafe, l_unsafe * 100 / l))
            print("Outside: \t\t%.2fm (%.2f%%)" %(l_outside, l_outside * 100 / l))
            # print("Response time: \t\t%.2fs" %(response))

            output.append([drone.id, l, t, l_safe, l_unsafe, l_outside])  # Changed from ["id"] to .id

        df = pd.DataFrame(output)
        current_folder = os.path.dirname(pickle_file)
        df.to_csv(current_folder + "/stats.csv", index=False, header=False)
    print("Longest flight: %.2fm" %(longest_flight))
def measure_traj_length(traj):
    return traj.length

def measure_straight_distance(destinations):
    # Deprecated
    t = []
    for dest in destinations:
        # Assuming destination is now a direct object with x, y, z attributes
        if isinstance(dest, float):
            continue
        point = [dest["x"], dest["y"], dest["z"]]  # Changed from dict access to attribute access
        t.append(point)
    if len(t) == 1:
        return 0
    ls = traj_to_linestring(t)
    return ls

def measure_total_time(traj):
    dt = 0.1
    return len(traj.coords) * dt

def measure_safe_distance(traj, safe, nfz):
    l = traj.length
    safe_dist = traj.intersection(safe).length
    unsafe = traj.intersection(nfz).length
    s = gp.GeoSeries([safe, nfz])
    combined_area = s.union_all()
    outside = traj.difference(combined_area).length
    return safe_dist, unsafe, outside

def measure_geofence_breaches(trajs):
    pass

def measure_link_loss(trajs):
    pass

def measure_emergency_landing_response(traj, born, ground_stop):
    T = timedelta(seconds = measure_total_time(traj))
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

main()