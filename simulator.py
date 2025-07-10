# Importing standard libraries
import time
# from datetime import datetime
import datetime
import random
from multiprocessing import Pool
from queue import Queue
# import threading
# Importing other libraries
import numpy as np
import geopandas as gp
from shapely.geometry import box, Point
import shapely
import gurobipy as grb
from numba import jit, njit

import drone
import environment
from sim_io import myio as io
from util import *

import os, sys
from sys import getsizeof
sys.path.append(os.path.join(os.path.dirname(__file__), "uam"))

import polygon_pathplanning
import sampling_pathplanning


# from uam.polygon_pathplanning import *

class simulator(drone.drone):
    def __init__(self):
        self.N = 50 # Prediction horizon
        self.delta_t = 0.1 # Time step
        self.total_iterations = 50000

        # Parameters
        self.n_vehicles = 10 # Starting number of vehicles
        self.K = 0 # Number of vehicles

        # Parameters for collision avoidance between vehicles
        self.d_x = 3  # Minimum horizontal distance
        self.d_y = 3  # Minimum vertical distance
        self.d_z = 3  # Minimum vertical distance

        self.collision = 3
        self.collision_warning = 5

        self.drn = [] # All drones objects
        self.drones = [] # Drone dictionaries
        self.drn_list = [] # Indices of alive drones
        self.env_list = [] # List of optimization enviroment for each drone

        self.vehicles = []
        self.full_traj = []
        self.obs = environment.env()

        today = datetime.date.today()
        # datetime(year, month, day, hour, minute, second, microsecond)
        self.sim_start = datetime.datetime(today.year, today.month, today.day, 12, 0, 0)

        now = datetime.datetime.now()
        date = now.strftime("%y-%m-%d-%H%M%S")
        self.sim_run = 'simulations/run_' + date + '/'
        self.sim_latest = 'simulations/last/'
        os.makedirs('plot/' + self.sim_run)
        self.obs.sim_run = self.sim_run
        self.obs.sim_latest = self.sim_latest

        # self.ppp = polygon_pathplanning.polygon_pp()
        self.ppp = sampling_pathplanning.sampling_pp()
        self.ppp.sim_run = self.sim_run
        self.ppp.sim_latest = self.sim_latest
        self.dummy_obstacles()
        self.missions = io.import_missions()

        self.max_vel = drone.drone().V_max # Max velocity

    def dummy_obstacles(self):
        # obstacles = [
        #     {'xmin': -1000, 'ymin': 600, 'zmin': -1000, 'xmax': -600, 'ymax': 1000, 'zmax': 1000},
        #     {'xmin': 7, 'ymin': 7, 'zmin': 2, 'xmax': 9, 'ymax': 8, 'zmax': 3},
        #     {'xmin': 6, 'ymin': 1, 'zmin': 0, 'xmax': 8, 'ymax': 2, 'zmax': 1},
        #     {'xmin': 9, 'ymin': 3, 'zmin': 0, 'xmax': 10, 'ymax': 5, 'zmax': np.random.uniform(5, 10)},
        #     {'xmin': 11, 'ymin': 6, 'zmin': 0, 'xmax': 13, 'ymax': 7, 'zmax': np.random.uniform(5, 10)},
        #     {'xmin': 14, 'ymin': 8, 'zmin': 0, 'xmax': 15, 'ymax': 10, 'zmax': np.random.uniform(5, 10)},
        #     {'xmin': 21, 'ymin': 15, 'zmin': 0, 'xmax': 23, 'ymax': 16, 'zmax': np.random.uniform(5, 10)}
        #     # {'xmin': 25, 'ymin': 17, 'zmin': 0, 'xmax': 26, 'ymax': 18, 'zmax': np.random.uniform(5, 10)}
        #
        # ]
        # Obstacle parameters

        obstacles = [
            {'xmin': -100, 'ymin': 20, 'zmin': -100, 'xmax': -20, 'ymax': 100, 'zmax': 100},
            {'xmin': 20, 'ymin': 20, 'zmin': -100, 'xmax': 100, 'ymax': 100, 'zmax': 100},
            {'xmin': -100, 'ymin': -100, 'zmin': -100, 'xmax': -20, 'ymax': -20, 'zmax': 100},
            {'xmin': 20, 'ymin': -100, 'zmin': -100, 'xmax': 100, 'ymax': -20, 'zmax': 100}
        ]
        temp_dict = []
        for obs in obstacles:
            boundary = [obs["xmin"], obs["ymin"], obs["xmax"], obs["ymax"]]
            p = box(boundary[0], boundary[1], boundary[2], boundary[3], ccw=True)
            # print(is_ccw(p))
            # print(shapely.is_ccw(p))
            # shapely.geometry.polygon.orient(p, sign=1)
            height = obs["zmax"] - obs["zmin"]
            edges = len(p.exterior.coords) - 1
            temp_dict.append({'geom': p, 'height': [obs["zmin"], obs["zmax"]], 'freq': 1, 'edges': edges})
        self.obstacles = temp_dict

    def check_new_missions(self):
        # Checking for missions that start at the current self.iteration
        # The missions list must be sorted by time
        if self.mission_counter < len(self.missions):
            condition = (self.missions[self.mission_counter]["iteration"] == self.iteration)
            while(condition == True):
                mission = self.missions[self.mission_counter]
                self.create_mission(mission)
                self.mission_counter += 1
                if self.mission_counter < len(self.missions):
                    condition = (self.missions[self.mission_counter]["iteration"] == self.iteration)
                if self.mission_counter == len(self.missions):
                    break

    def create_mission(self, mission):
        mission_type = mission["mission"]["type"]
        print(mission_type)
        if mission_type == "delivery":
            self.create_delivery_drone(mission)
        if mission_type == "firefighting":
            self.create_firefighting_drone(mission)
        if mission_type == "perimeter_patrol":
            self.create_general_mission(mission)
        if mission_type == "traffic_monitoring":
            self.create_traffic_monitoring_mission(mission)
        if mission_type == "research":
            self.create_general_mission(mission)
        if mission_type == "inspection":
            self.create_general_mission(mission)
        if mission_type == "recreational":
            self.create_general_mission(mission)

    def create_delivery_drone(self, mission):
        xi = mission["mission"]["destination"][0]["geometry"]
        xf = mission["mission"]["destination"][1]["geometry"].centroid
        pi = [xi.x, xi.y]
        pf = [xf.x, xf.y]
        waypoints = self.ppp.create_trajectory([pi, pf])
        waypoints = self.ppp.round_trip(waypoints)
        destination = int((len(waypoints) - 1) / 2)
        # if destination == 0:
        #     destination += 1 # To resolve an issue with firefighting drones
        destinations = [waypoints[0], waypoints[destination], waypoints[0]]
        self.create_drone(len(self.drn), mission["state"], waypoints, self.iteration, "delivery", "in progress", destinations)

    def create_firefighting_drone(self, mission):
        duration = int(10 * 60 / self.delta_t)
        destinations = mission["mission"]["destination"]
        waypoints = mission["mission"]["waypoints"]
        # Create a drone or multiple to go from the fire station to the location of the fire
        mission = self.create_drone(len(self.drn), mission["state"], waypoints, self.iteration, "firefighting", "in progress", destinations)
        # xi, waypoints, alert_drones, fire = self.obs.random_fire(self.drones, self.iteration)
        alert_drones, fire = self.obs.fire_response(mission, self.drones, self.iteration)
        # self.create_firefighting_drone(xi, waypoints)
        self.ppp.add_nfz(fire, duration)
        # Replan for the drones crossing the path and the area surrounding the location of the fire
        # Adding the fire to the no fly zones, so that other drones fly around it
        self.ppp.iteration = self.iteration
        for drn in alert_drones:
            current_state = self.drones[drn]["state"]
            print(self.drones[drn]["mission"]["destination"])
            home_state = self.drones[drn]["mission"]["destination"][0]
            dest_state = self.drones[drn]["mission"]["destination"][1]

            pi = [current_state['x'], current_state['y']]
            ph = [home_state['x'], home_state['y']]
            pf = [dest_state['x'], dest_state['y']]
            # If the delivery is not finished yet, finish it first, then go back to base
            status = self.drones[drn]["mission"]["status"]
            print(status)
            if status == "in progress":
                print("Recalculating trajectory")
                waypoints1 = self.ppp.create_trajectory([pi, pf])
                waypoints2 = self.ppp.create_trajectory([pf, ph])
                new_waypoints = waypoints1 + waypoints2
            # If the delivery is finished, just go back to base
            if status == "returning":
                print("Returning")
                # This was ph to pf
                new_waypoints = self.ppp.create_trajectory([pi, ph])
            if status == "waiting":
                print("Waiting")
                # This was ph to pf
                new_waypoints = self.ppp.create_trajectory([pi, ph])
            # # Putting the waypoints in the proper format
            # for i in range(len(new_waypoints)):
            #     new_waypoints[i] = list2state(new_waypoints[i])
            self.drones[drn]["mission"]["waypoints"] = new_waypoints

            traj = [state2list(state) for state in new_waypoints]
            ls = traj_to_linestring(traj)
            io.write_geom(transform_meter_global([ls]), "recalculated", "red")

            # Check this
            # Happened on iteration 573
            # self.drones[drn]["mission"]["progress"] = 0

    def create_traffic_monitoring_mission(self, mission):
        self.create_general_mission(mission)

    def create_general_mission(self, mission):
        self.create_drone(len(self.drn),
                          mission["state"],
                          mission["mission"]["waypoints"],
                          self.iteration, mission["mission"]["type"],
                          mission["mission"]["status"],
                          mission["mission"]["destination"])

    def create_drone(self, id, xi, waypoints, n, type, status, destinations):
        self.K += 1
        birthday = self.seconds_from_today(n)
        # Dictionary that contains all the data for the drone, except for the drone object
        d = {"id": id,
                 "birthday": birthday,
                 "iteration": n,
                 "trajs": [],
                 "alive": 1, # Alive, 0 is dead
                 "state": xi,
                 "factor": 0,
                 "mission": {
                     "type": type,
                     "destination": destinations,
                     "progress": 0,
                     "waypoints": waypoints, # Deliver to the final destination, then come back to the original place
                     "status":status
                 }}
        self.drones.append(d)
        traj_0 = self.make_full_traj(xi)
        self.drones[-1]["trajs"].append(traj_0)
        # Drone object
        self.drn.append(drone.drone())
        self.drn_list.append(len(self.drn) - 1)
        # self.env_list.append(grb.Env())
        return d

    def prepare_and_generate(self, k):
        t0 = time.time()
        self.proximity = 2 * self.N * self.delta_t * self.max_vel
        proximity_squared = self.proximity * self.proximity
        drone_prox_list = []
        # Finding the drones in proximity
        for i in self.drn_list:
            d = dist_squared(self.drones[i]["state"], self.drones[k]["state"])
            if d < proximity_squared and i != k:
                drone_prox_list.append(i)

        # Finding the obstacles in proximity
        state = self.drones[k]["state"]
        pos = [state['x'], state['y'], state['z']]
        # Temp solution
        obstacles = self.obs.nearby_obstacles(pos, self.proximity)
        # print(obstacles)
        obstacles = self.obstacles

        # Constructing the lists to be used as input to the function
        # Initial state
        xi = self.drones[k]["state"]
        # Final state

        progress = self.drones[k]["mission"]["progress"]
        # print("Drone %d, waypoint: %d, alive: %d" %(k, waypoint, self.drones[k]["alive"]))
        if self.drones[k]["mission"]["status"] == "waiting": # If the drone is waiting, stay at the last waypoint
            progress -= 1
        xf = self.drones[k]["mission"]["waypoints"][progress]
        # Other drones trajectories
        xi_1 = [self.drones[i]["trajs"][-1] for i in drone_prox_list]
        env = grb.Env()
        # Generating trajectories
        result = self.drn[k].generate_traj(env, xi, xf, xi_1, obstacles)
        env.close()

        t1 = time.time()
        print("T1 of drone: %.2f" % (t1 - t0))

        return result

    def m_start_simulation(self):
        """Optimized simulation function with Pool reuse and better resource management"""
        print("Starting Simulation")
        t0 = time.time()
        self.mission_counter = 0

        # Determine optimal number of processes - ensure it's never 0
        if hasattr(self, 'drones') and len(self.drones) > 0:
            max_processes = min(os.cpu_count(), len(self.drones))
        else:
            max_processes = os.cpu_count()  # Use all available cores when drone count is unknown/zero

        # Create pool once and reuse it throughout all iterations
        with Pool(processes=max_processes) as pool:
            # Main loop for optimization
            for self.iteration in range(self.total_iterations):
                print("%d============================" % (self.iteration))
                self.check_new_missions()

                # Build list of alive drones
                self.drn_list = []
                for k, drone in enumerate(self.drones):
                    if drone["alive"] == 1:
                        self.drn_list.append(k)

                K = len(self.drn_list)

                # Check termination conditions
                if K == 0 and (len(self.missions) == self.mission_counter):
                    print("No drones in simulation. Finishing up the run.")
                    break

                print(self.drn_list)
                t00 = time.time()

                if K == 0:
                    print("No drones to simulate. Skip")
                    continue

                # Process drones in parallel using the reused pool
                items = [(k, ) for k in self.drn_list]
                t02 = time.time()

                try:
                    # Use optimized prepare_and_generate function
                    results = pool.starmap(self.prepare_and_generate, items)

                    # Process results
                    for i, result in enumerate(results):
                        if i == 0:
                            t03 = time.time()

                        k = self.drn_list[i]
                        if result == -1:
                            self.drones[k]["alive"] = 0
                            self.drones[k]["mission"]["status"] = "Collided or infeasible"
                            print("Error in trajectory for drone %d" % k)
                        else:
                            self.drn[k].full_traj = result

                except Exception as e:
                    print(f"Error during parallel processing in iteration {self.iteration}: {e}")
                    # Continue with next iteration instead of crashing
                    continue

                # Update and log as before
                self.update()

                # io.log_to_json(self.drones, self.sim_run, self.sim_latest)
                # io.log_to_json_dict(self.drones, self.sim_run, self.sim_latest)
                io.log_to_json_dict_robust(self.drones, self.sim_run, self.sim_latest)

                t01 = time.time()
                print("Time of iteration: %.2f" % (t01 - t00))

    def update(self):
        # Check for collision
        self.check_collisions()
        # Update trajectories and the current state
        self.update_vehicle_state()
        # Apply random events
        # self.random_events()
        # Check the mission progress
        self.update_mission()
        # Update the trajectories

    def random_events(self):
        N = len(self.drn_list)
        if self.iteration == 5500:
            print("ABORT MISSION!!!!11!!!!!! MBS IS HERE!!!!")
            rnd_drn = random.choices(self.drn_list, k=3)
            print(rnd_drn)
            for drn in self.drn_list:
                self.drones[drn]["factor"] += 0.25
                if self.drones[drn]["mission"]["type"] != "firefighting":
                    if self.drones[drn]["mission"]["type"] != "emergency landing":
                        if self.drones[drn]["factor"] >= 0.125:
                            self.drones[drn]["mission"]["type"] = "emergency landing"
                            pos = state2list(self.drones[drn]["state"])
                            emergency_landing = self.ppp.closest_landing(pos)
                            el0 = point_to_waypoint(emergency_landing[0], 10)
                            el1 = point_to_waypoint(emergency_landing[1], 0)
                            el = [el0, el1]
                            self.drones[drn]["mission"]["waypoints"] += el
                            waypoints = self.drones[drn]["mission"]["waypoints"]
                            len(waypoints) - 2
                            progress = self.drones[drn]["mission"]["progress"]
                            self.drones[drn]["mission"]["progress"] = len(waypoints) - 2

    def make_full_traj(self, xi):
        # Making a full trajectories out of the initial state for initialization
        x = [xi['x'] for n in range(self.N)]
        y = [xi['y'] for n in range(self.N)]
        z = [xi['z'] for n in range(self.N)]
        xdot = [xi['xdot'] for n in range(self.N)]
        ydot = [xi['ydot'] for n in range(self.N)]
        zdot = [xi['zdot'] for n in range(self.N)]
        return [x, y, z, xdot, ydot, zdot]

    def update_vehicle_state(self):
        for k in self.drn_list:
            self.drones[k]["trajs"].append(self.drn[k].full_traj)
            # Extract positions and velocities from the model's solution
            x_position = self.drones[k]["trajs"][-1][0][0]
            y_position = self.drones[k]["trajs"][-1][1][0]
            z_position = self.drones[k]["trajs"][-1][2][0]
            x_velocity = self.drones[k]["trajs"][-1][3][0]
            y_velocity = self.drones[k]["trajs"][-1][4][0]
            z_velocity = self.drones[k]["trajs"][-1][5][0]
            # Update the initial conditions for the next iteration
            state = [x_position, y_position, z_position, x_velocity, y_velocity, z_velocity]
            self.drones[k]["state"] = list2state(state)

    def check_collisions(self):
        # i = 0
        # while i < len(self.drn_list):
        #     k = self.drn_list[i]
        #     if self.drn[k].get_drone_status() == False: # Drone collided
        #     if self.drn[k].get_drone_status() == False: # Drone collided
        #         self.drn_list.remove(k)
        #         print("Removed drone %d" % (k))
        #         i -= 1
        #     i += 1
        # Finding the drones in proximity
        collided_drones = set() # Set of collided drones
        for i in range(len(self.drn_list)):
            for j in range(i + 1, len(self.drn_list)):
                d1 = self.drn_list[i]
                d2 = self.drn_list[j]
                d = dist_squared(self.drones[d1]["state"], self.drones[d2]["state"])
                if d < self.collision_warning * self.collision_warning:
                    if d < self.collision * self.collision:
                        self.collide(d1, d2, d)
                        collided_drones.add(d1)
                        collided_drones.add(d2)

        for drone in collided_drones:
            # self.drn_list.remove(drone)
            pass

    def collide(self, d1, d2, d):
        # What to do when drones collide
        print("Collision between drone %d and %d, distance: %f" % (d1, d2, np.sqrt(d)))
        # self.drn_list.remove(d1)
        # self.drn_list.remove(d2)

    def update_mission(self):
        for k, drone in enumerate(self.drones):
            if drone["alive"]:
                drn = drone["state"]
                progress = drone["mission"]["progress"]
                if drone["mission"]["status"] == "waiting": # If the drone is waiting
                    print("waiting")
                    drone["mission"]["waypoints"][progress] -= 50 # Countdown the timer
                    print(drone["mission"]["waypoints"][progress])
                    if drone["mission"]["waypoints"][progress] <= 0: # If the timer is finished
                        drone["mission"]["progress"] += 1 # Go to the next step of the mission
                        drone["mission"]["status"] = "returning"
                    continue
                waypoint = drone["mission"]["waypoints"][progress]

                dist = np.sqrt(dist_squared(drn, waypoint))
                if dist < 5:
                    print("Drone %d reached destination!" %(k))
                    # drone["mission"]["progress"] += 1
                    progress = drone["mission"]["progress"]
                    geom = drone["mission"]["destination"][1]
                    destination = state2list(geom)
                    waypoint = [drone["mission"]["waypoints"][progress]['x'],
                                 drone["mission"]["waypoints"][progress]['y'],
                                   drone["mission"]["waypoints"][progress]['z']]
                    print("Mission progress: ", drone["mission"]["progress"])
                    print("Length of waypoints: ", len(drone["mission"]["waypoints"])-1)
                    if drone["mission"]["progress"] == len(drone["mission"]["waypoints"]) - 1: # If it completed the mission
                        drone["mission"]["status"] = "completed" # Mark it as completed
                        drone["alive"] = 0 # Mark it as dead/offline
                        print("MISSION COMPLETED WOOOHOOOOOO!!!1111!!!!!!11!1")
                        continue
                    # If delivered, change status to returning to base
                    elif waypoint == destination:
                        drone["mission"]["status"] = "returning"
                    drone["mission"]["progress"] += 1
                progress = drone["mission"]["progress"]
                waypoint = drone["mission"]["waypoints"][progress]
                if isinstance(waypoint, float) or isinstance(waypoint, int):
                    drone["mission"]["status"] = "waiting"

    def random_state(self, xi, xf):
        v = 2
        x = np.array(xi)
        y = np.zeros(3)
        vec = y - x
        vi = (v * vec / np.linalg.norm(vec)).tolist()
        x = np.array(xf)
        vec = x - y
        vf = (v * vec / np.linalg.norm(vec)).tolist()
        return vi, vf

    def seconds_from_today(self, iteration):
        s = iteration * self.delta_t
        # datetime.datetime.fromtimestamp(ms/1000.0)
        ts = self.sim_start.timestamp() + s
        mission_start = datetime.datetime.fromtimestamp(ts)
        s = mission_start.strftime("%Y-%m-%d %H:%M:%S")
        s = mission_start.strftime("%Y-%m-%d")
        return mission_start

    def set_initial_state(self):
        # Initial conditions for each vehicle
        self.xi = [
            {'x': -95, 'y': 0, 'z': 10, 'xdot': 3, 'ydot': 3, 'zdot': 2},
            {'x': 0, 'y': 95, 'z': 10, 'xdot': 3, 'ydot': 3, 'zdot': 2},
            {'x': 90, 'y': 0, 'z': 15, 'xdot': 3, 'ydot': 3, 'zdot': 2},
            {'x': 10, 'y': -95, 'z': 10, 'xdot': 3, 'ydot': 3, 'zdot': 2},

            {'x': 0, 'y': 0, 'z': -95, 'xdot': 3, 'ydot': 3, 'zdot': 2},
            {'x': 0, 'y': 0, 'z': 95, 'xdot': 3, 'ydot': 3, 'zdot': 2},
            {'x': 15, 'y': 95, 'z': 95, 'xdot': 3, 'ydot': 3, 'zdot': 2},
            {'x': -15, 'y': -95, 'z': 95, 'xdot': 3, 'ydot': 3, 'zdot': 2},

            {'x': 95, 'y': 15, 'z': 95, 'xdot': 3, 'ydot': 3, 'zdot': 2},
            {'x': -95, 'y': 15, 'z': 95, 'xdot': 3, 'ydot': 3, 'zdot': 2},
            {'x': -15, 'y': -95, 'z': -95, 'xdot': 3, 'ydot': 3, 'zdot': 2},
            {'x': 15, 'y': 95, 'z': -95, 'xdot': 3, 'ydot': 3, 'zdot': 2},

            {'x': 95, 'y': -15, 'z': 95, 'xdot': 3, 'ydot': 3, 'zdot': 2},
            {'x': -95, 'y': 15, 'z': -95, 'xdot': 3, 'ydot': 3, 'zdot': 2},

            {'x': -95, 'y': -15, 'z': 95, 'xdot': 3, 'ydot': 3, 'zdot': 2},
            {'x': -95, 'y': -15, 'z': -95, 'xdot': 3, 'ydot': 3, 'zdot': 2},

            {'x': 95, 'y': -15, 'z': -95, 'xdot': 3, 'ydot': 3, 'zdot': 2},
            {'x': 95, 'y': 15, 'z': -95, 'xdot': 3, 'ydot': 3, 'zdot': 2},

            {'x': -15, 'y': 95, 'z': -95, 'xdot': 3, 'ydot': 3, 'zdot': 2},
            {'x': 15, 'y': -95, 'z': 95, 'xdot': 3, 'ydot': 3, 'zdot': 2},

            {'x': 15, 'y': -95, 'z': -95, 'xdot': 3, 'ydot': 3, 'zdot': 2},
            {'x': -15, 'y': 95, 'z': 95, 'xdot': 3, 'ydot': 3, 'zdot': 2}

            # {'x': 2, 'y': 1, 'xdot': 0, 'ydot': 0}
        ]
        # Iterate over drones, set the final condition for each
        # for each drone
        # drone[i].set_initial_state(xi)

    def set_final_state(self):
        # Set final state
        self.final_conditions = [
            {'x': 95, 'y': 10, 'z': 10, 'xdot': 2, 'ydot': 2, 'zdot': 2},
            {'x': 10, 'y': -95, 'z': 10, 'xdot': 2, 'ydot': 2, 'zdot': 2},
            {'x': -95, 'y': 0, 'z': 10, 'xdot': 2, 'ydot': 2, 'zdot': 2},
            {'x': 0, 'y': 95, 'z': 10, 'xdot': 2, 'ydot': 2, 'zdot': 2},

            {'x': 0, 'y': 0, 'z': 95, 'xdot': 2, 'ydot': 2, 'zdot': 2},
            {'x': 0, 'y': 0, 'z': -95, 'xdot': 2, 'ydot': 2, 'zdot': 2},
            {'x': -15, 'y': -95, 'z': -95, 'xdot': 2, 'ydot': 2, 'zdot': 2},
            {'x': 15, 'y': 95, 'z': -95, 'xdot': 2, 'ydot': 2, 'zdot': 2},

            {'x': -95, 'y': -15, 'z': -95, 'xdot': 2, 'ydot': 2, 'zdot': 2},
            {'x': 95, 'y': -15, 'z': -95, 'xdot': 2, 'ydot': 2, 'zdot': 2},
            {'x': 15, 'y': 95, 'z': 95, 'xdot': 2, 'ydot': 2, 'zdot': 2},
            {'x': -15, 'y': -95, 'z': 95, 'xdot': 2, 'ydot': 2, 'zdot': 2},


            {'x': -95, 'y': 15, 'z': -95, 'xdot': 2, 'ydot': 2, 'zdot': 2},
            {'x': 95, 'y': -15, 'z': 95, 'xdot': 2, 'ydot': 2, 'zdot': 2},

            {'x': 95, 'y': 15, 'z': -95, 'xdot': 2, 'ydot': 2, 'zdot': 2},
            {'x': 95, 'y': 15, 'z': 95, 'xdot': 2, 'ydot': 2, 'zdot': 2},

            {'x': -95, 'y': 15, 'z': 95, 'xdot': 2, 'ydot': 2, 'zdot': 2},
            {'x': -95, 'y': -15, 'z': 95, 'xdot': 2, 'ydot': 2, 'zdot': 2},

            {'x': 15, 'y': -95, 'z': 95, 'xdot': 3, 'ydot': 3, 'zdot': 2},
            {'x': -15, 'y': 95, 'z': -95, 'xdot': 3, 'ydot': 3, 'zdot': 2},

            {'x': -15, 'y': 95, 'z': 95, 'xdot': 3, 'ydot': 3, 'zdot': 2},
            {'x': 15, 'y': -95, 'z': -95, 'xdot': 3, 'ydot': 3, 'zdot': 2}
            # {'x': 9, 'y': 10, 'xdot': 0, 'ydot': 0}
            # {'x': 8, 'y': 2, 'xdot': 0, 'ydot': 0},
            # {'x': 13, 'y': 9, 'xdot': 0, 'ydot': 0}
        ]

        # Iterate over drones, set the final condition for each
        # for each drone
        # drone[i].set_final_state(xf)

def main():
    optimization = simulator()
    optimization.m_start_simulation()

main()