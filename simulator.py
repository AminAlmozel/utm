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

import drone
import environment
from sim_io import myio as io

import os, sys
from sys import getsizeof
sys.path.append(os.path.join(os.path.dirname(__file__), "uam"))

import polygon_pathplanning


# from uam.polygon_pathplanning import *

class simulator(drone.drone):
    def __init__(self):
        self.N = 50 # Prediction horizon
        self.delta_t = 0.1 # Time step
        self.total_iterations = 5000

        # Parameters
        self.n_vehicles = 5 # Starting number of vehicles
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

        self.ppp = polygon_pathplanning.polygon_pp()
        self.ppp.sim_run = self.sim_run
        self.ppp.sim_latest = self.sim_latest
        self.dummy_obstacles()
        self.initialize_drones()

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

    def initialize_drones(self):
        # Initalize delivery drones
        for k in range(self.n_vehicles):
            self.create_delivery_drone(0)
        self.max_vel = self.drn[0].smax[3] # Max velocity

    def create_delivery_drone(self, n):
        xi, xf = self.obs.random_mission(0)
        xi = self.list2state(xi)
        xf = self.list2state(xf)
        # self.create_drone(self.xi[k], self.final_conditions[k], 0)
        pi = [xi['x'], xi['y']]
        pf = [xf['x'], xf['y']]
        # print([pi, pf])
        waypoints = self.ppp.create_trajectory([pi, pf])
        waypoints = self.ppp.round_trip(waypoints)
        for i in range(len(waypoints)):
            waypoints[i] = self.list2state(waypoints[i])
        # waypoints = [xi, xf]
        destination = int((len(waypoints) - 1) / 2)
        # print(destination)
        # if destination == 0:
        #     destination += 1 # To resolve an issue with firefighting drones
        destinations = [waypoints[0], waypoints[destination], waypoints[0]]

        self.create_drone(xi, waypoints, n, "delivery", "in progress", destinations)

    def create_firefighting_drone(self, xi, waypoints):
        xi = self.list2state(xi)
        for i in range(len(waypoints)):
            waypoints[i] = self.list2state(waypoints[i])
        # Create a drone or multiple to go from the fire station to the location of the fire
        destinations = waypoints
        self.create_drone(xi, waypoints, self.iteration, "firefighting", "in progress", destinations)

    def create_drone(self, xi, waypoints, n, type, status, destinations):
        self.K += 1
        born = self.seconds_from_today(n)
        # Dictionary that contains all the data for the drone, except for the drone object
        d = {"id": len(self.drn),
                 "born": born,
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

    def prepare_and_generate(self, k):
        t0 = time.time()
        self.proximity = 2 * self.N * self.delta_t * self.max_vel
        drone_prox_list = []
        # Finding the drones in proximity
        for i in self.drn_list:
            d = self.dist_squared(self.drones[i]["state"], self.drones[k]["state"])
            if d < self.proximity * self.proximity and i != k:
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
        waypoint = self.drones[k]["mission"]["progress"]
        # print("Drone %d, waypoint: %d, alive: %d" %(k, waypoint, self.drones[k]["alive"]))
        xf = self.drones[k]["mission"]["waypoints"][waypoint]
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
        print("Starting Simulation")
        t0 = time.time()

        # Main loop for optimization
        for self.iteration in range(self.total_iterations):
            print("%d============================" % (self.iteration))
            if self.iteration%10 == 0 and self.iteration < 200:
                self.create_delivery_drone(self.iteration)
                print("Created drUone")

            # K = len(self.drn_list)
            self.drn_list = []
            for k, drone in enumerate(self.drones):
                if drone["alive"] == 1:
                    self.drn_list.append(k)
            K = len(self.drn_list)
            if K == 0:
                print("No drones in simulation. Finishing up the run.")
                break
            print(self.drn_list)
            t00 = time.time()
            pool = Pool()
            with Pool() as pool:
                # prepare arguments
                # items = [(k, ) for k in range(K)]
                items = [(k, ) for k in self.drn_list]
                t02 = time.time()
                for i, result in enumerate(pool.starmap(self.prepare_and_generate, items)):
                    if i == 0:
                        t03 = time.time()
                    # report the value to show progress
                    k = self.drn_list[i]
                    if result == -1:
                        self.drones[k]["alive"] = 0
                        self.drones[k]["mission"]["status"] = "Collided or infeasible"
                        print("Error in trajectory for drone %d" %(k))
                    else:
                        self.drn[k].full_traj = result
            pool.close()
            pool.join()
            self.update()
            io.log_to_json(self.drones, self.sim_run, self.sim_latest)

            t01 = time.time()
            print("Time of iteration: %.2f" % (t01 - t00))
            print("T1 of iteration: %.2f" % (t03 - t02))


        io.log_to_json_dict(self.drones, self.sim_run, self.sim_latest)
        io.log_to_pickle(self.drones, self.sim_run, self.sim_latest)
        temp = io.read_log_pickle(self.sim_latest)
        # io.log_to_json(self.drones, self.sim_run, self.sim_latest)

        t1 = time.time()
        print("Time of execution: %.2f" % (t1 - t0))

    def update(self):
        # Check for collision
        self.check_collisions()
        # Update trajectories and the current state
        self.update_vehicle_state()
        # Apply random events
        self.random_events()
        # Check the mission progress
        self.update_mission()
        # Update the trajectories

    def random_events(self):
        N = len(self.drn_list)
        if self.iteration == 4: # Random fire emergency
            xi, waypoints, alert_drones, fire = self.obs.random_fire(self.drones, self.iteration)
            self.create_firefighting_drone(xi, waypoints)
            self.ppp.add_to_forbidden(fire)
            # Replan for the drones crossing the path and the area surrounding the location of the fire
            # Adding the fire to the no fly zones, so that other drones fly around it

            for drn in alert_drones:
                current_state = self.drones[drn]["state"]
                home_state = self.drones[drn]["mission"]["destination"][0]
                dest_state = self.drones[drn]["mission"]["destination"][1]

                pi = [current_state['x'], current_state['y']]
                px = [home_state['x'], home_state['y']]
                pf = [dest_state['x'], dest_state['y']]
                # If the delivery is not finished yet, finish it first, then go back to base
                if self.drones[drn]["mission"]["status"] == "in progress":
                    print("Recalculating trajectory")
                    waypoints1 = self.ppp.create_trajectory([pi, pf])
                    waypoints2 = self.ppp.create_trajectory([pf, px])
                    new_waypoints = waypoints1 + waypoints2
                # If the delivery is finished, just go back to base
                if self.drones[drn]["mission"]["status"] == "returning":
                    print("Returning")
                    new_waypoints = self.ppp.create_trajectory([px, pf])
                # Putting the waypoints in the proper format
                for i in range(len(new_waypoints)):
                    new_waypoints[i] = self.list2state(new_waypoints[i])
                self.drones[drn]["mission"]["waypoints"] = new_waypoints


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
                            pos = self.state2list(self.drones[drn]["state"])
                            emergency_landing = self.ppp.closest_landing(pos)
                            el0 = self.point_to_waypoint(emergency_landing[0], 10)
                            el1 = self.point_to_waypoint(emergency_landing[1], 0)
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
            self.drones[k]["state"] = self.list2state(state)

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
                d = self.dist_squared(self.drones[d1]["state"], self.drones[d2]["state"])
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
                dest = drone["mission"]["waypoints"][progress]
                dist = np.sqrt(self.dist_squared(drn, dest))
                if dist < 5:
                    print("Drone %d reached destination!" %(k))
                    drone["mission"]["progress"] += 1
                    progress = drone["mission"]["progress"]
                    if drone["mission"]["progress"] == len(drone["mission"]["waypoints"]): # If it completed the mission
                        drone["mission"]["status"] = "completed" # Mark it as completed
                        drone["alive"] = 0 # Mark it as dead/offline
                        print("MISSION COMPLETED WOOOHOOOOOO!!!1111!!!!!!11!1")
                    # If delivered, change status to returning to base
                    elif drone["mission"]["waypoints"][progress] == drone["mission"]["destination"][1]:
                        drone["mission"]["status"] = "returning"

    def random_drone(self):
        # Choosing a random entrace
        entrance_prob = [0.25, 0.25, 0.25, 0.25]
        i_index = random.choices(range(4), weights=entrance_prob)[0]
        # Choosing a random exit (except for the chosen entrace)
        exit_prob = [0.25, 0.25, 0.25, 0.25]
        exit_prob[i_index] = 0
        e_index = random.choices(range(4), weights=exit_prob)[0]
        pi = self.random_gate(i_index, 0)
        pf = self.random_gate(e_index, 0)
        vi, vf = self.random_state(pi, pf)
        xi = self.list2state(pi + vi)
        xf = self.list2state(pf + vf)
        return xi, xf

    def random_gate(self, index, z):
        x = 95
        y = 95
        z = 0
        gates = [[0, y, z],
                 [0, -y, z],
                 [x, 0, z],
                 [-x, 0, z]]
        return gates[index]

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

    def list2state(self, values):
        keys = ['x', 'y', 'z', 'xdot', 'ydot', 'zdot']
        return dict(zip(keys, values))

    def state2list(self, values):
        return [values["x"], values["y"], values["z"]]

    def point_to_waypoint(self, p, z):
        waypoint = [p.x, p.y, z, 0, 0, 0]
        return self.list2state(waypoint)

    def dist_squared(self, xi, xi_1):
        return (xi['x'] - xi_1['x'])**2 + (xi['y'] - xi_1['y'])**2 + (xi['z'] - xi_1['z'])**2

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