# Importing standard libraries
import time
import datetime
import random
import math
import concurrent.futures
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
import gc
sys.path.append(os.path.join(os.path.dirname(__file__), "uam"))

# import polygon_pathplanning as ppp
import sampling_pathplanning as spp


_worker_env = None
_worker_drone = None

def init_worker():
    """Initialize global environment and drone instance for each worker process."""
    global _worker_env, _worker_drone
    _worker_env = grb.Env()
    _worker_drone = drone.drone()

def process_drone_trajectory(drone_data):
    """
    External multiprocessing function to generate drone trajectories.
    Uses a global Gurobi environment and drone instance that are initialized once per worker.

    Args:
        drone_data: Dictionary containing all necessary data for trajectory generation

    Returns:
        result: Generated trajectory or -1 if failed
    """
    global _worker_env, _worker_drone

    try:
        t0 = time.time()
        # Extract data from the input dictionary
        drone_id = drone_data['drone_id']
        xi = drone_data['xi']
        xf = drone_data['xf']
        xi_1 = drone_data['xi_1']
        obstacles = drone_data['obstacles']
        drone_params = drone_data['drone_params']

        # Generate trajectory using the worker's environment and drone instance
        if drone_params["mission_status"] == "emergency landing":
            print(f"Drone {drone_id} is in emergency landing mode.")
            _worker_drone.engine_failure()  # Set the drone to engine failure mode
        result = _worker_drone.generate_traj(_worker_env, xi, xf, xi_1, obstacles)

        t1 = time.time()
        # print(f"Processed drone {drone_id} in {t1 - t0:.2f} seconds")
        return result

    except Exception as e:
        print(f"Error processing drone {drone_data.get('drone_id', 'unknown')}: {e}")
        return -1

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

class simulator(drone.drone):
    def __init__(self):
        self.N = 50 # Prediction horizon
        self.delta_t = 0.1 # Time step
        self.total_iterations = 50000

        self.emergency_landing = 45000

        # Parameters
        self.n_vehicles = 10 # Starting number of vehicles
        self.K = 0 # Number of vehicles

        # Parameters for collision avoidance between vehicles
        d_min = 5
        self.d_x = d_min  # Minimum horizontal distance
        self.d_y = d_min  # Minimum vertical distance
        self.d_z = d_min  # Minimum vertical distance

        self.collision = 2
        self.collision_warning = 5

        self.drn = [] # All drones objects
        self.drones = [] # Drone dictionaries
        self.drn_list = [] # Indices of alive drones
        self.env_list = [] # List of optimization enviroment for each drone

        self.vehicles = []
        self.full_traj = []
        print("Loading enviornment")
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
        print("Loading path planning")
        self.ppp = spp.sampling_pp()
        self.ppp.sim_run = self.sim_run
        self.ppp.sim_latest = self.sim_latest
        self.dummy_obstacles()
        print("Loading missions")
        self.missions = io.import_missions()

        self.max_vel = drone.drone().V_max # Max velocity
        self.max_acc = drone.drone().A_max # Max acceleration

    def prepare_drone_data(self, k):
        """
        Prepare data for multiprocessing function.

        Args:
            k: Drone index

        Returns:
            drone_data: Dictionary containing all necessary data for trajectory generation
        """
        # Calculate proximity for this drone
        self.proximity = 2 * self.N * self.delta_t * self.max_vel
        proximity_squared = self.proximity * self.proximity
        drone_prox_list = []

        # Finding the drones in proximity
        for i in self.drn_list:
            d = dist_squared(self.drones[i].state, self.drones[k].state)
            if d < proximity_squared and i != k:
                drone_prox_list.append(i)

        # Finding the obstacles in proximity
        state = self.drones[k].state
        pos = Point(state['x'], state['y'], state['z'])

        nfz = []
        if self.drones[k].mission_type == "delivery":
            obstacles, inside = self.ppp.nearby_nfz(pos, self.proximity)
            if len(obstacles) != 0:
                for obs in obstacles:
                    vertices = list(zip(*obs.exterior.coords.xy))
                    height = -1 # No height limit
                    nfz.append({'geom':vertices, 'height': [0, height], 'edges': len(vertices) - 1})
            if inside == 1:
                nfz = []  # If the drone is inside a no-fly zone, do not use it as an obstacle
        obstacles = nfz

        # Initial state
        xi = self.drones[k].state

        # Final state
        progress = self.drones[k].mission_progress
        if self.drones[k].mission_status == "waiting": # If the drone is waiting, stay at the last waypoint
            progress -= 1
        xf = self.drones[k].mission_waypoints[progress]

        # Other drones trajectories
        xi_1 = [self.drones[i].xi_1 for i in drone_prox_list]

        # Prepare drone parameters (if needed for trajectory generation)
        drone_params = {
            'N': self.N,
            'delta_t': self.delta_t,
            'max_vel': self.max_vel,
            'mission_status': self.drones[k].mission_status
        }

        return {
            'drone_id': k,
            'xi': xi,
            'xf': xf,
            'xi_1': xi_1,
            'obstacles': obstacles,
            'drone_params': drone_params
        }


    def m_start_simulation(self):
        """Optimized simulation function with external multiprocessing"""
        print("Starting Simulation")

        t0 = time.time()
        self.mission_counter = 0

        # Determine optimal number of processes - ensure it's never 0
        if hasattr(self, 'drones') and len(self.drones) > 0:
            max_workers = min(os.cpu_count(), len(self.drones))
        else:
            max_workers = os.cpu_count()  # Use all available cores when drone count is unknown/zero

        # Create executor pool with initialized workers and reuse it throughout all iterations
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=init_worker
        ) as executor:
            # Main loop for optimization
            for self.iteration in range(self.total_iterations):
                print("%d============================" % (self.iteration))
                if self.iteration < self.emergency_landing:
                    self.check_new_missions()

                # Build list of alive drones
                self.drn_list = []
                for k, drone in enumerate(self.drones):
                    if drone.alive == 1:
                        self.drn_list.append(k)

                K = len(self.drn_list)

                # Check termination conditions
                # if K == 0 and (len(self.missions) == self.mission_counter):
                if K == 0 and (self.iteration > self.emergency_landing):
                    print("No drones in simulation. Finishing up the run.")
                    break

                print(self.drn_list)
                t00 = time.time()

                if K == 0:
                    print("No drones to simulate. Skip")
                    continue

                # Prepare data for all drones
                drone_data_list = []
                future_to_drone = {}
                t02 = time.time()
                for k in self.drn_list:
                    drone_data = self.prepare_drone_data(k)
                    drone_data_list.append(drone_data)

                t03 = time.time()
                # print("Prepared data for %d drones in %.2f seconds" % (len(drone_data_list), t03 - t02))

                try:
                    # Submit all drone trajectory calculations to the executor
                    future_to_drone = {executor.submit(process_drone_trajectory, data): i
                                     for i, data in enumerate(drone_data_list)}

                    # Process results as they complete
                    for future in concurrent.futures.as_completed(future_to_drone):
                        i = future_to_drone[future]
                        k = self.drn_list[i]

                        try:
                            result = future.result()
                            if i == 0:
                                t03 = time.time()

                            if result == -1:
                                self.drones[k].alive = 0
                                self.drones[k].mission_status = "Collided or infeasible"
                                print("Error in trajectory for drone %d" % k)
                            else:
                                self.drn[k].full_traj = result
                        except Exception as exc:
                            print(f"Drone {k} generated an exception: {exc}")
                            self.drones[k].alive = 0
                            self.drones[k].mission_status = "Execution error"

                except Exception as e:
                    print(f"Error during parallel processing in iteration {self.iteration}: {e}")
                    # Continue with next iteration instead of crashing
                    continue

                # Update and log as before
                self.update()
                if self.iteration % 1000 == 0:
                    print("Writing")
                    io.log_to_pickle(self.drones, "missions", self.sim_run, self.sim_latest)
                    io.log_to_json_dict(self.drones, self.sim_run, self.sim_latest)
                # io.log_to_json_dict_robust(self.drones, self.sim_run, self.sim_latest)
                if self.iteration % 1000 == 0:
                    nfz = self.ppp.get_nfz()
                    io.log_dictionary(nfz, self.sim_run, self.sim_latest)

                gc.collect()
                t01 = time.time()
                print("Time of iteration: %.2f" % (t01 - t00))
        io.log_to_pickle(self.drones, "missions", self.sim_run, self.sim_latest)
        io.log_to_json_dict(self.drones, self.sim_run, self.sim_latest)
        nfz = self.ppp.get_nfz()
        io.log_dictionary(nfz, self.sim_run, self.sim_latest)
        self.ppp.write_area_costs()

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
        alert_drones, fire = self.obs.fire_response(mission, self.drones, self.iteration)
        self.ppp.add_nfz(fire, mission.id)
        # Replan for the drones crossing the path and the area surrounding the location of the fire
        # Adding the fire to the no fly zones, so that other drones fly around it
        self.ppp.iteration = self.iteration
        for drn in alert_drones:
            current_state = self.drones[drn].state
            home_state = self.drones[drn].mission_destination[0]
            dest_state = self.drones[drn].mission_destination[1]

            pi = [current_state['x'], current_state['y']]
            ph = [home_state['x'], home_state['y']]
            pf = [dest_state['x'], dest_state['y']]
            # If the delivery is not finished yet, finish it first, then go back to base
            status = self.drones[drn].mission_status
            print(status)
            if status == "in progress":
                waypoints1 = self.ppp.create_trajectory([pi, pf])
                waypoints2 = self.ppp.create_trajectory([pf, ph])
                new_waypoints = waypoints1 + waypoints2
            # If the delivery is finished, just go back to base
            if status == "returning":
                # This was ph to pf
                new_waypoints = self.ppp.create_trajectory([pi, ph])
            if status == "waiting":
                new_waypoints = self.ppp.create_trajectory([pi, ph])
            if status == "completed" or status == "collided":
                continue

            # TODO: Check other statuses
            progress = self.drones[drn].mission_progress
            self.drones[drn].mission_waypoints = self.drones[drn].mission_waypoints[:progress] + new_waypoints
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
        d = DroneData(
            id=id,
            birthday=birthday,
            iteration=n,
            state=xi,
            mission_type=type,
            mission_status=status,
            mission_destination=destinations,
            mission_waypoints=waypoints
        )
        self.drones.append(d)
        traj_0 = self.make_full_traj(xi)
        self.drones[-1].xi_1 = traj_0
        # Drone object
        self.drn.append(drone.drone())
        self.drn_list.append(len(self.drn) - 1)
        return d

    def update(self):
        # Check for collision
        # self.check_collisions()
        # Update trajectories and the current state
        self.update_vehicle_state()
        # Apply random events
        # self.random_events()
        # Check the mission progress
        self.update_mission()
        # Update the trajectories

    def update_mission(self):
        self.ppp.iteration = self.iteration
        for k, drone in enumerate(self.drones):
            if drone.alive:
                drn = drone.state
                progress = drone.mission_progress
                if drone.mission_status == "waiting": # If the drone is waiting
                    # print("waiting")
                    drone.mission_waypoints[progress] -= 1 # Countdown the timer
                    # print(drone.mission_waypoints[progress])
                    if drone.mission_waypoints[progress] <= 0: # If the timer is finished
                        drone.mission_progress += 1 # Go to the next step of the mission
                        drone.mission_status = "returning"
                    continue
                waypoint = drone.mission_waypoints[progress]

                dist = np.sqrt(dist_squared(drn, waypoint))
                if drone.mission_status == "emergency landing":
                    dist_threshold = 0.5
                else:
                    dist_threshold = 5
                if dist < dist_threshold:
                    print("Drone %d reached destination!" %(k))
                    progress = drone.mission_progress
                    geom = drone.mission_destination[1]
                    destination = state2list(geom)
                    waypoint = [drone.mission_waypoints[progress]['x'],
                              drone.mission_waypoints[progress]['y'],
                              drone.mission_waypoints[progress]['z']]
                    print("Mission progress: %d of %d" % (drone.mission_progress, len(drone.mission_waypoints)-1))
                    if drone.mission_progress == len(drone.mission_waypoints) - 1: # If it completed the mission
                        drone.mission_status = "completed" # Mark it as completed
                        drone.alive = 0 # Mark it as dead/offline
                        if drone.mission_type == "firefighting":
                            self.ppp.remove_nfz(drone.id)
                        print("MISSION COMPLETED WOOOHOOOOOO!!!1111!!!!!!11!1")
                        continue
                    # If delivered, change status to returning to base
                    elif waypoint == destination:
                        drone.mission_status = "returning"
                    drone.mission_progress += 1
                progress = drone.mission_progress
                waypoint = drone.mission_waypoints[progress]
                if isinstance(waypoint, float) or isinstance(waypoint, int):
                    drone.mission_status = "waiting"

    def emergency_events(self):
        # Add global events such as severe weather
        for k, drone in enumerate(self.drones):
            if drone.alive:
                if not hasattr(drone, 'emergency'):  # If emergency is not in slots, skip
                    continue
                drn_emergency = drone.emergency
                if drn_emergency == "none":
                    continue
                state = drone.state
                closest, inward, _ = self.ppp.closest_landing(Point(state['x'], state['y']))
                emergency_landing = [point_to_waypoint(closest, 30), point_to_waypoint(inward, 30)]
                home = drone.mission_destination[0]
                if drn_emergency == "gps_loss":
                    # Land, Switch to alternative navigation
                    pass

                if drn_emergency == "communication_loss":
                    # Return or loiter depending on battery
                    pass

                if drn_emergency == "low_battery":
                    # Return or land
                    pass

                if drn_emergency == "motor_failure":
                    # Land
                    pass

    def random_events(self):
        N = len(self.drn_list)
        if self.iteration == self.emergency_landing:
            print("ABORT MISSION!!!!11!!!!!! MBS IS HERE!!!!")
            print(self.drn_list)
            for drn in self.drn_list:
                self.drones[drn].mission_status = "emergency landing"
                # pos = state2list(self.drones[drn].state)
                pos = calculate_stopping_point(self.drones[drn].state, self.max_acc)
                emergency_landing = self.ppp.closest_landing(pos)
                print("Emergency landing for drone %d at %s" % (drn, emergency_landing))
                el0 = point_to_waypoint(emergency_landing[0], 10)
                el1 = point_to_waypoint(emergency_landing[1], 0)
                el = [el0, el1]
                progress = self.drones[drn].mission_progress
                self.drones[drn].mission_waypoints = self.drones[drn].mission_waypoints[:progress] + el

    def update_vehicle_state(self):
        for k in self.drn_list:
            self.drones[k].xi_1 = self.drn[k].full_traj
            # Extract positions and velocities from the model's solution
            x_position = self.drones[k].xi_1[0][0]
            y_position = self.drones[k].xi_1[1][0]
            z_position = self.drones[k].xi_1[2][0]
            x_velocity = self.drones[k].xi_1[3][0]
            y_velocity = self.drones[k].xi_1[4][0]
            z_velocity = self.drones[k].xi_1[5][0]
            # Update the initial conditions for the next iteration
            state = [x_position, y_position, z_position, x_velocity, y_velocity, z_velocity]
            state = add_state_noise(state)  # Add noise to the state
            speed = np.sqrt(x_velocity**2 + y_velocity**2 + z_velocity**2)
            print("Drone %d speed: %.2f" % (k, speed))
            self.drones[k].traj.append(state)
            self.drones[k].state = list2state(state)

    def check_collisions(self):
        # Finding the drones in proximity
        collided_drones = set() # Set of collided drones
        for i in range(len(self.drn_list)):
            for j in range(i + 1, len(self.drn_list)):
                d1 = self.drn_list[i]
                d2 = self.drn_list[j]
                d = dist_squared(self.drones[d1].state, self.drones[d2].state)
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
        self.drones[d1].alive = 0
        self.drones[d2].alive = 0
        self.drones[d1].mission_status = "collided"
        self.drones[d2].mission_status = "collided"

    def seconds_from_today(self, iteration):
        s = iteration * self.delta_t
        # datetime.datetime.fromtimestamp(ms/1000.0)
        ts = self.sim_start.timestamp() + s
        mission_start = datetime.datetime.fromtimestamp(ts)
        s = mission_start.strftime("%Y-%m-%d %H:%M:%S")
        s = mission_start.strftime("%Y-%m-%d")
        return mission_start

    def make_full_traj(self, xi):
        # Making a full trajectories out of the initial state for initialization
        x = [xi['x'] for n in range(self.N)]
        y = [xi['y'] for n in range(self.N)]
        z = [xi['z'] for n in range(self.N)]
        xdot = [xi['xdot'] for n in range(self.N)]
        ydot = [xi['ydot'] for n in range(self.N)]
        zdot = [xi['zdot'] for n in range(self.N)]
        return [x, y, z, xdot, ydot, zdot]

    def dummy_obstacles(self):
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

def add_state_noise(state):
    """
    Add multivariate normal noise to drone state.

    Args:
        state: List [x, y, z, xdot, ydot, zdot] or dict with these keys
        Q: 6x6 covariance matrix. If None, uses default diagonal matrix

    Returns:
        Noisy state in the same format as input (list or dict)
    """
    sigma_pos = 0.05 #0.01-0.1
    sigma_vel = 0.01 #0.01-0.05
    sigma_pv = 0.002
    Q = np.array([[sigma_pos*sigma_pos,    0,    0,  sigma_pv,    0,    0],
                  [   0, sigma_pos*sigma_pos,    0,     0, sigma_pv,    0],
                  [   0,    0, sigma_pos*sigma_pos,     0,    0, sigma_pv],
                  [sigma_pv,    0,    0, sigma_vel*sigma_vel,    0,    0],
                  [   0, sigma_pv,    0,     0, sigma_vel*sigma_vel,    0],
                  [   0,    0, sigma_pv,     0,    0, sigma_vel*sigma_vel]], dtype=np.float64)
    Q = np.array(Q, dtype=np.float64)
    # Convert dict to list if needed
    # state_array = np.array([state['x'], state['y'], state['z'],
    #                           state['xdot'], state['ydot'], state['zdot']], dtype=np.float64)
    state_array = np.array(state, dtype=np.float64)
    # Call the Numba-optimized function
    noisy_state = state_array + np.random.multivariate_normal(np.zeros(6), Q)
    noisy_state = clip_speed(noisy_state, 10)  # Clip speed to max 10 m/s
    # Convert back to original format
    return noisy_state

@njit
def clip_speed(state, max_speed):
    """
    state: [x, y, z, vx, vy, vz]
    max_speed: maximum allowed speed (m/s)
    """
    pos = state[:3]
    vel = state[3:6]

    current_speed = np.linalg.norm(vel)

    if current_speed > max_speed:
        # Scale velocity to max_speed
        vel = vel * (max_speed / current_speed)

    return np.concatenate((pos, vel))

def calculate_stopping_point(drone_state, max_acc):
        """
        Calculate the stopping point of a drone using maximum deceleration.

        Parameters:
        -----------
        drone_state : dict
            Dictionary with keys ['x', 'y', 'z', 'xdot', 'ydot', 'zdot']
            representing position and velocity in each axis
        max_acc : float
            Maximum acceleration/deceleration magnitude (positive value)

        Returns:
        --------
        dict
            Dictionary with keys ['x', 'y', 'z'] representing the stopping point
        """
        max_acc *= 0.8  # Use 80% of max acceleration
        # Extract current position and velocity
        x, y, z = drone_state['x'], drone_state['y'], drone_state['z']
        vx, vy, vz = drone_state['xdot'], drone_state['ydot'], drone_state['zdot']

        # Calculate current speed (magnitude of velocity vector)
        current_speed = math.sqrt(vx**2 + vy**2 + vz**2)

        # If drone is already stopped, return current position
        if current_speed == 0:
            return {'x': x, 'y': y, 'z': z}

        # Calculate time to stop using: t = v / a
        time_to_stop = current_speed / max_acc

        # Calculate stopping distance in each direction
        # Using kinematic equation: s = v*t - 0.5*a*t^2
        # Since we're decelerating opposite to velocity direction:
        # s = v*t - 0.5*a*t^2 where a = max_acc in direction of velocity

        # Unit velocity vector (direction of motion)
        vx_unit = vx / current_speed
        vy_unit = vy / current_speed
        vz_unit = vz / current_speed

        # Distance traveled during deceleration
        stopping_distance = current_speed * time_to_stop - 0.5 * max_acc * time_to_stop**2
        # This simplifies to: stopping_distance = current_speed^2 / (2 * max_acc)
        stopping_distance = current_speed**2 / (2 * max_acc)

        # Calculate stopping point
        stop_x = x + vx_unit * stopping_distance
        stop_y = y + vy_unit * stopping_distance
        stop_z = z + vz_unit * stopping_distance

        return Point(stop_x, stop_y, stop_z)

def pause():
    programPause = input("Press the <ENTER> key to continue...")

def main():
    optimization = simulator()
    optimization.m_start_simulation()

main()