import pickle as pkl
import numpy as np
import geopandas as gp
from shapely.geometry import Point, LineString

from datetime import timedelta
import random
import copy

import environment
from sim_io import myio as io
from util import *

def main():
    env = environment.env()
    lam = 0.0005 # lambda, the rate for the poisson distribution
    dt = 0.1
    timesteps = int(1 * 60 * 60 / dt)
    traffic = generate_traffic_schedule(env, timesteps)
    traffic = sort_traffic(traffic)
    # traffic = generate_vehicle_traffic(lam)        # Generate traffic data
    filename = "traffic"
    io.write_pickle("missions/" + filename + '.pkl', traffic) # Write the generated traffic into a pickle file (.pkl)
    traffic = io.read_pickle("missions/" + filename + '.pkl') # Load the traffic data from file
    # min_timedelta = 21.6 #s
    # traffic = filter_traffic(traffic, min_timedelta) # Removes any vehicles which have a time delta of less than min_timedelta
    # traffic_stats(traffic)

def generate_traffic_schedule(env, timesteps):
    display = []
    deliveries = []
    # Deliveries
    for index, restaurant in env.restaurants.iterrows():
        lam = restaurant["freq"]
        traffic = generate_vehicle_traffic(lam, timesteps)
        for i, time in enumerate(traffic):
            mission = create_delivery_mission(time, env, restaurant)
            deliveries.append(mission)
    # Firefighting
    firefighting = []
    # lam = 0.00002
    lam = 0.001
    traffic = generate_vehicle_traffic(lam, timesteps)
    for i, time in enumerate(traffic):
        mission = create_firefighting_mission(time, env)
        firefighting.append(mission)

    # Security
    security = []
    # lot = io.import_parking()
    # mission = create_parkinglot_monitoring(time, env, lot)

    # Perimeter patrol
    time = random.randint(0, timesteps) # To get one random perimeter section mission
    perimeter = io.import_perimeter()
    section = random.randint(0, len(perimeter)/2-1)
    mission = create_perimeter_patrol_mission(time, perimeter, section)
    security.append(mission)

    # Traffic zones
    # time = random.randint(0, timesteps) # To get one random perimeter section mission
    zones = io.import_traffic_zones()
    intersection = random.randint(0, len(zones)-1)
    mission = create_traffic_monitoring_mission(time, zones, intersection)
    security.append(mission)

    # Recreational
    recreational = []
    time = random.randint(0, timesteps) # To get one random inspection mission
    field = io.import_recreational()
    mission = create_recreational_mission(time, field)
    recreational.append(mission)

    # Solar panels inspection and maintenance
    inspection = []
    time = random.randint(0, timesteps) # To get one random inspection mission
    sites = io.import_inspection()
    inspection_site = random.randint(0, len(sites)-1)
    mission = create_inpsection_mission(time, sites, inspection_site)
    inspection.append(mission)

    # Research
    research = []
    time = random.randint(0, timesteps) # To get one random perimeter section mission
    sites = io.import_research()
    research_site = random.randint(0, 1)
    mission = create_research_mission(time, sites, research_site)
    research.append(mission)

    # display = transform_meter_global(display)
    # io.write_geom(display, "missions", "white")
    # return security + recreational + inspection + research
    return recreational + inspection + research

def generate_vehicle_traffic(lam, timesteps):
    """
    Simulates vehicle arrivals at an intersection from two directions (east and north),
    saves the arrival times into a pickle file, and prints summary statistics.
    """
    # generate traffic where
    # traffic = [[t_0, t_1, ...]
    #           [t_0, t_1, ...]]
    # and t_i is the time at at which vehicle i enters the intersection.
    # the first list (traffic[0][i]) is the vehicle i entering the intersection going east (x-axis),
    # and (traffic[1][i]) is the vehicle i entering the intersection going east (y-axis)

    # Generate Poisson-distributed arrivals for each direction
    schedule = np.random.poisson(lam=lam, size=timesteps)

    # Get time indices when vehicles arrive (i.e., when the Poisson output != 0)
    traffic = np.where(schedule != 0)[0].tolist() # Arrival times for eastbound vehicles

    # traffic_stats(traffic)
    return traffic

def create_delivery_mission(time, env, restaurant):
    # Get initial position and velocity
    restaurant_pos = [restaurant.geometry.x, restaurant.geometry.y, 20]
    initial_velocity = env.random_state(time, restaurant)
    initial_state = list2state(restaurant_pos + initial_velocity)

    # Get delivery location and desired final state
    delivery_location, delivery_pos, delivery_velocity = env.random_delivery_location(time, restaurant)
    final_state = list2state(delivery_pos + delivery_velocity)

    # Create state dictionaries with geometry attached
    start = copy.deepcopy(initial_state)
    end = copy.deepcopy(final_state)
    start["geometry"] = restaurant["geometry"]
    start["data"] = restaurant
    end["geometry"] = delivery_location["geometry"]
    end["data"] = delivery_location

    # Define the delivery mission
    destinations = [start, end, start]  # Go there and back
    mission = create_mission(initial_state, [], time, "delivery", "in progress", destinations, time)

    return mission

def create_firefighting_mission(time, env):
    # Get fire station info
    fire_station = env.fire_station.iloc[0]
    station_pos = [fire_station.geometry.x, fire_station.geometry.y, random.randint(0, 30)]
    station_velocity = env.random_state(time, fire_station)
    initial_state = list2state(station_pos + station_velocity)

    # Choose a random house for the fire
    house, house_pos = env.random_house(time, "placeholder")
    house_velocity = env.random_state(time, house)
    final_state = list2state(house_pos + house_velocity)

    # Attach geometry to states
    start = copy.deepcopy(initial_state)
    end = copy.deepcopy(final_state)
    start["geometry"] = fire_station["geometry"]
    start["data"] = fire_station
    end["geometry"] = house["geometry"]
    end["data"] = house

    # Time to stay at the fire (e.g., 3 minutes)
    dt = 0.1
    wait_duration = 3 * 60 / dt # seconds

    # Define mission route
    destinations = [start, end, wait_duration, start]
    radius = 50
    wp = sample_points_in_circle(house["geometry"].centroid, radius)
    waiting_times = generate_waiting_times(len(wp))
    z = 30
    waypoints = []
    for i in range(len(wp)):
        p = [wp[i].x, wp[i].y] + [z, 0, 0, 0]
        waypoints.append(list2state(p))
        if i == 0 or i == len(wp) - 1:
            continue
        waypoints.append(waiting_times[i])
    waypoints = [initial_state] + [final_state] + waypoints + [initial_state]
    mission = create_mission(initial_state, waypoints, time, "firefighting", "in progress", destinations, time)
    return mission

def create_recreational_mission(time, env):
    field = env["geometry"]
    wp = create_random_waypoints(field)
    waiting_times = generate_waiting_times(len(wp))
    z = 30
    waypoints = []
    for i in range(len(wp)):
        p = [wp.iloc[i].x, wp.iloc[i].y] + [30, 0, 0, 0]
        waypoints.append(list2state(p))
        if i == 0 or i == len(wp) - 1:
            continue
        waypoints.append(waiting_times[i])
    pi = field[0]
    xi = list2state([pi.x, pi.y] + [30, 0, 0, 0])
    destinations = [waypoints[0], waypoints[-3], waypoints[0]]
    waypoints = [xi] + waypoints + [xi]
    mission = create_mission(xi, waypoints, time, "recreational", "in progress", destinations, time)
    return mission

def create_inpsection_mission(time, env, inspection_site):
    points = env[inspection_site]["geometry"]
    wp = create_random_waypoints(points)
    waiting_times = generate_waiting_times(len(wp))
    z = 30
    waypoints = []
    for i in range(len(wp)):
        p = [wp.iloc[i].x, wp.iloc[i].y] + [z, 0, 0, 0]
        waypoints.append(list2state(p))
        if i == 0 or i == len(wp) - 1:
            continue
        waypoints.append(waiting_times[i])
    pi = points[0]
    xi = list2state([pi.x, pi.y] + [z, 0, 0, 0])
    destinations = [waypoints[0], waypoints[-3], waypoints[0]]
    waypoints = [xi] + waypoints + [xi]
    mission = create_mission(xi, waypoints, time, "inspection", "in progress", destinations, time)
    return mission

def create_research_mission(time, env, research_site):
    polygon = env["geometry"][research_site]
    wp = sample_grid_points(polygon, 50)
    waiting_times = generate_waiting_times(len(wp))
    z = 30
    waypoints = []
    for i in range(len(wp)):
        p = [wp[i].x, wp[i].y] + [z, 0, 0, 0]
        waypoints.append(list2state(p))
        if i == 0 or i == len(wp) - 1:
            continue
        waypoints.append(waiting_times[i])
    takeoff_point = int(len(env["geometry"])/2 + research_site)
    pi = env["geometry"][takeoff_point]
    xi = list2state([pi.x, pi.y] + [z, 0, 0, 0])
    destinations = [waypoints[0], waypoints[-3], waypoints[0]]
    waypoints = [xi] + waypoints + [xi]
    mission = create_mission(xi, waypoints, time, "research", "in progress", destinations, time)
    return mission

def create_parkinglot_monitoring(time, env, lot):
    pass

def create_perimeter_patrol_mission(time, env, section):
    perimeter = env["geometry"][section]
    wp = sample_points_from_linestring(perimeter)
    waiting_times = generate_waiting_times(len(wp))
    z = 30
    waypoints = []
    for i in range(len(wp)):
        p = [wp[i].x, wp[i].y] + [z, 0, 0, 0]
        waypoints.append(list2state(p))
        if i == 0 or i == len(wp) - 1:
            continue
        waypoints.append(waiting_times[i])
    takeoff_point = int(len(env["geometry"])/2 + section)
    pi = env["geometry"][takeoff_point]
    xi = list2state([pi.x, pi.y] + [z, 0, 0, 0])
    destinations = [waypoints[0], waypoints[-3], waypoints[0]]
    waypoints = [xi] + waypoints + [xi]
    mission = create_mission(xi, waypoints, time, "perimeter_patrol", "in progress", destinations, time)
    return mission

def create_traffic_monitoring_mission(time, env, intersection):
    zone = env["geometry"][intersection]
    radius = 50
    wp = sample_points_in_circle(zone, radius)
    waiting_times = generate_waiting_times(len(wp))
    z = 30
    waypoints = []
    for i in range(len(wp)):
        p = [wp[i].x, wp[i].y] + [z, 0, 0, 0]
        waypoints.append(list2state(p))
        if i == 0 or i == len(wp) - 1:
            continue
        waypoints.append(waiting_times[i])
    takeoff_point = 0
    pi = env["geometry"][takeoff_point]
    xi = list2state([pi.x, pi.y] + [z, 0, 0, 0])
    destinations = [waypoints[0], waypoints[-3], waypoints[0]]
    waypoints = [xi] + waypoints + [xi]
    mission = create_mission(xi, waypoints, time, "traffic_monitoring", "in progress", destinations, time)
    return mission

def create_mission(xi, waypoints, n, type, status, destinations, time):
    # Dictionary that contains all the data for the drone, except for the drone object
    d = {"id": -1,
                "birthday": time,
                "iteration": time,
                "trajs": [],
                "alive": 1, # Alive, 0 is dead
                "state": xi,
                "factor": 0,
                "mission": {
                    "type": type,
                    "destination": destinations, # Contains the destinations and waiting times
                    "progress": 0,
                    "waypoints": waypoints, # Deliver to the final destination, then come back to the original place
                    "status":status
                }}
    return d

def sort_traffic(traffic):
    missions_sorted = sorted(traffic, key=lambda mission: mission["iteration"])
    for mission in missions_sorted:
        time = mission["iteration"]
        if mission["mission"]["type"] == "delivery":
            restaurant = mission["mission"]["destination"][0]["data"]
            delivery_location = mission["mission"]["destination"][1]["data"]
            print(restaurant["name"], "\tto\t" + delivery_location["name"] + "\tat\t", end="")
            # print("%.1f" % (time * dt))
            milliseconds_to_hours(time * 100)
        if mission["mission"]["type"] == "firefighting":
            house = mission["mission"]["destination"][1]["data"]
            print("Fire at\t" + house["name"] + "\tat\t", end="")
            dt = 0.1
            milliseconds_to_hours(time * 100)
    return missions_sorted

def traffic_stats(traffic):
    if len(traffic) <= 1:
        return
    dt = 0.1
    # Analyze time gaps between consecutive vehicles
    times = []
    for vehicle in traffic:
        times.append(vehicle["iteration"])
    times = np.array(times)
    diff = np.ediff1d(times)       # Differences between consecutive arrival indices
    sorted_diff = np.sort(diff)         # Sort the time differences

    # Print traffic statistics
    print(sorted_diff)
    print("Total number of vehicles: %.0f in 4 hours" % len(times))
    print("A plane flying at 500 knots will cross 3 nautical miles in 21.6s")
    print("Minimum: \t%.1fs" % (diff.min() * dt))
    print("Mean:    \t%.1fs" % (diff.mean() * dt))
    print("Median:  \t%.1fs" % (np.median(diff) * dt))
    print("Std Dev: \t%.1f" % (diff.std() * dt))

def filter_traffic(arrivals, min_delta):
    dt = 0.1
    min_delta = int(min_delta / dt)
    if not arrivals:
        return []
    # print(arrivals)
    filtered = [arrivals[0]]
    for arrival_dict in arrivals[1:]:
        if arrival_dict["iteration"] - filtered[-1]["iteration"] >= min_delta:
            filtered.append(arrival_dict)
    return filtered

def milliseconds_to_hours(time):
    # Convert to timedelta
    td = timedelta(milliseconds=time)

    # Get total seconds and break into hh:mm:ss
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600 + 12
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60

    # Format as hh:mm:ss with leading zeros
    formatted_time = f"{hours:02}:{minutes:02}:{seconds:02}pm"

    print(formatted_time)

def sample_grid_points(polygon, spacing):
    minx, miny, maxx, maxy = polygon.bounds
    points = []
    y = miny
    row = 0
    while y <= maxy:
        x_range = np.arange(minx, maxx, spacing)
        if row % 2 == 1:  # zigzag for efficient pathing
            x_range = x_range[::-1]
        for x in x_range:
            p = Point(x, y)
            if polygon.contains(p):
                points.append(p)
        y += spacing
        row += 1
    return points

def sample_random_points(polygon, n_points):
    points = []
    minx, miny, maxx, maxy = polygon.bounds
    while len(points) < n_points:
        p = gp.Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
        if polygon.contains(p):
            points.append(p)
    return points

def create_random_waypoints(waypoints, avg_points=7, variation=2, altitude=40):
    """
    points: list of (x, y) or (x, y, z) tuples
    avg_points: average number of waypoints desired
    variation: maximum random variation in number of points
    altitude: altitude to assign if input points are 2D
    """
    points = range(1, len(waypoints))
    n_points = max(1, avg_points + random.randint(-variation, variation))
    selected_points = random.sample(points, min(n_points, len(points)))
    selected_points = [0] + selected_points + [0]
    return waypoints[selected_points]

def generate_waiting_times(n, mean=80, variance=40, min_time=30):
    """
    Generate n random integer waiting times with given mean and variance.

    Parameters:
        n (int): Number of waiting times.
        mean (float): Desired mean.
        variance (float): Desired variance.
        min_time (int): Minimum allowed waiting time.

    Returns:
        List[int]: Random waiting times.
    """
    std_dev = np.sqrt(variance)
    times = np.random.normal(loc=mean, scale=std_dev, size=n)

    # Round to nearest integer and clip to min_time
    times = np.round(times).astype(int)
    times = np.clip(times, min_time, None)

    return times.tolist()

def sample_points_from_linestring(line: LineString, avg_points=7, variation=2, mode="random") -> list[Point]:
    """
    Sample points along a LineString.

    Parameters:
        line (LineString): The input line to sample from.
        n_points (int): Number of points to sample.
        mode (str): 'even' for equally spaced, 'random' for randomly spaced.

    Returns:
        List[Point]: Sampled Shapely Point objects.
    """
    n_points = max(1, avg_points + random.randint(-variation, variation))
    if line.length == 0:
        return [Point(line.coords[0])] * n_points

    distances = []

    if mode == "even":
        distances = np.linspace(0, line.length, n_points + 2)[1:-1]  # exclude endpoints
    elif mode == "random":
        distances = np.sort(np.random.uniform(0, line.length, n_points))
    else:
        raise ValueError("Mode must be 'even' or 'random'.")
    return [line.interpolate(d) for d in distances]

def sample_points_in_circle(center: Point, radius: float, avg_points=7, variation=2) -> list[Point]:
    """
    Sample points uniformly distributed within a circle around a center point.

    Parameters:
        center (tuple): The (x, y) coordinates of the center point.
        radius (float): Radius of the circle.
        n_points (int): Number of points to sample.

    Returns:
        List[Point]: List of Shapely Point objects.
    """

    n_points = max(1, avg_points + random.randint(-variation, variation))
    cx, cy = center.x, center.y
    points = []

    for _ in range(n_points):
        r = radius * np.sqrt(np.random.uniform(0, 1))  # ensures uniform density
        theta = np.random.uniform(0, 2 * np.pi)
        x = cx + r * np.cos(theta)
        y = cy + r * np.sin(theta)
        points.append(Point(x, y))

    return points

def display_mission(mission):
    points = []
    for waypoint in mission["mission"]["waypoints"]:
        if isinstance(waypoint, int):
            continue
        wp = state2list(waypoint)
        points.append(wp)
    ls = traj_to_linestring(points)
    return ls

# Run the main function
main()