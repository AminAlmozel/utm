import pickle as pkl
import numpy as np

from datetime import timedelta
import random

import environment

def main():
    env = environment.env()

    lam = 0.0005 # lambda, the rate for the poisson distribution
    dt = 0.1
    timesteps = int(1 * 60 * 60 / dt)
    traffic = generate_traffic_schedule(env, timesteps)
    traffic = sort_traffic(traffic)
    # traffic = generate_vehicle_traffic(lam)        # Generate traffic data
    write_file("traffic", traffic) # Write the generated traffic into a pickle file (.pkl)
    traffic = read_vehicle_traffic()  # Load the traffic data from file
    # print(traffic)                 # Print the loaded traffic data
    min_timedelta = 21.6 #s
    # traffic = threshold_traffic(traffic, min_timedelta) # Removes any vehicles which have a time delta of less than min_timedelta
    # traffic_stats(traffic)

def generate_traffic_schedule(env, timesteps):
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
    lam = 0.00002
    traffic = generate_vehicle_traffic(lam, timesteps)
    for i, time in enumerate(traffic):
        mission = create_firefighting_mission(time, env)
        firefighting.append(mission)

    # Security

    # Recreational
    create_recreational_mission(time, env)

    # Solar panels inspection and maintenance
    create_inpsection_mission(time, env)

    # Research
    create_research_mission(time, env)

    return deliveries + firefighting

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
    dt = 0.1
    p = restaurant.geometry
    pi = [p.x, p.y, 20]
    delivery_location, pf, vf = env.random_delivery_location(time, restaurant)
    vi = env.random_state(time, restaurant)
    xi = list2state(pi + vi)
    xf = list2state(pf + vf)
    mission = {"mission": [restaurant, delivery_location], "state": [xi, xf], "time": time}
    destinations = [restaurant, delivery_location, restaurant]
    mission = create_mission(xi, [], time, "delivery", "in progress", destinations, time)
    return mission

def create_firefighting_mission(time, env):
    fire_station = env.fire_station.iloc[0]
    temp = fire_station.geometry
    zi = random.randint(0, 30)
    pi = [temp.x, temp.y, zi]
    vi = env.random_state(0, fire_station)
    xi = pi + vi

    # Choose a random house
    house, pf = env.random_house(12, "placeholder")
    vf = env.random_state(0, house)
    xf = pf + vf

    xi = list2state(xi)
    # Create a drone or multiple to go from the fire station to the location of the fire
    dt = 0.1
    wait = 3 * 60 / dt
    destinations = [fire_station, house, wait, fire_station]
    mission = create_mission(xi, [], time, "firefighting", "in progress", destinations, time)
    return mission

def create_recreational_mission(time, env):
    pass

def create_inpsection_mission(time, env):
    pass

def create_research_mission(time, env):
    pass

def create_mission(xi, waypoints, n, type, status, destinations, time):
    # Dictionary that contains all the data for the drone, except for the drone object
    d = {"id": -1,
                "born": time,
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
    missions_sorted = sorted(traffic, key=lambda mission: mission["born"])
    for mission in missions_sorted:
        if mission["mission"]["type"] == "delivery":
            restaurant = mission["mission"]["destination"][0]
            delivery_location = mission["mission"]["destination"][1]
            time = mission["born"]
            print(restaurant["name"], "\tto\t" + delivery_location["name"] + "\tat\t", end="")
            # print("%.1f" % (time * dt))
            milliseconds_to_hours(time * 100)
        if mission["mission"]["type"] == "firefighting":
            house = mission["mission"]["destination"][1]
            print("Fire at\t" + house["name"] + "\tat\t", end="")
            dt = 0.1
            milliseconds_to_hours(time * 100)
    return missions_sorted

def write_file(filename, traffic):
    # Save traffic data to file
    with open("missions/" + filename + '.pkl', 'wb') as fp:
        pkl.dump(traffic, fp, protocol=pkl.HIGHEST_PROTOCOL)

def read_vehicle_traffic():
    """
    Reads vehicle traffic data from a pickle file.
    """
    filename = "traffic"
    print("Reading traffic from file")
    with open("missions/" + filename + '.pkl', 'rb') as traffic_file:
        traffic = pkl.load(traffic_file)
    return traffic

def traffic_stats(traffic):
    if len(traffic) <= 1:
        return
    dt = 0.1
    # Analyze time gaps between consecutive vehicles on the eastbound lane
    diff = np.ediff1d(traffic)       # Differences between consecutive arrival indices
    sorted_diff = np.sort(diff)         # Sort the time differences

    # Print traffic statistics
    print(sorted_diff)
    print("Total number of vehicles: %.0f in 4 hours" % len(traffic))
    print("A plane flying at 500 knots will cross 3 nautical miles in 21.6s")
    print("Minimum: \t%.1fs" % (diff.min() * dt))
    print("Mean:    \t%.1fs" % (diff.mean() * dt))
    print("Median:  \t%.1fs" % (np.median(diff) * dt))
    print("Std Dev: \t%.1f" % (diff.std() * dt))

def threshold_traffic(traffic, min_delta):
    def filter_traffic(arrivals, min_delta):
        dt = 0.1
        min_delta = int(min_delta / dt)
        if not arrivals:
            return []

        filtered = [arrivals[0]]
        for time in arrivals[1:]:
            if time - filtered[-1] >= min_delta:
                filtered.append(time)
        return filtered

    return [filter_traffic(arrivals, min_delta) for arrivals in traffic]

def list2state(values):
    keys = ['x', 'y', 'z', 'xdot', 'ydot', 'zdot']
    return dict(zip(keys, values))

def state2list(values):
    return [values["x"], values["y"], values["z"]]

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

# Run the main function
main()