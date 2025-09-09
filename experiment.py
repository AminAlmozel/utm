import datetime

from shapely.geometry import LineString

from sim_io import myio as io
import sampling_pathplanning as spp
from util import *

print("Loading path planning")
ppp = spp.sampling_pp()
print("Finished loading path planning")
iteration = 0
today = datetime.date.today()
# datetime(year, month, day, hour, minute, second, microsecond)
sim_start = datetime.datetime(today.year, today.month, today.day, 12, 0, 0)

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

def create_all_missions():
    """
    Reads the missions file and creates all missions using existing code.
    """
    print("Loading missions")
    missions = io.import_missions()
    created_missions = []
    for idx, mission in enumerate(missions):
        print("Mission Number:", idx)
        created = create_mission(mission, idx)
        created_missions.append(created)
        if mission["mission"]["type"] == "delivery":
            start = mission["mission"]["destination"][0]["data"]
            end = mission["mission"]["destination"][1]["data"]
        else:
            start = {"name": "N/A"}
            end = {"name": "N/A"}
        print_mission_details(created, start, end)
        # print(f"Created mission: {created}")
        if idx == 20:
            break
    return created_missions

def create_mission(mission, drn_count):
    mission_type = mission["mission"]["type"]
    print(mission_type)
    if mission_type == "delivery":
        return create_delivery_drone(mission, drn_count)
    if mission_type == "firefighting":
        # create_firefighting_drone(mission, drn_count)
        pass
    if mission_type == "perimeter_patrol":
        return create_general_mission(mission, drn_count)
    if mission_type == "traffic_monitoring":
        return create_traffic_monitoring_mission(mission, drn_count)
    if mission_type == "research":
        return create_general_mission(mission, drn_count)
    if mission_type == "inspection":
        return create_general_mission(mission, drn_count)
    if mission_type == "recreational":
        return create_general_mission(mission, drn_count)

def create_delivery_drone(mission, drn_count):
    xi = mission["mission"]["destination"][0]["geometry"]
    xf = mission["mission"]["destination"][1]["geometry"].centroid
    pi = [xi.x, xi.y]
    pf = [xf.x, xf.y]
    waypoints = ppp.create_trajectory([pi, pf])
    waypoints = ppp.round_trip(waypoints)
    destination = int((len(waypoints) - 1) / 2)
    # if destination == 0:
    #     destination += 1 # To resolve an issue with firefighting drones
    destinations = [waypoints[0], waypoints[destination], waypoints[0]]
    return create_drone(drn_count, mission["state"], waypoints, iteration, "delivery", "in progress", destinations)

def create_traffic_monitoring_mission(mission, drn_count):
    return create_general_mission(mission, drn_count)

def create_general_mission(mission, drn_count):
    return create_drone(drn_count,
                        mission["state"],
                        mission["mission"]["waypoints"],
                        iteration, mission["mission"]["type"],
                        mission["mission"]["status"],
                        mission["mission"]["destination"])

def create_drone(id, xi, waypoints, n, type, status, destinations):
    # K += 1
    birthday = seconds_from_today(n)
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
    # drones.append(d)
    # traj_0 = make_full_traj(xi)
    # drones[-1].xi_1 = traj_0
    # # Drone object
    # drn.append(drone.drone())
    # drn_list.append(len(drn) - 1)
    return d

def seconds_from_today(iteration):
    delta_t = 0.1
    s = iteration * delta_t
    # datetime.datetime.fromtimestamp(ms/1000.0)
    ts = sim_start.timestamp() + s
    mission_start = datetime.datetime.fromtimestamp(ts)
    s = mission_start.strftime("%Y-%m-%d %H:%M:%S")
    s = mission_start.strftime("%Y-%m-%d")
    return mission_start

def print_mission_details(mission, start, end):
    # mission is a DroneData object
    mission_type = mission.mission_type
    waypoints = mission.mission_waypoints
    start = start["name"]
    end = end["name"]
    if waypoints and len(waypoints) >= 2:
        coords = [(wp['x'], wp['y'], wp['z']) for wp in waypoints if isinstance(wp, dict)]
        if len(coords) >= 2:
            linestring = LineString(coords)
            length = linestring.length
            print(f"Mission Type: {mission_type}")
            print(f"Start Point: {start}")
            print(f"End Point: {end}")
            print(f"Path Length: {length:.2f}")
        else:
            print("Waypoints do not contain enough valid coordinates.")
    else:
        print("Waypoints not available or insufficient for path length calculation.")

def choose_missions(missions, ids):
    selected = [missions[i] for i in ids if i < len(missions)]
    all_missions = [m.mission_waypoints for m in selected]
    geoms = [traj_to_linestring([state2list(wp) for wp in mission]) for mission in all_missions]
    return geoms

def main():
    missions = create_all_missions()
    mission_geometries = choose_missions(missions, [i for i in range(15)])
    io.write_geom(transform_meter_global(mission_geometries), "selected_missions.geojson", "white")


main()