import pandas as pd
import geopandas as gp
import numpy as np

from shapely.geometry import LineString, Point, Polygon, MultiPolygon, box
from shapely.ops import nearest_points
import glob
from math import radians, cos, sin, asin, atan2, sqrt, pi, ceil, exp, log
from time import time

import astar_uam as astar
import myio
import exploration
import terrain
import multidrone

import polygon_pathplanning
import skimage as sk

from matplotlib import cm
from matplotlib.ticker import LinearLocator
import matplotlib.pyplot as plt

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# 'EPSG:2154' for France
# epsg=4326  for equal area

class uam(myio.myio, exploration.exploration, terrain.terrain, multidrone.multi):
    def __init__(self):
        self.logging = True
        self.radius = 700
        self.disc = 7
        self.c = 0.0000124187
        self.m_adj = []
        self.m_heur = []
        self.se_ls = np.zeros((1,2)) # start end, concatenated with land spots
        self.fa = np.zeros((1,2))
        # self.colors = ["#0072BD", "#D95319", "#EDB120", "#7E2F8E", "#77AC30", "#4DBEEE", "#A2142F"]
        self.colors = ["#FF0000", "#00FF00", "#0000FF", "#00FFFF", "#FF00FF", "#FFFF00", "#000000"]
        self.colors = ["#4DBEEE", "#A2142F", "#FF00FF", "#7E2F8E", "#EDB120", "#D95319", "#77AC30"] # p1
        self.colors = ["#4DBEEE", "#FF00FF", "#A2142F", "#FF0000", "#0072BD", "#D95319", "#77AC30"] # p2

        self.project = "kaust/" # Which folder the data is read from
        self.terrain_file = "N21E039.hgt"
        self.lat_ = 21
        self.lon_ = 39
        self.import_hospitals()
        # self.insert_point(43.543981, 5.760011) # Given destination (in Toulouse)
        # self.insert_point(39.105643, 22.314860) # Kaust medical clinic

        self.import_landing()
        # self.import_lidar()
        self.visualize_landing()
        self.import_forbidden()
        self.import_terrain()
        # self.import_roads()
        self.import_factories()

        # self.use_center_of_ls()
        # self.use_ppp() # Polygon Path Planning
        self.use_ppp_jeddah() # Trajectory planning in Jeddah
        # self.test2()
        # self.sandbox_multi()
        routes = self.read_highways()
        self.write_highways(routes, "highways", "magenta")

    def sandbox(self):
        # Adding forbidden area for testing
        boundary = [5.497455, 43.519508, 5.521531, 43.493549]
        boundary = [5.530801, 43.497565, 5.508141, 43.543995]
        bbox = box(boundary[0], boundary[1], boundary[2], boundary[3])
        bbox = gp.GeoSeries(bbox)
        self.save_geoseries(bbox, "bbox", "red")
        self.add_to_forbidden(bbox)
        # # New path
        # print("New path")
        # path = self.find_path(6, 0)
        # self.write_path(path, "new_path", "#7E2F8E")
        # traj = self.path_to_traj(path)
        # self.measure_risk_airspace(traj)
        # # New trajectory
        # print("New trajectory")
        # path = self.find_path(6, 0)
        # traj = self.tune_path(path)
        # self.write_traj(traj, "new_traj", self.colors[0])
        # self.measure_risk_airspace(traj)
        # Baseline
        print("Baseline")
        path = self.find_baseline(6, 0)
        self.write_path(path, "baseline_traj", self.colors[2])
        traj = self.path_to_traj(path)
        self.measure_risk_airspace(traj)

        # Almost straight line
        print("Almost straight line")
        path = [6, 385, 522, 401, 590, 122, 578, 761, 182, 91, 235, 153, 29, 853, 236, 214, 213, 215, 158, 243, 237, 84, 184, 376, 255, 687, 0]
        traj = self.path_to_traj(path)
        self.measure_risk_airspace(traj)
        self.write_traj(traj, "straight_line", self.colors[3])

        # Tuned almost straight line
        print("Tuned almost straight line")
        traj = self.tune_path(path)
        self.measure_risk_airspace(traj)
        self.write_traj(traj, "straight_line_tuned", self.colors[4])

        # Straight line
        print("Straight line")
        path = [6, 0]
        traj = self.path_to_traj(path)
        self.measure_risk_airspace(traj)
        self.write_traj(traj, "line", self.colors[5])
        return 0

    def sandbox_multi(self):
        ppp = polygon_pathplanning.polygon_pp()
        # self.sandbox2()
        # Constructing the safe airspace
        sa = self.construct_airspace()
        # Finding the mountains
        mt = self.avoid_terrain()
        mt = gp.GeoSeries(mt)
        self.add_to_forbidden(mt)
        # Removing landing spots that are within a forbidden area
        self.remove_invalid_ls()
        sa = self.construct_airspace()
        # Trimming the airspace to the vicinity of the trajectory
        boundary = [5.7778, 43.5639, 5.4191, 43.4472]
        boundary = [5.4191, 43.4472, 5.4191, 43.4472]
        bbox = box(boundary[0], boundary[1], boundary[2], boundary[3])
        sa = sa.intersection(bbox)
        boundary = [5.7778, 43.5639, 5.4191, 43.4472]
        bbox = box(boundary[0], boundary[1], boundary[2], boundary[3])
        self.fa_gs = self.fa_gs.intersection(bbox)
        fa = self.fa_gs.unary_union # Geoseries to multipolygon
        # Approximating the polygonial airspace
        tolerance = 20 # Meters
        fa = self.simplify_polygon(fa, tolerance)

        # Outputting for testing
        self.write_geom([sa], "updated", "green")
        # sa = lidar.union(sa)
        self.write_geom([fa], "fa", "red")
        # self.write_geom([safa], "safa", "grey")
        ###
        tic = time()
        factories = self.fact_ls
        graph, ls = ppp.m_create_connectivity(sa, fa, factories)
        m_adj, m_heur = ppp.create_adjacency_matrix(graph, ls)
        self.m_adj = m_adj
        self.m_heur = m_heur
        self.ls = ls
        self.store_adjacency_matrix()
        # m_adj, m_heur, ls = self.load_adjacency_matrix()

        print("It took ", time() - tic)
        geom = self.multidrone_pathfinding(m_adj, m_heur, ls, factories)
        image = self.make_image(geom)
        tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360, endpoint=False)
        h, theta, d = sk.transform.hough_line(image, theta=tested_angles)
        fig, ax = plt.subplots()
        ax.imshow(image, cmap=cm.gray)
        print(h.shape)
        threshold = 0.3 * np.max(h)
        min_dist = 12
        min_ang = 1
        # for _, angle, dist in zip(*sk.transform.hough_line_peaks(h, theta, d, min_distance=min_dist)):
        # for _, angle, dist in zip(*sk.transform.hough_line_peaks(h, theta, d, min_angle=min_ang)):
        # for _, angle, dist in zip(*sk.transform.hough_line_peaks(h, theta, d, threshold=threshold)):
        #     (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
        #     ax.axline((x0, y0), slope=np.tan(angle + np.pi/2))

        # ax[0].imshow(image, cmap=cm.gray)
        plt.tight_layout()
        plt.show()

    def save_stuff(self):
        files = ["plot/airspace_green.geojson",
        "plot/explored_airspace.geojson",
        "plot/exp1.geojson",
        "plot/exp2.geojson",
        "plot/exp3.geojson",
        "plot/exp4.geojson",
        "plot/exp5.geojson"]
        self.combine_json_files(files, "trajectories_airspace")
        files = ["plot/wind_effect.geojson",
        "plot/buffer.geojson",
        "plot/mountains.geojson",
        "plot/venturi.geojson"]
        self.combine_json_files(files, "wind")
        files = ["plot/circles_green.geojson",
        "plot/original_path.geojson",
        "plot/original_traj.geojson"]
        self.combine_json_files(files, "original")
        # files = ["plot/airspace_red.geojson",
        # "plot/airspace_yellow.geojson",
        # "plot/airspace_green.geojson"]
        files = ["plot/airspace_yellow.geojson",
        "plot/airspace_green.geojson"]
        self.combine_json_files(files, "airspace")
        files = ["plot/airspace_green.geojson",
        "plot/original_traj.geojson",
        "plot/shifted.geojson"]
        self.combine_json_files(files, "simple_wind")
        files = ["plot/airspace_green.geojson",
        "plot/baseline_traj.geojson",
        "plot/original_path.geojson",
        "plot/line.geojson",
        "plot/straight_line.geojson"]
        self.combine_json_files(files, "paper1")
        files = ["plot/airspace_green.geojson",
        "plot/original_path.geojson",
        "plot/original_traj.geojson",
        "plot/straight_line.geojson",
        "plot/straight_line_tuned.geojson"]
        self.combine_json_files(files, "paper2")

    def use_ppp(self):
        self.radius = 700
        self.disc = 12
        ppp = polygon_pathplanning.polygon_pp()
        # self.sandbox2()
        # Constructing the safe airspace
        sa = self.construct_airspace()
        # Finding the mountains
        mt = self.avoid_terrain()
        mt = gp.GeoSeries(mt)
        self.add_to_forbidden(mt)
        # Removing landing spots that are within a forbidden area
        self.remove_invalid_ls()
        sa = self.construct_airspace()
        # Trimming the airspace to the vicinity of the trajectory
        boundary = [5.7778, 43.5639, 5.4191, 43.4472]
        bbox = box(boundary[0], boundary[1], boundary[2], boundary[3])
        sa = sa.intersection(bbox)
        self.fa_gs = self.fa_gs.intersection(bbox)
        fa = self.fa_gs.unary_union # Geoseries to multipolygon
        # Approximating the polygonial airspace
        tolerance = 30 # Meters
        fa = self.simplify_polygon(fa, tolerance)

        # safa = sa.union(fa)
        lidar = self.import_lidar()

        # Outputting for testing
        self.write_geom([sa], "updated", "green")
        # sa = lidar.union(sa)
        self.write_geom([fa], "fa", "red")
        # self.write_geom([safa], "safa", "grey")
        ###
        tic = time()
        start = [5.759125, 43.544123]
        end = [5.436492, 43.502888]
        start_end = [start, end]
        graph, ls = ppp.m_create_connectivity(sa, fa, start_end)
        m_adj, m_heur = ppp.create_adjacency_matrix(graph, ls)
        print("It took ", time() - tic)
        a = 0
        b = 1
        path = astar.a_star(m_adj, m_heur, a, b)
        traj = ppp.path_to_traj(path, ls)

        ppp.write_path(path, ls, "poly_traj", "blue")
        width = 50
        ppp.write_corridor(path, width, ls, "corridor", "green")
        traj_ls = self.traj_to_linestring(ppp.path_to_traj(path, ls))
        unknown_sections = self.scan_using_lidar(traj_ls, self.construct_airspace())
        self.write_geom(unknown_sections, "unknown_sections", "grey")

        # Exploration
        traj = ppp.path_to_traj(path, ls)
        tuned = self.modify_traj(traj, 2000)
        self.write_traj(tuned, "explored", "green")
        # Finding area to be scanned
        tuned = self.traj_to_linestring(tuned)
        unknown_sections = self.scan_using_lidar(tuned, self.construct_airspace())
        self.write_geom(unknown_sections, "unknown_sections", "yellow")

        # Straight line
        # Cleaning airspace
        gdf = self.gdf
        self.gdf = self.gdf[0:2]
        sa = self.construct_airspace()
        graph, ls = ppp.m_create_connectivity(sa, fa, start_end)
        m_adj, m_heur = ppp.create_adjacency_matrix(graph, ls)
        print("It took ", time() - tic)
        a = 0
        b = 1
        path = astar.a_star(m_adj, m_heur, a, b)
        print(path)
        ppp.write_path(path, ls, "straight_avoid", "green")
        traj_ls = self.traj_to_linestring(ppp.path_to_traj(path, ls))
        self.gdf = gdf
        unknown_sections = self.scan_using_lidar(traj_ls, self.construct_airspace())
        self.write_geom(unknown_sections, "unknown_sections", "yellow")

        self.radius = 700
        sa = self.construct_airspace()
        self.write_geom([sa], "original_airspace", "green")
        files = ["plot/original_airspace.geojson",
        "plot/shrunk_airspace.geojson",
        "plot/poly_traj.geojson",
        "plot/corridor.geojson"]
        self.combine_json_files(files, "poly")

    def use_ppp_jeddah(self):
        self.radius = 700
        self.disc = 12
        ppp = polygon_pathplanning.polygon_pp()
        # Constructing the safe airspace
        sa = self.construct_airspace()
        # Finding the mountains
        mt = self.avoid_terrain()
        mt = gp.GeoSeries(mt)
        self.add_to_forbidden(mt)
        # Removing landing spots that are within a forbidden area
        self.remove_invalid_ls()
        sa = self.construct_airspace()
        # Adding the roads to the safe airspace
        rds = self.import_roads()
        sa = sa.union(rds)
        # Trimming the airspace to the vicinity of the trajectory
        # boundary = [38.952054, 22.466675, 39.491757, 21.442638]
        boundary = [38.952054, 22.466675, 39.878717, 21.200736]
        # boundary = [5.7778, 43.5639, 5.4191, 43.4472]
        bbox = box(boundary[0], boundary[1], boundary[2], boundary[3])
        sa = sa.intersection(bbox)
        self.fa_gs = self.fa_gs.intersection(bbox)
        fa = self.fa_gs.unary_union # Geoseries to multipolygon
        # Approximating the polygonial airspace
        tolerance = 30 # Meters
        fa = self.simplify_polygon(fa, tolerance)

        # Outputting for testing
        self.write_geom([sa], "safe", "green")
        # sa = lidar.union(sa)
        self.write_geom([fa], "forbidden", "red")
        ###
        tic = time()
        start = [39.174135, 21.513528] # IMC
        end = [39.105643, 22.314860] # KAUST medical clinic
        start_end = [start, end]
        graph, ls = ppp.m_create_connectivity(sa, fa, start_end)
        m_adj, m_heur = ppp.create_adjacency_matrix(graph, ls)
        print("It took ", time() - tic)
        a = 0
        b = 1
        path = astar.a_star(m_adj, m_heur, a, b)
        traj = ppp.path_to_traj(path, ls)

        ppp.write_path(path, ls, "poly_traj", "blue")
        width = 50
        ppp.write_corridor(path, width, ls, "corridor", "green")
        traj_ls = self.traj_to_linestring(ppp.path_to_traj(path, ls))
        unknown_sections = self.scan_using_lidar(traj_ls, self.construct_airspace())
        self.write_geom(unknown_sections, "unknown_sections", "grey")

        self.radius = 700
        sa = self.construct_airspace()
        self.write_geom([sa], "original_airspace", "green")
        files = ["plot/original_airspace.geojson",
        "plot/shrunk_airspace.geojson",
        "plot/poly_traj.geojson",
        "plot/corridor.geojson"]
        self.combine_json_files(files, "poly")

    def use_center_of_ls(self):
        self.create_adjacency_matrix()
        # self.load_adjacency_matrix()
        self.construct_airspace()
        self.delete_node([27, 35, 79, 80, 82, 85, 86])

        self.path = self.find_path(6, 0) # Find path that goes through nodes
        self.traj = self.tune_path(self.path) # Tune path to make trajectory
        self.shift_traj(self.traj, pi/2, [100, 200]) # Generate the shifted trajectories for the wind
        self.write_path(self.path, "original_path", self.colors[0])
        self.write_traj(self.traj, "original_traj", self.colors[1])

        # self.store_adjacency_matrix()
        # self.explore(self.traj)
        # self.plot(self.traj)
        # self.sandbox()
        self.save_stuff()

    def visualize_landing(self):
        radius = 20
        landing = []
        color = "green"
        opacity = 0.9
        for i, row in self.gdf.iteritems():
            lon = self.gdf.geometry[i].x
            lat = self.gdf.geometry[i].y
            landing.append(self.draw_circle(lon, lat, radius))

        s = gp.GeoDataFrame(crs=4326, geometry=landing)
        mp = s.unary_union # Multipolygon
        s = gp.GeoDataFrame(crs=4326, geometry=[mp])
        s["fill-opacity"] = 0.3 * opacity
        s["stroke-opacity"] = opacity
        s["stroke"] = color
        s["marker-color"] = color
        s["fill"] = color
        s.to_file('plot/landing_points.geojson', driver='GeoJSON')
        mp_gs = gp.GeoSeries(mp)
        return mp_gs

    def process_forbidden(self):
        # Put them in the right shape (point to area, and buffer for certain areas)
        # Group them in a multipolygon for easy checks for intersection
        N = self.fa_gdf.geometry.size
        buff = 0.005 #0.008 is about the size of an airport
        fa = []
        for i in range(N):
            if self.fa_gdf.type[i] == "Point": # Point
                # Expand the point
                fa.append(self.fa_gdf.geometry[i].buffer(buff, 4))
            if self.fa_gdf.type[i] == "Polygon": # Polygon
                if i == "airport": # Doesn't work, fix this
                    fa.append(self.expand_polygon(self.fa_gdf.geometry[i], 25))
                else:
                    fa.append(self.fa_gdf.geometry[i])
            if self.fa_gdf.type[i] == "MultiPolygon":
                fa.append(self.fa_gdf.geometry[i])

        s = gp.GeoSeries(fa)
        p = gp.GeoSeries(s.unary_union)
        self.fa_gs = p

    def remove_invalid_ls(self):
        result = self.gdf.within(self.fa_gs.unary_union)
        self.gdf = self.gdf[~result]
        # print(self.gdf)
        # self.gdf = self.gdf[~self.gdf.is_empty]
        # print(self.gdf)
        return self.gdf

    def insert_point(self, x, y):
        # Adding it to se_ls
        a = np.array([y, x])
        self.se_ls = np.insert(self.se_ls, 0, a, axis=0)
        point = Point(y, x)
        # d = {"geometry":[point], "stroke":["green"]}
        # additional_point = gp.GeoDataFrame(index=[0], crs=4326, data=d)
        additional_point = gp.GeoSeries([point])
        self.se_gdf = pd.concat([additional_point, self.se_gdf], ignore_index=True)
        self.gdf = pd.concat([additional_point, self.gdf], ignore_index=True)
        # Coloring
        additional_point = gp.GeoDataFrame(geometry=additional_point)
        additional_point["stroke"] = "green"
        additional_point["marker-color"] = "green"
        additional_point["fill"] = "green"
        if self.logging:
            additional_point.to_file('plot/inserted_point.geojson', driver='GeoJSON')

    def create_adjacency_matrix(self):
        print("Constructing adjacency matrix. This may take few seconds")
        radius = [self.radius, 5 * self.radius, 200 * self.radius]
        stop_gap = 10
        N = self.se_ls.shape[0]
        self.m_adj = np.zeros((N, N))
        self.m_heur = np.zeros((N, N))
        for i in range(N):
            for j in range(i,N):
                dist = self.haversine(self.se_ls[i][0], self.se_ls[i][1], self.se_ls[j][0], self.se_ls[j][1])
                temp = dist
                if dist < radius[0]:
                    dist = dist
                elif dist < radius[1]:
                    dist = (dist + stop_gap) ** 2
                    # dist = 1.4 * (dist - radius[0]) + radius[0]
                    # dist = 5 * (dist - radius[0]) ** 2 + radius[0] ** 2
                elif dist < radius[2]:
                    dist = (dist + stop_gap) ** 3
                else:
                    dist = 0
                self.m_adj[i, j] = dist
                self.m_adj[j, i] = dist
                self.m_heur[i, j] = temp
                self.m_heur[j, i] = temp

        print("Constructed adjacency matrix")

    def find_path(self, a, b):
        print("Finding path")
        iterations = 400
        for i in range(iterations):
            # print("Started A*")
            path = astar.a_star(self.m_adj, self.m_heur, a, b)
            # print("Completed A*")
            if path == -1:
                print("Path not found")
                return -1
            # print(path)
            valid_path = self.check_path(path)
            print("[%i] Valid trajectory: %r" % (i, valid_path))
            if valid_path:
                break
        print(path)
        self.write_path(path, "path", "blue")
        self.write_circles(path, 10 * self.radius, "red", 0.5) #0.1
        self.write_circles(path, 5 * self.radius, "yellow", 1) #0.4
        self.write_circles(path, self.radius, "green", 1)
        # self.evaluate_traj(tuned) # Not accurate
        return path

    def find_baseline(self, a, b):
        # Store original matrices before changing them
        orig_adj = self.m_adj
        orig_heur = self.m_heur
        # Recalculate adjacency matrix and heuristic matrix
        # Assume that traveling inside the safe airspace is "free"
        # Traveling longer than the safe distance is the only cost
        free = 1 # Low cost
        mask = np.logical_and(self.m_adj < self.radius, self.m_adj > 0)
        self.m_adj[mask] = free
        # self.m_heur

        # Find optimal path
        path = self.find_path(a, b)

        # Restore original graphs
        self.m_adj = orig_adj
        self.m_heur = orig_heur
        return path

    def tune_path(self, path):
        print("Tuning path")
        leeway_m = 350
        traj = self.path_to_traj(path)
        return self.tune_traj(traj, leeway_m)

    def tune_traj(self, traj, leeway_m):
        tuned = []
        # leeway = 0.009
        leeway = 117 / 12966250 * leeway_m
        passes = 4
        original = np.array(traj)
        tuned = np.array(traj)
        unit_perpendicular = np.zeros([len(tuned), 3])
        # Finding the unit unit_perpendicular
        for i in range(len(tuned) - 2):
            unit_perpendicular[i + 1] = tuned[i + 2] - tuned[i]

        # Z unit vector
        z_vector = np.full([len(tuned), 3], [0 , 0, 1])
        # Finding unit perpendicular
        unit_perpendicular = np.cross(unit_perpendicular, z_vector)
        unit_perpendicular[0][2] = 1
        unit_perpendicular[len(tuned) - 1][2] = 1
        mag = np.linalg.norm(unit_perpendicular, axis=1, keepdims=True)
        unit_perpendicular /= mag

        gamma = 0.3
        for i in range(passes):
            # gamma *= gamma
            for j in range(len(traj) - 2):
                intersection = self.get_intersect(tuned[j][0:2], tuned[j + 2][0:2], tuned[j + 1][0:2], tuned[j + 1][0:2] + unit_perpendicular[j + 1][0:2])
                direction = intersection - tuned[j + 1]
                tuned[j + 1] = tuned[j + 1] + gamma * direction
                # scale = - np.sign(unit_perpendicular[j + 1].dot(tuned[j + 1]))
                # scale = np.linalg.norm(edge[j + 1] - tuned[j + 1])
                # tuned[j + 1] = tuned[j + 1] - gamma * scale * unit_perpendicular[j + 1]
                if np.linalg.norm(tuned[j + 1] - original[j + 1]) > leeway:
                    mag = np.linalg.norm(direction)
                    direction /= mag
                    tuned[j + 1] = original[j + 1] + leeway * direction
                # Checking the trajectory
                if(not self.check_traj(tuned)):
                    tuned[j + 1] = original[j + 1]
        print("tuned")
        self.write_traj(tuned, "traj", "blue")
        return tuned

    def shift_traj(self, traj, direction, distances):
        s_line = []
        points = []
        gdf_list = []
        colors = ["#D95319", "#7E2F8E"]
        dx = traj[0][0] - traj[-1][0]
        dy = traj[0][1] - traj[-1][1]
        direction = atan2(dy, dx) - pi/2
        reference_direction = 10 * round((90 - atan2(dy, dx) * 180 / pi) / 10)
        print("Reference direction: %03d" % reference_direction)
        for i in range(len(traj)):
            point = Point(traj[i][0], traj[i][1])
            points.append(point)
        s_line = LineString(points)
        for j in [-1, 1]:
            for i in range(2):
                distance = self.c * j * distances[i]
                shift = [distance * cos(direction), distance * sin(direction)]
                gs_line = gp.GeoSeries([s_line])
                gs_line = gs_line.translate(shift[0], shift[1])

                points = [Point(traj[0][0], traj[0][1])]
                for k in range(1, len(traj) - 1):
                    point = Point(gs_line.geometry[0].coords[k][0], gs_line.geometry[0].coords[k][1])
                    points.append(point)

                point = Point(traj[-1][0], traj[-1][1])
                points.append(point)
                line = LineString(points)
                shifted_traj = gp.GeoDataFrame(geometry=[line])

                shifted_traj["stroke"] = colors[i]
                shifted_traj["marker-color"] = colors[i]
                shifted_traj["fill"] = colors[i]
                shifted_traj["stroke-width"] = 3
                gdf_list.append(shifted_traj)
        comb_gdf = pd.concat(gdf_list, axis=0, ignore_index=True)
        print("Generated crosswind trajectories")

        filename = "plot/shifted.geojson"
        comb_gdf.to_file(filename, driver='GeoJSON')

    def check_path(self, path):
        # Check the entire path
        traj = self.path_to_traj(path)
        valid_trajectory = self.check_traj(traj)
        if (valid_trajectory):
            return valid_trajectory
        else: # If there's intersection with a forbidden area, remove invalid segments
            valid_trajectory = True
            for i in range(len(path)-1):
                # print("Segment number: ", i)
                a = path[i]
                b = path[i+1]
                is_valid = self.check_segment(a, b)
                if not is_valid:
                    self.remove_connection(a, b)
                    valid_trajectory = False
                    # print("Removed connection")
        return valid_trajectory

    def check_segment(self, a, b):
        path = [a, b]
        points = self.path_to_traj(path)
        s_line = LineString(points)
        return self.check_linestring(s_line)

    def check_traj(self, traj):
        s_line = self.traj_to_linestring(traj)
        return self.check_linestring(s_line)

    def check_linestring(self, linestring):
        # Make sure to call process_forbidden first
        for i in range(self.fa_gs.size):
            does_intersect = self.fa_gs[i].intersects(linestring)
            if does_intersect == True: # If there's intersection, trajectory is invalid
               return False # Invalid trajectory
        return True

    def remove_connection(self, a, b):
        no_connection = 0
        self.m_adj[a, b] = no_connection
        self.m_adj[b, a] = no_connection

    def evaluate_traj(self, traj):
        green = 0
        yellow = 0
        red = 0
        for i in range(len(traj) - 1):
            seg_len = self.haversine(traj[i][0], traj[i][1], traj[i + 1][0], traj[i + 1][1])
            if (seg_len > self.radius):
                if (seg_len > 5 * self.radius):
                    red += seg_len - 5 * self.radius
                yellow += seg_len - self.radius
            green += seg_len
        green -= yellow
        yellow -= red
        total = green + yellow + red
        traj_length = 0
        print("The total length of the trajectory:\t%f" % traj_length)
        print("The percentage of the generated trajectory that is within:")
        print("Green:\t%d%%" % round(100 * green / total))
        print("Yellow:\t%d%%" % round(100 * yellow / total))
        print("Red:\t%d%%" % round(100 * red / total))

    def measure_risk_airspace(self, traj):
        # print("Measuring risk")
        s = self.construct_airspace()
        # Generate lots of points along the trajectory, then check if they're inside the airspace
        s_line = self.traj_to_linestring(traj)
        sampled = self.sample_line(s_line)
        gs_line = self.linestring_to_points(sampled)

        counts = []
        for i in range(3):
            intersection = gs_line.within(s[i])
            count = intersection.cumsum()
            count = count[count.size - 1]
            counts.append(count)
        total = gs_line.size
        green = counts[0]
        yellow = counts[1] - counts[0]
        red = counts[2] - counts[1]

        traj_length = self.traj_length(s_line)[-1]
        print("The total length of the trajectory:\t%.1fkm" % traj_length)
        print("The percentage of the generated trajectory that is within:")
        print("Green:\t%.1f%%, \t%.2f" % ((100 * green / total), green / total * traj_length))
        print("Yellow:\t%.1f%%, \t%.2f" % ((100 * yellow / total), yellow / total * traj_length))
        print("Red:\t%.1f%%, \t%.2f" % ((100 * red / total), red / total * traj_length))
        return counts

    def scan_using_lidar(self, linestring, airspace):
        buffer_area = self.radius * self.c
        difference = linestring.difference(airspace)
        buffer = difference.buffer(buffer_area, 4)
        # return self.linestring_to_points(difference)
        return [buffer, difference]

    def simplify_polygon(self, fa, tolerance):
        return fa.simplify(tolerance * self.c)

    def path_to_traj(self, path):
        traj = []
        for i in range(len(path)):
            j = path[i]
            traj.append([self.gdf.geometry[j].x, self.gdf.geometry[j].y, 0])
        return traj

    def traj_to_linestring(self, traj):
        points = []
        for i in range(len(traj)):
            point = Point(traj[i][0], traj[i][1])
            points.append(point)
        s_line = LineString(points)
        return s_line

    def linestring_to_points(self, linestring):
        points = []
        for i in range(len(linestring.coords)):
            lon1 = linestring.coords[i][0]
            lat1 = linestring.coords[i][1]
            point = Point(lon1, lat1)
            points.append(point)
        gs_line = gp.GeoSeries(points)
        return gs_line

    def draw_circle(self, lon, lat, rad):
        l = []
        disc = self.disc
        ellipsis_factor = 0.72411296162
        c = self.c
        for i in range(disc):
            x = lon + c * rad * cos(2 * pi * i / disc)
            y = lat + c * rad * sin(2 * pi * i / disc) * ellipsis_factor
            l.append((x, y))
        p = Polygon(l)
        return p

    def get_intersect(self, a1, a2, b1, b2):
        # """
        # Returns the point of intersection of the lines passing through a2,a1 and b2,b1.
        # a1: [x, y] a point on the first line
        # a2: [x, y] another point on the first line
        # b1: [x, y] a point on the second line
        # b2: [x, y] another point on the second line
        # """
        s = np.vstack([a1,a2,b1,b2])        # s for stacked
        h = np.hstack((s, np.ones((4, 1)))) # h for homogeneous
        l1 = np.cross(h[0], h[1])           # get first line
        l2 = np.cross(h[2], h[3])           # get second line
        x, y, z = np.cross(l1, l2)          # point of intersection
        if z == 0:                          # lines are parallel
            return (float('inf'), float('inf'))
        return np.array([x/z, y/z, 0])

    def expand_polygon(self, airport, angle):
        # Expand parameters, in meters
        parallel = 500
        perpendicular = 0

        # Modify
        c = self.c
        parallel *= c
        perpendicular *= c
        # Setup
        pol = gp.GeoSeries([airport], crs=2154)
        # Rotate
        pol = pol.rotate(-angle)
        # pol.rotate(-angle, "center")
        # pol.rotate(-angle, "centroid")

        # Bound
        x1 = pol.bounds.minx[0]
        x2 = pol.bounds.maxx[0]
        y1 = pol.bounds.miny[0]
        y2 = pol.bounds.maxy[0]

        # Enlarge/Buffer
        x1 -= perpendicular
        x2 += perpendicular
        y1 -= parallel
        y2 += parallel

        # Envelope
        rectangle = gp.GeoSeries([Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])], crs=2154)

        # Rotate again
        rectangle = rectangle.rotate(angle, "centroid")
        # # print(rectangle.geometry[0])
        # bounding_box = gp.GeoSeries([airport], crs=2154)
        # # bounding_box.to_file('test_airport.geojson', driver='GeoJSON')
        # rec_df = gp.GeoDataFrame(geometry=rectangle, crs=2154)
        # rec_df["stroke"] = "red"
        # rec_df["marker-color"] = "red"
        # rec_df["fill"] = "red"
        # rec_df.to_file('plot/test_airport.geojson', driver='GeoJSON')

        return rectangle.geometry[0]

    def haversine(self, lon1, lat1, lon2, lat2):
        # Calculate the great circle distance between two points
        # on the earth (specified in decimal degrees)

        # convert decimal degrees to radians
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

        # haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        r = 6371000 # Radius of earth in meters
        return c * r

    def sample_line(self, traj):
        m = 30
        distance = self.c * m
        num_vert = int(round(traj.length / distance))
        if num_vert == 0:
            num_vert = 1
        return LineString(
            [traj.interpolate(float(n) / num_vert, normalized=True)
            for n in range(num_vert + 1)])

    def construct_airspace(self):
        radius = [self.radius, 7 * self.radius, 10 * self.radius]
        color = ["green", "yellow", "red"]
        opacity = [0.4, 0.4, 0.25]
        circles = []
        airspace = []
        for n in range(3):
            circles.append([])
            for i, row in self.gdf.iteritems():
                lon = self.gdf.geometry[i].x
                lat = self.gdf.geometry[i].y
                circles[n].append(self.draw_circle(lon, lat, radius[n]))

            s = gp.GeoDataFrame(crs=4326, geometry=circles[n])
            mp = s.unary_union # Multipolygon
            airspace.append(mp)
            s = gp.GeoDataFrame(crs=4326, geometry=[mp])
            s["fill-opacity"] = 0.3 * opacity[n]
            s["stroke-opacity"] = opacity[n]
            s["stroke"] = color[n]
            s["marker-color"] = color[n]
            s["fill"] = color[n]
            if self.logging:
                s.to_file('plot/airspace_' + color[n] + '.geojson', driver='GeoJSON')
        airspace_gs = gp.GeoSeries(airspace)
        return airspace_gs[0]

    def traj_info(self, traj):
        pass

def main():
    a = uam()
    print("Hello world")

main()