# from code import interact
# from platform import java_ver
# from re import L
# from socket import AI_PASSIVE
# from time import time
# from webbrowser import GenericBrowser
# import pandas as pd
import geopandas as gp
import numpy as np
# import astar_uam as astar

from shapely.geometry import LineString, Point, Polygon
from shapely.ops import nearest_points
import glob
from math import radians, cos, sin, asin, atan2, sqrt, pi, ceil, exp, log

# from matplotlib import cm
# from matplotlib.ticker import LinearLocator
# import matplotlib.pyplot as plt

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# 'EPSG:2154' for France
# epsg=4326  for equal area

class exploration:
    def explore(self, traj):
        # Don't risk too much (limit exploration)
        # Avoid forbidden areas
        # How to choose where to explore first?
        # Explored area
        # Unexplored area
        # Explored area starts with the known landing spots and radius
        # Slowly modify the trajectory to include a bit of unexplored area
        # Or choose less optimal path and modify it
        # Define the area worth exploring using the distance measurement, and the best already known path
        # Define an upper limit for risk
        # Then somehow adjust to find a more safe path that is within the risk parameters
        # Find a way to measure the risk of the newly generated path
        # for i in range(len(traj) - 1):
        #     print("%.0f\t%.0f" % (self.m_adj[traj[i]][traj[i + 1]], self.m_heur[traj[i]][traj[i + 1]]))
        # Maybe rewrite the tune_traj function to take in a direction for the gradient to make it more general

        # STEPS
        # Modify traj
        # tuned = self.modify_traj(traj, 350)
        # self.write_traj(tuned, "exp1", "blue")
        # risk = self.measure_risk_airspace(tuned)
        # tuned = self.modify_traj(traj, 500)
        # self.write_traj(tuned, "exp2", "green")
        # risk = self.measure_risk_airspace(tuned)
        # tuned = self.modify_traj(traj, 1000)
        # self.write_traj(tuned, "exp3", "yellow")
        # risk = self.measure_risk_airspace(tuned)
        # tuned = self.modify_traj(traj, 2000)
        # self.write_traj(tuned, "exp4", "orange")
        # risk = self.measure_risk_airspace(tuned)
        tuned = self.modify_traj(traj, 4000)
        self.write_traj(tuned, "exp5", "red")
        risk = self.measure_risk_airspace(tuned)
        # Measure risk
        # risk = self.measure_risk(tuned)
        # risk = self.measure_risk_airspace(tuned)
        # print("Risk of the trajectory is: %.0f%%" % (100 * risk))
        # Modify accordingly
        # Execute
        # Discover
        explored = self.explored_airspace(tuned, 2500)
        # Incorporate new data
        # Repeat
        return 0

    def explore2(self, path):
        # Use the heuristic matrix for exploration
        # Find a hybrid between the generated path using the normal A* and the purely heuristic A*
        # Try to find landing spots in the middle
        # Don't risk too much (limit exploration)
        # Avoid forbidden areas
        # How to choose where to explore first?
        # Explored area
        # Unexplored area
        # Explored area starts with the known landing spots and radius
        # Slowly modify the trajectory to include a bit of unexplored area
        # Or choose less optimal path and modify it
        # Allow for one "free" jump, ignore the penalty - can modify the adjacency matrix to allow for free jumps from one node
        # Define the area worth exploring using the distance measurement, and the best already known path
        # Define an upper limit for risk
        # self.m_adj, self.m_heur,
        # Follow heur to find "optimal" path, but not in terms of safety
        # Then somehow adjust to find a more safe path that is within the risk parameters
        # Make two different exploration algorithms, one to slightly modify the path
        # And the other is to use the heuristic methods to fill in the gaps
        # Find a way to measure the risk of the newly generated path
        for i in range(len(path) - 1):
            print("%.0f\t%.0f" % (self.m_adj[path[i]][path[i + 1]], self.m_heur[path[i]][path[i + 1]]))

    def modify_traj(self, traj, leeway_m):
        # Modify so we can resample trajectory every once in a while
        tuned = []
        # leeway = 0.009
        c = 117 / 12966250
        leeway = c * leeway_m
        passes = 30
        original = np.array(traj)
        tuned = np.array(traj)
        prev = np.array(traj)

        gamma = 0.6
        epsilon = 20 # in meters
        for i in range(passes):
            # gamma *= gamma
            for j in range(1, len(traj) - 1):
                if (j >= len(tuned) - 1):
                    break
                p = self.index_logic(j, len(tuned) - 1, 1)
                unit_perpendicular = self.gradient(tuned, j)
                intersection = self.get_intersect(tuned[p[0]][0:2], tuned[p[2]][0:2], tuned[p[1]][0:2], tuned[p[1]][0:2] + unit_perpendicular[0:2])
                direction = intersection - tuned[p[1]]

                mag = np.linalg.norm(direction)
                if mag / c < epsilon:
                    # print(p[1])
                    # print(tuned)
                    tuned = np.delete(tuned, p[1], axis=0)
                    original = np.delete(original, p[1], axis=0)
                    continue
                tuned[p[1]] = tuned[p[1]] + gamma * direction
                if np.linalg.norm(tuned[p[1]] - original[p[1]]) > leeway:
                    mag = np.linalg.norm(direction)
                    direction /= mag
                    tuned[p[1]] = original[p[1]] + leeway * direction
                # Checking the trajectory
                sub_traj = tuned[p]
                if(not self.check_traj(sub_traj)):
                    tuned[p[1]] = original[p[1]]
                    # tuned[p[1]] = prev[p[1]]
            prev = tuned
        return tuned

    def explored_airspace(self, traj, explore_range):
        explore_range = self.c * explore_range
        points = []
        for i in range(len(traj)):
            point = Point(traj[i][0], traj[i][1])
            points.append(point)
        s_line = gp.GeoSeries(LineString(points))
        explored = s_line.buffer(explore_range, resolution=3)
        s = gp.GeoDataFrame(crs=4326, geometry=explored)
        opacity = 1
        color = "grey"
        s["fill-opacity"] = 0.15 * opacity
        s["stroke-opacity"] = 0.5 * opacity
        s["stroke"] = color
        s["marker-color"] = color
        s["fill"] = color
        s.to_file('plot/explored_airspace.geojson', driver='GeoJSON')
        return explored

    def discover(self, explored):

        pass

    def gradient(self, tuned, i):
        c = 1
        p = self.index_logic(i, len(tuned) - 1, 1)
        unit_perpendicular = self.unit_perp(tuned[p[2]] - tuned[p[0]])
        # average = self.unit_perp(tuned[p[2]] - tuned[p[0]])
        # centerline = self.unit_perp(tuned[-1] - tuned[0])
        # unit_perpendicular = c * average + (1-c) * centerline
        return unit_perpendicular

    def measure_risk(self, traj):
        print("Calculating risk")
        # Measure distance to all landing spots
        # ls = self.sample_line(traj)
        points = []
        for i in range(len(traj)):
            point = Point(traj[i][0], traj[i][1])
            points.append(point)
        s_line = LineString(points)
        sampled = self.sample_line(s_line)
        # X = [0]
        # max = 0
        min = [1e10] * len(sampled.coords)
        for i in range(len(sampled.coords)):
            for j, row in self.gdf.iteritems():
                lon1 = sampled.coords[i][0]
                lat1 = sampled.coords[i][1]

                lon2 = self.gdf.geometry[j].x
                lat2 = self.gdf.geometry[j].y
                temp = self.haversine(lon1, lat1, lon2, lat2)
                if temp < min[i]:
                    min[i] = temp
        # Find minimum to each point in the trajectory
        # Use this measure to assess risk
        count = 0
        cutoff = self.radius
        for i in min:
            if i < cutoff:
                count += 1
        return count / len(min)

    def delete_node(self, a):
        # Deleting a node by increasing the cost in the adjacency matrix
        cost = 1e7
        N = self.se_ls.shape[0]
        for i in range(N):
            for j in a:
                self.m_adj[i, j] = 0
                self.m_adj[j, i] = 0
                self.m_heur[i, j] = cost
                self.m_heur[j, i] = cost
        self.gdf.drop(
            self.gdf.index[a],
            inplace=True
            )

    def unit_perp(self, v):
        # Z unit vector
        z_vector = np.array([0 , 0, 1])
        # Finding unit perpendicular
        unit_perpendicular = np.cross(v, z_vector)
        mag = np.linalg.norm(unit_perpendicular, keepdims=True)
        unit_perpendicular /= mag
        return unit_perpendicular

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

    def index_logic(self, i, size, width):
        p = [0] * 3
        p[0] = i - width
        p[1] = i
        p[2] = i + width
        if p[0] < 0:
            p[0] = 0
        if p[2] > size:
            p[2] = size

        return p