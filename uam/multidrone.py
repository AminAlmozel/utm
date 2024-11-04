# TODO
from time import time

import pandas as pd
import geopandas as gp
import numpy as np
# from skimage import measure, regionprops
import skimage as sk
import astar_uam as astar
# import terrain

import polygon_pathplanning

from shapely.geometry import LineString, Point, Polygon, MultiPolygon
from shapely.ops import nearest_points
import glob
from math import radians, cos, sin, asin, atan2, sqrt, pi as PI, ceil, exp, log, copysign

from matplotlib import cm
from matplotlib.ticker import LinearLocator
import matplotlib.pyplot as plt

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# 'EPSG:2154' for France
# epsg=4326  for equal area

class multi:
    def import_terrain(self):
        self.save_as_csv = False
        print("Importing terrain")
        #https://e4ftl01.cr.usgs.gov/provisional/MEaSUREs/NASADEM/Africa/hgt_merge/
        file = 'terrain/N43E005.hgt'   #Source of file is from the link above

        self.SAMPLES = 1201 # Change this to 1201 for SRTM3
        SAMPLES = self.SAMPLES

        with open(file, 'rb') as file:
            # Each data is 16bit signed integer(i2) - big endian(>)
            self.elv = np.fromfile(file, np.dtype('>i2'), SAMPLES*SAMPLES).reshape((SAMPLES, SAMPLES))
            self.elv = np.flipud(self.elv)

    def test(self):
        # Constructing test image
        image = np.zeros((200, 200))
        idx = np.arange(25, 175)
        image[idx, idx] = 255
        image[sk.draw.line(45, 25, 25, 175)] = 255
        image[sk.draw.line(25, 135, 175, 155)] = 255

        # Classic straight-line Hough transform
        # Set a precision of 0.5 degree.
        tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360, endpoint=False)
        h, theta, d = sk.transform.hough_line(image, theta=tested_angles)

        # Generating figure 1
        fig, axes = plt.subplots(1, 3, figsize=(15, 6))
        ax = axes.ravel()

        ax[0].imshow(image, cmap=cm.gray)
        ax[0].set_title('Input image')
        ax[0].set_axis_off()

        angle_step = 0.5 * np.diff(theta).mean()
        d_step = 0.5 * np.diff(d).mean()
        bounds = [np.rad2deg(theta[0] - angle_step),
                np.rad2deg(theta[-1] + angle_step),
                d[-1] + d_step, d[0] - d_step]
        ax[1].imshow(np.log(1 + h), extent=bounds, cmap=cm.gray, aspect=1 / 1.5)
        ax[1].set_title('Hough transform')
        ax[1].set_xlabel('Angles (degrees)')
        ax[1].set_ylabel('Distance (pixels)')
        ax[1].axis('image')

        ax[2].imshow(image, cmap=cm.gray)
        ax[2].set_ylim((image.shape[0], 0))
        ax[2].set_axis_off()
        ax[2].set_title('Detected lines')

        for _, angle, dist in zip(*sk.transform.hough_line_peaks(h, theta, d)):
            (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
            ax[2].axline((x0, y0), slope=np.tan(angle + np.pi/2))

        plt.tight_layout()
        plt.show()

    def test2(self):
        # Constructing test image
        image = np.zeros((200, 200))
        idx = np.arange(25, 175)
        image[idx, idx] = 255
        image[sk.draw.line(45, 25, 25, 175)] = 255
        image[sk.draw.line(25, 135, 175, 155)] = 255


        edges = sk.feature.canny(image, 2, 1, 25)
        lines = sk.transform.probabilistic_hough_line(edges, threshold=10, line_length=5,
                                        line_gap=40)

        # Generating figure 2
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
        ax = axes.ravel()

        ax[0].imshow(image, cmap=cm.gray)
        ax[0].set_title('Input image')

        ax[1].imshow(edges, cmap=cm.gray)
        ax[1].set_title('Canny edges')

        ax[2].imshow(edges * 0)
        for line in lines:
            p0, p1 = line
            ax[2].plot((p0[0], p1[0]), (p0[1], p1[1]))
        ax[2].set_xlim((0, image.shape[1]))
        ax[2].set_ylim((image.shape[0], 0))
        ax[2].set_title('Probabilistic Hough')

        for a in ax:
            a.set_axis_off()

        plt.tight_layout()
        plt.show()

    def multidrone_pathfinding(self, m_adj, m_heur, ls, factories):
        ppp = polygon_pathplanning.polygon_pp()
        print(len(factories))
        print(m_adj.shape[0])
        trajectories = []
        # for i in range(m_adj.shape[0] - len(factories), m_adj.shape[0], 2):
        for i in range(0, len(factories) - 1, 2):
            a = i
            b = i + 1
            path = astar.a_star(m_adj, m_heur, a, b)
            trajectory = ppp.path_to_traj(path, ls)
            linestring = self.traj_to_linestring(trajectory)
            trajectories.append(linestring)
        self.write_geom(trajectories, "drone_deliveries", "blue")
        return trajectories

    def make_image(self, geom):
        scale = 5000
        gs = gp.GeoSeries(geom)
        x0, y0, maxx, maxy = gs.total_bounds
        dimensions = [ceil(scale*(maxx - x0)),
                          ceil(scale*( maxy - y0))]
        image = np.zeros((dimensions[1], dimensions[0]))
        disc = 10 #meters
        list_xy = []
        for traj in geom:
            num_of_samples = int(100 * self.traj_length(traj)[-1] / disc)
            xy = self.sample_line_n(traj, num_of_samples)
            list_xy += self.linestring_to_list(xy)
        print(dimensions)
        for xy in list_xy:
            j = int(scale*(xy[0] - x0))
            i = int(scale*(xy[1] - y0))
            i = dimensions[1] - i - 1
            image[i, j] += 1
        image = sk.filters.gaussian(image, sigma=2)
        threshold = 0.01
        mask0 = np.where(image < threshold)
        mask1 = np.where(image >= threshold)
        image[mask0] = 0
        image[mask1] = 1
        return image

    def small_areas(self, Z, threshold):
        # Removing small mountains based on area
        arr = skimage.measure.label(Z > 0)

        # skimage.morphology.remove_small_holes
        # skimage.morphology.remove_small_objects
        for region in skimage.measure.regionprops(arr):
            if (region['area'] < threshold):
                coords = region['coords']
                arr[coords[:, 0], coords[:, 1]] = 0

        Z = self.threshold(arr, 1)
        # cluster_max = []
        # cluster_avg = []
        # for region in skimage.measure.regionprops(arr):
        #     coords = region['coords']
        #     cluster_max.append(self.max_height(coords))
        #     box = region['bbox']
        #     cluster_avg.append(self.average_height(box))
        return Z

        ###################################################################
        # # Plotting height
        # # plt.figure()
        # plt.subplots(nrows=2)
        # plt.subplot(211)

        # cruise_speed = 90
        # # f = distance to time
        # # g = time to distance
        # f = lambda x: x / cruise_speed * 60
        # g = lambda x: x * cruise_speed / 60
        # ax = plt.gca()
        # ax2 = ax.secondary_xaxis('top', functions=(f, g))
        # ax2.set_xlabel("Time (min)")
        # # X2 = np.linspace(0, distance / cruise_speed * 60, l)


        # # temp = []
        # # for i in range(len(elv)):
        # #     temp.append(Y[i] - elv[i])
        # plt.plot(X, elv)
        # plt.title("Elevation along the trajectory")
        # plt.xlabel("Distance (km)")
        # plt.ylabel("Elevation (m)")
        # ax = plt.gca()
        # # ax.set_ylim([0, 1000])
        # ax.yaxis.grid()
        # zeros = [0] * len(elv)
        # qnh = self.pressure(zeros, elv)
        # plt.subplot(212)
        # plt.plot(X, qnh)
        # plt.title("QNH")
        # plt.xlabel("Distance (km)")
        # plt.ylabel("Pressure (hPa)")

        ###################################################################
        # # Plotting pressure
        # plt.subplots(nrows=2)
        # plt.subplot(211)
        # traj_elv = self.plot_elevation_traj_3(elv)
        # pressure_at_altitude = self.pressure(zeros, traj_elv)
        # fixed_qnh = [qnh[0]] * len(elv)
        # height_qnh_fixed = self.height(fixed_qnh, pressure_at_altitude)
        # height_qnh_calc = self.height(qnh, pressure_at_altitude)

        # plt.plot(X, height_qnh_fixed, label='Fixed QNH')

        # plt.plot(X, height_qnh_calc, label='Calculated QNH')
        # plt.legend()
        # error = []
        # for i in range(len(elv)):
        #     error.append(height_qnh_calc[i] - height_qnh_fixed[i])
        # plt.subplot(212)
        # plt.plot(X, error)
        pass

    def traj_to_xy(self, traj):
        points = []
        for i in range(len(traj)):
            point = Point(traj[i][0], traj[i][1])
            points.append(point)
        s_line = LineString(points)
        sampled = self.sample_line(s_line)

        # 2154 3857 4326
        points = []
        for i in range(len(sampled.coords)):
            lon1 = sampled.coords[i][0]
            lat1 = sampled.coords[i][1]
            point = Point(lon1, lat1)
            points.append(point)
        s = gp.GeoDataFrame(geometry=points, crs=4326)
        s = s.to_crs(epsg=2154)
        # print(s)
        x = []
        y = []
        x0 = s.geometry[0].x
        y0 = s.geometry[0].y
        for i, row in s.iterrows():
            x.append((s.geometry[i].x - x0) / 1000)
            y.append((s.geometry[i].y - y0) / 1000)
        return (x, y)

    def traj_to_xy_n(self, traj, num_of_samples):
        points = []
        for i in range(len(traj)):
            point = Point(traj[i][0], traj[i][1])
            points.append(point)
        s_line = LineString(points)
        sampled = self.sample_line_n(s_line, num_of_samples)

        # 2154 4326
        points = []
        for i in range(len(sampled.coords)):
            lon1 = sampled.coords[i][0]
            lat1 = sampled.coords[i][1]
            point = Point(lon1, lat1)
            points.append(point)
        s = gp.GeoDataFrame(geometry=points, crs=4326)
        s = s.to_crs(epsg=2154)
        # print(s)
        x = []
        y = []
        x0 = s.geometry[0].x
        y0 = s.geometry[0].y
        for i, row in s.iterrows():
            x.append((s.geometry[i].x) / 1000)
            y.append((s.geometry[i].y) / 1000)
        return (x, y)

    def sample_line(self, traj):
        m = 30
        distance = self.c * m
        num_vert = int(round(traj.length / distance))
        if num_vert == 0:
            num_vert = 1
        return LineString(
            [traj.interpolate(float(n) / num_vert, normalized=True)
            for n in range(num_vert + 1)])

    def sample_line_n(self, traj, num_of_samples):
        num_vert = num_of_samples
        if num_vert == 0:
            num_vert = 1
        return LineString(
            [traj.interpolate(float(n) / num_vert, normalized=True)
            for n in range(num_vert + 1)])

    def linestring_to_list(self, linestring):
        return np.dstack(linestring.coords.xy).tolist()[0]

    def traj_length(self, s_line):
        sampled = self.sample_line(s_line)
        X = [0]
        max = 0
        for i in range(len(sampled.coords) - 1):
            lon1 = sampled.coords[i][0]
            lat1 = sampled.coords[i][1]
            lon2 = sampled.coords[i + 1][0]
            lat2 = sampled.coords[i + 1][1]
            temp = self.haversine(lon1, lat1, lon2, lat2)
            X.append(X[-1] + temp / 1000)
            if temp > max:
                max = temp
        return X

    def within(self, coords, shape):
        # shape = [xmin, ymin, xmax, ymax]
        if coords[0] > shape[1] and coords[0] < shape[3]:
            if coords[1] > shape[0] and coords[1] < shape[2]:
                return True
        return False

    def minecraft_line(self, dist, azimuth):
        if(dist < 1):
            dist = 1
        disc = 5 # Discretization
        length = np.linspace(0, dist, dist * disc)
        x = np.around(length * cos(azimuth))
        y = np.around(length * sin(azimuth))
        pattern = np.transpose(np.stack((x, y)))
        pattern = np.unique(pattern, axis=0)
        if (np.linalg.norm(pattern[0]) != 0):
            pattern = np.fliplr(np.transpose(pattern))
        else:
            pattern = np.transpose(pattern)
        return pattern

    def square_terrain_plot(self, lminx, lminy, lmaxx, lmaxy):
        dx = lmaxx - lminx
        dy = lmaxy - lminy
        cx = int(lminx + dx / 2)
        cy = int(lminy + dy / 2)
        square = dx
        if dy > dx:
            square = dx
        lminx = int(cx - square / 2)
        lmaxx = int(cx + square / 2)
        lminy = int(cy - square / 2)
        lmaxy = int(cy + square / 2)
        if lminx < 0:
            lminx = 0
            lmaxx = square
        if lminy < 0:
            lminy = 0
            lmaxy = square
        return lminx, lminy, lmaxx, lmaxy