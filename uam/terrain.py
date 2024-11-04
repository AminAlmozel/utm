# TODO
# Modify so the code can handle multiple terrain files in case the trajectory crosses the boundaries
# Resolve the case of the maximum size of the terrain in square terrain plot function
# Fly higher in areas where there's no safe landing spot nearby
from time import time
import pandas as pd
import geopandas as gp
import numpy as np
import skimage
import astar_uam as astar

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

class terrain:
    def import_terrain(self):
        self.save_as_csv = False
        print("Importing terrain")
        # https://bailu.ch/dem3/
        #https://e4ftl01.cr.usgs.gov/provisional/MEaSUREs/NASADEM/Africa/hgt_merge/
        file = self.project + "terrain/" + self.terrain_file  #Source of file is from the link above

        self.SAMPLES = 1201 # Change this to 1201 for SRTM3
        SAMPLES = self.SAMPLES

        with open(file, 'rb') as file:
            # Each data is 16bit signed integer(i2) - big endian(>)
            self.elv = np.fromfile(file, np.dtype('>i2'), SAMPLES*SAMPLES).reshape((SAMPLES, SAMPLES))
            self.elv = np.flipud(self.elv)

    def add_to_forbidden(self, area):
        list_of_df = [self.fa_gs, area]
        self.fa_gs = pd.concat(list_of_df, ignore_index=True)
        self.fa_gs = gp.GeoSeries(self.fa_gs.unary_union)
        print("Added area to forbidden")
        return self.fa_gs.unary_union

    def avoid_terrain(self):
        traj = [[5.7851, 43.5761],
        [5.4173, 43.4035]]
        p, mountain_buffer, wind_effect, venturi = self.plot_ctow(traj)
        list_of_gs = [p, mountain_buffer, wind_effect, venturi]
        mt = pd.concat(list_of_gs, ignore_index=True)
        mt = mt.unary_union
        self.write_geom([mt], "mt", "red")
        return mt

    def plot(self, traj):
        # Min and Max height
        self.min_h = 20
        self.max_h = 120
        # Sample trajectory
        points = []
        for i in range(len(traj)):
            point = Point(traj[i][0], traj[i][1])
            points.append(point)
        s_line = LineString(points)

        X = self.traj_length(s_line)
        _, _, elv = self.get_terrain_traj(s_line)
        print("Elevation gain: %dm" % (elv[-1] - elv[0]))

        # Plotting
        # self.plot_altitude(elv, X)
        # self.plot_top_side(elv, traj, X)
        # self.plot_3d_traj(elv, traj, X)
        # self.plot_terrain_slope(traj)
        # self.plot_contours(traj)
        # self.plot_terrain_shadow(traj)
        # self.plot_ctow(traj)
        # plt.show()
        return 0

    def plot_terrain(self, traj):
        print("Plotting terrain")
        # 3D plot of trajectory
        SAMPLES = 1201 # Change this to 3601 for SRTM1

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        p1 = [5.332591, 43.179651]
        p2 = [5.779597, 43.571692]
        # traj = [p1, p2]
        X, Y, Z = self.get_terrain_section(traj)

        # Plot the surface.
        # surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
        #                     linewidth=1, antialiased=True)
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                cmap='viridis', antialiased=True)
        # surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
        #         cmap='viridis', edgecolor='none', antialiased=True, alpha=1)

        # Customize the z axis.
        scale = 3
        ax.set_zlim(0, scale * Z.max())
        ax.zaxis.set_major_locator(LinearLocator(10))
        # A StrMethodFormatter is used automatically
        ax.zaxis.set_major_formatter('{x:.02f}')

        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.3, aspect=10)
        fig.tight_layout()

        # Saving the terrain
        if(self.save_as_csv):
            xyz = [X.flatten('F'), Y.flatten('F'), Z.flatten('F')]
            df = pd.DataFrame(xyz)
            df.to_csv("plot/terrain.csv", index=False, header=False)
        return ax

    def plot_altitude(self, elv, X):
        ###################################################################
        # Plotting altitude
        Y = elv
        plt.plot(X, Y)
        plt.title("Elevation of the Terrain Along the Trajectory")
        plt.xlabel("Distance (km)")
        plt.ylabel("Elevation (m)")
        ax = plt.gca()
        ax.set_ylim([0, 1000])
        ax.yaxis.grid()
        ax.axhline(y=elv[0], linestyle = 'dashed')
        ax.axhline(y=elv[-1], linestyle = 'dashed')
        ax.axhline(y=max(elv), linestyle = 'dashed')

        ax.annotate("HPP", (X[0], Y[0]), horizontalalignment='center', textcoords="offset points", xytext=(0, -10))
        ax.annotate("CEAM", (X[-1], Y[-1]), horizontalalignment='center', textcoords="offset points", xytext=(0, 5))

        min_height = []
        max_height = []

        for i in range(len(elv)):
            min_height.append(elv[i] + self.min_h)
            max_height.append(elv[i] + self.max_h)
        plt.plot(X, min_height, linestyle = 'dashed', color='grey')
        plt.plot(X, max_height, linestyle = 'dashed', color='grey')

        # traj_elv = self.plot_elevation_traj(elv)
        # Y = traj_elv
        # plt.plot(X, Y)

        # traj_elv = self.plot_elevation_traj_2(elv)
        # Y = traj_elv
        # plt.plot(X, Y)

        traj_elv = self.plot_elevation_traj_3(elv)
        Y = traj_elv
        plt.plot(X, Y)

        # traj_elv = self.plot_elevation_traj_4(elv)
        # Y = traj_elv
        # plt.plot(X, Y)

        return 0

    def plot_top_side(self, elv, traj, X):
        ###################################################################
        # Plotting side and top view
        plt.subplots(nrows=2)
        plt.subplot(211)
        ax = plt.gca()
        ax.yaxis.grid()

        # Min and Max height
        self.min_h = 20
        self.max_h = 120
        min_height = []
        max_height = []

        for i in range(len(elv)):
            min_height.append(elv[i] + self.min_h)
            max_height.append(elv[i] + self.max_h)
        plt.plot(X, elv, linestyle = 'dashed', color='C0')
        plt.plot(X, min_height, linestyle = 'dashed', color='grey')
        plt.plot(X, max_height, linestyle = 'dashed', color='grey')
        plt.title("Side view of the trajectory")
        plt.xlabel("Distance (km)")
        plt.ylabel("Altitude (m)")

        # traj_elv = self.plot_elevation_traj(elv)
        # Y = traj_elv
        # plt.plot(X, Y)

        # traj_elv = self.plot_elevation_traj_2(elv)
        # Y = traj_elv
        # plt.plot(X, Y)
        traj_elv = self.plot_elevation_traj_3(elv)
        Y = traj_elv
        plt.plot(X, Y, color="C1")

        # traj_elv = self.plot_elevation_traj_4(elv)
        # Y = traj_elv
        # plt.plot(X, Y)
        # x, y = self.traj_to_xy(traj)
        x, y = self.traj_to_xy_n(traj, len(X) - 1)
        plt.subplot(212)
        ax = plt.gca()
        ax.grid()
        ax.set_aspect('equal')
        ax.annotate("HPP", (x[0], y[0]), horizontalalignment='center', textcoords="offset points", xytext=(0, -10))
        ax.annotate("CEAM", (x[-1], y[-1]), horizontalalignment='center', textcoords="offset points", xytext=(0, 5))
        plt.title("Top view of the trajectory")
        plt.xlabel("Position (km)")
        plt.ylabel("Position (km)")

        plt.plot(x, y)

    def plot_3d_traj(self, elv, traj, X):
        ###################################################################
        # 3D plot of trajectory
        # Plotting the terrain
        ax = self.plot_terrain(traj)

        points = []
        for i in range(len(traj)):
            point = Point(traj[i][0], traj[i][1])
            points.append(point)
        s_line = LineString(points)

        # Plotting the 3D trajectory
        X, Y, Z = self.get_terrain_traj(s_line)
        Z = self.plot_elevation_traj_4(elv) # Get waypoint representation of the trajectory for plotting
        # Z = self.plot_debug(elv)
        if(self.save_as_csv):
            xyz = [X, Y, Z]
            df = pd.DataFrame(np.array(xyz))
            df.to_csv("plot/traj3d.csv", index=False, header=False)

        ax.plot3D(X, Y, Z, 'red')
        # ax.view_init(elev=70, azim=-70)
        # ax.view_init(elev=90, azim=0)
        ax.view_init(elev=90, azim=-90)

    def plot_terrain_slope(self, traj):
        X, Y, Z = self.terrain_slope(traj)
        self.save_mat(Z, [X[0][0], Y[0][0]], "grad", "red")

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                cmap='viridis', antialiased=True)
        ax.view_init(elev=90, azim=-90)

        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.3, aspect=10)
        fig.tight_layout()

        gs = self.mat_to_geoseries(Z, [X[0][0], Y[0][0]])
        threshold = 20
        p = self.small_areas_geoseries(gs, threshold)
        self.save_geoseries(p, "geo", "orange")

        # Saving the terrain
        if (self.save_as_csv):
            xyz = [X.flatten('F'), Y.flatten('F'), Z.flatten('F')]
            df = pd.DataFrame(xyz)
            df.to_csv("plot/gradient.csv", index=False, header=False)

    def plot_terrain_shadow(self, traj):
        X, Y, Z = self.terrain_slope(traj)
        # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        # surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
        #         cmap='viridis', antialiased=True)
        # ax.view_init(elev=90, azim=-90)

        # # Add a color bar which maps values to colors.
        # fig.colorbar(surf, shrink=0.3, aspect=10)
        # fig.tight_layout()
        c = 15
        Z = self.small_areas(Z, c)
        self.save_mat(Z, [X[0][0], Y[0][0]], "mountains", "brown")
        sun = 60 * PI / 180 # 60 degrees, altitude angle
        Z = self.mountain_shadow(Z, [X[0][0], Y[0][0]], PI / 6 , sun)
        tic = time()
        self.save_mat(Z, [X[0][0], Y[0][0]], "shadow_of_mordor", "grey")
        print(time() - tic)
        # Saving the terrain
        if (self.save_as_csv):
            xyz = [X.flatten('F'), Y.flatten('F'), Z.flatten('F')]
            df = pd.DataFrame(xyz)
            df.to_csv("plot/shadow.csv", index=False, header=False)

    def plot_ctow(self, traj):
        # Christian Theory of Wind
        X, Y, Z = self.terrain_slope(traj)
        wind_direction = 80
        wind_strength = 15 #m/s
        buff = 10
        # Mountain buffer
        threshold = 15
        D = self.small_areas(Z, threshold)
        p = self.mat_to_geoseries(D, [X[0][0], Y[0][0]])
        rad = 0.0002 * buff
        mountain_buffer = p.buffer(rad, 3)

        # Wind effect
        wind_effect_mat = self.christian_theory_of_wind(traj, Z, [X[0][0], Y[0][0]], wind_direction, wind_strength, buff)
        wind_effect = self.mat_to_geoseries(wind_effect_mat, [X[0][0], Y[0][0]])

        # Venturi

        # self.venturi_effect_geoseries(Z, [X[0][0], Y[0][0]], 300)
        venturi_mat = self.venturi_effect(traj, Z, [X[0][0], Y[0][0]], 300)
        venturi = self.mat_to_geoseries(venturi_mat, [X[0][0], Y[0][0]])
        # self.buff(Z, 5)

        # Combine
        # p = mountain_buffer.union(wind_effect)
        self.save_geoseries(p, "mountain", "brown")
        self.save_geoseries(mountain_buffer, "mountain_buffer", "grey")
        self.save_geoseries(wind_effect, "wind_effect", "purple")
        self.save_geoseries(venturi, "venturi", "red")
        # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        # surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
        #         cmap='viridis', antialiased=True)
        # ax.view_init(elev=90, azim=-90)

        # # Add a color bar which maps values to colors.
        # fig.colorbar(surf, shrink=0.3, aspect=10)
        # fig.tight_layout()
        files = ["plot/mountain.geojson",
        "plot/mountain_buffer.geojson",
        "plot/wind_effect.geojson",
        "plot/venturi.geojson"]
        self.combine_json_files(files, "ctow")
        return p, mountain_buffer, wind_effect, venturi

    def get_terrain_section(self, traj):
        SAMPLES = self.SAMPLES
        # Getting the boundaries
        points = []
        for i in range(len(traj)):
            point = Point(traj[i][0], traj[i][1])
            points.append(point)
        t = LineString(points)
        minx, miny, maxx, maxy = t.bounds
        # minx, miny, maxx, maxy = bounds
        lminx = int(round((minx - int(minx)) * (SAMPLES - 1), 0))
        lminy = int(round((miny - int(miny)) * (SAMPLES - 1), 0))
        lmaxx = int(round((maxx - int(maxx)) * (SAMPLES - 1), 0))
        lmaxy = int(round((maxy - int(maxy)) * (SAMPLES - 1), 0))
        # lminy = SAMPLES - lminy
        # lmaxy = SAMPLES - lmaxy
        lminx, lminy, lmaxx, lmaxy = self.square_terrain_plot(lminx, lminy, lmaxx, lmaxy)
        # Getting a section of terrain to work with
        section = self.elv[lminy:lmaxy, lminx:lmaxx]
        X = range(lminx, lmaxx)
        Y = range(lminy, lmaxy)
        X, Y = np.meshgrid(X, Y)
        return X, Y, section

    def get_terrain_traj(self, traj):
        sampled = self.sample_line(traj)
        # Find elevation at samples
        # https://e4ftl01.cr.usgs.gov/provisional/MEaSUREs/NASADEM/Africa/hgt_merge/
        file = 'terrain/N43E005.hgt'   #Source of file is from the link above
        SAMPLES = self.SAMPLES # Change this to 3601 for SRTM1
        X = []
        Y = []
        elv = []
        with open(file, 'rb') as file:
            # Each data is 16bit signed integer(i2) - big endian(>)
            for i in range(len(sampled.coords)):
                lon = sampled.coords[i][0]
                lat = sampled.coords[i][1]
                lon_row = int(round((lon - int(lon)) * (SAMPLES - 1), 0))
                lat_row = int(round((lat - int(lat)) * (SAMPLES - 1), 0))
                # lon_row = SAMPLES - lon_row
                # lat_row = SAMPLES - lat_row
                X.append(lon_row)
                Y.append(lat_row)
                elv.append(self.elv[lat_row][lon_row])
        return X, Y, elv

    def terrain_slope(self, traj):
        X, Y, section = self.get_terrain_section(traj)
        slope = np.gradient(section)
        Z = np.sqrt(np.square(slope[0]) + np.square(slope[1]))
        c = 25 #30 is good too
        Z = self.threshold(Z, c)
        # Use blur to soften the edges, and fill smaller areas ontop of mountains
        return X, Y, Z

    def threshold(self, Z, threshold):
        mask0 = np.where(Z < threshold)
        mask1 = np.where(Z >= threshold)
        Z[mask0] = 0
        Z[mask1] = 1
        return Z

    def mat_to_geoseries(self, Z, p_i):
        SAMPLES = self.SAMPLES

        rad = 0.002
        # rad = 0.0007
        result = np.where(Z == 1)
        my = self.lat_ + (result[0] + p_i[1]) / SAMPLES
        mx = self.lon_ + (result[1] + p_i[0]) / SAMPLES
        return gp.GeoSeries(gp.GeoSeries.from_xy(mx, my).buffer(rad, resolution=3).unary_union)

    def save_mat(self, Z, p_i, name, color):
        p = self.mat_to_geoseries(Z, p_i)
        self.save_geoseries(p, name, color)

    def save_geoseries(self, p, name, color):
        opacity = 0.6
        s = gp.GeoDataFrame(crs=4326, geometry=p)
        s["fill-opacity"] = 0.3 * opacity
        s["stroke-opacity"] = opacity
        s["stroke"] = color
        s["marker-color"] = color
        s["fill"] = color
        s.to_file('plot/' + name + '.geojson', driver='GeoJSON')

    def small_areas_geoseries(self, p, threshold):
        threshold *= 1e-5
        large = []
        for i, row in p.iteritems():
            # mp = row.explode(ignore_index=True)
            for i in range(len(row.geoms)):
                # does_intersect = self.fa_gs[i].intersects(linestring)
                if row.geoms[i].area > threshold:
                    large.append(row.geoms[i])

        return gp.GeoSeries(MultiPolygon(large))

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

    def mountain_shadow(self, Z, p_i, azimuth, altitude):
        # Where the mountains are, which way the wind is blowing from, and how high the sun is in the sky (how far the shadow reaches)
        arr = skimage.measure.label(Z > 0)
        shape = [0, 0] + list(arr.shape)
        altitude = altitude * 5 # Intended as angle, but can be seen as percentage
        cell_size = 3600 / (self.SAMPLES - 1) * 10
        max_diff = 1000 # Max predicted height difference to generate the whole pattern
        dist = int(altitude * max_diff / cell_size)
        pattern = self.minecraft_line(dist, azimuth)
        # pattern = [range(dist), [0] * dist]
        for region in skimage.measure.regionprops(arr):
            box = region['bbox']
            avg_h = self.average_height(box)
            for index in region['coords']:
                h = self.elv[index[0] + p_i[0]][index[1] + p_i[1]] - avg_h
                # Every cell is 30m, and we want to move by 40% of the height
                dist = altitude * h / cell_size
                for i in range(pattern.shape[1]):
                    if dist * dist < pattern[0][i] * pattern[0][i] + pattern[1][i] * pattern[1][i]:
                        break
                    next_row = int(index[0] + pattern[0][i])
                    next_col = int(index[1] + pattern[1][i])
                    if (self.within([next_row, next_col], shape)):
                        arr[next_row][next_col] = 1

        Z = self.threshold(arr, 1)
        return Z

    def christian_theory_of_wind(self, traj, Z, p_i, wind_direction, wind_strength, buff):
        wind_strength *= 1.5
        downwind = 0.4
        upwind = 0.1
        c = 15
        Z = self.small_areas(Z, c)
        # self.venturi_effect_geoseries(Z, p_i, 300)
        # self.venturi_effect(traj, Z, p_i, 300)
        # self.buff(Z, 5)
        azimuth = wind_direction * PI / 180 + PI
        altitude = wind_strength * PI / 180 # 60 degrees, altitude angle
        # Downwind effect
        Z1 = self.mountain_shadow(Z, [p_i[0], p_i[1]], azimuth , altitude * downwind)
        # Upwind effect
        azimuth = azimuth + PI
        Z2 = self.mountain_shadow(Z, [p_i[0], p_i[1]], azimuth , altitude * upwind)
        Z = np.logical_or(Z1, Z2)
        return Z

    def venturi_effect_geoseries(self, mountains, p_i, buffer):
        buffer *= 1e-5
        # Buffer each mountain indiviually
        mount = self.mat_to_geoseries(mountains, p_i).explode(ignore_index=True).buffer(buffer)
        # Check for intersections
        intersections = []
        for i in range(len(mount)):
            for j in range(i + 1, len(mount)):
                intersections.append(mount[i].intersection(mount[j]))
        # Save intersections
        venturi = gp.GeoSeries(intersections)
        self.save_geoseries(venturi, "venturi_geoseries", "red")
        return venturi

    def venturi_effect(self, traj, Z, p_i, buffer):
        c = 15
        mountains = self.small_areas(Z, c)
        buffer = 5
        M = self.buffer_areas(mountains, buffer)
        Z = np.zeros(mountains.shape)
        # Finding the intersection of the area between the mountains after adding the buffer area
        N = len(M)
        for i in range(N):
            for j in range(i + 1, N):
                intersection = np.logical_and(M[i], M[j])
                # Adding the intersection to the matrix of areas between mountains
                Z = np.logical_or(Z, intersection)
        self.save_mat(Z, p_i, "venturi", "red")
        return Z

    def valley(self, traj, Z, p_i):
        c = 15
        mountains = self.small_areas(Z, c)
        print("Started water simulation")
        _, _, elv = self.get_terrain_section(traj)
        water = self.trough(Z, elv)
        # water = self.remove_mountains(water, mountains)
        print("Finished water simulation")
        self.save_mat(water, p_i, "water", "blue")

        # mountains_enhanced = np.zeros(mountains.shape)
        # M = self.buffer_areas(mountains, 1)
        # for i in range(N):
        #     mountains_enhanced = np.logical_or(mountains_enhanced, M[i])
        mountains_enhanced = self.fill_areas(mountains)
        self.save_mat(mountains_enhanced, p_i, "mountains", "brown")

        water = self.remove_mountains(water, mountains_enhanced)
        self.save_mat(water, p_i, "enhanced", "blue")

        # valley = self.filter_areas(mountains_enhanced, water)
        # self.save_mat(valley, p_i, "valley", "purple")

        # all_valleys = self.all_areas_between_mountains(mountains)
        # self.save_mat(all_valleys, p_i, "all_valleys", "green")
        return water

    def trough(self, Z, elv):
        sim_duration = 16
        water = np.zeros(Z.shape)
        for i in range(sim_duration):
            if (i < 5):
                water = self.add_water(Z, water)
            water = self.simulate_water(elv, water)
        water = water > 0.1
        return water

    def simulate_water(self, Z, water):
        viscosity = 0.5
        # Find gradient
        total_elv = Z + water
        # grad = np.gradient(total_elv)
        pattern = [[0, 1],
        [1, 0],
        [0, -1],
        [-1, 0]]
        # Check movement
        # Move water
        difference = np.zeros([4, 1])
        for i in range(1, total_elv.shape[0] - 1):
            for j in range(1, total_elv.shape[1] - 1):
                if water[i][j] != 0:
                    for idx, k in enumerate(pattern):
                        difference[idx] = total_elv[i][j] - total_elv[i + k[0]][j + k[1]]
                    difference[difference < 0] = 0
                    sum = np.sum(difference)
                    if (sum == 0):
                        continue # No water to move because it's the lowest point
                    for idx, k in enumerate(pattern):
                        water[i + k[0]][j + k[1]] += viscosity * difference[idx] / sum * water[i][j]
                    water[i][j] -= viscosity * water[i][j]
        return water

    def add_water(self, Z, water):
        c = 1
        Z = self.threshold(Z, 1)
        water += c * Z
        return water

    def buffer_areas(self, Z, buffer):
        # Returns a list of matricies, with each one being a mountain with a buffer
        arr, N = skimage.measure.label(Z > 0, return_num=True)
        m = np.zeros(Z.shape)
        M = []
        # Finding the mountains, and adding a buffer to each mountains
        for i in range(N):
            coords = skimage.measure.regionprops(arr)[i]['coords']
            m = np.zeros(Z.shape)
            m[coords[:, 0], coords[:, 1]] = 1
            m = self.buff(m, buffer)
            M.append(m)
        return M

    def buff(self, Z, buffer):
        pattern = skimage.morphology.disk(buffer)
        Z = skimage.morphology.dilation(Z, pattern)
        return Z

    def remove_mountains(self, water, mountains):
        # Remove the water ontop of mountains
        mountains = np.logical_not(mountains)
        water = np.logical_and(mountains, water)
        return water

    def fill_areas(self, Z):
        arr, N = skimage.measure.label(Z > 0, return_num=True)
        for i in range(N):
            bbox = skimage.measure.regionprops(arr)[i]['bbox']
            Z[bbox[0]:bbox[2], bbox[1]:bbox[3]] = skimage.measure.regionprops(arr)[i]['image_filled']
        return Z

    def filter_areas(self, mountains, water):
        # Only keep the water between the mountains
        # Max distance between mountains
        max_dist = 1000 #m
        max_dist = max_dist / 30 # Every element is 30m
        max_dist = max_dist * max_dist # Square the distance
        arr, N = skimage.measure.label(mountains > 0, return_num=True)
        valley = np.zeros(mountains.shape)
        temp = np.zeros(mountains.shape)
        for i in range(N):
            for j in range(i + 1, N):
                m1 = skimage.measure.regionprops(arr)[i]
                m2 = skimage.measure.regionprops(arr)[j]
                # Measure the distance between the mountains
                dist = (m1['centroid'][0] - m2['centroid'][0]) * (m1['centroid'][0] - m2['centroid'][0]) +\
                    (m1['centroid'][1] - m2['centroid'][1]) * (m1['centroid'][1] - m2['centroid'][1])
                if dist < max_dist:
                    temp.fill(0) # Reset the array
                    # Combine the two mountains
                    bbox = m1['bbox']
                    temp[bbox[0]:bbox[2], bbox[1]:bbox[3]] = m1['image_filled'] # Use image if there are any issues
                    bbox = m2['bbox']
                    temp[bbox[0]:bbox[2], bbox[1]:bbox[3]] = m2['image_filled']

                    # Intersect it with water
                    convex = skimage.morphology.convex_hull_image(temp)
                    intersection = np.logical_and(convex, water)

                    # Add water to output
                    valley = np.logical_or(intersection, valley)
        return valley

    def all_areas_between_mountains(self, mountains):
        # Only keep the water between the mountains
        # Max distance between mountains
        max_dist = 1000 #m
        max_dist = max_dist / 30 # Every element is 30m
        max_dist = max_dist * max_dist # Square the distance
        arr, N = skimage.measure.label(mountains > 0, return_num=True)
        valley = np.zeros(mountains.shape)
        temp = np.zeros(mountains.shape)
        mountains = np.logical_not(mountains)
        for i in range(N):
            for j in range(i + 1, N):
                m1 = skimage.measure.regionprops(arr)[i]
                m2 = skimage.measure.regionprops(arr)[j]
                # Measure the distance between the mountains
                dist = (m1['centroid'][0] - m2['centroid'][0]) * (m1['centroid'][0] - m2['centroid'][0]) +\
                    (m1['centroid'][1] - m2['centroid'][1]) * (m1['centroid'][1] - m2['centroid'][1])
                if dist < max_dist:
                    temp.fill(0) # Reset the array
                    # Combine the two mountains
                    bbox = m1['bbox']
                    temp[bbox[0]:bbox[2], bbox[1]:bbox[3]] = m1['image_filled'] # Use image if there are any issues
                    bbox = m2['bbox']
                    temp[bbox[0]:bbox[2], bbox[1]:bbox[3]] = m2['image_filled']

                    # Intersect it with water
                    convex = skimage.morphology.convex_hull_image(temp)
                    # Remove the mountains by A&~B
                    # ~B is before the start of the for loop
                    intersection = np.logical_and(convex, mountains)

                    # Add water to output
                    valley = np.logical_or(intersection, valley)
        return valley

    def max_height(self, indices):
        max_h = 0
        for index in indices:
            if (self.elv[index[0]][index[1]] > max_h):
                max_h = self.elv[index[0]][index[1]]
        return max_h

    def find_peaks(self, area):
        # 1.5km
        # around peaks, 1.5x
        # Local maximum

        return 0

    def average_height(self, indices):
        cumsum = 0
        for i in range(indices[0], indices[2]):
            for j in range(indices[1], indices[3]):
                cumsum = cumsum + self.elv[i][j]
        total = (indices[2] - indices[0]) * (indices[3] - indices[1])
        # return cumsum / total
        return 0

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

    def plot_pressure(self, elv, traj, X):
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

        # print(self.haversine(traj[0][0], traj[0][1], traj[1][0], traj[1][1]))
        # print(sqrt((s.geometry[0].x - s.geometry[1].x) * (s.geometry[0].x - s.geometry[1].x) +
        #  (s.geometry[0].y - s.geometry[1].y) * (s.geometry[0].y - s.geometry[1].y)))

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

        # print(self.haversine(traj[0][0], traj[0][1], traj[1][0], traj[1][1]))
        # print(sqrt((s.geometry[0].x - s.geometry[1].x) * (s.geometry[0].x - s.geometry[1].x) +
        #  (s.geometry[0].y - s.geometry[1].y) * (s.geometry[0].y - s.geometry[1].y)))

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
            x.append((s.geometry[i].x - x0) / 1000)
            y.append((s.geometry[i].y - y0) / 1000)
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

    def plot_elevation_traj(self, elv):
        num_samples = 20
        min_height = self.min_h
        max_height = self.max_h
        traj_elv = [0] * len(elv)
        # Takeoff
        traj_elv[0:num_samples] = [min_height + max(elv[0:num_samples])] * num_samples
        # Landing
        traj_elv[-num_samples:] = [min_height + max(elv[-num_samples:-1])] * num_samples
        for i in range(num_samples, len(elv) - num_samples):
            traj_elv[i] = min_height + max(elv[i - num_samples:i + num_samples])

        for i in range(len(elv) - num_samples, len(elv)):
            traj_elv[i] = min_height * ((len(elv) - i)/num_samples) + max(elv[i:])

        return traj_elv

    def plot_elevation_traj_2(self, elv):
        num_samples = 20
        min_height = self.min_h
        max_height = self.max_h
        angle = 10
        levels = 30
        short_dip = 2.5 * angle

        traj_elv = [0] * len(elv)

        # # Takeoff
        # traj_elv[0:num_samples] = [min_height + max(elv[0:num_samples])] * num_samples
        # # Landing
        # traj_elv[-num_samples:] = [min_height + max(elv[-num_samples:-1])] * num_samples

        # Takeoff
        traj_elv[0:num_samples] = [min_height + max(elv[0:num_samples])] * num_samples

        flat = 30
        vertical = 100
        takeoff_angle = 10
        takeoff_altitude = 20
        for i in range(takeoff_angle):
            traj_elv[i] = elv[0] + vertical + takeoff_altitude / takeoff_angle * i
        # The flat part
        for i in range(takeoff_angle, takeoff_altitude + takeoff_angle):
            traj_elv[i] = elv[0] + vertical + takeoff_altitude


        # Landing
        traj_elv[-num_samples:] = [min_height + max(elv[-num_samples:-1])] * num_samples
        for i in range(len(elv) - num_samples, len(elv)):
            traj_elv[i] = min_height * ((len(elv) - i)/num_samples) + max(elv[i:])
        # Final approach
        flat = 30
        landing_angle = 10
        landing_altitude = 20

        # The flat part
        for i in range(len(elv) - (flat + landing_angle), len(elv) - landing_angle):
            traj_elv[i] = elv[-1] + landing_altitude

        for i in range(len(elv) - (landing_angle), len(elv)):
            traj_elv[i] = elv[-1] + landing_altitude / landing_angle * (len(elv) - i)

        takeoff = takeoff_altitude + takeoff_angle
        landing = flat + landing_angle
        # Middle part
        for i in range(num_samples, len(elv) - num_samples):
            traj_elv[i] = min_height + max(elv[i - num_samples:i + num_samples])

        # Landing
        for i in range(len(elv) - num_samples, len(elv)):
            traj_elv[i] = min_height * ((len(elv) - i)/num_samples) + max(elv[i:])

        # Trigger altitude
        for i in range(num_samples, len(elv) - num_samples):
            traj_elv[i] = ceil(traj_elv[i] / levels) * levels

        # Cleaning short dips
        dip = 0
        for i in range(num_samples, len(elv) - num_samples):
            if traj_elv[i] > traj_elv[i + 1]:
                dip = i - 3
            if traj_elv[i] < traj_elv[i + 1]:
                if dip != 0 and i - dip < short_dip:
                    traj_elv[dip:i + 1] = [traj_elv[i + 1]] * (i - dip + 1)
                    dip = 0

        orig = traj_elv
        rate = levels / angle

        # Descending
        for i in range(num_samples, len(traj_elv) - num_samples):
            if traj_elv[i] > orig[i + 1]:
                traj_elv[i + 1] = traj_elv[i] - rate
        # Ascending
        for i in range(len(traj_elv) - num_samples, num_samples, -1):
            if traj_elv[i] > orig[i - 1]:
                traj_elv[i - 1] = traj_elv[i] - rate


        return traj_elv

    def plot_elevation_traj_3(self, elv):
        num_samples = 20
        min_height = self.min_h
        max_height = self.max_h
        angle = 20
        levels = 50
        height = 350
        traj_elv = [0] * len(elv)

        # Takeoff
        traj_elv[0:num_samples] = [min_height + max(elv[0:num_samples])] * num_samples

        flat = 30
        vertical = 100
        takeoff_angle = 10
        takeoff_altitude = 20
        for i in range(takeoff_angle):
            traj_elv[i] = elv[0] + vertical + takeoff_altitude / takeoff_angle * i
        # The flat part
        for i in range(takeoff_angle, takeoff_altitude + takeoff_angle):
            traj_elv[i] = elv[0] + vertical + takeoff_altitude


        # Landing
        traj_elv[-num_samples:] = [min_height + max(elv[-num_samples:-1])] * num_samples
        for i in range(len(elv) - num_samples, len(elv)):
            traj_elv[i] = min_height * ((len(elv) - i)/num_samples) + max(elv[i:])
        # Final approach
        flat = 30
        landing_angle = 10
        landing_altitude = 20

        # The flat part
        for i in range(len(elv) - (flat + landing_angle), len(elv) - landing_angle):
            traj_elv[i] = elv[-1] + landing_altitude

        for i in range(len(elv) - (landing_angle), len(elv)):
            traj_elv[i] = elv[-1] + landing_altitude / landing_angle * (len(elv) - i)

        takeoff = takeoff_altitude + takeoff_angle
        landing = flat + landing_angle

        # Middle part
        for i in range(takeoff, len(elv) - landing):
            traj_elv[i] = min_height + max(elv[i - num_samples:i + num_samples])

        # Trigger altitude
        for i in range(takeoff, len(elv) - landing):
            traj_elv[i] = ceil(traj_elv[i] / levels) * levels
            if traj_elv[i] < height:
                traj_elv[i] = height

        orig = traj_elv
        rate = levels / angle

        # Descending
        # The one was landing
        for i in range(takeoff, len(traj_elv) - 1):
            if traj_elv[i] - rate > orig[i + 1]:
                traj_elv[i + 1] = traj_elv[i] - rate
        # Ascending
        # The zero was takeoff
        for i in range(len(traj_elv) - landing, 0, -1):
            if traj_elv[i] - rate > orig[i - 1]:
                traj_elv[i - 1] = traj_elv[i] - rate

        traj_elv[0] = elv[0]
        return traj_elv

    def plot_elevation_traj_4(self, elv):
        num_samples = 20
        min_height = self.min_h
        max_height = self.max_h
        angle = 20
        levels = 30
        height = 350
        short_dip = 2.5 * angle

        traj_elv = [0] * len(elv)

        # Takeoff
        traj_elv[0:num_samples] = [min_height + max(elv[0:num_samples])] * num_samples

        flat = 30
        vertical = 100
        takeoff_angle = 10
        takeoff_altitude = 20
        for i in range(takeoff_angle):
            traj_elv[i] = elv[0] + vertical + takeoff_altitude / takeoff_angle * i
        # The flat part
        for i in range(takeoff_angle, takeoff_altitude + takeoff_angle):
            traj_elv[i] = elv[0] + vertical + takeoff_altitude


        # Landing
        traj_elv[-num_samples:] = [min_height + max(elv[-num_samples:-1])] * num_samples
        for i in range(len(elv) - num_samples, len(elv)):
            traj_elv[i] = min_height * ((len(elv) - i)/num_samples) + max(elv[i:])
        # Final approach
        flat = 30
        landing_angle = 10
        landing_altitude = 20

        # The flat part
        for i in range(len(elv) - (flat + landing_angle), len(elv) - landing_angle):
            traj_elv[i] = elv[-1] + landing_altitude

        for i in range(len(elv) - (landing_angle), len(elv)):
            traj_elv[i] = elv[-1] + landing_altitude / landing_angle * (len(elv) - i)

        takeoff = takeoff_altitude + takeoff_angle
        landing = flat + landing_angle

        # Middle part
        for i in range(takeoff, len(elv) - landing):
            traj_elv[i] = min_height + max(elv[i - num_samples:i + num_samples])

        # Trigger altitude
        for i in range(takeoff, len(elv) - landing):
            traj_elv[i] = ceil(traj_elv[i] / levels) * levels

        # Cleaning short dips
        dip = 0
        for i in range(num_samples, len(elv) - num_samples):
            if traj_elv[i] > traj_elv[i + 1]:
                dip = i - 3
            if traj_elv[i] < traj_elv[i + 1]:
                if dip != 0 and i - dip < short_dip:
                    traj_elv[dip:i + 1] = [traj_elv[i + 1]] * (i - dip + 1)
                    dip = 0

        orig = traj_elv
        rate = levels / angle

        # Descending
        # The one was landing
        for i in range(takeoff, len(traj_elv) - 1):
            if traj_elv[i] - rate > orig[i + 1]:
                traj_elv[i + 1] = traj_elv[i] - rate
        # Ascending
        # The zero was takeoff
        for i in range(len(traj_elv) - landing, 0, -1):
            if traj_elv[i] - rate > orig[i - 1]:
                traj_elv[i - 1] = traj_elv[i] - rate

        traj_elv[0] = elv[0]
        return traj_elv

    def plot_contours(self, traj):
        X, Y, Z = self.terrain_slope(traj)
        c = 15
        Z = self.small_areas(Z, c)
        sun = 60 * PI / 180 # 60 degrees, altitude angle
        Z = self.mountain_shadow(Z, [X[0][0], Y[0][0]], PI / 6 , sun)
        arr = skimage.measure.label(Z > 0)
        scale = 68 / 1000 # 30m per cell, 1000m per km
        contours = skimage.measure.find_contours(arr, 0.8)
        # contours = contours * scale

        ###################################################################
        # Plotting side and top view
        plt.subplots(nrows=1)
        plt.subplot(111)
        ax = plt.gca()
        ax.yaxis.grid()

        for i in range(len(contours)):
            x = (contours[i][:, 1]) * scale
            y = (contours[i][:, 0] - len(arr) / 2) * scale
            plt.plot(x, y, color='grey')
            # plt.fill(x, y, color='grey')

        x, y = self.traj_to_xy_n(traj, len(X) - 1)
        plt.plot(x, y)

        ax = plt.gca()
        ax.grid()
        ax.set_aspect('equal')
        plt.title("Top view of the trajectory")
        plt.xlabel("Position")
        plt.ylabel("Position")
        return 0

    def plot_debug(self, elv):
        traj_elv = []

        for i in range(len(elv)):
            traj_elv.append(elv[i] + 20)
        return traj_elv

    def pressure(self, reference, h):
        Pb = 101325
        g0 = 9.80665
        M = 0.0289644
        # h = 0 # height
        hb = 0 # elv
        Tb = 288.15 # 15C
        R = 8.3144598
        height = []
        for i in range(len(reference)):
            height.append(h[i] - reference[i])

        pressure = []
        for i in range(len(reference)):
            pressure.append(Pb*exp(-g0 * M * (h[i] - reference[i])/ (R* Tb)))
        return pressure

    def height(self, qnh, pressure):
        Pb = 101325
        g0 = 9.80665
        M = 0.0289644
        h = 0 # height
        hb = 0 # elv
        Tb = 288.15 # 15C
        R = 8.3144598
        height = []
        for i in range(len(qnh)):
            height.append(-log(pressure[i] / qnh[i]) * R*Tb / (g0 * M) + hb)
        return height

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