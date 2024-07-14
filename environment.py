# Importing standard libraries
from math import radians, cos, sin, asin, sqrt
import time
from datetime import datetime
import random
from multiprocessing import Pool

import numpy as np
import geopandas as gp
import pandas as pd
import shapely
from shapely.geometry import box, Point
import glob

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.collections
import mpl_toolkits.mplot3d as a3
# from matplotlib._png import read_png
from matplotlib.cbook import get_sample_data

class env():
    def __init__(self):
        self.read_houses()
        self.read_apartments()
        self.add_height()
        self.transform_coords()
        # self.plot()
        nearby = self.nearby_obstacles([4, 5, 8], 10)
        # print(nearby)

    def read_houses(self):
        # Reading the file
        filename = "env/houses.geojson"
        file = open(filename)
        houses_df = gp.read_file(file)
        # for col in houses_df.columns:
        #     print(col)
        # print(apartments_df.loc[:, "addr:housenumber"].to_string())

        # Name, polygon, freq
        # Putting the houses into a dictionary
        N = houses_df.geometry.size
        houses_dict = []
        for index, row in houses_df.iterrows():
            if row.geometry.geom_type == "Polygon":
                p = row.geometry.convex_hull
                edges = len(p.exterior.coords) - 1
                if row["addr:housenumber"] != None:
                    houses_dict.append({'name': row["addr:housenumber"], 'geom': p, 'freq': 1, 'n_edges': edges})

                if row["name"] != None:
                    houses_dict.append({'name': row["name"], 'geom': p, 'freq': 1, 'n_edges': edges})

        df = pd.DataFrame(houses_dict)
        gdf = gp.GeoDataFrame(df, crs=4326, geometry=df['geom'])
        del gdf['geom']
        self.houses = gdf

        color = "green"
        name = "houses"
        gdf["stroke"] = color
        gdf["marker-color"] = color
        gdf["fill"] = color
        gdf.to_file('plot/' + name + '.geojson', driver='GeoJSON')

    def read_apartments(self):
        # Reading the file
        filename = "env/apartments.geojson"
        file = open(filename)
        apartments_df = gp.read_file(file)

        # Name, polygon, freq
        # Putting the apartments into a dictionary
        N = apartments_df.geometry.size
        apt_dict = []
        for index, row in apartments_df.iterrows():
            if row.geometry.geom_type == "Polygon":
                p = row.geometry.convex_hull
                if row["addr:housenumber"] != None:
                    apt_dict.append({'name': row["addr:housenumber"], 'geom': p, 'freq': 1})

                if row["name"] != None:
                    apt_dict.append({'name': row["name"], 'geom': p, 'freq': 1})

        df = pd.DataFrame(apt_dict)
        gdf = gp.GeoDataFrame(df, crs=4326, geometry=df['geom'])
        del gdf['geom']
        self.apt = gdf
        color = "orange"
        name = "apts"
        gdf["stroke"] = color
        gdf["marker-color"] = color
        gdf["fill"] = color
        gdf.to_file('plot/' + name + '.geojson', driver='GeoJSON')

    def add_height(self):
        placeholder_height = 20
        # placeholder_height = np.array([0, placeholder_height])
        self.apt.insert(2, "zmin", 0)
        self.apt.insert(2, "zmax", placeholder_height)
        name = "apts_h"
        self.apt.to_file('plot/' + name + '.geojson', driver='GeoJSON')

        self.houses.insert(2, "zmin", 0)
        self.houses.insert(2, "zmax", placeholder_height)
        # self.houses.insert(2, "Height", placeholder_height)
        name = "houses_h"
        self.houses.to_file('plot/' + name + '.geojson', driver='GeoJSON')
        # for index, row in self.apt.iterrows():

    def transform_coords(self):
        gdf = self.houses
        # Converting to meters projection
        gdf.to_crs(epsg=20437, inplace=True)
        # Finding the geometric center of the geometries
        offset = gdf.unary_union.centroid

        # Centering the area around that point (centroid)
        gdf.geometry = gdf.translate(-offset.x, -offset.y)
        self.write_geom(gdf, "translated", "blue")
        self.houses = gdf
        return gdf

    def write_geom(self, gdf, name, color):
        s = gdf
        s["stroke"] = color
        s["marker-color"] = color
        s["fill"] = color
        s.to_file('plot/' + name + '.geojson', driver='GeoJSON')

    def plot(self):
        # Initial plot setup
        # self.fig = plt.figure(figsize=(10, 8))
        plt.rcParams["figure.autolayout"] = True
        self.fig = plt.figure(tight_layout=True)
        self.ax = self.fig.add_subplot(111, projection='3d')  # 'self.ax' to make it accessible in other methods
        #(-1701.4044408848858, -3020.0373695367016, 2851.8315827711485, 1473.981712395791)
        # self.ax.set_xlim([-1000, 1000])
        # self.ax.set_ylim([-1000, 1000])
        # self.ax.set_zlim([-100, 300])
        self.ax.set_xlabel('X Axis')
        self.ax.set_ylabel('Y Axis')
        self.ax.set_zlabel('Z Axis')

        for index, row in self.houses.iterrows():
            building = row.geometry
            height = row.height
            # self.draw_polyhedron(self.ax, building, height)  # Pass 'self.ax' here
            # print(index)
        # Initialize lines for each vehicle
        # self.lines = [self.ax.plot([], [], [], 'o-', linewidth=1, markersize=1)[0] for _ in range(len(self.drn))]
        img = plt.imread("plot/Figure_1.png")
        img = img[0:800, 0:800, :]
        rot = 0
        trans = [0, 0, 0]
        scale = 5
        X1, Y1, Z1 = self.transform_img(img, rot, scale, trans)


        # self.ax.set_aspect('auto')
        # self.ax.plot_surface(X1, Y1, Z1, rstride=1, cstride=1, facecolors=img, shade=False)
        # self.ax.plot_surface(X1, Y1, Z1, facecolors=img, shade=False)
        self.ax.imshow(img)


        # plt.show(block=False)
        plt.show()

    def draw_polyhedron(self, ax, obs, height, color='gray', alpha=0.3):
        # height = 20
        """Draws a 3D polyhedron representing an obstacle."""
        x, y = obs.exterior.coords.xy
        z = np.zeros_like(x)
        vb = tuple(zip(x,y,z))
        z = z + height
        vt = tuple(zip(x,y,z))
        faces = []
        N = len(x)
        for i in range(N):
            faces.append([i, (i+1)%N])

        for i in np.arange(len(faces)):
            square=[vb[faces[i][0]], vb[faces[i][1]], vt[faces[i][1]], vt[faces[i][0]]]
            face = a3.art3d.Poly3DCollection([square])
            face.set_edgecolor('k')
            face.set_alpha(alpha)
            ax.add_collection3d(face)

        # Top and bottom
        face = a3.art3d.Poly3DCollection([vt])
        face.set_edgecolor('k')
        face.set_alpha(alpha)
        ax.add_collection3d(face)

        face = a3.art3d.Poly3DCollection([vb])
        face.set_edgecolor('k')
        face.set_alpha(alpha)
        ax.add_collection3d(face)

        # plt.show()

    def transform_img(self, img, rot, scale, trans):
        boundary = [1, -1, -1, 1]
        bbox = gp.GeoSeries(box(boundary[0], boundary[1], boundary[2], boundary[3]))
        bbox = bbox.rotate(rot)
        bbox = bbox.scale(scale, scale)
        bbox = bbox.translate(trans[0], trans[1])
        t = bbox[0]
        print(t)
        # Make rectangle in geopandas
        # Translate and rotate
        # Take the output and put it in the following function

        # 10 is equal length of x and y axises of your surface
        # img = np.zeros((img.shape[0], img.shape[1]))
        stepX, stepY = 10. / img.shape[0], 10. / img.shape[1]
        t = list(t.exterior.coords)
        xi = t[0][0]
        yi = t[0][1]
        xf = t[2][0]
        yf = t[2][1]
        print(xi, xf, yi, yf)
        X1 = np.arange(xi, xf, stepX)
        Y1 = np.arange(yi, yf, stepY)
        X1, Y1 = np.meshgrid(X1, Y1)
        # Z1 = -2.01 * np.ones((X1.shape[0], Y1.shape[1]))
        Z1 = 0.1 * np.ones((X1.shape[0], Y1.shape[1]))

        # stride args allows to determine image quality
        # stride = 1 work slow
        print(img.shape)
        print(X1.shape)
        print(Y1.shape)
        print(Z1.shape)
        return X1, Y1, Z1

    def nearby_obstacles(self, pos, range):
        position = Point(pos[0], pos[1], pos[2])
        dist = self.houses.distance(position)
        return self.houses[dist<=range]


# def main():
#     enviroment = env()

# main()