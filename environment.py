# Importing standard libraries
from math import radians, cos, sin, asin, sqrt
import time
from datetime import datetime
import random
from multiprocessing import Pool

import geopandas as gp
import pandas as pd
import numpy as np
import shapely
import glob

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.collections
import mpl_toolkits.mplot3d as a3

class env():
    def __init__(self):
        self.read_houses()
        self.read_apartments()
        self.transform_coords()
        self.plot()

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
                if row["addr:housenumber"] != None:
                    houses_dict.append({'name': row["addr:housenumber"], 'geom': p, 'freq': 1})

                if row["name"] != None:
                    houses_dict.append({'name': row["name"], 'geom': p, 'freq': 1})

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

    def transform_coords(self):
        gdf = self.houses
        # Converting to meters projection
        gdf.to_crs(epsg=20437, inplace=True)
        # Finding the geometric center of the geometries
        offset = gdf.unary_union.centroid

        # Centering the area around that point (centroid)
        gdf.geometry = gdf.translate(-offset.x, -offset.y)
        print(gdf.unary_union.bounds)
        self.write_geom(gdf, "translated", "blue")
        self.houses = gdf
        return gdf

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

    def distance(self, lon1, lat1, lon2, lat2):
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

        # haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        return sqrt(dlon**2 + dlat**2)

    def write_geom(self, gdf, name, color):
        s = gdf
        s["stroke"] = color
        s["marker-color"] = color
        s["fill"] = color
        s.to_file('plot/' + name + '.geojson', driver='GeoJSON')

    def plot(self):
        # Initial plot setup
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')  # 'self.ax' to make it accessible in other methods
        #(-1701.4044408848858, -3020.0373695367016, 2851.8315827711485, 1473.981712395791)
        self.ax.set_xlim([-1000, 1000])
        self.ax.set_ylim([-1000, 1000])
        self.ax.set_zlim([-100, 300])
        self.ax.set_xlabel('X Axis')
        self.ax.set_ylabel('Y Axis')
        self.ax.set_zlabel('Z Axis')
        for index, row in self.houses.iterrows():
            building = row.geometry
            self.draw_polyhedron(self.ax, building)  # Pass 'self.ax' here
            print(index)
        # Initialize lines for each vehicle
        # self.lines = [self.ax.plot([], [], [], 'o-', linewidth=1, markersize=1)[0] for _ in range(len(self.drn))]

        # plt.show(block=False)
        plt.tight_layout()
        plt.show()


    def draw_polyhedron(self, ax, obs, color='gray', alpha=0.3):
        height = 20
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


def main():
    enviroment = env()

main()