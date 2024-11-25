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
        self.read_restaurants()
        self.read_fire_station()
        self.add_height()
        self.transform_coords()
        self.random_fire()
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
        self.apts = gdf
        color = "orange"
        name = "apts"
        gdf["stroke"] = color
        gdf["marker-color"] = color
        gdf["fill"] = color
        gdf.to_file('plot/' + name + '.geojson', driver='GeoJSON')

    def read_restaurants(self):
        # Reading the file
        filename = "env/restaurants.geojson"
        # filename = "env/cafe.geojson"
        file = open(filename)
        orig_df = gp.read_file(file)

        # Name, point, freq
        # Putting the restaurants into a dictionary
        restaurant_dict = []
        for index, row in orig_df.iterrows():
            if row.geometry.geom_type == "Polygon":
                p = row.geometry.centroid
            else:
                p = row.geometry
            if row["amenity"] == "restaurant":
                restaurant_dict.append({'name': row["name"], 'geom': p, 'freq': 1, 'amenity': "restaurant"})

            if row["amenity"] == "fast_food":
                restaurant_dict.append({'name': row["name"], 'geom': p, 'freq': 1, 'amenity': "fast_food"})

        filename = "env/cafe.geojson"
        file = open(filename)
        orig_df = gp.read_file(file)
        for index, row in orig_df.iterrows():
            if row.geometry.geom_type == "Polygon":
                p = row.geometry.centroid
            else:
                p = row.geometry
            if row["amenity"] == "cafe":
                restaurant_dict.append({'name': row["name"], 'geom': p, 'freq': 1, 'amenity': "cafe"})

        df = pd.DataFrame(restaurant_dict)
        gdf = gp.GeoDataFrame(df, crs=4326, geometry=df['geom'])
        del gdf['geom']
        self.restaurants = gdf
        color = "orange"
        name = "restaurants"
        gdf["stroke"] = color
        gdf["marker-color"] = color
        gdf["fill"] = color
        gdf.to_file('plot/' + name + '.geojson', driver='GeoJSON')

    def read_fire_station(self):
        # Reading the file
        filename = "env/fire_station.geojson"
        file = open(filename)
        fire_station_df = gp.read_file(file)
        # Name, polygon, freq
        # Putting the houses into a dictionary
        N = fire_station_df.geometry.size
        fire_station_dict = []
        for index, row in fire_station_df.iterrows():
            fire_station_dict.append({'name': row["name"], 'geom': row["geometry"]})
        df = pd.DataFrame(fire_station_dict)
        gdf = gp.GeoDataFrame(df, crs=4326, geometry=df['geom'])
        del gdf['geom']
        self.fire_station = gdf
        print(self.fire_station)

        color = "green"
        name = "fire_station"
        gdf["stroke"] = color
        gdf["marker-color"] = color
        gdf["fill"] = color
        gdf.to_file('plot/' + name + '.geojson', driver='GeoJSON')

    def add_height(self):
        placeholder_height = 20
        # placeholder_height = np.array([0, placeholder_height])
        self.apts.insert(2, "zmin", 0)
        self.apts.insert(2, "zmax", placeholder_height)
        name = "apts_h"
        self.apts.to_file('plot/' + name + '.geojson', driver='GeoJSON')

        self.houses.insert(2, "zmin", 0)
        self.houses.insert(2, "zmax", placeholder_height)
        # self.houses.insert(2, "Height", placeholder_height)
        name = "houses_h"
        self.houses.to_file('plot/' + name + '.geojson', driver='GeoJSON')
        # for index, row in self.apts.iterrows():

    def transform_coords(self):
        # Converting to meters projection
        self.houses.to_crs(epsg=20437, inplace=True)
        self.apts.to_crs(epsg=20437, inplace=True)
        self.restaurants.to_crs(epsg=20437, inplace=True)
        self.fire_station.to_crs(epsg=20437, inplace=True)

    def write_geom(self, gdf, name, color):
        s = gdf
        s["stroke"] = color
        s["marker-color"] = color
        s["fill"] = color
        s.to_file('plot/' + name + '.geojson', driver='GeoJSON')

    def nearby_obstacles(self, pos, range):
        position = Point(pos[0], pos[1], pos[2])
        dist = self.houses.distance(position)
        return self.houses[dist<=range]

    def random_mission(self, tod):
        restaurant, pi = self.random_restaurant(tod)
        vi = self.random_state(tod, restaurant)

        # apt, pf = self.random_apt(tod, restaurant)
        house, pf = self.random_house(tod, restaurant)
        vf = self.random_state(tod, house)
        print("Going from %s to %s" % (restaurant["name"], house["name"]))
        xi = pi + vi
        xf = pf + vf
        return xi, xf

    def random_restaurant(self, tod):
        zmin = 20
        zmax = 30
        zi = random.randint(zmin, zmax)
        n = self.restaurants.shape[0] - 1
        # Using uniform distribution
        i = random.randint(0, n)
        temp = self.restaurants.iloc[i].geometry
        p = [temp.x, temp.y, zi]
        return self.restaurants.iloc[i], p

    def random_house(self, tod, restaurant):
        zmin = 20
        zmax = 30
        zi = random.randint(zmin, zmax)
        n = self.houses.shape[0] - 1
        # Using uniform distribution
        i = random.randint(0, n)
        temp = self.houses.iloc[i].geometry.centroid
        p = [temp.x, temp.y, zi]
        return self.houses.iloc[i], p

    def random_apt(self, tod, restaurant):
        zmin = 20
        zmax = 30
        zi = random.randint(zmin, zmax)
        n = self.apts.shape[0] - 1
        # Using uniform distribution
        i = random.randint(0, n)
        temp = self.apts.iloc[i].geometry.centroid
        p = [temp.x, temp.y, 30]
        return self.apts.iloc[i], p

    def random_state(self, tod, loc):
        min_v = 0
        max_v = 1
        v = []
        for i in range(3):
            v.append(random.uniform(min_v, max_v))
        v = [0, 0, 0]
        return v

    def random_fire(self):
        fire_station = self.fire_station.iloc[0]
        pi = fire_station.geometry
        print(fire_station)
        print(pi)
        # Choose a random house
        house, pf = self.random_house(12, "placeholder")
        print(house)
        print(pf)

# def main():
#     enviroment = env()

# main()