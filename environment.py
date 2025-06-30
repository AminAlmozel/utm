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
from shapely.geometry import box, Point, LineString
import glob

from sim_io import myio as io

class env():
    def __init__(self):
        self.flow = 0.0003
        self.read_houses()
        self.read_apartments()
        self.read_restaurants()
        self.read_fire_station()
        self.add_height()
        self.transform_coords()
        self.sim_run = ""
        self.sim_latest = ""
        self.mission_log = []
        self.mission_progress = 0
        self.repeat = 1

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
                    houses_dict.append({'name': row["addr:housenumber"], 'geom': p, 'freq': 1 * self.flow, 'n_edges': edges})

                if row["name"] != None:
                    houses_dict.append({'name': row["name"], 'geom': p, 'freq': 1 * self.flow, 'n_edges': edges})

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
                    apt_dict.append({'name': row["addr:housenumber"], 'geom': p, 'freq': 1 * self.flow})

                if row["name"] != None:
                    apt_dict.append({'name': row["name"], 'geom': p, 'freq': 1 * self.flow})

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
        # Name, point, freq
        # Putting the restaurants into a dictionary
        restaurant_dict = []

        # Reading the file
        filename = "env/albaik.geojson"
        file = open(filename)
        orig_df = gp.read_file(file)
        for index, row in orig_df.iterrows():
            if row.geometry.geom_type == "Polygon":
                p = row.geometry.centroid
            else:
                p = row.geometry
            if row["amenity"] == "fast_food":
                restaurant_dict.append({'name': row["name"], 'geom': p, 'freq': 1 * self.flow, 'amenity': "cafe"})

        filename = "env/restaurants.geojson"
        file = open(filename)
        orig_df = gp.read_file(file)
        for index, row in orig_df.iterrows():
            if row.geometry.geom_type == "Polygon":
                p = row.geometry.centroid
            else:
                p = row.geometry
            if row["amenity"] == "restaurant":
                restaurant_dict.append({'name': row["name"], 'geom': p, 'freq': 0.5 * self.flow, 'amenity': "restaurant"})

            if row["amenity"] == "fast_food":
                restaurant_dict.append({'name': row["name"], 'geom': p, 'freq': 1 * self.flow, 'amenity': "fast_food"})

        filename = "env/cafe.geojson"
        file = open(filename)
        orig_df = gp.read_file(file)
        for index, row in orig_df.iterrows():
            if row.geometry.geom_type == "Polygon":
                p = row.geometry.centroid
            else:
                p = row.geometry
            if row["amenity"] == "cafe":
                restaurant_dict.append({'name': row["name"], 'geom': p, 'freq': 0.1 * self.flow, 'amenity': "cafe"})


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

    def transform_meter_global(self, geom):
        gdf = gp.GeoDataFrame(geometry=geom, crs="EPSG:20437")
        gdf.to_crs(epsg=4326, inplace=True)
        return gdf.geometry

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
        if self.repeat:
            missions = io.read_pickle(self.sim_latest, "mission_log")
            i = self.mission_progress
            restaurant = missions[i]["mission"][0]
            house = missions[i]["mission"][1]
            xi = missions[i]["state"][0]
            xf = missions[i]["state"][1]
            print("Going from %s to %s" % (restaurant["name"], house["name"]))
            self.mission_progress += 1

        else:
            restaurant, pi = self.random_restaurant(tod)
            vi = self.random_state(tod, restaurant)
            house, pf, vf = self.random_delivery_location(tod, restaurant)
            print("Going from %s to %s" % (restaurant["name"], house["name"]))
            xi = pi + vi
            xf = pf + vf
            mission = {"mission": [restaurant, house], "state": [xi, xf], "time": tod}
            self.mission_log.append(mission)
            io.log_to_pickle(self.mission_log, "mission_log", self.sim_run, self.sim_latest)
        return xi, xf

    def random_restaurant(self, tod):
        zmin = 20
        zmax = 30
        zi = random.randint(zmin, zmax)
        n = self.restaurants.shape[0] - 1
        # Using uniform distribution
        i = random.randint(0, n)
        i = 0
        temp = self.restaurants.iloc[i].geometry
        p = [temp.x, temp.y, zi]
        return self.restaurants.iloc[i], p

    def random_delivery_location(self, tod, restaurant):
        apt_chance = 70 #%
        if random.randint(0, 100) < apt_chance:
            house, pf = self.random_apt(tod, restaurant)
            vf = self.random_state(tod, house)
        else:
            house, pf = self.random_house(tod, restaurant)
            vf = self.random_state(tod, house)
        return house, pf, vf

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

    def random_fire(self, drones, iteration):
        # if self.repeat:
        #     missions = io.read_pickle(self.sim_latest, "mission_log")
        #     i = self.mission_progress
        #     fire_station = missions[i]["mission"][0]
        #     temp = fire_station.geometry
        #     house = missions[i]["mission"][1]
        #     xi = missions[i]["state"][0]
        #     xf = missions[i]["state"][1]
        #     pi = missions[i]["position"][0]
        #     pf = missions[i]["position"][1]
        #     self.mission_progress += 1
        # else:
        #     fire_station = self.fire_station.iloc[0]
        #     temp = fire_station.geometry
        #     zi = random.randint(0, 30)
        #     pi = [temp.x, temp.y, zi]
        #     vi = self.random_state(0, fire_station)
        #     xi = pi + vi

        #     # Choose a random house
        #     house, pf = self.random_house(12, "placeholder")
        #     vf = self.random_state(0, house)
        #     xf = pf + vf
        #     mission = {"mission": [fire_station, house], "state": [xi, xf], "position": [pi, pf], "time": iteration}
        #     self.mission_log.append(mission)
        #     io.log_to_pickle(self.mission_log, "mission_log", self.sim_run, self.sim_latest)
        print("Fire at %s" % (house["name"]))
        # Make a buffered line going from the fire station to that house
        # and the area surrounding the location of the fire
        ls = self.traj_to_linestring([pi, pf])
        g = gp.GeoSeries([ls.buffer(30), temp.buffer(50), Point(pf).buffer(100)])
        avoid = g.unary_union
        avoid = self.transform_meter_global([avoid])[0]
        io.write_geom([avoid], self.sim_run + "avoid", "red")
        io.write_geom([avoid], self.sim_latest + "avoid", "red")
        fire_duration = 1000 # Timesteps, which is 1000 * dt seconds
        start = iteration
        end = iteration + fire_duration
        io.log_timed_geom([avoid], [[start, end]], self.sim_run, self.sim_latest)

        # Reconstruct the trajectories of all the other drones
        trajs = []
        for drn in drones:
            if drn["alive"]:
                progress = drn["mission"]["progress"]
                waypoints = drn["mission"]["waypoints"]
                traj = self.waypoints_to_traj(waypoints)
                ls = self.traj_to_linestring(traj)
                trajs.append(ls)
        trajs = self.transform_meter_global(trajs)
        io.write_geom(trajs, "rerouting", "white")

        # Check for intersection with the fire response trajectory
        trajs = gp.GeoSeries(trajs)
        result = trajs.intersects(avoid)
        indices = np.where(result)[0]
        return xi, [xf, xi], indices, avoid

    def traj_to_linestring(self, traj):
        points = []
        for i in range(len(traj)):
            point = Point(traj[i][0], traj[i][1])
            points.append(point)
        s_line = LineString(points)
        return s_line

    def waypoints_to_traj(self, values):
        traj = []
        for waypoint in values:
            traj.append([waypoint["x"], waypoint["y"]])
        return traj
# def main():
#     enviroment = env()

# main()