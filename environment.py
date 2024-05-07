# Importing standard libraries
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



class env():
    def __init__(self):
        pass

    def read(self):
        filename = "env/*.geojson"
        list_of_files = glob.glob(filename)
        print(list_of_files)
        filename = "env/houses.geojson"
        file = open(filename)
        houses_df = gp.read_file(file)
        filename = "env/apartments.geojson"
        file = open(filename)
        apartments_df = gp.read_file(file)
        # for col in apartments_df.columns:
        #     print(col)

        # print(apartments_df.loc[:, "addr:housenumber"].to_string())
        # Name, polygon, freq,
        N = apartments_df.geometry.size
        apartments = []
        for index, row in apartments_df.iterrows():
            if row.geometry.geom_type == "Polygon":
                p = row.geometry.convex_hull
                if row["addr:housenumber"] != None:
                    apartments.append({'name': row["addr:housenumber"], 'geom': p, 'freq': 1})

                if row["name"] != None:
                    apartments.append({'name': row["name"], 'geom': p, 'freq': 1})

        s = []
        for apt in apartments:
            s.append(apt['geom'])

        df = gp.GeoDataFrame(crs=4326, geometry=s)
        color = "green"
        name = "updated"
        df["stroke"] = color
        df["marker-color"] = color
        df["fill"] = color
        df.to_file('plot/' + name + '.geojson', driver='GeoJSON')
            #     pass
            #

        # ls = np.zeros((1,2))
        # list_of_df = []
        # for filename in list_of_files:
            # file = open(filename)
            # df = gp.read_file(file)

        #     N = df.geometry.size
        #     gdf = df.to_crs(crs=4326)

        #     gdf = df.centroid

        #     list_of_df.append(gdf)

        #     ls_temp = np.zeros((N,2))
        #     for i in range(N):
        #         ls_temp[i, 0] = gdf[i].x
        #         ls_temp[i, 1] = gdf[i].y
        #     ls = np.concatenate((ls, ls_temp), axis=0)
        # ls = np.delete(ls, 0, 0)
        # self.se_ls = np.concatenate((self.se_ls, ls), axis=0)
        # self.ls_gdf = pd.concat(list_of_df, ignore_index=True)
        # self.gdf = pd.concat([self.se_gdf, self.ls_gdf], ignore_index=True)
        # print("Imported landing")

def main():
    enviroment = env()
    # optimization.start_simulation() # 205s
    enviroment.read() # 44s
    # optimization.m2_start_simulation() # 293s (invalid)
main()