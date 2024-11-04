from code import interact
from platform import java_ver
from re import L
from socket import AI_PASSIVE
from time import time
from webbrowser import GenericBrowser
import pandas as pd
import geopandas as gp
import numpy as np
import astar_uam as astar

from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import nearest_points
import glob
from math import radians, cos, sin, asin, atan2, sqrt, pi, ceil, exp, log

from matplotlib import cm
from matplotlib.ticker import LinearLocator
import matplotlib.pyplot as plt

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# 'EPSG:2154' for France
# epsg=4326  for equal area

class process:
    def __init__(self):
        # N43E005
        c1 = [5, 43]
        c2 = [6, 44]
        # c1 = [5.332591, 43.179651]
        # c2 = [5.779597, 43.571692]
        c1 = [5.461063, 43.402681]
        c2 = [5.765934, 43.566590]

        p = self.create_box(c1, c2)
        self.readfile()
        self.save_intersection(p)

    def readfile(self):
        # filename = "powerlines/reseau-aerien-basse-tension-bt.geojson"
        filename = "powerlines/powerlines.geojson"
        # filename = "plot/fa.geojson"
        file = open(filename)
        df = gp.read_file(file)

        N = df.geometry.size
        print(N)
        # gdf = df.to_crs(crs=4326)
        self.powerlines = df

    def create_box(self, c1, c2):
        # Getting the boundaries
        points = [Point(c1[0], c1[1]), Point(c2[0], c2[1])]
        t = LineString(points)
        minx, miny, maxx, maxy = t.bounds
        return box(minx, miny, maxx, maxy, ccw=True)

    def save_intersection(self, polygon):
        name = "powerlines"
        color = "red"
        opacity = 0.6
        intersection = self.powerlines.within(polygon)
        # intersection = self.powerlines.intersection(polygon)
        s = self.powerlines[intersection]
        # s["fill-opacity"] = 0.3 * opacity
        # s["stroke-opacity"] = opacity
        # s["stroke"] = color
        # s["marker-color"] = color
        # s["fill"] = color
        s.to_file('plot/' + name + '.geojson', driver='GeoJSON')



def main():
    a = process()
    print("Hello world")

main()