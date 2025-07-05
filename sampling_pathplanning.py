# TODO:
import pandas as pd
import geopandas as gp
import numpy as np
import poisson_disc

from shapely.geometry import LineString, Point, Polygon, MultiPolygon, MultiLineString, box
from shapely.ops import nearest_points
from shapely.prepared import prep
from shapely.strtree import STRtree

from itertools import combinations
import glob
from math import radians, cos, sin, asin, atan2, sqrt, pi, ceil, exp, log

# import astar_uam as astar
# import exploration
# import terrain
from sim_io import myio as io
from util import *

from time import time
import random
from multiprocessing import Pool
from typing import List

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

class sampling_pp(io):
    def __init__(self):
        self.radius = 700

        self.logging = True
        self.project = "uam/kaust/"
        # self.sa = self.import_custom("landing")
        # self.sa = self.sa.geometry.unary_union
        # fa = self.import_forbidden()
        # Approximating the polygonial airspace
        tolerance = 30 # Meters
        # self.fa = self.simplify_polygon(fa.unary_union, tolerance)
        self.sim_run = ""
        self.sim_latest = ""
        self.mp_areas = []
        self.mp_costs = []
        self.nfz = []
        self.prepare()

    def prepare(self):
        # Get multipolygons
        comm = io.import_communication()
        r = 400
        comm = comm["geometry"].buffer(r).union_all()
        io.write_geom(transform_meter_global([comm]), "comm", "yellow")
        self.mp_areas = []
        self.mp_areas.append(comm)
        self.mp_costs = []
        minx, miny, maxx, maxy = comm.bounds
        bounds = comm.bounds
        # Sample airspace
        n_points = 300
        samples = samples_poisson(n_points, bounds)
        # samples = samples_biased(n_points, self.mp_areas, bounds, self.nfz, 0.7)
        # samples = samples_uniform(n_points, self.mp_areas, bounds, self.nfz)



        # Construct lines from samples
        max_distance = 500
        lines = connect_close_points(samples, max_distance)
        # Intersect lines with each of the polygons
        t0 = time()
        lengths = line_intersection_lengths(lines, comm)
        print(sum(lengths))
        t1 = time()
        # print(lengths)
        lengths = calculate_intersection_lengths_vectorized(lines, comm)
        t2 = time()
        print(sum(lengths))
        print("T0: %.2f" % (t1 - t0))
        print("T1: %.2f" % (t2 - t1))

        # Construct adjacency and
        # samples = gp.GeoSeries(samples).buffer(5)
        samples = transform_meter_global(samples)
        io.write_geom(samples, "samples", "yellow")

def samples_poisson(n_points, bounds):
    r = 0.05
    dims2d = np.array([1.0,1.0])
    samples = poisson_disc.Bridson_sampling(dims=dims2d, radius=r, k=n_points, hypersphere_sample=poisson_disc.hypersphere_surface_sample)
    samples = transpose_points(samples, bounds)
    return samples

def samples_biased(n_points, preferred_polygons, bounds, no_fly_polygons, bias_ratio=0.7):
    samples = []
    for _ in range(n_points):
        samples.append(sample_biased(preferred_polygons, bounds, no_fly_polygons, bias_ratio=bias_ratio))
    return samples

def samples_uniform(n_points, preferred_polygons, bounds, no_fly_polygons):
    samples = []
    for _ in range(n_points):
        samples.append(sample_uniform(bounds, no_fly_polygons))
    return samples

def sample_biased(preferred_polygons, bounds, no_fly_polygons, bias_ratio=0.7):
    if np.random.rand() < bias_ratio and len(preferred_polygons) > 0:
        # Biased sample
        poly = random.choice(preferred_polygons)
        minx, miny, maxx, maxy = poly.bounds
        while True:
            x = np.random.uniform(minx, maxx)
            y = np.random.uniform(miny, maxy)
            p = Point(x, y)
            if poly.contains(p):
                return p
    else:
        return sample_uniform(bounds, no_fly_polygons)

def sample_uniform(bounds, no_fly_polygons):
    while True:
        x = np.random.uniform(bounds[0], bounds[2])
        y = np.random.uniform(bounds[1], bounds[3])
        p = Point(x, y)
        if not any(poly.contains(p) for poly in no_fly_polygons):
            return p

def transpose_points(points, bounds):
    """
    Transpose a list of (x, y) points from [0, 1] space to [minx, miny, maxx, maxy] bounds.

    Args:
        points (list of tuple): List of (x, y) tuples with values in [0, 1].
        minx (float): Minimum x value of the target bounding box.
        miny (float): Minimum y value of the target bounding box.
        maxx (float): Maximum x value of the target bounding box.
        maxy (float): Maximum y value of the target bounding box.

    Returns:
        list of tuple: List of (x, y) tuples mapped to the new bounds.
    """
    minx, miny, maxx, maxy = bounds
    width = maxx - minx
    height = maxy - miny

    return [Point(minx + x * width, miny + y * height) for x, y in points]

def connect_close_points(points, max_distance):
    """
    Connects each pair of Shapely Points with a LineString if they are within max_distance.

    Args:
        points (list of shapely.geometry.Point): The list of points.
        max_distance (float): Maximum distance allowed to connect two points.

    Returns:
        list of shapely.geometry.LineString: Lines connecting points within the given distance.
    """
    lines = []
    for p1, p2 in combinations(points, 2):
        if p1.distance(p2) <= max_distance:
            lines.append(LineString([p1, p2]))
    return lines

def line_intersection_lengths(lines, multipolygon):
    """
    For each line, compute the length of the segment that intersects the MultiPolygon.

    Args:
        lines (list of LineString): List of Shapely LineString geometries.
        multipolygon (MultiPolygon): Shapely MultiPolygon geometry.

    Returns:
        list of float: Lengths of intersections for each line (0 if no intersection).
    """
    # Prepare geometry for faster repeated intersection checks
    prepared_multipolygon = prep(multipolygon)

    lengths = []
    for line in lines:
        # Fast rejection
        if not prepared_multipolygon.intersects(line):
            lengths.append(0.0)
            continue

        # Compute actual intersection geometry
        intersection = multipolygon.intersection(line)

        # Sum length if result is LineString or MultiLineString
        if intersection.is_empty:
            lengths.append(0.0)
        elif intersection.geom_type == 'LineString':
            lengths.append(intersection.length)
        elif intersection.geom_type == 'MultiLineString':
            lengths.append(sum(segment.length for segment in intersection.geoms))
        else:
            lengths.append(0.0)  # Intersection not a line segment

    return lengths


def calculate_intersection_lengths_vectorized(lines: List[LineString], multipolygon: MultiPolygon) -> np.ndarray:
    """
    Alternative vectorized approach for very large datasets.
    May be faster when dealing with thousands of lines.
    """
    if not lines:
        return np.array([])

    lengths = np.zeros(len(lines), dtype=np.float64)

    # Create bounds arrays for vectorized operations
    line_bounds = np.array([line.bounds for line in lines])

    for poly in multipolygon.geoms:
        poly_bounds = poly.bounds

        # Vectorized bounds check to filter lines that can't possibly intersect
        mask = (
            (line_bounds[:, 0] <= poly_bounds[2]) &  # minx <= poly_maxx
            (line_bounds[:, 2] >= poly_bounds[0]) &  # maxx >= poly_minx
            (line_bounds[:, 1] <= poly_bounds[3]) &  # miny <= poly_maxy
            (line_bounds[:, 3] >= poly_bounds[1])    # maxy >= poly_miny
        )

        # Only process lines that pass the bounds check
        for i in np.where(mask)[0]:
            try:
                intersection = lines[i].intersection(poly)
                if intersection.geom_type == 'LineString':
                    lengths[i] += intersection.length
                elif intersection.geom_type == 'MultiLineString':
                    lengths[i] += sum(geom.length for geom in intersection.geoms)
            except Exception:
                continue

    return lengths

def main():
    spp = sampling_pp()
    print("Done")

main()
