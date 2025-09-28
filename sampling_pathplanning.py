# TODO:
import os
import pandas as pd
import geopandas as gp
import numpy as np
import poisson_disc

from shapely.geometry import LineString, Point, Polygon, MultiPolygon, MultiLineString, box
from shapely.ops import nearest_points
from shapely.prepared import prep
from shapely.strtree import STRtree
from scipy.spatial.distance import cdist

from itertools import combinations
import glob
from math import radians, cos, sin, asin, atan2, sqrt, pi, ceil, exp, log

from dijkstra import *
from sim_io import myio as io
from util import *
import environment

from time import time
import random
from multiprocessing import Pool
from typing import List, Tuple, Union, Dict

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

class areas:
    __slots__ = ['id', 'geometry', 'type', 'cost', 'iteration', 'length', 'm_adj']

    def __init__(self, id, geometry, type, cost, iteration, length,
                 m_adj):
        self.id = id
        self.geometry = geometry
        self.type = type
        self.cost = cost
        self.iteration = iteration
        self.length = length
        self.m_adj = m_adj


class sampling_pp(io):
    def __init__(self):
        self.radius = 700
        self.sim_run = ""
        self.sim_latest = ""
        self.adj = []
        self.heur = []
        self.nfz_costs = 10000000
        self.nodes = []
        self.iteration = 0
        self.areas = []
        self.el = 0
        self.env = environment.env()
        self.initalize()

    def initalize(self):
        kaust = io.load_geojson_files("env/kaust.geojson", concat=True)["geometry"][0]
        self.kaust = make_mp(kaust)
        # Sample airspace
        n_points = 3500
        bounds = self.kaust.bounds
        samples = samples_poisson(n_points, bounds)
        # samples = samples_biased(n_points, self.mp_areas, bounds, self.nfz, 0.7)
        # samples = samples_uniform(n_points, self.mp_areas, bounds, self.nfz)

        # Remove extra points outside of kaust airspace
        samples = [p for p in samples if p.within(self.kaust)]
        # print(min_distance_brute_force(samples))
        self.nodes = samples
        # Get multipolygons
        t0 = time()
        self.add_area(id=-1, geometry=self.kaust, type=1, cost=1,
                      iteration=0, length=1000000, m_adj=None)

        # Safe landing spots
        sa = io.load_geojson_files("env/landing/*.geojson", concat=True)
        sa = make_mp(sa.geometry.union_all())
        self.el = sa
        self.add_area(id=-1, geometry=sa, type=1, cost=-0.9, iteration=0,
                      length=1000000, m_adj=None)

        # Communication/GPS constraints
        comm = io.import_communication()
        r = 900
        comm = construct_3way_intersection(comm["geometry"].buffer(r))
        # comm = comm["geometry"].buffer(r).union_all()
        comm = make_mp(comm)
        # self.add_area(id=-1, geometry=comm, type=1, cost=-0.3,
        #               iteration=0, length=1000000, m_adj=None)
        # io.write_geom(transform_meter_global([comm]), "comm", "yellow")
        # NFZs
        fa = io.load_geojson_files("env/forbidden/*.geojson", concat=True)
        fa = make_mp(fa.geometry.union_all())
        self.add_area(id=-1, geometry=fa, type=0, cost=self.nfz_costs,
                      iteration=0, length=1000000, m_adj=None)
        t1 = time()
        self.write_area_costs()
        print(f"Created adjacency matrices in {t1 - t0:.2f} seconds")

    def process_area(self, index = -1):
        """
        Process the areas and create adjacency matrices.
        This is called after adding new nodes or areas.
        """
        # Create adjacency matrices for all areas
        max_distance = 500  # m
        nodes = self.nodes
        lines, node_pairs = connect_points_within_distance(nodes, max_distance)
        # Intersect lines with each of the polygons
        lengths = calculate_intersection_lengths_vectorized(lines, self.areas[index].geometry)
        lengths *= self.areas[index].cost
        # Construct adjacency lists
        m_adj = create_adjacency_matrix_vectorized(lengths, node_pairs, nodes)
        self.areas[index].m_adj = m_adj
        return m_adj

    def create_trajectory(self, coords):
        # Prepare the areas
        # Adjust the costs of the noise zones

        # # Calculating the graphs
        coords = [Point(coord) for coord in coords]
        coords = self.valid_takeoff(coords)
        nodes = self.add_nodes(coords)
        # Incrementally update adjacency matrices
        adj = self.update_adj_matrices(coords)
        m_adj = sum(adj)

        # Finding the optimal trajectory
        path = dijkstra(m_adj, 0, 1)
        traj = [nodes[p] for p in path]
        # ls = LineString(traj)
        # io.write_geom(transform_meter_global([ls]), "traj", "blue")
        z = 50
        result = [point_to_waypoint(p, z) for p in traj]

        takeoff_height = 5
        # Insert duplicate of first element at beginning with z=0
        result.insert(0, {**result[0], 'z': takeoff_height})

        # Append duplicate of last element at end with z=0
        result.append({**result[-1], 'z': takeoff_height})
        return result

    def add_nodes(self, new_nodes):
        return new_nodes + self.nodes

    def create_adj_matrices(self):
        max_distance = 500  # m
        nodes = self.nodes
        lines, node_pairs = connect_points_within_distance(nodes, max_distance)
        adj = []
        for area in self.areas:
            if area.iteration + area.length > self.iteration:
                # Intersect lines with each of the polygons
                geom = area.geometry
                # if area.type == 0:  # No-fly zone
                #     geom = area.geometry.buffer(5)  # Small buffer to avoid precision issues
                lengths = calculate_intersection_lengths_vectorized(lines, geom)
                lengths *= area.cost
                # Construct adjacency lists
                m_adj = create_adjacency_matrix_vectorized(lengths, node_pairs, nodes)
                area.m_adj = m_adj
                adj.append(m_adj)

        # samples = gp.GeoSeries(nodes).buffer(5)
        # samples = transform_meter_global(samples)
        # io.write_geom(samples, "samples", "yellow")
        # lines = transform_meter_global(lines)
        # io.write_geom(lines, "lines", "white")
        return adj

    def update_adj_matrices(self, new_nodes):
        """
        Incrementally update existing adjacency matrices by adding new nodes.
        Only calculates lengths for new connections involving new nodes.
        New nodes are inserted at the beginning of the matrix.
        """
        max_distance = 500  # m

        # Combined node list - NEW NODES FIRST
        all_nodes = new_nodes + self.nodes
        num_new_nodes = len(new_nodes)
        num_existing_nodes = len(self.nodes)
        total_nodes = len(all_nodes)

        # Get only NEW connections (involving at least one new node)
        new_lines, new_node_pairs = connect_multiple_points_to_network(self.nodes, new_nodes, max_distance)

        adj = []
        for area in self.areas:
            if area.iteration + area.length > self.iteration:
                # Get existing matrix
                old_matrix = area.m_adj

                # Create expanded matrix with new nodes at the beginning
                new_matrix = np.full((total_nodes, total_nodes), np.inf, dtype=np.float64)
                np.fill_diagonal(new_matrix, 0)

                # Copy existing connections to the bottom-right corner
                new_matrix[num_new_nodes:, num_new_nodes:] = old_matrix

                # Calculate lengths only for new connections
                # new_lengths = calculate_intersection_lengths_vectorized(new_lines, area.geometry)
                geom = area.geometry
                # if area.type == 0:  # No-fly zone
                #     geom = area.geometry.buffer(5)  # Small buffer to avoid precision issues
                new_lengths = calculate_intersection_lengths_vectorized(new_lines, geom)
                new_lengths *= area.cost

                # Add new connections to the matrix
                for (i, j), length in zip(new_node_pairs, new_lengths):
                    # Map indices to new matrix layout
                    if i < len(self.nodes):  # existing node
                        matrix_i = i + num_new_nodes
                    else:  # new node
                        matrix_i = i - len(self.nodes)

                    if j < len(self.nodes):  # existing node
                        matrix_j = j + num_new_nodes
                    else:  # new node
                        matrix_j = j - len(self.nodes)

                    new_matrix[matrix_i, matrix_j] = length
                    new_matrix[matrix_j, matrix_i] = length  # assuming symmetric

                # Update the stored matrix and add to results
                # area.m_adj = new_matrix
                adj.append(new_matrix)

        return adj

    def add_area(self, id, geometry, type, cost, iteration, length, m_adj=None):
        self.areas.append(areas(
            id=id,
            geometry=geometry,
            type=type,  # 0: No-fly zone, 1: Safe area
            cost=cost,
            iteration=iteration,
            length=length,
            m_adj=[],
        ))
        self.process_area()

    def add_nfz(self, nfz, id=-1):
        duration = 100000
        nfz = make_mp(nfz)
        self.areas.append(areas(
            id=id,
            geometry=nfz,
            type=0,  # 0: No-fly zone, 1: Safe area
            cost=self.nfz_costs,
            iteration=self.iteration,
            length=duration,
            m_adj=[],
        ))
        self.process_area()

    def remove_nfz(self, id):
        for area in self.areas:
            if area.id == id:
                area.length = self.iteration - area.iteration

    def get_nfz(self):
        """
        Get all no-fly zones as a MultiPolygon.
        """
        nfz = []
        for area in self.areas:
            if area.type == 0:
                nfz.append({"geometry": area.geometry, "iteration:": area.iteration, "length": area.length})
        return nfz

    def nearby_nfz(self, point, radius=1000):
        """
        Get nearby no-fly zone polygons within a certain radius.
        Returns a list of Polygons that are within the radius of the point.

        Args:
            point (Point): The reference point
            radius (float): Search radius in meters

        Returns:
            list: List of Polygons that are within the radius of the point
        """
        inside = 0
        nearby = []
        for area in self.areas:
            if area.type == 0:  # No-fly zone
                # For each polygon in the MultiPolygon
                for polygon in area.geometry.geoms:
                    if polygon.distance(point) < radius:
                        nearby.append(polygon)
                        if polygon.contains(point):
                            inside = 1

        return nearby, inside

    def closest_landing(self, target_point, inward_distance=10.0):
        """
        Optimized version that pre-filters multipolygons by distance.
        If target_point is inside a multipolygon, returns both points inside that multipolygon.
        """
        # Pre-filter multipolygons by rough distance check
        print(target_point)

        multipolygons = [self.areas[1].geometry]  # Assuming the second area is the safe landing spots
        multipolygons = [self.el]
        candidates = []

        # First check if target point is inside any multipolygon
        for idx, mp in enumerate(multipolygons):
            if mp.contains(target_point):
                # Target is inside this multipolygon
                print(f"Target point is inside multipolygon {idx}")

                # Find a point that's inward_distance away from the target
                # We'll try to move inward from the nearest boundary
                closest_boundary_point = nearest_points(target_point, mp.boundary)[1]

                # Calculate direction from boundary to target (inward direction)
                inward_direction_x = target_point.x - closest_boundary_point.x
                inward_direction_y = target_point.y - closest_boundary_point.y

                # Normalize the direction
                direction_length = (inward_direction_x**2 + inward_direction_y**2)**0.5
                if direction_length > 0:
                    unit_x = inward_direction_x / direction_length
                    unit_y = inward_direction_y / direction_length

                    # Create a point that's inward_distance away from target in the inward direction
                    inward_point_x = target_point.x + unit_x * inward_distance
                    inward_point_y = target_point.y + unit_y * inward_distance
                    inward_point = Point(inward_point_x, inward_point_y)

                    # Check if the inward point is still inside the multipolygon
                    if mp.contains(inward_point):
                        return target_point, inward_point, idx
                    else:
                        # If inward point is outside, try moving in the opposite direction
                        inward_point_x = target_point.x - unit_x * inward_distance
                        inward_point_y = target_point.y - unit_y * inward_distance
                        inward_point = Point(inward_point_x, inward_point_y)

                        if mp.contains(inward_point):
                            return target_point, inward_point, idx
                        else:
                            # If both directions fail, find the centroid or use target as both points
                            centroid = mp.centroid
                            if mp.contains(centroid):
                                return target_point, centroid, idx
                            else:
                                # Last resort: return target point for both
                                return target_point, target_point, idx
                else:
                    # If target is exactly on boundary, use centroid
                    centroid = mp.centroid
                    if mp.contains(centroid):
                        return target_point, centroid, idx
                    else:
                        return target_point, target_point, idx

        # If target is not inside any multipolygon, proceed with original logic
        for idx, mp in enumerate(multipolygons):
            # Quick distance check using bounds
            bounds_distance = target_point.distance(Point(mp.bounds[0], mp.bounds[1]))
            candidates.append((bounds_distance, idx, mp))

        # Sort by distance and check closest ones first
        candidates.sort()

        min_distance = float('inf')
        best_closest_point = None
        best_inward_point = None
        best_multipolygon_idx = None

        for bounds_dist, mp_idx, multipolygon in candidates:
            # Skip if bounds distance is already larger than best found
            if bounds_dist > min_distance:
                break

            # Get actual closest point
            closest_point = nearest_points(target_point, multipolygon.boundary)[1]
            distance = target_point.distance(closest_point)

            if distance < min_distance:
                inward_point = find_inward_point(closest_point, multipolygon, inward_distance)

                if inward_point is not None:
                    min_distance = distance
                    best_closest_point = closest_point
                    best_inward_point = inward_point
                    best_multipolygon_idx = mp_idx

        return best_closest_point, best_inward_point, best_multipolygon_idx

    def round_trip(self, one_way):
        # Return trip
        return_path = one_way[::-1]
        one_way.pop() # Remove the last element to make a full trip without duplicates
        round_trip = one_way + return_path
        return round_trip

    def write_area_costs(self, filepath=None):
        """
        Write the costs of all areas into a text file.

        Args:
            filepath (str, optional): The path to the text file where costs will be written.
                                    If None, will use sim_run path with 'area_costs.txt'
        """
        if filepath is None:
            filepath = os.path.join("plot/" + self.sim_run, 'params.txt')
            print(filepath)

        with open(filepath, 'w') as f:
            # Write header
            f.write("area_id,area_type,cost,iteration,length,area_size\n")

            # Write data for each area
            for area in self.areas:
                # Calculate area size in square meters
                area_size = area.geometry.area
                line = f"{area.id},{area.type},{area.cost},{area.iteration},{area.length},{area_size:.2f}\n"
                f.write(line)

        print(f"Area costs written to {filepath}")

    def valid_takeoff(self, points: List[Point]) -> List[Point]:
        """Check if takeoff points are valid and adjust them if they're within obstacles.
        If a point is within an obstacle, attempts to move it to a safe location outside.

        Args:
            points: List of Points to validate

        Returns:
            List of valid Points (either original or adjusted)
        """
        valid_points = []
        for p in points:
            nearby = self.env.nearby_all_obstacles(p, 5)
            if nearby.empty:  # Check if the GeoDataFrame is empty
                valid_points.append(p)
                continue

            # Create a union of all nearby geometries
            obstacles = nearby.geometry.unary_union
            if not p.within(obstacles):
                valid_points.append(p)
                continue

            # Try to find a safe point nearby through random sampling
            valid_point_found = False
            for _ in range(10):  # Try 10 times to find a safe point
                angle = random.uniform(0, 2 * pi)
                distance = random.uniform(5, 10)  # Sample between 5-10m away
                new_x = p.x + distance * cos(angle)
                new_y = p.y + distance * sin(angle)
                new_point = Point(new_x, new_y)

                # Check if the new point is valid
                if self.kaust.contains(new_point):
                    check_nearby = self.env.nearby_all_obstacles(new_point, 5)
                    if check_nearby.empty or not new_point.within(check_nearby.geometry.unary_union):
                        valid_points.append(new_point)
                        valid_point_found = True
                        break

            if not valid_point_found:
                # If random sampling failed, try using the nearest point on obstacle boundary
                boundary_point = nearest_points(obstacles.boundary, p)[0]
                direction = (boundary_point.x - p.x, boundary_point.y - p.y)
                distance = sqrt(direction[0]**2 + direction[1]**2)

                if distance < 1e-10:  # Point is exactly on boundary
                    valid_points.append(p)
                else:
                    # Move 5 meters away from obstacle
                    scale = 5.0 / distance
                    new_point = Point(
                        boundary_point.x + direction[0] * scale,
                        boundary_point.y + direction[1] * scale
                    )
                    valid_points.append(new_point)

        return valid_points

def samples_poisson(n_points, bounds):
    r = get_radius(n_points)
    dims2d = np.array([1.0,1.0])
    samples = poisson_disc.Bridson_sampling(dims=dims2d, radius=r, k=30, hypersphere_sample=poisson_disc.hypersphere_surface_sample)
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

def find_inward_point(boundary_point, multipolygon, inward_distance):
    """
    Find a point inward from the boundary point.

    Args:
        boundary_point: Point on the multipolygon boundary
        multipolygon: The multipolygon containing the boundary point
        inward_distance: Distance to move inward

    Returns:
        Point: Inward point, or None if not found
    """
    # Method 1: Use negative buffer to find inward direction
    try:
        # Create a small buffer around the boundary point
        point_buffer = boundary_point.buffer(0.1)

        # Find the intersection with the multipolygon interior
        interior_intersection = multipolygon.intersection(point_buffer)

        if not interior_intersection.is_empty:
            # Get the centroid of the intersection as reference for inward direction
            interior_centroid = interior_intersection.centroid

            # Calculate direction vector from boundary to interior
            dx = interior_centroid.x - boundary_point.x
            dy = interior_centroid.y - boundary_point.y

            # Normalize and scale by inward_distance
            length = np.sqrt(dx*dx + dy*dy)
            if length > 0:
                dx_norm = dx / length * inward_distance
                dy_norm = dy / length * inward_distance

                # Create inward point
                inward_point = Point(boundary_point.x + dx_norm, boundary_point.y + dy_norm)

                # Verify point is inside multipolygon
                if multipolygon.contains(inward_point):
                    return inward_point
    except:
        pass

    # Method 2: Sample points in multiple directions and pick the best one
    return find_inward_point_sampling(boundary_point, multipolygon, inward_distance)

def find_inward_point_sampling(boundary_point, multipolygon, inward_distance):
    """
    Find inward point by sampling multiple directions.
    """
    best_point = None
    max_distance_from_boundary = 0

    # Try 8 directions around the boundary point
    for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
        dx = np.cos(angle) * inward_distance
        dy = np.sin(angle) * inward_distance

        candidate_point = Point(boundary_point.x + dx, boundary_point.y + dy)

        # Check if point is inside multipolygon
        if multipolygon.contains(candidate_point):
            # Measure distance from boundary (higher is more inward)
            distance_from_boundary = candidate_point.distance(multipolygon.boundary)

            if distance_from_boundary > max_distance_from_boundary:
                max_distance_from_boundary = distance_from_boundary
                best_point = candidate_point

    return best_point

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

def connect_points_within_distance(points: List[Point], max_distance: float) -> Tuple[List[LineString], List[Tuple[int, int]]]:
    """
    Connect pairs of Shapely Points with LineStrings if they are within max_distance.

    Args:
        points: List of Shapely Point objects
        max_distance: Maximum distance for connecting points

    Returns:
        Tuple containing:
        - List of LineString objects connecting points within distance
        - List of tuples containing indices of connected points (i, j) where i < j
    """
    if len(points) < 2:
        return [], []

    # Extract coordinates for vectorized distance calculations
    coords = np.array([[point.x, point.y] for point in points])

    # Calculate all pairwise distances at once using scipy
    distances = cdist(coords, coords, metric='euclidean')

    # Get indices of point pairs within max_distance
    # Use upper triangle to avoid duplicates (i < j)
    i_indices, j_indices = np.where(
        (distances <= max_distance) & (distances > 0) &
        (np.triu(np.ones_like(distances), k=1) == 1)
    )

    # Create LineStrings for valid connections
    lines = []
    indices = []

    for i, j in zip(i_indices, j_indices):
        line = LineString([coords[i], coords[j]])
        lines.append(line)
        indices.append((int(i), int(j)))

    return lines, indices

def connect_multiple_points_to_network(existing_points: List[Point],
                                     new_points: List[Point],
                                     max_distance: float,
                                     start_index: int = None) -> Tuple[List[LineString], List[Tuple[int, int]]]:
    """
    Connect multiple new points to existing points within max_distance.
    Does not connect new points to each other.

    Args:
        existing_points: List of existing Shapely Point objects
        new_points: List of new Shapely Point objects to connect to the network
        max_distance: Maximum distance for connecting points
        start_index: Starting index for new points. If None, uses len(existing_points)

    Returns:
        Tuple containing:
        - List of LineString objects connecting new points to existing points within distance
        - List of tuples containing indices of connected points (existing_index, new_index)
    """
    if len(existing_points) == 0 or len(new_points) == 0:
        return [], []

    # Assign starting index for new points if not provided
    if start_index is None:
        start_index = len(existing_points)

    # Extract coordinates for vectorized distance calculations
    existing_coords = np.array([[point.x, point.y] for point in existing_points])
    new_coords = np.array([[point.x, point.y] for point in new_points])

    # Calculate distances from all new points to all existing points
    distances = cdist(new_coords, existing_coords, metric='euclidean')

    # Create LineStrings for valid connections
    lines = []
    indices = []

    for new_idx, new_point_distances in enumerate(distances):
        new_point_index = start_index + new_idx

        # Get indices of existing points within max_distance for this new point
        valid_existing_indices = np.where(new_point_distances <= max_distance)[0]

        for existing_idx in valid_existing_indices:
            line = LineString([existing_coords[existing_idx], new_coords[new_idx]])
            lines.append(line)
            indices.append((int(existing_idx), new_point_index))

    return lines, indices

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

def create_adjacency_matrix_vectorized(lengths: np.ndarray,
                                     node_pairs: np.ndarray,
                                     node_coordinates: Union[np.ndarray, List[Point]] = None,
                                     num_nodes: int = None,
                                     symmetric: bool = True,
                                     return_heuristic: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Vectorized version for better performance with large datasets.

    Args:
        lengths: NumPy array of line lengths
        node_pairs: NumPy array of shape (n_connections, 2) containing node pairs
        node_coordinates: Either:
                         - NumPy array of shape (num_nodes, 2) containing (x, y) coordinates
                         - List of Shapely Point objects
                         Required if return_heuristic=True
        num_nodes: Total number of nodes. If None, inferred from max node index + 1
        symmetric: If True, creates symmetric matrix (undirected graph)
        return_heuristic: If True, also returns heuristic matrix with Euclidean distances

    Returns:
        If return_heuristic=False: numpy.ndarray (adjacency matrix)
        If return_heuristic=True: Tuple[numpy.ndarray, numpy.ndarray] (adjacency matrix, heuristic matrix)
    """
    if len(lengths) != len(node_pairs):
        raise ValueError("lengths and node_pairs must have the same length")

    if return_heuristic and node_coordinates is None:
        raise ValueError("node_coordinates must be provided when return_heuristic=True")

    # Convert to numpy arrays if not already
    lengths = np.asarray(lengths, dtype=np.float64)
    node_pairs = np.asarray(node_pairs, dtype=np.int32)

    if return_heuristic:
        # Convert Shapely Points to NumPy array if needed
        if isinstance(node_coordinates, list) and len(node_coordinates) > 0 and isinstance(node_coordinates[0], Point):
            node_coordinates = np.array([[point.x, point.y] for point in node_coordinates])
        else:
            node_coordinates = np.asarray(node_coordinates, dtype=np.float64)

        if node_coordinates.shape[1] != 2:
            raise ValueError("node_coordinates must have shape (num_nodes, 2) or be a list of Shapely Points")

    if lengths.size == 0:
        if num_nodes is None:
            raise ValueError("num_nodes must be specified when no connections are provided")
        adj_matrix = np.full((num_nodes, num_nodes), np.inf, dtype=np.float64)
        np.fill_diagonal(adj_matrix, 0)

        if return_heuristic:
            heuristic_matrix = _create_heuristic_matrix(node_coordinates, num_nodes)
            return adj_matrix, heuristic_matrix
        return adj_matrix

    # Determine number of nodes if not specified
    if num_nodes is None:
        num_nodes = np.max(node_pairs) + 1

    # Validate node indices
    if np.max(node_pairs) >= num_nodes:
        raise ValueError(f"Node index {np.max(node_pairs)} exceeds num_nodes-1 ({num_nodes-1})")

    if return_heuristic and len(node_coordinates) != num_nodes:
        raise ValueError(f"node_coordinates length ({len(node_coordinates)}) must match num_nodes ({num_nodes})")

    # Initialize adjacency matrix with infinity
    adj_matrix = np.full((num_nodes, num_nodes), np.inf, dtype=np.float64)

    # Set diagonal to 0
    np.fill_diagonal(adj_matrix, 0)

    # Extract i, j indices
    i_indices = node_pairs[:, 0]
    j_indices = node_pairs[:, 1]

    # Fill in the connections using advanced indexing
    adj_matrix[i_indices, j_indices] = lengths

    if symmetric:
        adj_matrix[j_indices, i_indices] = lengths

    if return_heuristic:
        heuristic_matrix = _create_heuristic_matrix(node_coordinates, num_nodes)
        return adj_matrix, heuristic_matrix

    return adj_matrix

def _create_heuristic_matrix(node_coordinates: np.ndarray, num_nodes: int) -> np.ndarray:
    """
    Create a heuristic matrix with Euclidean distances between all node pairs.

    Args:
        node_coordinates: NumPy array of shape (num_nodes, 2) containing (x, y) coordinates
        num_nodes: Total number of nodes

    Returns:
        numpy.ndarray: Heuristic matrix with Euclidean distances
    """
    # Calculate all pairwise Euclidean distances
    # Using broadcasting: (n, 1, 2) - (1, n, 2) -> (n, n, 2)
    coords_expanded = node_coordinates[:, np.newaxis, :]  # Shape: (n, 1, 2)
    coords_broadcast = node_coordinates[np.newaxis, :, :]  # Shape: (1, n, 2)

    # Calculate squared differences
    diff_squared = (coords_expanded - coords_broadcast) ** 2

    # Sum over the coordinate dimension and take square root
    euclidean_distances = np.sqrt(np.sum(diff_squared, axis=2))

    return euclidean_distances

def get_radius(n_points):
    data = [
    (0.01, 8313),
    (0.012, 5776),
    (0.014, 4252),
    (0.016, 3271),
    (0.018, 2591),
    (0.02, 2089),
    (0.022, 1727),
    (0.024, 1475),
    (0.026, 1252),
    (0.028, 1070),
    (0.03, 948),
    (0.032, 823),
    (0.034, 741),
    (0.036, 661),
    (0.038, 588),
    (0.04, 540),
    (0.042, 482),
    (0.044, 438),
    (0.046, 405),
    (0.048, 371),
    (0.05, 342),
    (0.052, 321),
    (0.054, 296),
    (0.056, 280),
    (0.058, 261),
    (0.06, 245),
    (0.062, 229),
    (0.064, 213),
    (0.066, 199),
    (0.068, 190),
    (0.07, 180),
    (0.072, 176),
    (0.074, 159),
    (0.076, 156),
    (0.078, 146),
    (0.08, 141),
    (0.082, 131),
    (0.084, 125),
    (0.086, 124),
    (0.088, 116),
    (0.09, 112),
    (0.092, 105),
    (0.094, 102),
    (0.096, 99),
    (0.098, 91),
    (0.14, 50),
    (0.20, 25),
    (0.30, 10),
    (0.45, 5)]
    prev_point = data[0]
    for point in data:
        if point[1] < n_points:
            return prev_point[0]
        prev_point = point
    return 0.5

def min_distance_brute_force(points):
    """Calculate minimum distance between any two points"""
    min_dist = float('inf')
    closest_pair = None

    for p1, p2 in combinations(points, 2):
        dist = p1.distance(p2)
        if dist < min_dist:
            min_dist = dist
            closest_pair = (p1, p2)

    return min_dist

def make_mp(polygon):
    """
    Create a MultiPolygon from a Polygon and assign an ID.
    """
    if isinstance(polygon, Polygon):
        polygon = MultiPolygon([polygon])
    return polygon

def construct_comm_intersection(points):
    # Collect all non-empty intersections
    all_intersections = []
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            intersection = points.iloc[i].intersection(points.iloc[j])
            if not intersection.is_empty:
                all_intersections.append(intersection)
    # Union all intersections
    return shapely.unary_union(all_intersections)

def construct_3way_intersection(geoseries):
    # Get all 3-way intersections
    intersections = []
    for i, j, k in combinations(range(len(geoseries)), 3):
        intersection = (geoseries.iloc[i]
                    .intersection(geoseries.iloc[j])
                    .intersection(geoseries.iloc[k]))
        if not intersection.is_empty:
            intersections.append(intersection)

    # Union all 3-way intersections
    return shapely.unary_union(intersections)

def main():
    spp = sampling_pp()

    coords = [[510783.21359357954, 2467813.5549285114], [512190.1320467823, 2468948.2965289974]]
    # spp.create_trajectory(coords)
    print("Done")

# main()