# TODO:
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

import uam.astar_uam as astar
from dijkstra import *
# import exploration
# import terrain
from sim_io import myio as io
from util import *

from time import time
import random
from multiprocessing import Pool
from typing import List, Tuple, Union

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

class sampling_pp(io):
    def __init__(self):
        self.radius = 700
        self.sim_run = ""
        self.sim_latest = ""
        self.mp_areas = []
        self.mp_costs = []
        self.adj = []
        self.heur = []
        self.nfz = []
        self.nfz_costs = 10000000
        self.nodes = []
        self.iteration = 0
        self.initalize()
        # p = Point([511083.21359357954, 2467813.5549285114])
        # boundary, inward, _ = self.closest_landing(p)
        # io.write_geom(transform_meter_global([boundary, inward, p]), "closest", "blue")

    def initalize(self):
        # Get multipolygons
        kaust = io.load_geojson_files("env/kaust.geojson", concat=True)["geometry"][0]
        self.kaust = MultiPolygon([kaust])
        self.mp_areas.append(self.kaust)
        self.mp_costs.append(1)

        # Safe landing spots
        sa = io.load_geojson_files("env/landing/*.geojson", concat=True)
        sa = sa.geometry.union_all()
        self.add_area(sa, -0.3)

        # Communication/GPS constraints
        comm = io.import_communication()
        r = 400
        comm = comm["geometry"].buffer(r).union_all()
        # self.add_area(comm, -0.1)
        # io.write_geom(transform_meter_global([comm]), "comm", "yellow")

        # NFZs
        fa = io.load_geojson_files("env/forbidden/*.geojson", concat=True)
        fa = fa.geometry.union_all()
        self.add_nfz(fa)

        bounds = self.kaust.bounds

        # Sample airspace
        n_points = 3500
        # n_points = 500
        samples = samples_poisson(n_points, bounds)
        # samples = samples_biased(n_points, self.mp_areas, bounds, self.nfz, 0.7)
        # samples = samples_uniform(n_points, self.mp_areas, bounds, self.nfz)

        # Remove extra points outside of kaust airspace
        samples = [p for p in samples if p.within(self.kaust)]
        self.nodes = samples

    def create_trajectory(self, coords):
        # Prepare the areas
        # Adjust the costs of the noise zones

        # # Calculating the graphs
        coords = [Point(coord) for coord in coords]
        nodes = self.add_nodes(coords)
        # nodes = self.nodes
        # start = closest_node(coords, nodes)
        # end = closest_node(coords, nodes)
        adj = self.update_graphs(nodes)
        m_adj = sum(adj)
        # m_heur = sum(heur)

        # Finding the optimal trajectory
        path = dijkstra(m_adj, 0, 1)
        traj = [nodes[p] for p in path]
        ls = LineString(traj)
        io.write_geom(transform_meter_global([ls]), "traj", "blue")
        z = 30
        result = [point_to_waypoint(p, z) for p in traj]
        return result

    def add_nodes(self, new_nodes):
        return new_nodes + self.nodes

    def update_graphs(self, nodes):
        # Construct lines from samples
        adj = []
        heur = []
        max_distance = 500 #km
        lines, node_pairs = connect_points_within_distance(nodes, max_distance)
        for i in range(len(self.mp_areas)):
            t0 = time()
            # Intersect lines with each of the polygons
            # lengths = line_intersection_lengths(lines, comm)
            lengths = calculate_intersection_lengths_vectorized(lines, self.mp_areas[i])
            lengths *= self.mp_costs[i]
            # Construct adjacency and heuristic matrices
            m_adj = create_adjacency_matrix_vectorized(lengths, node_pairs, nodes)
            adj.append(m_adj)
            # heur.append(m_heur)

        for i in range(len(self.nfz)):
            if self.nfz[i]["iteration"] + self.nfz[i]["length"] > self.iteration:
                # Intersect lines with each of the polygons
                # lengths = line_intersection_lengths(lines, comm)
                lengths = calculate_intersection_lengths_vectorized(lines, self.nfz[i]["geometry"])
                lengths *= self.nfz_costs
                # Construct adjacency and heuristic matrices
                m_adj = create_adjacency_matrix_vectorized(lengths, node_pairs, nodes)
                adj.append(m_adj)
                # heur.append(m_heur)

        # samples = gp.GeoSeries(nodes).buffer(5)
        # samples = transform_meter_global(samples)
        # io.write_geom(samples, "samples", "yellow")
        # lines = transform_meter_global(lines)
        # io.write_geom(lines, "lines", "white")
        return adj#, heur

    def add_area(self, area, cost):
        # area = self.kaust.difference(area)
        if isinstance(area, Polygon):
            area = MultiPolygon([area])
        self.mp_areas.append(area)
        self.mp_costs.append(cost)

    def add_nfz(self, nfz, id=-1):
        duration = 100000
        if isinstance(nfz, Polygon):
            nfz = MultiPolygon([nfz])
        nfz = {"geometry": nfz, "id": id, "iteration": self.iteration, "length": duration}
        self.nfz.append(nfz)

    def remove_nfz(self, id):
        for i in range(len(self.nfz)):
            if self.nfz[i]["id"] == id:
                self.nfz[i]["length"] = self.iteration - self.nfz[i]["iteration"]

    def closest_landing(self, target_point, inward_distance=10.0):
        """
        Optimized version that pre-filters multipolygons by distance.
        """
        # Pre-filter multipolygons by rough distance check
        multipolygons = self.mp_areas
        candidates = []

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

def closest_node(target_point, point_list):
    # Convert to numpy arrays once
    coords = np.array([(p.x, p.y) for p in point_list])
    target_coords = np.array([target_point.x, target_point.y])

    # Vectorized distance calculation
    distances = np.sum((coords - target_coords)**2, axis=1)
    return np.argmin(distances)

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

def main():
    spp = sampling_pp()

    coords = [[510783.21359357954, 2467813.5549285114], [512190.1320467823, 2468948.2965289974]]
    # spp.create_trajectory(coords)
    print("Done")

# main()