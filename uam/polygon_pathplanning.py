# TODO:
import pandas as pd
import geopandas as gp
import numpy as np

from shapely.geometry import LineString, Point, Polygon, MultiPolygon, MultiLineString, box
from shapely.ops import nearest_points
import glob
from math import radians, cos, sin, asin, atan2, sqrt, pi, ceil, exp, log

import astar_uam as astar
import exploration
import terrain
import myio
# from myio import myio as io

from time import time
from multiprocessing import Pool

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

class polygon_pp(myio.myio):
    def __init__(self):
        self.radius = 700
        self.disc = 12
        self.c = 0.0000124187
        self.logging = True
        self.project = "uam/kaust/"
        self.sa = self.import_custom("landing")
        self.sa = self.sa.geometry.unary_union
        # self.sa = gp.GeoSeries()
        fa = self.import_forbidden()
        # Approximating the polygonial airspace
        tolerance = 30 # Meters
        self.fa = self.simplify_polygon(fa.unary_union, tolerance)

    def create_trajectory(self, coords):
        pi = self.transform_coords(coords[0])
        pf = self.transform_coords(coords[1])
        coords = [pi, pf]
        # Process airspace
        sa = self.sa.buffer(self.c * 100)
        sa = self.simplify_polygon(sa, 5)
        self.write_geom([sa], "Buffered area", "green")
        graph, ls = self.m_create_connectivity(sa, self.fa, coords)
        m_adj, m_heur = self.create_adjacency_matrix(graph, ls)
        a = 0
        b = 1
        path = astar.a_star(m_adj, m_heur, a, b)
        traj = self.path_to_traj(path, ls)

        traj = self.transform_coords_meters(traj)
        for i in range(len(traj)):
            traj[i] += [30, 0, 0, 0]
        return traj

    def closest_landing(self, coords):
        # Ignore the height
        coords = [coords[0], coords[1]]
        p = Point(coords)

        # If it's already within a landing spot, just glide to the center
        # Finding the closest landing area
        gs = gp.GeoSeries(self.sa).explode(ignore_index=True)
        # Transform global coords to meters
        gs = self.project_global_to_meter(gs.geometry)
        distances = gs.distance(p)
        min = 0
        for i, dist in enumerate(distances):
            if dist < distances[min]:
                min = i

        # Finding the closest point on that area
        lr = gs.exterior
        proj = lr.project(p)
        p_exterior = lr[min].interpolate(proj[min])
        p_center = gs[min].centroid
        geoms = [p, p_exterior, gs[min], p_center]
        gs = self.project_meter_to_global(geoms)
        self.write_geom(gs, "emergency_landing", "red")

        return [p_exterior, p_center]

    def sandbox1(self):
        # Constructing the safe airspace
        sa = self.construct_airspace()
        # Finding the mountains
        mt = self.avoid_terrain()
        mt = gp.GeoSeries(mt)
        self.add_to_forbidden(mt)
        # Removing landing spots that are within a forbidden area
        self.remove_invalid_ls()
        sa = self.construct_airspace()
        # Adding the roads to the safe airspace
        rds = self.import_roads()
        sa = sa.union(rds)
        # Trimming the airspace to the vicinity of the trajectory
        # boundary = [38.952054, 22.466675, 39.491757, 21.442638]
        boundary = [38.952054, 22.466675, 39.878717, 21.200736]
        # boundary = [5.7778, 43.5639, 5.4191, 43.4472]
        bbox = box(boundary[0], boundary[1], boundary[2], boundary[3])
        sa = sa.intersection(bbox)
        self.fa_gs = self.fa_gs.intersection(bbox)
        fa = self.fa_gs.unary_union # Geoseries to multipolygon

        fa = self.simplify_polygon(fa, tolerance)

        # Outputting for testing
        self.write_geom([sa], "safe", "green")
        # sa = lidar.union(sa)
        self.write_geom([fa], "forbidden", "red")
        ###
        tic = time()
        start = [39.174135, 21.513528] # IMC
        end = [39.105643, 22.314860] # KAUST medical clinic
        start_end = [start, end]
        graph, ls = self.m_create_connectivity(sa, fa, start_end)
        m_adj, m_heur = self.create_adjacency_matrix(graph, ls)
        print("It took ", time() - tic)
        a = 0
        b = 1
        path = astar.a_star(m_adj, m_heur, a, b)
        traj = self.path_to_traj(path, ls)

        self.write_path(path, ls, "poly_traj", "blue")
        width = 50
        self.write_corridor(path, width, ls, "corridor", "green")
        traj_ls = self.traj_to_linestring(self.path_to_traj(path, ls))
        unknown_sections = self.scan_using_lidar(traj_ls, self.construct_airspace())
        self.write_geom(unknown_sections, "unknown_sections", "grey")

        self.radius = 700
        sa = self.construct_airspace()
        self.write_geom([sa], "original_airspace", "green")
        files = ["plot/original_airspace.geojson",
        "plot/shrunk_airspace.geojson",
        "plot/poly_traj.geojson",
        "plot/corridor.geojson"]
        self.combine_json_files(files, "poly")

    def import_lidar(self):
        filename = "lidar.geojson"
        file = open(filename)
        df = gp.read_file(file)
        r = self.c * 700
        mpb = df.geometry.buffer(r, 4)

        # Temporary shift to somewhere along the trajectory
        point = [5.71292, 43.52178]
        center = mpb.centroid
        dx = point[0] - center.x
        dy = point[1] - center.y
        mpb = mpb.translate(dx, dy)
        # end of shift
        s = gp.GeoDataFrame(geometry=mpb)
        s["stroke"] = "green"
        s["marker-color"] = "green"
        s["fill"] = "green"
        s.to_file('plot/lidar_buffer.geojson', driver='GeoJSON')
        print("Imported lidar")
        return mpb

    def sandbox2(self):
        # Construct test airspace
        coords = [[5.760011, 43.543981],
                [5.767577, 43.530686]]
        sa = self.construct_airspace(coords)
        hole = self.draw_circle(coords[0][0], coords[0][1], self.radius / 2)
        sa = sa.difference(hole)
        boundary = [5.7654, 43.5298, 5.7691, 43.5333]
        fa = box(boundary[0], boundary[1], boundary[2], boundary[3])
        boundary = [5.7487, 43.5485, 5.7530, 43.5518]
        bbox = box(boundary[0], boundary[1], boundary[2], boundary[3])
        fa = fa.union(bbox)
        boundary = [5.7755, 43.5451, 5.7815, 43.5415]
        bbox = box(boundary[0], boundary[1], boundary[2], boundary[3])
        fa = fa.union(bbox)
        self.write_geom([sa], "test_sa", "green")
        self.write_geom([fa], "test_fa", "red")

        start = [5.759125, 43.544123]
        end = [5.436492, 43.502888]
        start_end = [start, end]
        # Process airspace
        graph, ls = self.m_create_connectivity(sa, fa, start_end)
        m_adj, m_heur = self.create_adjacency_matrix(graph, ls)
        a = 0
        b = 1
        path = astar.a_star(m_adj, m_heur, a, b)
        print(path)
        io.write_path(path, ls, "test_poly_traj", "green")
        files = ["plot/test_sa.geojson",
        "plot/test_fa.geojson",
        "plot/touchy.geojson",
        "plot/inside.geojson",
        "plot/test_poly_traj.geojson"]
        self.combine_json_files(files, "poly")

    def process_airspace(self, mp):
        # Assuming mp is multipolygon
        # Assuming construct airspace is used, which generates 3 multipolygons,
        # [0] corresponding to the areas within landing spots
        # Turn the polygon airspace into points in a vector
        ls = np.zeros((1,2))
        if mp.geom_type == "Polygon":
            print("It's a polygon")
            ls = np.delete(ls, 0, 0)
            return ls
        for p in mp.geoms:
            # if p.
            temp = self.exterior_to_vector(p)
            ls = np.concatenate((ls, temp), axis=0)
            temp = self.interiors_to_vector(p)
            ls = np.concatenate((ls, temp), axis=0)
        ls = np.delete(ls, 0, 0)
        self.se_ls = ls
        return ls

    def exterior_to_vector(self, p):
        pe = p.exterior # Polygon exterior/perimeter
        xx, yy = pe.coords.xy
        ls = np.stack([xx, yy]).transpose()
        ls = np.delete(ls, 0, 0)
        return ls

    def interiors_to_vector(self, p):
        # Repeats the first element
        pi = p.interiors # Polygon interiors
        ls = np.zeros((1,2))
        for interior in pi:
            xx, yy = interior.coords.xy
            temp = np.stack([xx, yy]).transpose()
            temp = np.delete(temp, 0, 0)
            ls = np.concatenate((ls, temp), axis=0)
        ls = np.delete(ls, 0, 0)
        return ls

    def process_forbidden(self):
        # Put them in the right shape (point to area, and buffer for certain areas)
        # Group them in a multipolygon for easy checks for intersection
        N = self.fa_gdf.geometry.size
        buff = 0.005 #0.008 is about the size of an airport
        fa = []
        for i in range(N):
            if self.fa_gdf.type[i] == "Point": # Point
                # Expand the point
                fa.append(self.fa_gdf.geometry[i].buffer(buff, 4))
            if self.fa_gdf.type[i] == "Polygon": # Polygon
                if i == "airport": # Doesn't work, fix this
                    fa.append(self.expand_polygon(self.fa_gdf.geometry[i], 25))
                else:
                    fa.append(self.fa_gdf.geometry[i])
            if self.fa_gdf.type[i] == "MultiPolygon":
                fa.append(self.fa_gdf.geometry[i])

        s = gp.GeoSeries(fa)
        p = gp.GeoSeries(s.unary_union)
        self.fa_gs = p

    def add_to_forbidden(self, area):
        list_of_df = [self.fa_gs, area]
        self.fa_gs = pd.concat(list_of_df, ignore_index=True)
        self.fa_gs = gp.GeoSeries(self.fa_gs.unary_union)
        print("Added area to forbidden")
        return self.fa_gs.unary_union

    def m_create_connectivity(self, sa, fa, additional_points):
        # Safe Airspace, Forbidden Airspace
        ls = self.process_airspace(sa.symmetric_difference(fa))
        # Start and end point
        for i in range(len(additional_points) - 1, -1, -1):
            point = additional_points[i]
            ls = self.insert_point(ls, point)
        # print(ls.shape[0])
        # print("Creating connectivity matrix")
        pool = Pool()
        tic = time()
        N = ls.shape[0]

        graph = np.zeros((N, N))
        with Pool() as pool:
            # prepare arguments
            items = [(i, sa, fa, ls) for i in range(N)]
            # call the same function with different data in parallel
            for i, result in enumerate(pool.starmap(self.m_single_node_connectivity, items)):
                # report the value to show progress
                # print(result)
                graph[i, :] = result
        pool.close()
        # print("Created connectivity matrix in ", time() - tic)
        # Bidirectional graph
        graph = graph + graph.transpose()
        return graph, ls

    def m_single_node_connectivity(self, i, sa, fa, ls):
        buff = 0.000001
        sab = sa.buffer(buff, 0)
        N = ls.shape[0]
        graph = np.zeros(N)
        start = np.tile(ls[i], (N - i - 1, 1))
        coords = np.zeros((N - i - 1, 2, 2))
        coords[:, 0, :] = start
        coords[:, 1, :] = ls[i + 1:]
        lines = MultiLineString(coords.tolist())
        gs = gp.GeoSeries(lines)
        gs = gs.explode(index_parts=False, ignore_index=True)

        within = gs.within(sab)
        touches = gs.touches(sa) & ~within | gs.disjoint(sa)
        forbidden = gs.within(fa) | gs.crosses(fa)

        within = within & ~forbidden
        touches = touches & ~forbidden

        contained = [within, touches, forbidden]
        for j in range(gs.size):
            k = i + j + 1
            if (contained[0][j] == True): # If it's completely within a polygon
                graph[k] = 1
            elif (contained[1][j] == True): # If it touches a polygon, but does not overlap with it
                graph[k] = 2
            elif (contained[2][j] == True):
                graph[k] = 0
        # print(i)
        return graph

    def create_connectivity(self, sa, fa):
        print("Creating connectivity matrix")
        tic = time()
        buff = 0.000001

        airspace = sa.symmetric_difference(fa) # A combination of both safe and forbidden, includes all the vertices
        self.write_geom([airspace], "symm_diff", "black")
        ls = self.process_airspace(airspace)
        sab = sa.buffer(buff, 0)
        # fab = fa.buffer(buff, 0)
        point = [5.739326, 43.556763]
        ls = self.insert_point(ls, point)
        point = [5.776577, 43.486763]
        ls = self.insert_point(ls, point)
        N = ls.shape[0]

        graph = np.zeros((N, N))
        inside = []
        touch = []
        forbid = []
        for i in range(N - 1):
            print(i)
            start = np.tile(ls[i], (N - i - 1, 1))
            coords = np.zeros((N - i - 1, 2, 2))
            coords[:, 0, :] = start
            coords[:, 1, :] = ls[i + 1:]
            lines = MultiLineString(coords.tolist())
            gs = gp.GeoSeries(lines)
            gs = gs.explode(index_parts=False, ignore_index=True)

            within = gs.within(sab)
            touches = gs.touches(sa) & ~within | gs.disjoint(sa)
            forbidden = gs.within(fa) | gs.crosses(fa)

            within = within & ~forbidden
            touches = touches & ~forbidden

            contained = [within, touches, forbidden]
            for j in range(gs.size):
                k = i + j + 1
                if (contained[0][j] == True): # If it's completely within a polygon
                    graph[i][k] = 1
                    graph[k][i] = 1
                    inside.append(gs[j])
                elif (contained[1][j] == True): # If it touches a polygon, but does not overlap with it
                    graph[i][k] = 2
                    graph[k][i] = 2
                    touch.append(gs[j])
                elif (contained[2][j] == True): # Gibbrish that is only used with the other two
                    graph[i][k] = 0
                    graph[k][i] = 0
                    forbid.append(gs[j])
        if (len(touch) > 0):
            io.write_linestring(touch, "touchy", "red")
        if (len(inside) > 0):
            io.write_linestring(inside, "inside", "blue")
        print(len(forbid))
        if (len(forbid) > 0):
            io.write_linestring(forbid, "forbid", "yellow")
        print("Created connectivity matrix in ", time() - tic)
        return graph, ls

    def create_adjacency_matrix(self, graph, ls):
        # Turn the vector of points into a matrix
        # Assign the correct cost values
        # print("Constructing adjacency matrix. This may take few minutes")
        N = ls.shape[0]
        m_adj = np.zeros((N, N))
        m_heur = np.zeros((N, N))
        for i in range(N - 1):
            for j in range(i, N):
                dist = self.haversine(ls[i][0], ls[i][1], ls[j][0], ls[j][1])
                temp = dist
                if (graph[i][j] == 1): # If it's completely within a polygon
                    dist = dist
                elif (graph[i][j] == 2): # If it touches a polygon, but does not overlap with it
                    # dist = (dist + stop_gap) ** 2
                    # dist = 1.4 * (dist - radius[0]) + radius[0]
                    # dist = 5 * (dist - radius[0]) ** 2 + radius[0] ** 2
                    # dist = (dist + stop_gap) ** 2
                    # dist = 1.4 * (dist - radius[0]) + radius[0]
                    # dist = 100 * (dist - radius[0]) ** 2 + radius[0] ** 2
                    # dist = 500 * dist + 100 * dist * dist
                    dist = 500 * dist
                else:
                    dist = 0
                m_adj[i, j] = dist
                m_adj[j, i] = dist
                m_heur[i, j] = temp
                m_heur[j, i] = temp
        # print("Constructed adjacency matrix")
        return m_adj, m_heur

    def insert_point(self, ls, point):
        # Adding it to se_ls
        a = np.array([point[0], point[1]])
        ls = np.insert(ls, 0, a, axis=0)
        return ls

    def path_to_traj(self, path, ls):
        traj = []
        for i in range(len(path)):
            j = path[i]
            traj.append([ls[j][0], ls[j][1], 0])
        return traj

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

    def construct_airspace(self, coords):
        radius = [self.radius, 7 * self.radius, 10 * self.radius]
        color = ["green", "yellow", "red"]
        opacity = [0.4, 0.4, 0.25]
        n = 0
        circles = []
        airspace = []
        circles = []
        for i in range(len(coords)):
            lon = coords[i][0]
            lat = coords[i][1]
            circles.append(self.draw_circle(lon, lat, radius[n]))

        s = gp.GeoDataFrame(crs=4326, geometry=circles)
        mp = s.unary_union # Multipolygon
        airspace.append(mp)
        s = gp.GeoDataFrame(crs=4326, geometry=[mp])
        s["fill-opacity"] = 0.3 * opacity[n]
        s["stroke-opacity"] = opacity[n]
        s["stroke"] = color[n]
        s["marker-color"] = color[n]
        s["fill"] = color[n]
        s.to_file('plot/test_airspace_' + color[n] + '.geojson', driver='GeoJSON')
        airspace_gs = gp.GeoSeries(airspace)
        return airspace_gs[0]

    def draw_circle(self, lon, lat, rad):
        l = []
        disc = self.disc
        ellipsis_factor = 0.72411296162
        c = self.c
        for i in range(disc):
            x = lon + c * rad * cos(2 * pi * i / disc)
            y = lat + c * rad * sin(2 * pi * i / disc) * ellipsis_factor
            l.append((x, y))
        p = Polygon(l)
        # print(self.haversine(lon, lat, l[0][0], l[0][1]))

        return p

    def simplify_polygon(self, fa, tolerance):
        return fa.simplify(tolerance * self.c)

    def project_meter_to_global(self, geom):
        gdf = gp.GeoDataFrame(crs="EPSG:20437", geometry=geom)
        gdf.to_crs(crs=4326, inplace=True)
        return gdf.geometry

    def project_global_to_meter(self, geom):
        gdf = gp.GeoDataFrame(crs=4326, geometry=geom)
        gdf.to_crs(crs="EPSG:20437", inplace=True)
        return gdf.geometry

    def transform_coords(self, coords):
        # Converting to global projection
        point = Point(coords)
        dummy_gdf = gp.GeoDataFrame(geometry=[point], crs="EPSG:20437")
        dummy_gdf.to_crs(crs=4326, inplace=True)
        p = [dummy_gdf.geometry[0].x, dummy_gdf.geometry[0].y]
        return p

    def transform_coords_meters(self, waypoints):
        result = []
        for coord in waypoints:
            result.append(self.transform_coord_meters(coord))
        return result

    def transform_coord_meters(self, coords):
        # Converting to meters projection
        point = Point(coords)
        dummy_gdf = gp.GeoDataFrame(geometry=[point], crs=4326)
        dummy_gdf.to_crs(crs="EPSG:20437", inplace=True)
        p = [dummy_gdf.geometry[0].x, dummy_gdf.geometry[0].y]
        return p

    # def write_linestring(self, linestring, name, color):
    #     gs = gp.GeoSeries(linestring)
    #     traj_gdf = gp.GeoDataFrame(crs=4326, geometry=gs)
    #     traj_gdf["stroke"] = color
    #     traj_gdf["stroke-width"] = 3
    #     traj_gdf.to_file("plot/" + name + ".geojson", driver='GeoJSON')

    # def write_traj(self, traj, name, color):
    #     s_line = self.traj_to_linestring(traj)
    #     traj_gdf = gp.GeoDataFrame(index=[0], crs=4326, geometry=[s_line])
    #     traj_gdf["stroke"] = color
    #     traj_gdf["stroke-width"] = 3
    #     traj_gdf.to_file("plot/" + name + ".geojson", driver='GeoJSON')

    # def write_path(self, path, ls, name, color):
    #     traj = self.path_to_traj(path, ls)
    #     self.write_traj(traj, name, color)

    def write_corridor(self, path, width, ls, name, color):
        traj = self.path_to_traj(path, ls)
        s_line = self.traj_to_linestring(traj)
        buff = self.c * width
        corridor = s_line.buffer(buff, 4)
        self.write_geom([corridor], name, color)

    def write_geom(self, geom, name, color):
        s = gp.GeoDataFrame(crs=4326, geometry=geom)
        s["stroke"] = color
        s["marker-color"] = color
        s["fill"] = color
        s.to_file('plot/' + name + '.geojson', driver='GeoJSON')

    def traj_to_linestring(self, traj):
        points = []
        for i in range(len(traj)):
            point = Point(traj[i][0], traj[i][1])
            points.append(point)
        s_line = LineString(points)
        return s_line

    def combine_json_files(self, list, output):
        concat_gdf = []
        for filename in list:
            file = open(filename)
            df = gp.read_file(file)
            concat_gdf.append(df)
        comb_gdf = pd.concat(concat_gdf, axis=0, ignore_index=True)
        comb_gdf.to_file('plot/' + output + '.geojson', driver='GeoJSON')


def main():
    a = polygon_pp()
    print("Hello world")

main()