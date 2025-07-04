import datetime
import glob

import geopandas as gp
import pandas as pd
import numpy as np
import pickle as pkl

# from data_analysis import *

from shapely.geometry import LineString, Point, Polygon, box

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# 'EPSG:2154' for France
# epsg=4326  for equal area

class myio:
    def import_hospitals(self):
        filename = self.project + "hospitals/*.geojson"
        list_of_files = glob.glob(filename)
        print(list_of_files)
        se = np.zeros((1,2))
        list_of_df = []
        for filename in list_of_files:
            file = open(filename)
            df = gp.read_file(file)

            N = df.geometry.size
            gdf = df.to_crs(crs=4326)

            list_of_df.append(gdf)
            gdf = df.centroid
            se_temp = np.zeros((N,2))
            for i in range(N):
                se_temp[i, 0] = gdf[i].x
                se_temp[i, 1] = gdf[i].y
            se = np.concatenate((se, se_temp), axis=0)
        se = np.delete(se, 0, 0)
        self.se_ls = np.concatenate((self.se_ls, se), axis=0)
        color = pd.concat(list_of_df, ignore_index=True)
        self.se_gdf = color.centroid
        self.gdf = self.se_gdf
        print("Imported hospitals")

        # Coloring
        color["stroke"] = "green"
        color["marker-color"] = "green"
        color["fill"] = "green"
        if self.logging:
            color.to_file('plot/hospitals.geojson', driver='GeoJSON')

    def import_landing(self):
        filename = self.project + "landing/*.geojson"
        list_of_files = glob.glob(filename)
        print(list_of_files)
        ls = np.zeros((1,2))
        list_of_df = []
        for filename in list_of_files:
            file = open(filename)
            df = gp.read_file(file)

            N = df.geometry.size
            gdf = df.to_crs(crs=4326)

            gdf = df.centroid

            list_of_df.append(gdf)

            ls_temp = np.zeros((N,2))
            for i in range(N):
                ls_temp[i, 0] = gdf[i].x
                ls_temp[i, 1] = gdf[i].y
            ls = np.concatenate((ls, ls_temp), axis=0)
        ls = np.delete(ls, 0, 0)
        self.se_ls = np.concatenate((self.se_ls, ls), axis=0)
        self.ls_gdf = pd.concat(list_of_df, ignore_index=True)
        self.gdf = pd.concat([self.se_gdf, self.ls_gdf], ignore_index=True)
        print("Imported landing")

        ## Coloring
        color = self.ls_gdf
        color = gp.GeoDataFrame(geometry=color)
        color["stroke"] = "gray"
        color["marker-color"] = "gray"
        color["fill"] = "gray"
        color["marker-size"] = "small"
        if self.logging:
            color.to_file('plot/ls.geojson', driver='GeoJSON')

    def import_factories(self):
        filename = self.project + "factories/*.geojson"
        list_of_files = glob.glob(filename)
        if not list_of_files:
            print("No factories to import")
            return 0
        list_of_df = []

        for filename in list_of_files:
            file = open(filename)
            df = gp.read_file(file)
            # df = df.to_crs(crs=4326)

            list_of_df.append(df)

        self.fact = pd.concat(list_of_df, ignore_index=True)
        fact = self.fact.explode(ignore_index=True).centroid # Multipolygon to polygon to points at the center

        # Removing the padding
        # Trimming the airspace to the vicinity of the trajectory
        # boundary = [5.5488, 43.5254, 5.4728, 43.5301]
        boundary = [5.7778, 43.5639, 5.4191, 43.4472]
        bbox = box(boundary[0], boundary[1], boundary[2], boundary[3])
        self.fact = self.fact.intersection(bbox) # Squares
        self.fact = self.fact[~self.fact.is_empty]
        fact = fact.intersection(bbox) # Points
        fact = fact[~fact.is_empty]
        # self.lidar = self.lidar.geometry.unary_union

        # End of trimming

        self.gdf = pd.concat([self.gdf, fact], ignore_index=True)

        # self.fact = gp.GeoDataFrame(geometry=self.fact)
        self.fact = gp.GeoDataFrame(geometry=fact)
        self.fact["stroke"] = "blue"
        self.fact["marker-color"] = "blue"
        self.fact["fill"] = "blue"
        if self.logging:
            self.fact.to_file('plot/fact.geojson', driver='GeoJSON')
        print("Imported factories")

        points = []
        for i, row in fact.iteritems():
            lon = fact.geometry[i].x
            lat = fact.geometry[i].y
            point = [lon, lat]
            points.append(point)
        self.fact_ls = points
        # self.store_factories()
        return points

    def import_lidar(self):
        filename = self.project + "lidar/*.geojson"
        list_of_files = glob.glob(filename)
        if not list_of_files:
            print("No factories to import")
            return 0
        list_of_df = []

        for filename in list_of_files:
            file = open(filename)
            df = gp.read_file(file)
            # df = df.to_crs(crs=4326)

            list_of_df.append(df)

        self.lidar = pd.concat(list_of_df, ignore_index=True)
        lidar = self.lidar.explode(ignore_index=True).centroid # Multipolygon to polygon to points at the center

        # Removing the padding
        # Trimming the airspace to the vicinity of the trajectory
        boundary = [5.5488, 43.5254, 5.4728, 43.5301]
        bbox = box(boundary[0], boundary[1], boundary[2], boundary[3])
        self.lidar = self.lidar.difference(bbox) # Squares
        self.lidar = self.lidar[~self.lidar.is_empty]
        lidar = lidar.difference(bbox) # Points
        lidar = lidar[~lidar.is_empty]
        # self.lidar = self.lidar.geometry.unary_union

        # End of trimming

        self.gdf = pd.concat([self.gdf, lidar], ignore_index=True)

        self.lidar = gp.GeoDataFrame(geometry=self.lidar)
        self.lidar["stroke"] = "green"
        self.lidar["marker-color"] = "green"
        self.lidar["fill"] = "green"
        if self.logging:
            self.lidar.to_file('plot/lidar.geojson', driver='GeoJSON')
        print("Imported lidar")
        return self.lidar

    def import_roads(self):
        filename = self.project + "roads/*.geojson"
        list_of_files = glob.glob(filename)
        print(list_of_files)
        list_of_df = []
        for filename in list_of_files:
            file = open(filename)
            df = gp.read_file(file)

            N = df.geometry.size
            gdf = df.to_crs(crs=4326)
            gdf = gdf.buffer(self.c * self.radius)
            list_of_df.append(gdf)
            gdf = df.centroid
            rds_temp = np.zeros((N,2))
            for i in range(N):
                rds_temp[i, 0] = gdf[i].x
                rds_temp[i, 1] = gdf[i].y
            self.se_ls = np.concatenate((self.se_ls, rds_temp), axis=0)
        # self.se_ls = np.delete(self.rds, 0, 0)

        self.rds_gdf = pd.concat(list_of_df, ignore_index=True)
        print("Imported roads")
        s = gp.GeoSeries(self.rds_gdf)
        p = gp.GeoSeries(s.unary_union)
        self.rds_gdf = p
        ## Coloring
        color = self.rds_gdf
        color = gp.GeoDataFrame(geometry=color)
        color["stroke"] = "blue"
        color["marker-color"] = "blue"
        color["fill"] = "blue"
        if self.logging:
            color.to_file('plot/rds.geojson', driver='GeoJSON')
        return s.unary_union

    def import_large_areas(self):
        # INCOMPLETE
        filename = self.project + "areas/*.geojson"
        list_of_files = glob.glob(filename)
        print(list_of_files)
        ls = np.zeros((1,2))
        list_of_df = []
        for filename in list_of_files:
            file = open(filename)
            df = gp.read_file(file)

            N = df.geometry.size
            gdf = df.to_crs(crs=4326)

            list_of_df.append(gdf)

        #     ls_temp = np.zeros((N,2))
        #     for i in range(N):
        #         ls_temp[i, 0] = gdf[i].x
        #         ls_temp[i, 1] = gdf[i].y
        #     ls = np.concatenate((ls, ls_temp), axis=0)
        # ls = np.delete(ls, 0, 0)
        self.se_ls = np.concatenate((self.se_ls, ls), axis=0)
        self.ls_gdf = pd.concat(list_of_df, ignore_index=True)
        self.gdf = pd.concat([self.se_gdf, self.ls_gdf], ignore_index=True)
        print("Imported landing")

        ## Coloring
        color = self.ls_gdf
        color = gp.GeoDataFrame(geometry=color)
        color["stroke"] = "gray"
        color["marker-color"] = "gray"
        color["fill"] = "gray"
        color["marker-size"] = "small"
        if self.logging:
            color.to_file('ls.geojson', driver='GeoJSON')

    def import_forbidden(self):
        filename = self.project + "forbidden/*.geojson"
        list_of_files = glob.glob(filename)
        print(list_of_files)
        list_of_df = []
        for filename in list_of_files:
            file = open(filename)
            df = gp.read_file(file)

            N = df.geometry.size
            gdf = df.to_crs(crs=4326)

            list_of_df.append(gdf)
            gdf = df.centroid
            fa_temp = np.zeros((N,2))
            for i in range(N):
                fa_temp[i, 0] = gdf[i].x
                fa_temp[i, 1] = gdf[i].y
            self.fa = np.concatenate((self.fa, fa_temp), axis=0)
        self.fa = np.delete(self.fa, 0, 0)

        self.fa_gdf = pd.concat(list_of_df, ignore_index=True)
        print("Imported forbidden")
        self.process_forbidden()

        ## Coloring
        color = self.fa_gdf
        color["stroke"] = "red"
        color["marker-color"] = "red"
        color["fill"] = "red"
        if self.logging:
            color.to_file('plot/fa.geojson', driver='GeoJSON')

    def import_missions():
        """
        Reads vehicle traffic data from a pickle file.
        """
        filename = "traffic"
        print("Reading traffic from file")
        with open("missions/" + filename + '.pkl', 'rb') as traffic_file:
            traffic = pkl.load(traffic_file)
        return traffic


    def load_geojson_files(pattern: str, concat: bool = False, crs_epsg: int = 20437):
        """
        Load and transform GeoJSON files matching a pattern.

        Parameters:
            pattern (str): Glob pattern for filenames.
            concat (bool): If True, returns a single concatenated GeoDataFrame.
            crs_epsg (int): EPSG code to reproject the data to.

        Returns:
            list[GeoDataFrame] or GeoDataFrame: Loaded and reprojected data.
        """
        files = glob.glob(pattern)
        gdfs = [gp.read_file(f).to_crs(epsg=crs_epsg) for f in files]
        return pd.concat(gdfs, ignore_index=True) if concat else gdfs

    def import_inspection():
        return myio.load_geojson_files("env/solar*.geojson", concat=False)

    def import_research():
        return myio.load_geojson_files("env/mangrove.geojson", concat=True)

    def import_recreational():
        return myio.load_geojson_files("env/recreational.geojson", concat=True)

    def import_perimeter():
        return myio.load_geojson_files("env/perimeter.geojson", concat=True)

    def import_traffic_zones():
        return myio.load_geojson_files("env/intersections.geojson", concat=True)

    def store_adjacency_matrix(self):
        print("Saving adjacency matrix to file")
        df = pd.DataFrame(self.m_adj)
        df.to_csv("adj.csv", index=False, header=False)

        df = pd.DataFrame(self.m_heur)
        df.to_csv("heur.csv", index=False, header=False)

        df = pd.DataFrame(self.ls)
        df.to_csv("ls.csv", index=False, header=False)
        print("Done saving")

    def load_adjacency_matrix(self):
        print("Reading adjacency matrix from file")
        m_adj = pd.read_csv("adj.csv", delimiter=",", header=None).to_numpy()
        m_heur = pd.read_csv("heur.csv", delimiter=",", header=None).to_numpy()
        ls = pd.read_csv("ls.csv", delimiter=",", header=None).to_numpy()
        return m_adj, m_heur, ls

    def store_factories(self):
        print("Saving factories matrix to file")
        print(len(self.fact_ls))
        m_pair = []
        # for i in range(m_adj.shape[0] - len(factories), m_adj.shape[0], 2):
        for i in range(len(self.fact_ls)):
            pair = [self.fact_ls[i][0], self.fact_ls[i][1]]
            m_pair.append(pair)

        df = pd.DataFrame(m_pair)
        df.to_csv("factories.csv", index=False, header=False)

        print("Done saving")

    def read_highways(self):
        filename = self.project + "points.csv"
        routes = pd.read_csv(filename, delimiter=",", header=None).to_numpy()
        print(routes)
        return routes

    def traj_to_linestring(traj):
        if len(traj) <= 1:
            return
        points = []
        for i in range(len(traj)):
            point = Point(traj[i][0], traj[i][1], traj[i][2]) # 3D trajectory
            # point = Point(traj[i][1], traj[i][0]) #2D trajectory
            points.append(point)
        s_line = LineString(points)
        return s_line

    def transform_meter_global(geom):
        gdf = gp.GeoDataFrame(geometry=geom, crs="EPSG:20437")
        gdf.to_crs(epsg=4326, inplace=True)
        return gdf.geometry

    def transform_global_meter(geom):
        gdf = gp.GeoDataFrame(geometry=geom, crs="EPSG:4326")
        gdf.to_crs(epsg=20437, inplace=True)
        return gdf.geometry

    def write_highways(self, routes, name, color):
        geom = []
        for i in range(len(routes)):
            s = [routes[i][0], routes[i][1]]
            e = [routes[i][2], routes[i][3]]
            traj = [s, e]
            linestring = self.traj_to_linestring(traj)
            geom.append(linestring)
        self.write_geom(geom, name, color)

    def write_path(self, path, name, color):
        traj = self.path_to_traj(path)
        self.write_traj(traj, name, color)

    def write_traj(self, traj, name, color):
        s_line = self.traj_to_linestring(traj)
        traj_gdf = gp.GeoDataFrame(index=[0], crs=4326, geometry=[s_line])
        traj_gdf["stroke"] = color
        traj_gdf["stroke-width"] = 3
        traj_gdf.to_file("plot/" + name + ".geojson", driver='GeoJSON')

    def write_geom(geom, name, color):
        s = gp.GeoDataFrame(crs=4326, geometry=geom)
        s["stroke"] = color
        s["marker-color"] = color
        s["fill"] = color
        s.to_file('plot/' + name + '.geojson', driver='GeoJSON')

    def write_adj(self, a):
        # Used for testing and seeing the connections between the nodes
        z = a
        concat_gdf = []
        point1 = Point(self.gdf.geometry[z].x, self.gdf.geometry[z].y)
        point2 = point1
        s_line = LineString([point1, point2])
        traj = gp.GeoDataFrame(index=[0], crs=4326, geometry=[s_line])
        concat_gdf.append(traj)
        print(len(self.m_adj))
        for i in range(1, len(self.m_adj)):
            # for j in range(i):
            if (self.m_adj[i, z] != 0):
                # print("None zero")
                point2 = Point(self.gdf.geometry[i].x, self.gdf.geometry[i].y)
                s_line = LineString([point1, point2])
                traj = gp.GeoDataFrame(index=[0], crs=4326, geometry=[s_line])
                concat_gdf.append(traj)
        comb_gdf = pd.concat(concat_gdf, axis=0, ignore_index=True)
        comb_gdf.to_file('plot/adj.geojson', driver='GeoJSON')

    def write_circle(self, a):
        radius = self.radius

        lon = self.gdf.geometry[a].x
        lat = self.gdf.geometry[a].y
        circle = self.draw_circle(lon, lat, radius)

        s = gp.GeoDataFrame(index=[0], crs=4326, geometry=[circle])

        s.to_file('circle.geojson', driver='GeoJSON')

    def write_circles(self, path, rad, color, opacity):
        radius = rad
        circles = []
        for i in range(len(path)):
            lon = self.gdf.geometry[path[i]].x
            lat = self.gdf.geometry[path[i]].y
            circles.append(self.draw_circle(lon, lat, radius))
        s = gp.GeoDataFrame(crs=4326, geometry=circles)
        p = gp.GeoSeries(s.unary_union)
        s = gp.GeoDataFrame(crs=4326, geometry=p)
        s["fill-opacity"] = 0.15 * opacity
        s["stroke-opacity"] = 0.5 * opacity
        s["stroke"] = color
        s["marker-color"] = color
        s["fill"] = color
        s.to_file('plot/circles_' + color + '.geojson', driver='GeoJSON')

    def read_log_pickle(last):
        print("Reading trajectories from file")
        filename = 'plot/' + last + '*' + ".pkl"
        list_of_files = glob.glob(filename)
        mission_pickle = list_of_files[0]
        print(mission_pickle)
        mission_file = open(mission_pickle, 'rb')
        missions_dict = pkl.load(mission_file)
        mission_file.close()
        return missions_dict

    def read_pickle(last, name):
        # print("Reading missions from file")
        filename = 'plot/' + last + name + ".pkl"
        list_of_files = glob.glob(filename)
        mission_pickle = list_of_files[0]
        mission_file = open(mission_pickle, 'rb')
        missions_dict = pkl.load(mission_file)
        mission_file.close()
        return missions_dict

    def log_to_json(drones, run, last):
        trajs = []
        for drone in drones:
            t = []
            for traj in drone["trajs"]:
                point = [traj[0][0], traj[1][0], traj[2][0]]
                t.append(point)
            if len(t) == 1:
                continue
            ls = myio.traj_to_linestring(t)
            trajs.append(ls)
        df = gp.GeoDataFrame(geometry=trajs, crs="EPSG:20437")
        df.to_crs(crs=4326, inplace=True)
        trajs = df.geometry
        myio.write_geom(trajs, run + "trajs", "blue")
        myio.write_geom(trajs, last + "trajs", "blue")

    def log_to_json_dict(drones, run, last):
        if len(drones) == 0:
            print("No drones to log")
            return
        trajs = []
        dt = 0.1
        for drone in drones:
            t = []
            for traj in drone["trajs"]:
                # point = Point(traj[0][0], traj[1][0], traj[2][0])
                point = [traj[0][0], traj[1][0], traj[2][0]]
                t.append(point)
            ls = myio.traj_to_linestring(t)
            # trajs.append(ls)
            s = drone["birthday"]
            # 100 milliseconds for each timestep
            T = datetime.timedelta(milliseconds=100*(len(t)-1)) # Duration of the flight in seconds
            e = s + T
            # [safe_dist, outside, unsafe] = measure_safe_distance(traj, safe, nfz)
            trajs.append({'geometry': ls, 'start_datetime': s, 'end_datetime': e, 'iteration': drone["iteration"], 'length': len(t)})

        df = pd.DataFrame(trajs)
        gdf = gp.GeoDataFrame(df, crs="EPSG:20437")
        gdf.to_crs(crs=4326, inplace=True)

        name = "trajs_time"
        # gdf.to_file('plot/' + name + '.geojson', driver='GeoJSON')
        # now = datetime.datetime.now()
        # date = now.strftime("%y-%m-%d-%H%M%S")
        # gdf.to_file('plot/trajs/' + name + date + '.geojson', driver='GeoJSON')

        gdf.to_file('plot/' + run + name + '.geojson', driver='GeoJSON')
        gdf.to_file('plot/' + last + name + '.geojson', driver='GeoJSON')

    def log_to_pickle(dictionary, name, run, last):
        # now = datetime.datetime.now()
        # date = now.strftime("%y-%m-%d-%H%M%S")
        with open('plot/' + run + name + '.pkl', 'wb') as fp:
            pkl.dump(dictionary, fp, protocol=pkl.HIGHEST_PROTOCOL)

        with open('plot/' + last + name + '.pkl', 'wb') as fp:
            pkl.dump(dictionary, fp, protocol=pkl.HIGHEST_PROTOCOL)

    def log_timed_geom(geoms, time, run, last):
        gs = []
        dt = 0.1
        for k, geom in enumerate(geoms):
            s = time[k][0]
            e = time[k][1]
            # T = datetime.timedelta(milliseconds=100*(len(t)-1)) # Duration of the flight in seconds
            T = e - s # Placeholder
            gs.append({'geometry': geom, 'start_datetime': s, 'end_datetime': e, 'length': T})

        df = pd.DataFrame(gs)
        gdf = gp.GeoDataFrame(df, crs="EPSG:20437")
        gdf.to_crs(crs=4326, inplace=True)

        name = "avoid_time"

        gdf.to_file('plot/' + run + name + '.geojson', driver='GeoJSON')
        gdf.to_file('plot/' + last + name + '.geojson', driver='GeoJSON')

    def combine_json_files(self, list, output):
        concat_gdf = []
        for filename in list:
            file = open(filename)
            df = gp.read_file(file)
            concat_gdf.append(df)
        comb_gdf = pd.concat(concat_gdf, axis=0, ignore_index=True)
        comb_gdf.to_file('plot/' + output + '.geojson', driver='GeoJSON')