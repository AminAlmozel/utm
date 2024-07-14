import gurobipy as gp
import numpy as np
from gurobipy import GRB
import matplotlib.pyplot as plt
import matplotlib.animation as animation
# from mayavi import mlab
import time
import warnings
from math import copysign
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore')


class drone():
    def __init__(self):
        self.N = 40
        self.delta_t = 0.1 # Time step
        self.N_polygon = 8 # Number of sides for the polygon approximation
        self.not_collided = True

        # Parameters
        self.K = 1  # Number of vehicles
        self.K_other = 21 # number of other drones in proximity
        # self.L = 4  # Number of stationary obstacles
        self.min_dist = 1

        # Parameters for polygon approximation
        self.V_min = 0  # Minimum velocity
        self.V_max = 20 # Maximum velocity (for bounding purposes)
        self.A_max = 3  # Maximum acceleration (already defined)
        self.theta = [2 * np.pi * k / self.N_polygon for k in range(1, self.N_polygon+1)]  # Polygon vertex angles

        # Bounds for states and controls
        self.smin = [- 100, - 100, -100, - self.V_max, - self.V_max, - self.V_max]  # Position in m, Velocity in m/s
        self.smax = [100, 100, 100, self.V_max,  self.V_max, self.V_max] # Position in m, Velocity in m/s
        self.umin = [-self.A_max, -self.A_max, -self.A_max]  # Acceleration in m/s^2
        self.umax = [self.A_max, self.A_max, self.A_max]  # Acceleration in m/s^2

        # Define weights for the objective function
        self.qo = [1] * 6  # State deviation weights
        self.ro = [1] * 3  # Control effort weights
        self.po = [100] * 6  # Terminal state weights

        # Parameters for collision avoidance between vehicles
        d_min = 3
        self.d_x = d_min  # Minimum horizontal distance
        self.d_y = d_min  # Minimum vertical distance
        self.d_z = d_min  # Minimum vertical distance

        self.M = 100000  # Large positive number for 'big-M' method

        self.vehicles_positions = []
        self.vehicles = []
        self.full_traj = []

        self.hlines = []

        self.obstacles = []

        # self.env = gp.Env()
        # # self.m = gp.Model(env=env)

    def generate_traj(self, xi, xi_1, obstacles):
        # Clear the model from previously setup constraints
        self.env = gp.Env()
        self.env.setParam(GRB.Param.OutputFlag, 0)
        self.env.setParam(GRB.Param.LogToConsole, 0)
        self.m = gp.Model(env=self.env)
        # import obstacles and position of other drones
        self.set_initial_condition(xi)
        self.set_obstacles(obstacles)
        self.set_other_drones_trajectories(xi_1)

        # Setup optimization program for the drone
        self.setup_variables()
        self.setup_constraints()
        self.setup_objective()

        # Solve
        self.optimize()

        # return the full trajectory
        self.update_vehicle_state()
        # print(self.full_traj)
        return self.full_traj

    def set_initial_condition(self, xi):
        self.initial_conditions = []
        self.initial_conditions.append(xi)

    def set_final_condition(self, xf):
        self.final_conditions = []
        self.final_conditions.append(xf)

    def set_obstacles(self, obstacles):
        self.obstacles = obstacles
        # Set the maximum number of edges in a polygon
        max_edges = 0
        for obs in obstacles:
            if obs["edges"] > max_edges:
                max_edges = obs["edges"]
        self.max_edges = max_edges

        # for obs in obstacles:
        #      for j, key in enumerate(['xmin', 'ymin', 'zmin', 'xmax', 'ymax', 'zmax']):
        #         obstacle[key] = obs[key]
        #     self.obstacles.append(obstacle)

    def set_other_drones_trajectories(self, xi_1):
        self.k_traj = xi_1
        self.K = len(self.k_traj)

    def initialize(self, update='False'):
        # Obstacle parameters
        self.obstacles = [
            {'xmin': -100, 'ymin': 20, 'zmin': -100, 'xmax': -20, 'ymax': 100, 'zmax': 100},
            {'xmin': 20, 'ymin': 20, 'zmin': -100, 'xmax': 100, 'ymax': 100, 'zmax': 100},
            {'xmin': -100, 'ymin': -100, 'zmin': -100, 'xmax': -20, 'ymax': -20, 'zmax': 100},
            {'xmin': 20, 'ymin': -100, 'zmin': -100, 'xmax': 100, 'ymax': -20, 'zmax': 100}
        ]

        # Initial and final conditions for each vehicle
        if update == 'False':
            self.initial_conditions = [
                {'x': -95, 'y': 0, 'z': 10, 'xdot': 3, 'ydot': 3, 'zdot': 2},
                {'x': 0, 'y': 95, 'z': 10, 'xdot': 3, 'ydot': 3, 'zdot': 2},
                {'x': 90, 'y': 0, 'z': 15, 'xdot': 3, 'ydot': 3, 'zdot': 2},
                {'x': 10, 'y': -95, 'z': 10, 'xdot': 3, 'ydot': 3, 'zdot': 2},

                {'x': 0, 'y': 0, 'z': -95, 'xdot': 3, 'ydot': 3, 'zdot': 2},
                {'x': 0, 'y': 0, 'z': 95, 'xdot': 3, 'ydot': 3, 'zdot': 2},
                {'x': 15, 'y': 95, 'z': 95, 'xdot': 3, 'ydot': 3, 'zdot': 2},
                {'x': -15, 'y': -95, 'z': 95, 'xdot': 3, 'ydot': 3, 'zdot': 2},

                {'x': 95, 'y': 15, 'z': 95, 'xdot': 3, 'ydot': 3, 'zdot': 2},
                {'x': -95, 'y': 15, 'z': 95, 'xdot': 3, 'ydot': 3, 'zdot': 2},
                {'x': -15, 'y': -95, 'z': -95, 'xdot': 3, 'ydot': 3, 'zdot': 2},
                {'x': 15, 'y': 95, 'z': -95, 'xdot': 3, 'ydot': 3, 'zdot': 2},

                {'x': 95, 'y': -15, 'z': 95, 'xdot': 3, 'ydot': 3, 'zdot': 2},
                {'x': -95, 'y': 15, 'z': -95, 'xdot': 3, 'ydot': 3, 'zdot': 2},

                {'x': -95, 'y': -15, 'z': 95, 'xdot': 3, 'ydot': 3, 'zdot': 2},
                {'x': -95, 'y': -15, 'z': -95, 'xdot': 3, 'ydot': 3, 'zdot': 2},

                {'x': 95, 'y': -15, 'z': -95, 'xdot': 3, 'ydot': 3, 'zdot': 2},
                {'x': 95, 'y': 15, 'z': -95, 'xdot': 3, 'ydot': 3, 'zdot': 2},

                {'x': -15, 'y': 95, 'z': -95, 'xdot': 3, 'ydot': 3, 'zdot': 2},
                {'x': 15, 'y': -95, 'z': 95, 'xdot': 3, 'ydot': 3, 'zdot': 2},

                {'x': 15, 'y': -95, 'z': -95, 'xdot': 3, 'ydot': 3, 'zdot': 2},
                {'x': -15, 'y': 95, 'z': 95, 'xdot': 3, 'ydot': 3, 'zdot': 2}

                # {'x': 2, 'y': 1, 'xdot': 0, 'ydot': 0}
                ]
            # print("================== at first", self.initial_conditions)
        # else:
        #     print("================== iteration", self.initial_conditions)

        self.final_conditions = [
            {'x': 95, 'y': 10, 'z': 10, 'xdot': 2, 'ydot': 2, 'zdot': 2},
            {'x': 10, 'y': -95, 'z': 10, 'xdot': 2, 'ydot': 2, 'zdot': 2},
            {'x': -95, 'y': 0, 'z': 10, 'xdot': 2, 'ydot': 2, 'zdot': 2},
            {'x': 0, 'y': 95, 'z': 10, 'xdot': 2, 'ydot': 2, 'zdot': 2},

            {'x': 0, 'y': 0, 'z': 95, 'xdot': 2, 'ydot': 2, 'zdot': 2},
            {'x': 0, 'y': 0, 'z': -95, 'xdot': 2, 'ydot': 2, 'zdot': 2},
            {'x': -15, 'y': -95, 'z': -95, 'xdot': 2, 'ydot': 2, 'zdot': 2},
            {'x': 15, 'y': 95, 'z': -95, 'xdot': 2, 'ydot': 2, 'zdot': 2},

            {'x': -95, 'y': -15, 'z': -95, 'xdot': 2, 'ydot': 2, 'zdot': 2},
            {'x': 95, 'y': -15, 'z': -95, 'xdot': 2, 'ydot': 2, 'zdot': 2},
            {'x': 15, 'y': 95, 'z': 95, 'xdot': 2, 'ydot': 2, 'zdot': 2},
            {'x': -15, 'y': -95, 'z': 95, 'xdot': 2, 'ydot': 2, 'zdot': 2},


            {'x': -95, 'y': 15, 'z': -95, 'xdot': 2, 'ydot': 2, 'zdot': 2},
            {'x': 95, 'y': -15, 'z': 95, 'xdot': 2, 'ydot': 2, 'zdot': 2},

            {'x': 95, 'y': 15, 'z': -95, 'xdot': 2, 'ydot': 2, 'zdot': 2},
            {'x': 95, 'y': 15, 'z': 95, 'xdot': 2, 'ydot': 2, 'zdot': 2},

            {'x': -95, 'y': 15, 'z': 95, 'xdot': 2, 'ydot': 2, 'zdot': 2},
            {'x': -95, 'y': -15, 'z': 95, 'xdot': 2, 'ydot': 2, 'zdot': 2},

            {'x': 15, 'y': -95, 'z': 95, 'xdot': 3, 'ydot': 3, 'zdot': 2},
            {'x': -15, 'y': 95, 'z': -95, 'xdot': 3, 'ydot': 3, 'zdot': 2},

            {'x': -15, 'y': 95, 'z': 95, 'xdot': 3, 'ydot': 3, 'zdot': 2},
            {'x': 15, 'y': -95, 'z': -95, 'xdot': 3, 'ydot': 3, 'zdot': 2}
            # {'x': 9, 'y': 10, 'xdot': 0, 'ydot': 0}
            # {'x': 8, 'y': 2, 'xdot': 0, 'ydot': 0},
            # {'x': 13, 'y': 9, 'xdot': 0, 'ydot': 0}
        ]

    def setup_variables(self):
        self.vehicles = []
        # Create variables for the vehicle
        p = 0
        vehicle = {
            's': self.m.addVars(self.N, 6, lb=[self.smin] * self.N, ub=[self.smax] * self.N, name=f"s_{p}"),
            'u': self.m.addVars(self.N, 3, lb=[self.umin] * self.N, ub=[self.umax] * self.N, name=f"u_{p}"),
            'w': self.m.addVars(self.N, 6, name=f"w_{p}"),
            'v': self.m.addVars(self.N, 3, name=f"v_{p}"),
            't': [self.m.addVars(self.N, self.max_edges + 2, vtype=GRB.BINARY, name=f"t_{p}_obstacle_{c}") for c in range(len(self.obstacles))],
            'tc': self.m.addVars(self.N, self.N_polygon, vtype=GRB.BINARY, name=f"tc_{p}_accel")
        }
        self.vehicles.append(vehicle)

        # Create binary variables for collision avoidance between each pair of vehicles
        self.b_vars = {}
        # print(self.K)
        for p in range(self.K): # 0, 1
            self.b_vars[p] = self.m.addVars(self.N, 6, vtype=GRB.BINARY, name=f"b_{p}")

        # print("Done setup_variable")
        return 0

    def setup_constraints(self):
        self.state_constraints()
        self.control_constraints()
        self.state_transition_constraints()
        self.initial_final_condition_constraints()
        # self.obstacle_avoidance_constraints()
        self.general_obstacle_avoidance_constraints()
        # self.vehicle_collision_avoidance_constraints()
        # self.fixed_vehicle_collision_avoidance_constraints()
        # print("done setup constraints")

    def state_constraints(self):
        # Add constraints and objective components for each vehicle
        for p, vehicle in enumerate(self.vehicles):
            s, u, w, v_vars, t_vars, tc_vars = vehicle['s'], vehicle['u'], vehicle['w'], vehicle['v'], vehicle['t'], vehicle['tc']

            # State and velocity constraints with slack variables
            for i in range(1, self.N):  # i = 1 to N
                for j in range(6):  # Position and velocity deviation constraints
                    self.m.addConstr(
                        s[i, j] - self.final_conditions[p][['x', 'y', 'z', 'xdot', 'ydot', 'zdot'][j]] <= w[i, j],
                        f"State_Dev_{j}_{p}_{i}")
                    self.m.addConstr(
                        -s[i, j] + self.final_conditions[p][['x', 'y', 'z', 'xdot', 'ydot', 'zdot'][j]] <= w[i, j],
                        f"State_Dev_Neg_{j}_{p}_{i}")

            # Do we need this?
            for i in range(1, self.N):  # State bounds 1 to N
                for j in range(3):
                    self.m.addConstr(s[i, j] >= self.smin[j], f"State_Lower_Bound_{j}_{p}_{i}")
                    self.m.addConstr(s[i, j] <= self.smax[j], f"State_Upper_Bound_{j}_{p}_{i}")

            # Velocity magnitude limitation using polygon approximation
            for i in range(1, self.N):
                for k, angle in enumerate(self.theta):

                    self.m.addConstr(np.sin(angle) * s[i, 3] + np.cos(angle) * s[i, 4] <= self.V_max,
                                    f"Velocity_Poly_{k}_{p}_{i}")
                    self.m.addConstr(np.sin(angle) * s[i, 3] + np.cos(angle) * s[i, 4] >= self.V_min- self.M * tc_vars[i, k],
                                    f"Min_Vel_{k}_{p}_{i}")
                # Binary Variable of V_min
                self.m.addConstr(sum(tc_vars[i, k] for k in range(self.N_polygon)) <= self.N_polygon - 1,
                                f"Min_Velocity_Constraint_{p}_{i}")

            # Do we need this?
            # Bounded Velocity along Z-Axis (Decoupled)
            for i in range(1, self.N):
                # self.m.addConstr(smin[5] <= s[i, 5] <= smax[5], f"Min_MAX_z_Vel_{p}_{i}")

                self.m.addConstr(s[i, 5] >= self.smin[5], f"z_Vel_Lower_Bound_{p}_{i}")
                self.m.addConstr(s[i, 5] <= self.smax[5], f"z_Vel_Upper_Bound_{p}_{i}")

    def control_constraints(self):

        # Add constraints and objective components for each vehicle
        for p, vehicle in enumerate(self.vehicles):
            s, u, w, v_vars, t_vars, tc_vars = vehicle['s'], vehicle['u'], vehicle['w'], vehicle['v'], vehicle['t'], vehicle['tc']

            # Control constraints with slack variables
            for i in range(self.N):  # i = 0 to N - 1
                for j in range(3):  # Control effort constraints
                    self.m.addConstr(u[i, j] <= v_vars[i, j], f"Control_Effort_{j}_{p}_{i}")
                    self.m.addConstr(-u[i, j] <= v_vars[i, j], f"Control_Effort_Neg_{j}_{p}_{i}")

            # Bounded Acceleration along Z-Axis (Decoupled)
            for i in range(0, self.N-1):
                # self.m.addConstr(umin[2] <= u[i, 2] <= umax[2], f"Min_MAX_z_Acc_{p}_{i}")
                self.m.addConstr(u[i, 2] >= self.umin[2], f"z_Acc_Lower_Bound_{p}_{i}")
                self.m.addConstr(u[i, 2] <= self.umax[2], f"z_Acc_Upper_Bound_{p}_{i}")

            # Acceleration magnitude limitation using polygon approximation, no A_min
            for i in range(0, self.N - 1):
                for k, angle in enumerate(self.theta):
                    self.m.addConstr(np.sin(angle) * u[i, 0] + np.cos(angle) * u[i, 1] <= self.A_max,
                                         f"Accel_Poly_{k}_{p}_{i}")

    def state_transition_constraints(self):
        # Add constraints and objective components for each vehicle
        for p, vehicle in enumerate(self.vehicles):
            s, u, w, v_vars, t_vars, tc_vars = vehicle['s'], vehicle['u'], vehicle['w'], vehicle['v'], vehicle['t'], vehicle['tc']

            # Dynamics constraints
            for i in range(0, self.N-1):# 0 to N-1
                self.m.addConstr(s[i + 1, 0] == s[i, 0] + self.delta_t * s[i, 3] + ((self.delta_t ** 2)/2)*u[i, 0], f"Dynamics_x_{p}_{i}")
                self.m.addConstr(s[i + 1, 1] == s[i, 1] + self.delta_t * s[i, 4] + ((self.delta_t ** 2)/2)*u[i, 1], f"Dynamics_y_{p}_{i}")
                self.m.addConstr(s[i + 1, 2] == s[i, 2] + self.delta_t * s[i, 5] + ((self.delta_t ** 2)/2)*u[i, 2], f"Dynamics_z_{p}_{i}")
                self.m.addConstr(s[i + 1, 3] == s[i, 3] + self.delta_t * u[i, 0], f"Dynamics_xdot_{p}_{i}")
                self.m.addConstr(s[i + 1, 4] == s[i, 4] + self.delta_t * u[i, 1], f"Dynamics_ydot_{p}_{i}")
                self.m.addConstr(s[i + 1, 5] == s[i, 5] + self.delta_t * u[i, 2], f"Dynamics_ydot_{p}_{i}")

    def initial_final_condition_constraints(self):
        # Add constraints and objective components for each vehicle
        for p, vehicle in enumerate(self.vehicles):
            s, u, w, v_vars, t_vars, tc_vars = vehicle['s'], vehicle['u'], vehicle['w'], vehicle['v'], vehicle['t'], vehicle['tc']

            # Initial and final conditions
            self.m.addConstrs((s[0, j] == self.initial_conditions[p][key] for j, key in enumerate(['x', 'y', 'z', 'xdot', 'ydot', 'zdot'])), f"Initial_{p}")
            # self.m.addConstrs((s[N - 1, j] == self.final_conditions[p][key] for j, key in enumerate(['x', 'y', 'z', 'xdot', 'ydot', 'zdot'])), f"Final_{p}")

    def obstacle_avoidance_constraints(self):
        # Add constraints and objective components for each vehicle
        for p, vehicle in enumerate(self.vehicles):
            s, u, w, v_vars, t_vars, tc_vars = vehicle['s'], vehicle['u'], vehicle['w'], vehicle['v'], vehicle['t'], vehicle['tc']
            # Obstacle avoidance constraints for each vehicle 'p'
            for c, obs in enumerate(self.obstacles):
                t = t_vars[c]
                for i in range(1, self.N):  # i = 1 to N
                    self.m.addConstr(s[i, 0] <= obs['xmin'] - self.min_dist + self.M * t[i, 0], f"Obstacle_x_{c}_{p}_{i}")
                    self.m.addConstr(-s[i, 0] <= -obs['xmax'] - self.min_dist + self.M * t[i, 1], f"Obstacle_Neg_x_{c}_{p}_{i}")
                    self.m.addConstr(s[i, 1] <= obs['ymin'] - self.min_dist + self.M * t[i, 2], f"Obstacle_y_{c}_{p}_{i}")
                    self.m.addConstr(-s[i, 1] <= -obs['ymax'] - self.min_dist + self.M * t[i, 3], f"Obstacle_Neg_y_{c}_{p}_{i}")
                    self.m.addConstr(s[i, 2] <= obs['zmin'] - self.min_dist + self.M * t[i, 4], f"Obstacle_z_{c}_{p}_{i}")
                    self.m.addConstr(-s[i, 2] <= -obs['zmax'] - self.min_dist + self.M * t[i, 5], f"Obstacle_Neg_z_{c}_{p}_{i}")
                    self.m.addConstr(t[i, 0] + t[i, 1] + t[i, 2] + t[i, 3] + t[i, 4] + t[i, 5] <= 5, f"Obstacle_Sum_{c}_{p}_{i}")

    def general_obstacle_avoidance_constraints(self):
         # Add constraints and objective components for each vehicle
        for p, vehicle in enumerate(self.vehicles):
            s, u, w, v_vars, t_vars, tc_vars = vehicle['s'], vehicle['u'], vehicle['w'], vehicle['v'], vehicle['t'], vehicle['tc']
            # Obstacle avoidance constraints for each vehicle 'p'
            for c, obs in enumerate(self.obstacles):
                t = t_vars[c]
                p = obs["geom"]
                # height = [0, obs["height"]]
                height = obs["height"]
                vertices = list(zip(*p.exterior.coords.xy))
                for n in range (1, self.N):
                    binary_sum = 0
                    for i in range(len(vertices) - 1):
                        p1 = [vertices[i][0], vertices[i][1]]
                        p2 = [vertices[i + 1][0], vertices[i + 1][1]]
                        a, b, c, sign = self.get_line_coeff(p1, p2)
                        self.m.addConstr(sign * (a * s[n, 1] + b * s[n, 0] + c) <= self.M * t[n, i])
                        binary_sum += t[n, i]

                    i = self.max_edges + 2
                    self.m.addConstr(-s[n, 2] <= -height[0] - self.min_dist + self.M * t[n, i - 2], f"Obstacle_Neg_z_{c}_{p}_{i}")
                    self.m.addConstr(s[n, 2] <= height[1] - self.min_dist + self.M * t[n, i - 1], f"Obstacle_z_{c}_{p}_{i}")
                    binary_sum += t[n, i - 2] + t[n, i - 1]
                    total_sum = len(vertices)
                    self.m.addConstr(binary_sum  <= total_sum + 2)


    def vehicle_collision_avoidance_constraints(self):
        # Collision avoidance constraints between vehicles
        for p in range(self.K):
            for q in range(p + 1, self.K):
                for i in range(1, self.N):  # i = 1 to N
                    self.m.addConstr(self.vehicles[p]['s'][i, 0] - self.vehicles[q]['s'][i, 0] >= self.d_x - self.M * self.b_vars[(p, q)][i, 0], f"Collision_x_{p}_{q}_{i}")
                    self.m.addConstr(self.vehicles[q]['s'][i, 0] - self.vehicles[p]['s'][i, 0] >= self.d_x - self.M * self.b_vars[(p, q)][i, 1], f"Collision_Neg_x_{p}_{q}_{i}")
                    self.m.addConstr(self.vehicles[p]['s'][i, 1] - self.vehicles[q]['s'][i, 1] >= self.d_y - self.M * self.b_vars[(p, q)][i, 2], f"Collision_y_{p}_{q}_{i}")
                    self.m.addConstr(self.vehicles[q]['s'][i, 1] - self.vehicles[p]['s'][i, 1] >= self.d_y - self.M * self.b_vars[(p, q)][i, 3], f"Collision_Neg_y_{p}_{q}_{i}")
                    self.m.addConstr(self.vehicles[p]['s'][i, 2] - self.vehicles[q]['s'][i, 2] >= self.d_z - self.M * self.b_vars[(p, q)][i, 4],
                                    f"Collision_z_{p}_{q}_{i}")
                    self.m.addConstr(self.vehicles[q]['s'][i, 2] - self.vehicles[p]['s'][i, 2] >= self.d_z - self.M * self.b_vars[(p, q)][i, 5],
                                    f"Collision_Neg_z_{p}_{q}_{i}")
                    self.m.addConstr(self.b_vars[(p, q)][i, 0] + self.b_vars[(p, q)][i, 1] + self.b_vars[(p, q)][i, 2] + self.b_vars[(p, q)][i, 3] + self.b_vars[(p, q)][i, 4]+ self.b_vars[(p, q)][i, 5] <= 5, f"Collision_Sum_{p}_{q}_{i}")

    def fixed_vehicle_collision_avoidance_constraints(self):
        # Collision avoidance constraints between vehicles
        for p, vehicle in enumerate(self.k_traj):
            q = 0
            N = min(len(vehicle[0]), self.N - 1) # The smaller of the prediction horizon and the given trajectories
            for n in range(1, N + 1):  # i = 1 to N
                self.m.addConstr(vehicle[0][n - 1] - self.vehicles[q]['s'][n, 0] >= self.d_x - self.M * self.b_vars[p][n, 0], f"Collision_x_{p}_{q}_{n}")
                self.m.addConstr(self.vehicles[q]['s'][n, 0] - vehicle[0][n - 1] >= self.d_x - self.M * self.b_vars[p][n, 1], f"Collision_Neg_x_{p}_{q}_{n}")
                self.m.addConstr(vehicle[1][n - 1] - self.vehicles[q]['s'][n, 1] >= self.d_y - self.M * self.b_vars[p][n, 2], f"Collision_y_{p}_{q}_{n}")
                self.m.addConstr(self.vehicles[q]['s'][n, 1] - vehicle[1][n - 1] >= self.d_y - self.M * self.b_vars[p][n, 3], f"Collision_Neg_y_{p}_{q}_{n}")
                self.m.addConstr(vehicle[2][n - 1] - self.vehicles[q]['s'][n, 2] >= self.d_z - self.M * self.b_vars[p][n, 4],
                                f"Collision_z_{p}_{q}_{n}")
                self.m.addConstr(self.vehicles[q]['s'][n, 2] - vehicle[2][n - 1] >= self.d_z - self.M * self.b_vars[p][n, 5],
                                f"Collision_Neg_z_{p}_{q}_{n}")
                self.m.addConstr(self.b_vars[p][n, 0] + self.b_vars[p][n, 1] + self.b_vars[p][n, 2] + self.b_vars[p][n, 3] + self.b_vars[p][n, 4]+ self.b_vars[p][n, 5] <= 5, f"Collision_Sum_{p}_{q}_{n}")

    def setup_objective(self):
        # Initialize the objective function
        self.obj = gp.LinExpr()

        # Add constraints and objective components for each vehicle
        for p, vehicle in enumerate(self.vehicles):
            s, u, w, v_vars, t_vars, tc_vars = vehicle['s'], vehicle['u'], vehicle['w'], vehicle['v'], vehicle['t'], vehicle['tc']

            # Add state deviation costs to the objective
            for i in range(1, self.N-1):  # i = 1 to N - 1
                for j in range(6):
                    self.obj.add(self.qo[j] * w[i, j])

            # Add control effort costs to the objective
            for i in range(0, self.N-1):  # i = 0 to N - 1
                for j in range(3):
                    self.obj.add(self.ro[j] * v_vars[i, j])

            # Add terminal state costs to the objective
            for j in range(6):
                self.obj.add(self.po[j] * w[self.N-1, j])
        # print("done setup objective")

    def optimize(self):
        # Set the objective function in the model
        self.m.setObjective(self.obj, GRB.MINIMIZE)
        self.m.setParam(GRB.Param.OutputFlag, 0)
        # self.m.setParam('Threads', 0)
        self.m.update()
        # Optimize the model
        self.m.optimize()

        total_computational_time = self.m.Runtime

        # print("Total Computational Time(s):", total_computational_time, "seconds")
        # for plotting purpose

        tory = 0
        # if self.m.status == GRB.OPTIMAL:
        #     # print("find the optimal")
        #     self.update_visualization_positions()
            # self.update_plot()

            # iteration_positions = []  # Temporary list to store positions for all vehicles in the current iteration
            # for vehicle in self.vehicles:
            #     x_positions = [vehicle['s'][i, 0].X for i in range(self.N)]
            #     y_positions = [vehicle['s'][i, 1].X for i in range(self.N)]
            #     z_positions = [vehicle['s'][i, 2].X for i in range(self.N)]
            #     iteration_positions.append((x_positions, y_positions, z_positions))
            # self.vehicles_positions.append(iteration_positions)

                # # Used for Debugging or for validation
                # xd = [vehicle['s'][i, 3].X for i in range(N)]
                # yd = [vehicle['s'][i, 4].X for i in range(N)]
                # zd = [vehicle['s'][i, 5].X for i in range(N)]
                #
                # vel = np.sqrt(np.square(xd) + np.square(yd))
                # print("==================================================\n")
                # print('vehicle', tory)
                # print("Velocity: ", vel)
                # print("velocity in z", zd)
                #
                # xdd = [vehicle['u'][i, 0].X for i in range(N)]
                # ydd = [vehicle['u'][i, 1].X for i in range(N)]
                # zdd = [vehicle['u'][i, 2].X for i in range(N)]
                #
                # acc = np.sqrt(np.square(xdd) + np.square(ydd))
                #
                # print("Acceleration: ", acc)
                # print("Acceleration in z", zdd, "\n")
                # print("==================================================\n")
            # print("old vehicle", self.vehicles_positions)

            # total_fuel_consumption = sum(
            #     abs(vehicle['u'][i, 0].X) + abs(vehicle['u'][i, 1].X) + abs(vehicle['u'][i, 2].X) for i in range(N))

            # for p, vehicle in enumerate(self.vehicles):
            #     print(f"\nVehicle {p + 1} Solution:")
            #     for i in range(self.N):
            #         # Extract and print state vector
            #         state = [vehicle['s'][i, j].X for j in range(6)]
            #         print(f"Time Step {i}: State = {state}")
            #
            #         # Extract and print control input if it's not the last step
            #         if i < self.N - 1:
            #             control = [vehicle['u'][i, j].X for j in range(3)]
            #             print(f"Time Step {i}: Control = {control}")

    def update_vehicle_state(self):
        self.full_traj = [[], [], [], [], [], []]
        if (self.m.Status == 2):
            for i, vehicle in enumerate(self.vehicles):
                for n in range(1, self.N):
                    # Extract positions and velocities from the model's solution
                    x_pos = vehicle['s'][n, 0].X
                    y_pos = vehicle['s'][n, 1].X
                    z_pos = vehicle['s'][n, 2].X
                    x_vel = vehicle['s'][n, 3].X
                    y_vel = vehicle['s'][n, 4].X
                    z_vel = vehicle['s'][n, 5].X

                    # self.full_traj.append([x_pos, y_pos, z_pos])
                    self.full_traj[0].append(x_pos)
                    self.full_traj[1].append(y_pos)
                    self.full_traj[2].append(z_pos)
                    self.full_traj[3].append(x_vel)
                    self.full_traj[4].append(y_vel)
                    self.full_traj[5].append(z_vel)
        else:
            print("Model is infeasible")
            self.not_collided = False
            self.full_traj = -1

    def get_drone_status(self):
        return self.not_collided

    def get_line_coeff(self, p1, p2):
        normal = [-(p1[1] - p2[1]), p1[0] - p2[0]]
        if (p2[0] - p1[0] != 0):
            a = 1
            b = - (p2[1] - p1[1]) / (p2[0] - p1[0])
            c = - (p1[1] + b * p1[0])
        else:
            a = 0
            b = 1
            c = -p1[0]
        # Positive sign => larger
        sign = copysign(1.0, a * (p1[1] + normal[1]) + b * (p1[0] + normal[0]) + c)
        return a, b, c, sign

    # def update_plot(self):
    #     # Check if there is data to plot
    #     if not self.vehicles_positions:
    #         return  # No data to plot
    #
    #     # Ensure the number of position sets matches the number of lines
    #     latest_positions = self.vehicles_positions[-1]  # Get the latest positions
    #     if len(latest_positions) != len(self.lines):
    #         raise ValueError("Mismatch between the number of vehicles and plot lines.")
    #
    #
    #     for idx, vehicle_position in enumerate(latest_positions):
    #         x, y, z = vehicle_position
    #         self.lines[idx].set_data([x], [y])
    #         self.lines[idx].set_3d_properties([z])
    #
    #     for idx, vehicle in enumerate(self.vehicles_horizon):
    #         # print(len(vehicle))
    #         x, y, z = vehicle
    #         self.lines[idx].set_data(x, y)
    #         self.lines[idx].set_3d_properties(z)
    #
    #
    #     plt.draw()
    #     plt.pause(0.05)  # Pause to update the plot