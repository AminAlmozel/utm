import gurobipy as gp
import numpy as np
from gurobipy import GRB
import time
import warnings
from math import copysign
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore')


class DroneVehicle:
    __slots__ = ['s', 'u', 'w', 'v', 't', 'tc', 's_c']

    def __init__(self, s, u, w, v, t, tc, s_c):
        self.s = s
        self.u = u
        self.w = w
        self.v = v
        self.t = t
        self.tc = tc
        self.s_c = s_c


class drone():
    def __init__(self):
        self.N = 50
        self.delta_t = 0.1 # Time step
        self.N_polygon = 24 # Number of sides for the polygon approximation
        self.not_collided = True

        # Parameters
        self.K = 1
        self.min_dist = 1

        # Parameters for polygon approximation
        self.V_min = 0  # Minimum velocity
        self.V_max = 10 # Maximum velocity (for bounding purposes) / was 10 m/s
        self.A_max = 2  # Maximum acceleration (already defined) / was 3 m/s^2
        self.theta = [2 * np.pi * k / self.N_polygon for k in range(1, self.N_polygon+1)]  # Polygon vertex angles

        # Bounds for states and controls
        c = 50000000
        self.smin = [- c, - c, -c, - self.V_max, - self.V_max, - self.V_max]  # Position in m, Velocity in m/s
        self.smax = [c, c, c, self.V_max,  self.V_max, self.V_max] # Position in m, Velocity in m/s
        self.umin = [-self.A_max, -self.A_max, -self.A_max]  # Acceleration in m/s^2
        self.umax = [self.A_max, self.A_max, self.A_max]  # Acceleration in m/s^2

        # Define weights for the objective function
        self.qo = [1] * 6  # State deviation weights
        self.ro = [1] * 3  # Control effort weights
        self.po = [100] * 6  # Terminal state weights

        # Parameters for collision avoidance between vehicles
        d_min = 5
        self.collision_penality = 1000
        self.d_x = d_min  # Minimum horizontal distance
        self.d_y = d_min  # Minimum vertical distance
        self.d_z = d_min  # Minimum vertical distance

        self.M = 10000000  # Large positive number for 'big-M' method

        self.vehicles_positions = []
        self.vehicles = []
        self.full_traj = []

        self.obstacles = []

        # # self.m = gp.Model(env=env)

    def generate_traj(self, env, xi, xf, xi_1, obstacles):
        # Clear the model from previously setup constraints

        env.setParam(GRB.Param.OutputFlag, 0)
        env.setParam(GRB.Param.LogToConsole, 0)
        env.setParam(GRB.Param.MIPFocus, 1)  # Focus on finding good feasible solutions
        env.setParam(GRB.Param.Heuristics, 0.2)  # Increase heuristic time
        env.setParam(GRB.Param.Cuts, 2)  # Aggressive cuts
        env.setParam(GRB.Param.Presolve, 2)  # Aggressive presolve
        env.setParam(GRB.Param.MIPGap, 0.01)  # Allow 1% optimality gap
        env.setParam(GRB.Param.TimeLimit, 30)  # Set time limit
        # self.m = gp.Model(env=self.env)
        with gp.Model(env=env) as self.m:

            # import obstacles and position of other drones
            self.set_initial_condition(xi)
            self.set_final_condition(xf)
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

        # Clearing and freeing things up
        self.m.close()
        # self.env.close()
        env.close()

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

    def setup_variables(self):
        self.vehicles = []
        # Create variables for the vehicle
        p = 0
        vehicle = DroneVehicle(
            s=self.m.addVars(self.N, 6, lb=[self.smin] * self.N, ub=[self.smax] * self.N, name=f"s_{p}"),
            u=self.m.addVars(self.N, 3, lb=[self.umin] * self.N, ub=[self.umax] * self.N, name=f"u_{p}"),
            w=self.m.addVars(self.N, 6, name=f"w_{p}"),
            v=self.m.addVars(self.N, 3, name=f"v_{p}"),
            t=[self.m.addVars(self.N, self.max_edges + 2, vtype=GRB.BINARY, name=f"t_{p}_obstacle_{c}") for c in range(len(self.obstacles))],
            tc=self.m.addVars(self.N, self.N_polygon, vtype=GRB.BINARY, name=f"tc_{p}_accel"),
            s_c=self.m.addVars(self.N, lb=[0] * self.N, ub=[self.d_x] * self.N)
        )
        self.vehicles.append(vehicle)

        # Create binary variables for collision avoidance between each pair of vehicles
        self.b_vars = {}
        for p in range(self.K): # 0, 1
            self.b_vars[p] = self.m.addVars(self.N, 6, vtype=GRB.BINARY, name=f"b_{p}")
        return 0

    def setup_constraints(self):
        # self.state_constraints()
        self.state_constraints_compact()
        # self.control_constraints()
        self.control_constraints_compact()
        self.state_transition_constraints()
        self.initial_final_condition_constraints()
        self.general_obstacle_avoidance_constraints()
        self.fixed_vehicle_collision_avoidance_constraints()

    def state_constraints(self):
        # Add constraints and objective components for each vehicle
        for p, vehicle in enumerate(self.vehicles):
            s, w = vehicle.s, vehicle.w

            # State and velocity constraints with slack variables
            for i in range(1, self.N, 2):  # i = 1 to N
                for j in range(6):  # Position and velocity deviation constraints
                    self.m.addLConstr(
                        s[i, j] - self.final_conditions[p][['x', 'y', 'z', 'xdot', 'ydot', 'zdot'][j]] <= w[i, j])
                    self.m.addLConstr(
                        -s[i, j] + self.final_conditions[p][['x', 'y', 'z', 'xdot', 'ydot', 'zdot'][j]] <= w[i, j])

            # Do we need this?
            for i in range(1, self.N):  # State bounds 1 to N
                for j in range(3):
                    self.m.addLConstr(s[i, j] >= self.smin[j])
                    self.m.addLConstr(s[i, j] <= self.smax[j])

            # Velocity magnitude limitation using polygon approximation
            for i in range(1, self.N):
                for k, angle in enumerate(self.theta):

                    self.m.addLConstr(np.sin(angle) * s[i, 3] + np.cos(angle) * s[i, 4] <= self.V_max)
                    # self.m.addLConstr(np.sin(angle) * s[i, 3] + np.cos(angle) * s[i, 4] >= self.V_min- self.M * tc_vars[i, k],
                    #                 f"Min_Vel_{k}_{p}_{i}")
                # Binary Variable of V_min
                # self.m.addLConstr(sum(tc_vars[i, k] for k in range(self.N_polygon)) <= self.N_polygon - 1,
                #                 f"Min_Velocity_Constraint_{p}_{i}")

            # Do we need this?
            # Bounded Velocity along Z-Axis (Decoupled)
            for i in range(1, self.N):
                # self.m.addLConstr(smin[5] <= s[i, 5] <= smax[5], f"Min_MAX_z_Vel_{p}_{i}")

                self.m.addLConstr(s[i, 5] >= self.smin[5])
                self.m.addLConstr(s[i, 5] <= self.smax[5])

    def control_constraints(self):
        # Add constraints and objective components for each vehicle
        for p, vehicle in enumerate(self.vehicles):
            u, v_vars = vehicle.u, vehicle.v

            # Control constraints with slack variables
            for i in range(self.N):  # i = 0 to N - 1
                for j in range(3):  # Control effort constraints
                    self.m.addLConstr(u[i, j] <= v_vars[i, j])
                    self.m.addLConstr(-u[i, j] <= v_vars[i, j])

            # Bounded Acceleration along Z-Axis (Decoupled)
            for i in range(0, self.N-1):
                # self.m.addLConstr(umin[2] <= u[i, 2] <= umax[2], f"Min_MAX_z_Acc_{p}_{i}")
                self.m.addLConstr(u[i, 2] >= self.umin[2])
                self.m.addLConstr(u[i, 2] <= self.umax[2])

            # Acceleration magnitude limitation using polygon approximation, no A_min
            for i in range(0, self.N - 1):
                for k, angle in enumerate(self.theta):
                    self.m.addLConstr(np.sin(angle) * u[i, 0] + np.cos(angle) * u[i, 1] <= self.A_max)

    def state_constraints_compact(self):
        """
        Compact version of state_constraints using the helper function
        """
        for p, vehicle in enumerate(self.vehicles):

            # 1. Slack variable constraints
            A, b, x_vars, sense = self.build_constraint_matrix("slack", p)
            if A is not None:
                self.m.addMConstr(A, x_vars, sense, b, name=f"slack_constraints_vehicle_{p}")

            # 2. Position bounds - lower bounds
            A, _, x_vars, _ = self.build_constraint_matrix("position_bounds", p)
            if A is not None:
                b_lower = np.tile(self.smin[:3], len(range(1, self.N)))
                self.m.addMConstr(A, x_vars, GRB.GREATER_EQUAL, b_lower,
                                name=f"position_lower_bounds_vehicle_{p}")

                # Position bounds - upper bounds
                b_upper = np.tile(self.smax[:3], len(range(1, self.N)))
                self.m.addMConstr(A, x_vars, GRB.LESS_EQUAL, b_upper,
                                name=f"position_upper_bounds_vehicle_{p}")

            # 3. Velocity magnitude constraints
            A, b, x_vars, sense = self.build_constraint_matrix("velocity_magnitude", p)
            if A is not None:
                self.m.addMConstr(A, x_vars, sense, b, name=f"velocity_magnitude_vehicle_{p}")

            # 4. Z-velocity bounds
            A, _, x_vars, _ = self.build_constraint_matrix("z_velocity_bounds", p)
            if A is not None:
                # Lower bounds
                b_lower = np.full(len(x_vars), self.smin[5])
                self.m.addMConstr(A, x_vars, GRB.GREATER_EQUAL, b_lower,
                                name=f"z_velocity_lower_vehicle_{p}")

                # Upper bounds
                b_upper = np.full(len(x_vars), self.smax[5])
                self.m.addMConstr(A, x_vars, GRB.LESS_EQUAL, b_upper,
                                name=f"z_velocity_upper_vehicle_{p}")

    def control_constraints_compact(self):
        """
        Compact version of control_constraints using the helper function
        """
        for p, vehicle in enumerate(self.vehicles):

            # 1. Control constraints with slack variables
            A, b, x_vars, sense = self.build_control_constraint_matrix("control_slack", p)
            if A is not None:
                self.m.addMConstr(A, x_vars, sense, b, name=f"control_slack_vehicle_{p}")

            # 2. Z-acceleration bounds - lower bounds
            A, _, x_vars, _ = self.build_control_constraint_matrix("z_acceleration_bounds", p)
            if A is not None:
                b_lower = np.full(len(x_vars), self.umin[2])
                self.m.addMConstr(A, x_vars, GRB.GREATER_EQUAL, b_lower,
                                name=f"z_accel_lower_bounds_vehicle_{p}")

                # Z-acceleration bounds - upper bounds
                b_upper = np.full(len(x_vars), self.umax[2])
                self.m.addMConstr(A, x_vars, GRB.LESS_EQUAL, b_upper,
                                name=f"z_accel_upper_bounds_vehicle_{p}")

            # 3. Acceleration magnitude constraints
            A, b, x_vars, sense = self.build_control_constraint_matrix("acceleration_magnitude", p)
            if A is not None:
                self.m.addMConstr(A, x_vars, sense, b, name=f"acceleration_magnitude_vehicle_{p}")

    def state_transition_constraints(self):
        # Add constraints and objective components for each vehicle
        for p, vehicle in enumerate(self.vehicles):
            s, u = vehicle.s, vehicle.u

            # Dynamics constraints
            for i in range(0, self.N-1):# 0 to N-1
                self.m.addLConstr(s[i + 1, 0] == s[i, 0] + self.delta_t * s[i, 3])
                self.m.addLConstr(s[i + 1, 1] == s[i, 1] + self.delta_t * s[i, 4])
                self.m.addLConstr(s[i + 1, 2] == s[i, 2] + self.delta_t * s[i, 5])
                self.m.addLConstr(s[i + 1, 3] == s[i, 3] + self.delta_t * u[i, 0])
                self.m.addLConstr(s[i + 1, 4] == s[i, 4] + self.delta_t * u[i, 1])
                self.m.addLConstr(s[i + 1, 5] == s[i, 5] + self.delta_t * u[i, 2])

    def initial_final_condition_constraints(self):
        # Add constraints and objective components for each vehicle
        for p, vehicle in enumerate(self.vehicles):
            s = vehicle.s
            # Initial and final conditions
            for j, key in enumerate(['x', 'y', 'z', 'xdot', 'ydot', 'zdot']):
                self.m.addLConstr(s[0, j] == self.initial_conditions[p][key])
            # self.m.addConstrs((s[0, j] == self.initial_conditions[p][key] for j, key in enumerate(['x', 'y', 'z', 'xdot', 'ydot', 'zdot'])), f"Initial_{p}")
            # self.m.addLConstrs((s[N - 1, j] == self.final_conditions[p][key] for j, key in enumerate(['x', 'y', 'z', 'xdot', 'ydot', 'zdot'])), f"Final_{p}")

    def obstacle_avoidance_constraints(self):
        # Add constraints and objective components for each vehicle
        for p, vehicle in enumerate(self.vehicles):
            s, u, w, v_vars, t_vars, tc_vars = vehicle.s, vehicle.u, vehicle.w, vehicle.v, vehicle.t, vehicle.tc
            # Obstacle avoidance constraints for each vehicle 'p'
            for c, obs in enumerate(self.obstacles):
                t = t_vars[c]
                for i in range(1, self.N):  # i = 1 to N
                    self.m.addLConstr(s[i, 0] <= obs['xmin'] - self.min_dist + self.M * t[i, 0], f"Obstacle_x_{c}_{p}_{i}")
                    self.m.addLConstr(-s[i, 0] <= -obs['xmax'] - self.min_dist + self.M * t[i, 1], f"Obstacle_Neg_x_{c}_{p}_{i}")
                    self.m.addLConstr(s[i, 1] <= obs['ymin'] - self.min_dist + self.M * t[i, 2], f"Obstacle_y_{c}_{p}_{i}")
                    self.m.addLConstr(-s[i, 1] <= -obs['ymax'] - self.min_dist + self.M * t[i, 3], f"Obstacle_Neg_y_{c}_{p}_{i}")
                    self.m.addLConstr(s[i, 2] <= obs['zmin'] - self.min_dist + self.M * t[i, 4], f"Obstacle_z_{c}_{p}_{i}")
                    self.m.addLConstr(-s[i, 2] <= -obs['zmax'] - self.min_dist + self.M * t[i, 5], f"Obstacle_Neg_z_{c}_{p}_{i}")
                    self.m.addLConstr(t[i, 0] + t[i, 1] + t[i, 2] + t[i, 3] + t[i, 4] + t[i, 5] <= 5, f"Obstacle_Sum_{c}_{p}_{i}")

    def general_obstacle_avoidance_constraints(self):
         # Add constraints and objective components for each vehicle
        for p, vehicle in enumerate(self.vehicles):
            s, u, w, v_vars, t_vars, tc_vars = vehicle.s, vehicle.u, vehicle.w, vehicle.v, vehicle.t, vehicle.tc
            # Obstacle avoidance constraints for each vehicle 'p'
            for c, obs in enumerate(self.obstacles):
                t = t_vars[c]
                vertices = obs["geom"]
                # height = [0, obs["height"]]
                height = obs["height"]
                for n in range (1, self.N):
                    binary_sum = 0
                    # print(vertices)
                    for i in range(len(vertices) - 1):
                        p1 = [vertices[i][0], vertices[i][1]]
                        p2 = [vertices[i + 1][0], vertices[i + 1][1]]
                        a, b, c, sign = self.get_line_coeff(p1, p2)
                        self.m.addLConstr(sign * (a * s[n, 1] + b * s[n, 0] + c) <= self.M * t[n, i])
                        binary_sum += t[n, i]

                    total_sum = len(vertices) - 1 # -1 Because shapley rings double count a vertex
                    if height[1] != -1:  # If there is a height limit
                        i = self.max_edges + 2
                        self.m.addLConstr(-s[n, 2] <= -height[0] - self.min_dist + self.M * t[n, i - 2])
                        self.m.addLConstr(s[n, 2] <= height[1] - self.min_dist + self.M * t[n, i - 1])
                        binary_sum += t[n, i - 2] + t[n, i - 1]
                        total_sum += 2 # for the top and bottom constraints (height)

                    # Total sum has to be less than the number of half-plane constraints by one
                    self.m.addLConstr(binary_sum  <= total_sum - 1)

    def vehicle_collision_avoidance_constraints(self):
        # Collision avoidance constraints between vehicles
        for p in range(self.K):
            for q in range(p + 1, self.K):
                for i in range(1, self.N):  # i = 1 to N
                    self.m.addLConstr(self.vehicles[p].s[i, 0] - self.vehicles[q].s[i, 0] >= self.d_x - self.M * self.b_vars[(p, q)][i, 0])
                    self.m.addLConstr(self.vehicles[q].s[i, 0] - self.vehicles[p].s[i, 0] >= self.d_x - self.M * self.b_vars[(p, q)][i, 1])
                    self.m.addLConstr(self.vehicles[p].s[i, 1] - self.vehicles[q].s[i, 1] >= self.d_y - self.M * self.b_vars[(p, q)][i, 2])
                    self.m.addLConstr(self.vehicles[q].s[i, 1] - self.vehicles[p].s[i, 1] >= self.d_y - self.M * self.b_vars[(p, q)][i, 3])
                    self.m.addLConstr(self.vehicles[p].s[i, 2] - self.vehicles[q].s[i, 2] >= self.d_z - self.M * self.b_vars[(p, q)][i, 4])
                    self.m.addLConstr(self.vehicles[q].s[i, 2] - self.vehicles[p].s[i, 2] >= self.d_z - self.M * self.b_vars[(p, q)][i, 5])
                    self.m.addLConstr(self.b_vars[(p, q)][i, 0] + self.b_vars[(p, q)][i, 1] + self.b_vars[(p, q)][i, 2] + self.b_vars[(p, q)][i, 3] + self.b_vars[(p, q)][i, 4]+ self.b_vars[(p, q)][i, 5] <= 5, f"Collision_Sum_{p}_{q}_{i}")

    def fixed_vehicle_collision_avoidance_constraints(self):
        # Collision avoidance constraints between vehicles
        for p, vehicle in enumerate(self.k_traj):
            q = 0
            N = min(len(vehicle[0]), self.N - 1) # The smaller of the prediction horizon and the given trajectories
            for n in range(1, N + 1):  # i = 1 to N
                # self.m.addLConstr(vehicle[0][n - 1] - self.vehicles[q].s[n, 0] >= self.d_x - self.M * self.b_vars[p][n, 0])
                # self.m.addLConstr(self.vehicles[q].s[n, 0] - vehicle[0][n - 1] >= self.d_x - self.M * self.b_vars[p][n, 1])
                # self.m.addLConstr(vehicle[1][n - 1] - self.vehicles[q].s[n, 1] >= self.d_y - self.M * self.b_vars[p][n, 2])
                # self.m.addLConstr(self.vehicles[q].s[n, 1] - vehicle[1][n - 1] >= self.d_y - self.M * self.b_vars[p][n, 3])
                # self.m.addLConstr(vehicle[2][n - 1] - self.vehicles[q].s[n, 2] >= self.d_z - self.M * self.b_vars[p][n, 4])
                # self.m.addLConstr(self.vehicles[q].s[n, 2] - vehicle[2][n - 1] >= self.d_z - self.M * self.b_vars[p][n, 5])

                self.m.addLConstr(vehicle[0][n - 1] - self.vehicles[q].s[n, 0] + self.vehicles[q].s_c[n] >= self.d_x - self.M * self.b_vars[p][n, 0])
                self.m.addLConstr(self.vehicles[q].s[n, 0] - vehicle[0][n - 1] + self.vehicles[q].s_c[n] >= self.d_x - self.M * self.b_vars[p][n, 1])
                self.m.addLConstr(vehicle[1][n - 1] - self.vehicles[q].s[n, 1] + self.vehicles[q].s_c[n] >= self.d_y - self.M * self.b_vars[p][n, 2])
                self.m.addLConstr(self.vehicles[q].s[n, 1] - vehicle[1][n - 1] + self.vehicles[q].s_c[n] >= self.d_y - self.M * self.b_vars[p][n, 3])
                self.m.addLConstr(vehicle[2][n - 1] - self.vehicles[q].s[n, 2] + self.vehicles[q].s_c[n] >= self.d_z - self.M * self.b_vars[p][n, 4])
                self.m.addLConstr(self.vehicles[q].s[n, 2] - vehicle[2][n - 1] + self.vehicles[q].s_c[n] >= self.d_z - self.M * self.b_vars[p][n, 5])

                self.m.addLConstr(self.b_vars[p][n, 0] + self.b_vars[p][n, 1] + self.b_vars[p][n, 2] + self.b_vars[p][n, 3] + self.b_vars[p][n, 4]+ self.b_vars[p][n, 5] <= 5)

    def setup_objective(self):
        # Initialize the objective function
        self.obj = gp.LinExpr()

        # Add constraints and objective components for each vehicle
        for p, vehicle in enumerate(self.vehicles):
            s, u, w, v_vars, s_c = vehicle.s, vehicle.u, vehicle.w, vehicle.v, vehicle.s_c

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

            for i in range(self.N):
                self.obj.add(self.collision_penality * s_c[i])
        # print("done setup objective")

    def optimize(self):
        # Set the objective function in the model
        self.m.setObjective(self.obj, GRB.MINIMIZE)
        self.m.setParam(GRB.Param.OutputFlag, 0)
        # self.m.setParam('Threads', 0)
        self.m.setParam('TimeLimit', 5)

        self.m.update()
        # self.provide_warm_start()
        # Optimize the model
        # print("Starting optimization...")
        self.m.optimize()

        total_computational_time = self.m.Runtime
        # print("Total Computational Time(s):", total_computational_time, "seconds")

        # if self.m.status == GRB.OPTIMAL:
        #     # print("find the optimal")

    def update_vehicle_state(self):
        self.full_traj = [[], [], [], [], [], []]
        if (self.m.Status == 2):
            for i, vehicle in enumerate(self.vehicles):
                for n in range(1, self.N):
                    # Extract positions and velocities from the model's solution
                    x_pos = vehicle.s[n, 0].X
                    y_pos = vehicle.s[n, 1].X
                    z_pos = vehicle.s[n, 2].X
                    x_vel = vehicle.s[n, 3].X
                    y_vel = vehicle.s[n, 4].X
                    z_vel = vehicle.s[n, 5].X

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

    def provide_warm_start(self):
        """Provide a warm start with straight-line trajectory"""
        try:
            for p, vehicle in enumerate(self.vehicles):
                s = vehicle.s

                # Linear interpolation between initial and final positions
                x_init = np.array([self.initial_conditions[p]['x'],
                                 self.initial_conditions[p]['y'],
                                 self.initial_conditions[p]['z']])
                x_final = np.array([self.final_conditions[p]['x'],
                                  self.final_conditions[p]['y'],
                                  self.final_conditions[p]['z']])

                for i in range(self.N):
                    alpha = i / (self.N - 1)
                    pos = x_init + alpha * (x_final - x_init)

                    s[i, 0].Start = pos[0]
                    s[i, 1].Start = pos[1]
                    s[i, 2].Start = pos[2]

                    # Simple velocity estimate
                    if i < self.N - 1:
                        vel = (x_final - x_init) / (self.N * self.delta_t)
                        s[i, 3].Start = vel[0]
                        s[i, 4].Start = vel[1]
                        s[i, 5].Start = vel[2]
        except:
            pass  # If warm start fails, continue without it

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

    def build_constraint_matrix(self, constraint_type, vehicle_idx):
        """
        Helper function to build constraint matrices more systematically
        """
        vehicle = self.vehicles[vehicle_idx]
        s, w = vehicle.s, vehicle.w

        if constraint_type == "slack":
            # Build slack variable constraints
            time_steps = list(range(1, self.N, 2))
            if not time_steps:
                return None, None, None, None

            num_constraints = len(time_steps) * 6 * 2
            num_vars = len(time_steps) * 6 * 2  # s and w variables

            A = np.zeros((num_constraints, num_vars))
            b = np.zeros(num_constraints)

            final_conditions_keys = ['x', 'y', 'z', 'xdot', 'ydot', 'zdot']

            for idx, i in enumerate(time_steps):
                for j in range(6):
                    # Upper bound constraint: s[i,j] - w[i,j] <= final_conditions
                    constraint_idx = idx * 12 + j * 2
                    s_var_idx = idx * 6 + j
                    w_var_idx = len(time_steps) * 6 + idx * 6 + j

                    A[constraint_idx, s_var_idx] = 1
                    A[constraint_idx, w_var_idx] = -1
                    b[constraint_idx] = self.final_conditions[vehicle_idx][final_conditions_keys[j]]

                    # Lower bound constraint: -s[i,j] - w[i,j] <= -final_conditions
                    constraint_idx = idx * 12 + j * 2 + 1
                    A[constraint_idx, s_var_idx] = -1
                    A[constraint_idx, w_var_idx] = -1
                    b[constraint_idx] = -self.final_conditions[vehicle_idx][final_conditions_keys[j]]

            # Build variable list
            x_vars = []
            for i in time_steps:
                for j in range(6):
                    x_vars.append(s[i, j])
            for i in time_steps:
                for j in range(6):
                    x_vars.append(w[i, j])

            return A, b, x_vars, GRB.LESS_EQUAL

        elif constraint_type == "position_bounds":
            # Build position bounds constraints
            time_steps = list(range(1, self.N))
            if not time_steps:
                return None, None, None, None

            num_constraints = len(time_steps) * 3  # Only position dimensions
            num_vars = len(time_steps) * 3

            A = np.zeros((num_constraints, num_vars))
            b = np.zeros(num_constraints)

            for idx, i in enumerate(time_steps):
                for j in range(3):  # Only position (x, y, z)
                    constraint_idx = idx * 3 + j
                    var_idx = idx * 3 + j
                    A[constraint_idx, var_idx] = 1
                    # We'll return both bounds separately, so b will be set in the caller

            # Build variable list
            x_vars = []
            for i in time_steps:
                for j in range(3):
                    x_vars.append(s[i, j])

            return A, b, x_vars, None  # Return sense as None, will be set by caller

        elif constraint_type == "velocity_magnitude":
            # Build velocity magnitude constraints
            if not hasattr(self, 'theta') or len(self.theta) == 0:
                return None, None, None, None

            time_steps = list(range(1, self.N))
            if not time_steps:
                return None, None, None, None

            num_constraints = len(time_steps) * len(self.theta)
            num_vars = len(time_steps) * 2  # Only s[i,3] and s[i,4]

            A = np.zeros((num_constraints, num_vars))
            b = np.full(num_constraints, self.V_max)

            for idx, i in enumerate(time_steps):
                for k, angle in enumerate(self.theta):
                    constraint_idx = idx * len(self.theta) + k

                    # s[i,3] coefficient (x velocity)
                    s3_var_idx = idx * 2 + 0
                    A[constraint_idx, s3_var_idx] = np.sin(angle)

                    # s[i,4] coefficient (y velocity)
                    s4_var_idx = idx * 2 + 1
                    A[constraint_idx, s4_var_idx] = np.cos(angle)

            # Build variable list
            x_vars = []
            for i in time_steps:
                x_vars.append(s[i, 3])  # x velocity
                x_vars.append(s[i, 4])  # y velocity

            return A, b, x_vars, GRB.LESS_EQUAL

        elif constraint_type == "z_velocity_bounds":
            # Build z-velocity bounds
            time_steps = list(range(1, self.N))
            if not time_steps:
                return None, None, None, None

            num_constraints = len(time_steps)
            num_vars = len(time_steps)

            A = np.eye(num_constraints)  # Identity matrix
            b = np.zeros(num_constraints)  # Will be set by caller

            # Build variable list
            x_vars = []
            for i in time_steps:
                x_vars.append(s[i, 5])  # z velocity

            return A, b, x_vars, None  # Return sense as None, will be set by caller

        return None, None, None, None

    def build_control_constraint_matrix(self, constraint_type, vehicle_idx):
        """
        Helper function to build control constraint matrices systematically
        """
        vehicle = self.vehicles[vehicle_idx]
        u, v_vars = vehicle.u, vehicle.v

        if constraint_type == "control_slack":
            # Build control constraints with slack variables
            # u[i,j] <= v_vars[i,j] and -u[i,j] <= v_vars[i,j] for all i,j

            time_steps = list(range(self.N))  # 0 to N-1
            if not time_steps:
                return None, None, None, None

            num_constraints = len(time_steps) * 3 * 2  # 3 control dims, 2 constraints each
            num_vars = len(time_steps) * 3 * 2  # u and v_vars variables

            A = np.zeros((num_constraints, num_vars))
            b = np.zeros(num_constraints)

            for idx, i in enumerate(time_steps):
                for j in range(3):  # Control dimensions
                    # Upper bound constraint: u[i,j] - v_vars[i,j] <= 0
                    constraint_idx = idx * 6 + j * 2
                    u_var_idx = idx * 3 + j
                    v_var_idx = len(time_steps) * 3 + idx * 3 + j

                    A[constraint_idx, u_var_idx] = 1      # u[i,j]
                    A[constraint_idx, v_var_idx] = -1     # v_vars[i,j]
                    b[constraint_idx] = 0

                    # Lower bound constraint: -u[i,j] - v_vars[i,j] <= 0
                    constraint_idx = idx * 6 + j * 2 + 1
                    A[constraint_idx, u_var_idx] = -1     # u[i,j]
                    A[constraint_idx, v_var_idx] = -1     # v_vars[i,j]
                    b[constraint_idx] = 0

            # Build variable list: [u[0,:], u[1,:], ..., u[N-1,:], v_vars[0,:], v_vars[1,:], ...]
            x_vars = []
            for i in time_steps:
                for j in range(3):
                    x_vars.append(u[i, j])
            for i in time_steps:
                for j in range(3):
                    x_vars.append(v_vars[i, j])

            return A, b, x_vars, GRB.LESS_EQUAL

        elif constraint_type == "z_acceleration_bounds":
            # Build z-acceleration bounds: umin[2] <= u[i,2] <= umax[2]
            time_steps = list(range(self.N - 1))  # 0 to N-2
            if not time_steps:
                return None, None, None, None

            num_constraints = len(time_steps)
            num_vars = len(time_steps)

            A = np.eye(num_constraints)  # Identity matrix
            b = np.zeros(num_constraints)  # Will be set by caller

            # Build variable list: [u[0,2], u[1,2], ..., u[N-2,2]]
            x_vars = []
            for i in time_steps:
                x_vars.append(u[i, 2])  # z acceleration

            return A, b, x_vars, None  # Return sense as None, will be set by caller

        elif constraint_type == "acceleration_magnitude":
            # Build acceleration magnitude constraints using polygon approximation
            # sin(angle) * u[i,0] + cos(angle) * u[i,1] <= A_max

            if not hasattr(self, 'theta') or len(self.theta) == 0:
                return None, None, None, None

            time_steps = list(range(self.N - 1))  # 0 to N-2
            if not time_steps:
                return None, None, None, None

            num_constraints = len(time_steps) * len(self.theta)
            num_vars = len(time_steps) * 2  # Only u[i,0] and u[i,1]

            A = np.zeros((num_constraints, num_vars))
            b = np.full(num_constraints, self.A_max)

            for idx, i in enumerate(time_steps):
                for k, angle in enumerate(self.theta):
                    constraint_idx = idx * len(self.theta) + k

                    # u[i,0] coefficient (x acceleration)
                    u0_var_idx = idx * 2 + 0
                    A[constraint_idx, u0_var_idx] = np.sin(angle)

                    # u[i,1] coefficient (y acceleration)
                    u1_var_idx = idx * 2 + 1
                    A[constraint_idx, u1_var_idx] = np.cos(angle)

            # Build variable list: [u[0,0], u[0,1], u[1,0], u[1,1], ...]
            x_vars = []
            for i in time_steps:
                x_vars.append(u[i, 0])  # x acceleration
                x_vars.append(u[i, 1])  # y acceleration

            return A, b, x_vars, GRB.LESS_EQUAL

        return None, None, None, None
