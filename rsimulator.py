import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
# from mayavi import mlab
from datetime import datetime
import matplotlib.collections
import time
import drone

from multiprocessing import Pool


class simulator(drone.drone):
    def __init__(self):
        self.N = 20
        self.delta_t = 0.1 # Time step
        self.N_polygon = 8 # Number of sides for the polygon approximation
        self.total_iterations = 20

        # Parameters
        self.K = 10  # Number of vehicles
        self.L = 4  # Number of stationary obstacles
        self.min_dist = 1

        # Parameters for collision avoidance between vehicles
        self.d_x = 3  # Minimum horizontal distance
        self.d_y = 3  # Minimum vertical distance
        self.d_z = 3  # Minimum vertical distance

        self.collision = 3
        self.collision_warning = 5

        self.vehicles_positions = []
        self.drn = [] # All drones
        self.drn_list = [] # Indices of alive drones

        self.hlines = []

        # self.obstacles = [
        #     {'xmin': -1000, 'ymin': 600, 'zmin': -1000, 'xmax': -600, 'ymax': 1000, 'zmax': 1000},
        #     {'xmin': 7, 'ymin': 7, 'zmin': 2, 'xmax': 9, 'ymax': 8, 'zmax': 3},
        #     {'xmin': 6, 'ymin': 1, 'zmin': 0, 'xmax': 8, 'ymax': 2, 'zmax': 1},
        #     {'xmin': 9, 'ymin': 3, 'zmin': 0, 'xmax': 10, 'ymax': 5, 'zmax': np.random.uniform(5, 10)},
        #     {'xmin': 11, 'ymin': 6, 'zmin': 0, 'xmax': 13, 'ymax': 7, 'zmax': np.random.uniform(5, 10)},
        #     {'xmin': 14, 'ymin': 8, 'zmin': 0, 'xmax': 15, 'ymax': 10, 'zmax': np.random.uniform(5, 10)},
        #     {'xmin': 21, 'ymin': 15, 'zmin': 0, 'xmax': 23, 'ymax': 16, 'zmax': np.random.uniform(5, 10)}
        #     # {'xmin': 25, 'ymin': 17, 'zmin': 0, 'xmax': 26, 'ymax': 18, 'zmax': np.random.uniform(5, 10)}
        #
        # ]

        # Obstacle parameters
        self.obstacles = [
            {'xmin': -100, 'ymin': 20, 'zmin': -100, 'xmax': -20, 'ymax': 100, 'zmax': 100},
            {'xmin': 20, 'ymin': 20, 'zmin': -100, 'xmax': 100, 'ymax': 100, 'zmax': 100},
            {'xmin': -100, 'ymin': -100, 'zmin': -100, 'xmax': -20, 'ymax': -20, 'zmax': 100},
            {'xmin': 20, 'ymin': -100, 'zmin': -100, 'xmax': 100, 'ymax': -20, 'zmax': 100}
        ]

        self.initialize()

    def initialize(self):

        self.vehicles_positions = []
        self.vehicles = []
        self.full_traj = []
        self.initialize_drones()

    def initialize_drones(self):
        self.set_initial_state()
        self.set_final_state()
        self.initial_conditions = []
        for k in range(self.K):
            self.create_drone(self.xi[k], self.final_conditions[k])

        self.max_vel = self.drn[0].smax[3] # Max velocity

    def create_drone(self, xi, xf):
        self.drn.append(drone.drone())
        self.initial_conditions.append(xi)
        self.drn[-1].set_final_condition(xf)
        self.drn_list.append(len(self.drn) - 1)
        self.set_full_traj(xi)

    def start_simulation(self):
        print("Starting Simulation")
        t0 = time.time()
        # Main loop for optimization
        for iteration in range(self.total_iterations):
            print("%d============================" % (iteration))
            if iteration%10 == 0:
                self.create_drone(self.xi[14], self.final_conditions[14])
                print("Created drone")
            # Generate new trajectories for each drone
            for k in (self.drn_list):
                self.prepare_and_generate(k)
            # Check for collision
            self.check_collisions()
            print(self.drn_list)

            # Update positions
            self.update_vehicle_state()
            self.update_visualization_positions()  # Update the plot after each iteration
        t1 = time.time()
        print("Time of execution: %f" % (t1 - t0))
        # Optionally, keep the final plot open
        # Initialize the plot first
        self.plot()
        self.create_animation()

    def prepare_and_generate(self, k):
        self.proximity = 2 * self.N * self.delta_t * self.max_vel
        drone_prox_list = []
        # Finding the drones in proximity
        for i in self.drn_list:
            d = self.dist_squared(self.initial_conditions[i], self.initial_conditions[k])
            if d < self.proximity * self.proximity and i != k:
                drone_prox_list.append(i)

        # Finding the obstacles in proximity
        # Temp solution
        obstacles = self.obstacles

        # Constructing the lists to be used as input to the function
        xi = self.initial_conditions[k]
        xi_1 = [self.full_traj[i] for i in drone_prox_list]

        # Generating trajectories
        self.drn[k].generate_traj(xi, xi_1, obstacles)

    def m_start_simulation(self):
        print("Starting Simulation")
        t0 = time.time()
        # Main loop for optimization
        for iteration in range(self.total_iterations):
            print("============================")
            # Generate new trajectories for each drone
            # for k in range(self.K):
            K = self.K
            pool = Pool()
            with Pool() as pool:
                # prepare arguments
                items = [(k, k) for k in range(K)]
                pool.starmap(self.m_prepare_and_generate, items)
                    # graph[i, :] = result
            pool.close()
            # for k in range(self.K):
            #     self.full_traj[k] = self.drn[k].full_traj

            # # Check for collision
            # self.check_collisions()
            # # Update positions
            # self.update_vehicle_state()
            # self.update_visualization_positions()  # Update the plot after each iteration
        # t1 = time.time()
        # print("Time of execution: %f" % (t1 - t0))
        # # Optionally, keep the final plot open
        # self.create_animation()

    def set_initial_state(self):
        # Initial conditions for each vehicle
        self.xi = [
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
        # Iterate over drones, set the final condition for each
        # for each drone
        # drone[i].set_initial_state(xi)

    def set_final_state(self):
        # Set final state
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

        # Iterate over drones, set the final condition for each
        # for each drone
        # drone[i].set_final_state(xf)

    def set_full_traj(self, xi):
        x = [xi['x'] for n in range(self.N)]
        y = [xi['y'] for n in range(self.N)]
        z = [xi['z'] for n in range(self.N)]
        xdot = [xi['xdot'] for n in range(self.N)]
        ydot = [xi['ydot'] for n in range(self.N)]
        zdot = [xi['zdot'] for n in range(self.N)]
        self.full_traj.append([x, y, z, xdot, ydot, zdot])

    def update_vehicle_state(self):
        for k in self.drn_list:
            self.full_traj[k] = self.drn[k].full_traj
        for i, vehicle in enumerate(self.full_traj):
            # Extract positions and velocities from the model's solution
            x_position = vehicle[0][0]
            y_position = vehicle[1][0]
            z_position = vehicle[2][0]
            x_velocity = vehicle[3][0]
            y_velocity = vehicle[4][0]
            z_velocity = vehicle[5][0]

            # Update the initial conditions for the next iteration
            self.initial_conditions[i] = {
                'x': x_position, 'y': y_position, 'z': z_position, 'xdot': x_velocity, 'ydot': y_velocity, 'zdot': z_velocity
            }

    def check_collisions(self):
        # for k in (self.drn_list):
        i = 0
        while i < len(self.drn_list):
            k = self.drn_list[i]
            if self.drn[k].get_drone_status() == False: # Drone collided
                self.drn_list.remove(k)
                print("Removed drone %d" % (k))
                i -= 1
            i += 1
        # Finding the drones in proximity
        for i in range(len(self.drn)):
            for j in range(i + 1, len(self.drn)):
                d = self.dist_squared(self.initial_conditions[i], self.initial_conditions[j])
                if d < self.collision_warning * self.collision_warning:
                    if d < self.collision * self.collision:
                        self.collide(i, j, d)

    def collide(self, d1, d2, d):
        # What to do when drones collide
        print("Collision between drone %d and %d, distance: %f" % (d1, d2, np.sqrt(d)))
        # self.drn_list.remove(d1)
        # self.drn_list.remove(d2)
        # if (self.drn[d1])

    def update_visualization_positions(self):
        iteration_positions = []  # Temporary list for the current iteration
        for vehicle in self.full_traj:
            x_positions = vehicle[:][0]
            y_positions = vehicle[:][1]
            z_positions = vehicle[:][2]
            iteration_positions.append((x_positions, y_positions, z_positions))

        # Append the positions of this iteration to the main list
        self.vehicles_positions.append(iteration_positions)

    def plot(self):
        # Initial plot setup
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')  # 'self.ax' to make it accessible in other methods
        self.ax.set_xlim([-100, 100])
        self.ax.set_ylim([-100, 100])
        self.ax.set_zlim([-100, 100])
        self.ax.set_xlabel('X Axis')
        self.ax.set_ylabel('Y Axis')
        self.ax.set_zlabel('Z Axis')
        for obs in self.obstacles:
            self.draw_box(self.ax, obs)  # Pass 'self.ax' here

        # Initialize lines for each vehicle
        self.lines = [self.ax.plot([], [], [], 'o-', linewidth=1, markersize=1)[0] for _ in range(len(self.drn))]

        plt.show(block=False)

    def update(self, frame):
        total_frames = len(self.vehicles_positions)

        # Start and end elevations
        start_elev = 0
        end_elev = 70

        # Calculate the current elevation for this frame
        current_elev = start_elev + (end_elev - start_elev) * (frame / total_frames)

        # Rotate the plot by changing the azimuth angle
        angle = frame % 360  # This will continuously rotate the plot
        # self.ax.view_init(elev=current_elev, azim=1.25*angle)

        # Initialize lines for each vehicle
        # self.lines = [self.ax.plot([], [], [], 'o-', linewidth=1, markersize=1)[0] for _ in range(self.K)]
        # Clear existing lines and points
        for line in self.ax.lines[:]:
            line.remove()
        # Remove scatter objects
        for scatter in list(self.ax.collections):
            if isinstance(scatter, matplotlib.collections.PathCollection):
                scatter.remove()

        # Update the data for each line based on the current frame
        for line, vehicle in zip(self.lines, self.vehicles_positions[frame]):
            # Separate x, y, z data
            x_data, y_data, z_data = vehicle

            # Check if there is more than one point to draw a line
            if len(x_data) > 1 and len(y_data) > 1 and len(z_data) > 1:
                # Line from the second point to the last for each axis
                line.set_data(x_data[1:], y_data[1:])
                line.set_3d_properties(z_data[1:])
                self.ax.add_line(line)

            # Sphere for the first point
            self.ax.scatter(x_data[0], y_data[0], z_data[0], s=12, c='red', depthshade=True)

    def create_animation(self):
        # Check if there is data to animate
        if self.vehicles_positions:
            # Format current time
            current_time = datetime.now().strftime("%H-%M-%S")

            # Create filename with time
            filename = f"plot/animation_{current_time}.gif"
            ani = animation.FuncAnimation(self.fig, self.update, frames=len(self.vehicles_positions), repeat=True)
            ani.save(filename, writer='imagemagick', fps=10)
            plt.show()
        else:
            print("No data available for animation.")

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

    # def create_animation(self):
    #     def update(frame):
    #         # Clear previous lines
    #         # Remove existing lines from the plot
    #         for line in self.ax.lines[:]:
    #             line.remove()
    #
    #
    #         for vehicle in self.vehicles_positions[frame]:
    #             x, y, z = vehicle
    #             self.ax.plot(x, y, z, 'o-',linewidth=1, markersize=1)
    #
    #     if self.vehicles_positions:
    #         ani = animation.FuncAnimation(self.fig, update, frames=len(self.vehicles_positions), repeat=False)
    #         plt.show()
    #     else:
    #         print("No data available for animation.")

    def draw_box(self, ax, obs, color='gray', alpha=0.3):
        """Draws a 3D box (cuboid) representing an obstacle."""
        # Define the corners of the obstacle
        x_corners = [obs['xmin'], obs['xmax'], obs['xmax'], obs['xmin'], obs['xmin']]
        y_corners = [obs['ymin'], obs['ymin'], obs['ymax'], obs['ymax'], obs['ymin']]
        z_bottom = obs['zmin']
        z_top = obs['zmax']

        # Draw the bottom and top faces
        x = np.array([[obs['xmin'], obs['xmax']], [obs['xmin'], obs['xmax']]])
        y = np.array([[obs['ymin'], obs['ymin']], [obs['ymax'], obs['ymax']]])
        z = np.array([[z_bottom, z_bottom], [z_bottom, z_bottom]])
        ax.plot_surface(x, y, z, color=color, alpha=alpha)

        z = np.array([[z_top, z_top], [z_top, z_top]])
        ax.plot_surface(x, y, z, color=color, alpha=alpha)

        # Draw the side faces
        for i in range(4):
            x = np.array([x_corners[i:i+2], x_corners[i:i+2]])
            y = np.array([y_corners[i:i+2], y_corners[i:i+2]])
            z = np.array([[z_bottom, z_bottom], [z_top, z_top]])
            z = np.array([[z_bottom, z_bottom], [z_top, z_top]])
            ax.plot_surface(x, y, z, color=color, alpha=alpha)

    def dist_squared(self, xi, xi_1):
        return (xi['x'] - xi_1['x'])**2 + (xi['y'] - xi_1['y'])**2 + (xi['z'] - xi_1['z'])**2

    def m_prepare_and_generate(self, k, b):
        print(k)
        return k

def main():
    optimization = simulator()
    optimization.start_simulation()
    # optimization.m_start_simulation()

main()