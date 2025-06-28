import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from datetime import datetime
import matplotlib.collections
import time

class simulator(drone.drone):
    def __init__(self):
        self.N = 20
        self.delta_t = 0.1 # Time step
        self.N_polygon = 8 # Number of sides for the polygon approximation
        self.total_iterations = 3

        # Parameters
        self.K = 15  # Number of vehicles
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

        self.vehicles_positions = []
        self.vehicles = []
        self.full_traj = []

        self.hlines = []

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