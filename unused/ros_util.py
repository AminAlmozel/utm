# import numpy as np
#ROS
import rospy
from std_msgs.msg import String
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped, Polygon, Point32

def setup_publisher(name, topic, msg):
    pub = rospy.Publisher(topic, msg, queue_size=10)
    return pub

def setup_subscriber(name, topic, callback):
    rospy.Subscriber(topic, String, callback)

def state2msg(xi):
    msg = Odometry()
    msg.pose.pose.position.x = xi['x']
    msg.pose.pose.position.y = xi['y']
    msg.pose.pose.position.z = xi['z']
    msg.twist.twist.linear.x = xi['xdot']
    msg.twist.twist.linear.y = xi['ydot']
    msg.twist.twist.linear.z = xi['zdot']
    return msg

def msg2state(msg):
    keys = ['x', 'y', 'z', 'xdot', 'ydot', 'zdot']
    values = [msg.pose.pose.position.x,
    msg.pose.pose.position.y,
    msg.pose.pose.position.z,
    msg.twist.twist.linear.x,
    msg.twist.twist.linear.y,
    msg.twist.twist.linear.z]
    return dict(zip(keys, values))

def traj2msg(xi_1):
    msg = Path()
    for vehicle in xi_1:
        for n in range(len(vehicle[0])):
            # print("Hello")
            state = PoseStamped()
            state.pose.position.x = vehicle[0][n]
            state.pose.position.y = vehicle[1][n]
            state.pose.position.z = vehicle[2][n]
            state.pose.orientation.x = vehicle[3][n]
            state.pose.orientation.y = vehicle[4][n]
            state.pose.orientation.z = vehicle[5][n]
            state.pose.orientation.w = 0
            msg.poses.append(state)
        state = PoseStamped()
        state.pose.position.x = 0
        state.pose.position.y = 0
        state.pose.position.z = 0
        state.pose.orientation.x = 0
        state.pose.orientation.y = 0
        state.pose.orientation.z = 0
        state.pose.orientation.w = 1
        msg.poses.append(state)
    return msg

def msg2traj(msg):
    xi_1 = []
    x = []
    y = []
    z = []
    xdot = []
    ydot = []
    zdot = []
    for pose in msg.poses:
        if (pose.pose.orientation.w == 0):
            x.append(pose.pose.position.x)
            y.append(pose.pose.position.y)
            z.append(pose.pose.position.z)
            xdot.append(pose.pose.orientation.x)
            ydot.append(pose.pose.orientation.y)
            zdot.append(pose.pose.orientation.z)
        else:
            vehicle = [x, y, z, xdot, ydot, zdot]
            xi_1.append(vehicle)
            x = []
            y = []
            z = []
            xdot = []
            ydot = []
            zdot = []
    return xi_1

def obstacles2msg(obstacles):
    msg = Polygon()
    for obs in obstacles:
        point = Point32()
        point.x = obs['xmin']
        point.y = obs['ymin']
        point.z = obs['zmin']
        msg.points.append(point)

        point = Point32()
        point.x = obs['xmax']
        point.y = obs['ymax']
        point.z = obs['zmax']
        msg.points.append(point)

    return msg

def msg2obstacles(msg):
    obstacles = []
    keys = ['xmin', 'ymin', 'zmin', 'xmax', 'ymax', 'zmax']
    N = int(len(msg.points)/2)
    for i in range(N):
        values = [msg.points[2*i].x,
                  msg.points[2*i].y,
                  msg.points[2*i].z,
                  msg.points[2*i + 1].x,
                  msg.points[2*i + 1].y,
                  msg.points[2*i + 1].z]
        obs = dict(zip(keys, values))
        obstacles.append(obs)
    return obstacles