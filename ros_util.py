# import numpy as np
#ROS
import rospy
from std_msgs.msg import String
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped

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


