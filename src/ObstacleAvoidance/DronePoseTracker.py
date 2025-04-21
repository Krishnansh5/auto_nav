'''
This should be completely movied to separate node which publishes estimated drone pose using sensor data on a topic
'''

import numpy as np
import rospy
import tf2_ros
from nav_msgs.msg import Odometry

class DronePoseTracker:
    def __init__(self, map_frame="world", drone_frame="base_link", odom_topic="/iris/odom"):
        self.map_frame = map_frame
        self.drone_frame = drone_frame

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.odom_topic = odom_topic
        self.latest_odom = None

        self.odom_sub = rospy.Subscriber(
            self.odom_topic,
            Odometry,
            self.odom_callback
        )
        
    def get_latest_pose_from_transform(self):
        '''
            Returns the transform between map frame to drone frame (base_link) using simulation transforms
        '''
        try:
            transform = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.drone_frame,
                rospy.Time(0),  
                rospy.Duration(1.0)  
            )
            
            position = np.array([
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.translation.z
            ])
            
            orientation = np.array([
                transform.transform.rotation.x,
                transform.transform.rotation.y,
                transform.transform.rotation.z,
                transform.transform.rotation.w
            ])
            
            return position, orientation
            
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn(f"TF lookup failed: {e}")
            return np.zeros(3), np.array([0, 0, 0, 1])

    def odom_callback(self, msg):
        self.latest_odom = msg
    
    def get_latest_pose_from_odom_msg(self):
        '''
            Returns the transform between map frame to drone frame (base_link) using odometry messages
        '''
        if self.latest_odom is None:
            rospy.logwarn("No odometry message received yet")
            return np.zeros(3), np.array([0, 0, 0, 1])
            
        position = np.array([
            self.latest_odom.pose.pose.position.x,
            self.latest_odom.pose.pose.position.y,
            self.latest_odom.pose.pose.position.z
        ])
        
        orientation = np.array([
            self.latest_odom.pose.pose.orientation.x,
            self.latest_odom.pose.pose.orientation.y,
            self.latest_odom.pose.pose.orientation.z,
            self.latest_odom.pose.pose.orientation.w
        ])
        
        return position, orientation
