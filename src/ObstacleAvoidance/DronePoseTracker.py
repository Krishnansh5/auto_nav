'''
This should be completely movied to separate node which publishes estimated drone pose using sensor data on a topic
'''

import numpy as np
import rospy
import tf2_ros

class DronePoseTracker:
    def __init__(self, map_frame="world", drone_frame="base_link"):
        self.map_frame = map_frame
        self.drone_frame = drone_frame
        
        # Set up TF listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
    def get_latest_pose_from_sim(self):
        '''
            Returns the transform between map frame to drone frame (base_link) using simulation transforms published by robot_state_publisher
        '''
        try:
            # Look up the transform from map to drone
            transform = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.drone_frame,
                rospy.Time(0),  # Get latest transform
                rospy.Duration(1.0)  # Wait up to 1 second
            )
            
            # Extract position
            position = np.array([
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.translation.z
            ])
            
            # Extract orientation
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
    
    def get_latest_pose_from_sensors():
        '''
            Returns the transform between map frame to drone frame (base_link) using sensors to estimate pose
        '''
        pass
