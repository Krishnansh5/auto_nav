#!/usr/bin/env python  
import roslib
roslib.load_manifest('auto_nav')

import rospy
import tf
import math
from nav_msgs.msg import Odometry

if __name__ == '__main__':
    rospy.init_node('dynamic_tf_broadcaster')
    br = tf.TransformBroadcaster()
    rate = rospy.Rate(10.0)
    latest_odom = None
    def odom_callback(msg):
        global latest_odom
        latest_odom = msg

    odom_sub = rospy.Subscriber(
            "/odom",
            Odometry,
            odom_callback
        )

    while not rospy.is_shutdown():
        if latest_odom is None:
            continue
        x,y,z = (
            latest_odom.pose.pose.position.x,
            latest_odom.pose.pose.position.y,
            latest_odom.pose.pose.position.z)
        
        qx,qy,qz,qw = (
            latest_odom.pose.pose.orientation.x,
            latest_odom.pose.pose.orientation.y,
            latest_odom.pose.pose.orientation.z,
            latest_odom.pose.pose.orientation.w)
        print(x,y,z,qx,qy,qz,qw)
        t = rospy.Time.now().to_sec() * math.pi
        br.sendTransform((x, y, z),
                        (qx, qy, qz, qw),
                        rospy.Time.now(),
                        "base_link",
                        "world")
        rate.sleep()