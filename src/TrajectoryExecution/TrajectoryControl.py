import rospy
from geometry_msgs.msg import PoseStamped, TwistStamped
from mavros_msgs.msg import State, PositionTarget
from mavros_msgs.srv import CommandBool, SetMode, SetModeRequest, CommandBoolRequest, CommandTOL
from std_msgs.msg import Header
from nav_msgs.msg import Odometry
import time
from TrajectoryPlanners import *

import matplotlib.pyplot as plt

class DroneController:
    def __init__(self,takeoff_height):
        rospy.init_node('trajectory_controller', anonymous=True)

        self.current_trajectory = None
        
        self.state_sub = rospy.Subscriber('mavros/state', State, self.state_cb)
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)
        self.local_pos_pub = rospy.Publisher('mavros/setpoint_raw/local', PositionTarget, queue_size=10)
        # self.local_vel_pub = rospy.Publisher('mavros/setpoint_velocity/cmd_vel', TwistStamped, queue_size=10)
        
        rospy.wait_for_service('mavros/cmd/arming')
        self.arming_client = rospy.ServiceProxy('mavros/cmd/arming', CommandBool)

        rospy.wait_for_service('mavros/set_mode')
        self.set_mode_client = rospy.ServiceProxy('mavros/set_mode', SetMode)

        rospy.wait_for_service('/mavros/cmd/takeoff')
        self.takeoff_client = rospy.ServiceProxy('/mavros/cmd/takeoff', CommandTOL)
        
        self.current_state = State()
        self.current_pos = None
        self.rate = rospy.Rate(110)  # 20Hz

        self.threshold = 0.3
    
    def odom_callback(self, msg):
        self.current_pos = msg.pose.pose.position

    def state_cb(self, msg):
        self.current_state = msg

    def set_flight_mode(self,mode):
        try:
            mode_response = self.set_mode_client(custom_mode=mode)
            return mode_response.mode_sent
        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)

    def arm_drone(self):
        try:
            arm_response = self.arming_client(True)
            return arm_response.success
        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)

    def takeoff_drone(self,altitude):
        try:
            takeoff_response = self.takeoff_client(altitude=altitude)
            return takeoff_response.success
        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)

    def move_drone(self,x, y, z, yaw, vx=0, vy=0, vz=0):
        position_target = PositionTarget()
        position_target.header = Header()
        position_target.header.stamp = rospy.Time.now()
        position_target.coordinate_frame = PositionTarget.FRAME_LOCAL_NED
        position_target.type_mask = PositionTarget.IGNORE_AFX | PositionTarget.IGNORE_AFY | PositionTarget.IGNORE_AFZ | PositionTarget.FORCE | PositionTarget.IGNORE_YAW_RATE
        position_target.position.x = x
        position_target.position.y = y
        position_target.position.z = z
        position_target.velocity.x = vx
        position_target.velocity.y = vy
        position_target.velocity.z = vz
        position_target.yaw = yaw
        self.local_pos_pub.publish(position_target)

    def update_trajectory(self,wp,max_velocity):
        planner = CubicTrajectorySplinePlanner(wp, max_velocity=max_velocity, max_yaw_rate=np.pi/2)
        self.current_trajectory = planner.generate_trajectory()

        print(self.current_trajectory.positions)

        # Plot the x-y trajectory
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(self.current_trajectory.positions[:, 0], self.current_trajectory.positions[:, 1], label='Trajectory')
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_title('Drone Trajectory')
        ax.legend()
        plt.savefig('trajectory_plot.png')


    def execute_trajectory(self):
        if self.current_trajectory is not None:
            for point, yaw, velocity in zip(self.current_trajectory.positions, self.current_trajectory.yaw, self.current_trajectory.velocities):
                x,y,z = point[0],point[1],point[2]
                yw = yaw
                vx, vy, vz = velocity[0], velocity[1], velocity[2]
                self.move_drone(x,y,z,yw, vx=vx, vy=vy, vz=vz)
                # print("Current: ",self.current_pos)
                # print("Target: ",x,y,z)
                count = 0
                threshold = self.threshold
                # if self.current_pos is not None:
                #     # distance = np.round(np.linalg.norm(np.array([self.current_pos.y, self.current_pos.x]) - np.array([-x, y])),2)
                #     while(np.round(np.linalg.norm(self.current_pos.y+x),3) > self.threshold or np.round(np.linalg.norm(self.current_pos.x-y),3) > self.threshold):
                #     # while(distance > threshold):
                #         # distance = np.round(np.linalg.norm(np.array([self.current_pos.y, self.current_pos.x]) - np.array([-x, y])),2)
                #         rospy.loginfo(f"Moving to position: x={x}, y={y}, z={z}, dist={np.round(np.linalg.norm(self.current_pos.y+x),2)}, disty:{np.round(np.linalg.norm(self.current_pos.x-y),2)}")
                #         self.rate.sleep()
                #         if count > 5:
                #             break
                #         count += 1
                        
                self.rate.sleep()

    # def set_waypoints(self,waypoints):
    #     self.waypoints = waypoints


if __name__ == '__main__':
    try:
        drone_controller = DroneController(takeoff_height=2)
        # drone_controller.set_flight_mode("GUIDED")

        # drone_controller.arm_drone()

        # drone_controller.takeoff_drone(2)
        
        start = np.array([0.0, 0, 2.0])
        mid = np.array([3.0,-3.0,2.0])
        mid2 = np.array([7.0,1.0,2.0])
        mid3 = np.array([10.0,-3.0,2.0])
        end = np.array([15.0,0.0,2.0])
        ls1 = np.array([start,mid,mid2,mid3,end])
        ls2 = ls1[::-1]
        wp = np.array(ls1)

        duration = [0, 8, 24, 34,40]
        
        # Set velocity limits
        max_velocity = np.array([1.0, 1.0, 1.0])

        drone_controller.update_trajectory(wp, max_velocity)
        start_time = time.time()
        drone_controller.execute_trajectory()
        end_time = time.time()
        print("Exec time: ", end_time-start_time)

        rospy.spin()

    except rospy.ROSInterruptException:
        pass

        