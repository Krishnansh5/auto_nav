#!/usr/bin/env python3
import time
import math
import rospy
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs import point_cloud2
import open3d as o3d

import matplotlib.pyplot as plt

from geometry_msgs.msg import Point32
from visualization_msgs.msg import Marker, MarkerArray
from auto_nav.msg import BoundingBoxList, BoundingBox

from ObstacleAvoidance.DronePoseTracker import DronePoseTracker
from ObstacleAvoidance.Open3DVis import Visualizer
from ObstacleAvoidance.utils import convert_rgbFloat_to_tuple, convert_rgbUint32_to_tuple

MAP_FRAME="world"

DEPTH_IMAGE_TOPIC = '/camera/depth/image_raw'
POINT_CLOUD_TOPIC = '/camera/depth/points'

BBX_PUB_TOPIC = '/auto_nav/obstacle_avoidance/bounding_boxes'

VOXEL_SIZE=0.05

VIS_UPDATE_INTERVAL = 1

DRONE_DIM = {
    'x': (-0.33, 0.33),    # Front-back axis
    'y': (-0.42, 0.42),    # Left-right axis
    'z': (-0.21, 0.434)    # Vertical axis
}

aspect_ratio = 480 / 640  # height/width
vertical_fov = 2 * math.atan(math.tan(1.047/2) * aspect_ratio)
max_range = 3.0
max_vertical_view = max_range * math.cos(vertical_fov/2)+0.2

class ObstacleDetectionNode:
    def __init__(self):
        rospy.init_node('ObstacleDetectionNode', anonymous=True)

        self.bridge = CvBridge()
        
        self.depth_image_sub = rospy.Subscriber(DEPTH_IMAGE_TOPIC, 
                                         Image, 
                                         self.depth_callback)
        self.point_sub = rospy.Subscriber(POINT_CLOUD_TOPIC, PointCloud2, self.pcl_callback)

        self.bbx_pub = rospy.Publisher(BBX_PUB_TOPIC,BoundingBoxList,queue_size=5)

        self.marker_pub = rospy.Publisher('/auto_nav/visualization/bboxes', MarkerArray, queue_size=10)

        self.voxel_map = Visualizer("Voxel Map")
        self.pcd_vis = Visualizer("Point Cloud")
        self.drone_pose_tracker = DronePoseTracker()

        self.pcd = o3d.geometry.PointCloud()
        self.pcd_down = None # downsampled/voxelized point cloud
        self.ground_plane = None
        self.obstacles = None
        self.bbxs = []

        self.drone_bbx = o3d.geometry.OrientedBoundingBox(
            R=o3d.geometry.get_rotation_matrix_from_xyz([0, 0, 0]),
            extent=[DRONE_DIM['x'][1]-DRONE_DIM['x'][0],
                    DRONE_DIM['y'][1]-DRONE_DIM['y'][0],
                    DRONE_DIM['z'][1]-DRONE_DIM['z'][0]],
            center=[0, 0, 0]
        )

        self.timer = rospy.Timer(rospy.Duration(VIS_UPDATE_INTERVAL), self.update_voxel_map)
        self.timer = rospy.Timer(rospy.Duration(VIS_UPDATE_INTERVAL), self.update_pcd_visualisation)
        # self.timer = rospy.Timer(rospy.Duration(VIS_UPDATE_INTERVAL), self.update_drone_bbx)

        rospy.loginfo("depth node initialized")

        rospy.spin()

    def depth_callback(self, data): 
        """Callback function that processes incoming depth images"""
        try:
            # Convert ROS Image message to OpenCV image
            # For depth images, use 32FC1 or 16UC1 depending on the encoding
            depth_image = self.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
            
            # Get image dimensions
            height, width = depth_image.shape
            
            # Get depth value at the center of the image
            center_idx = (height // 2, width // 2)
            center_depth = depth_image[center_idx]
            
            # Log the depth value at the center of the image
            # rospy.loginfo("Depth at center: %.2f meters", center_depth)
            
            # Process depth image as needed
            # For example, you could find the minimum and maximum depths
            min_depth = np.nanmin(depth_image)
            max_depth = np.nanmax(depth_image)
            avg_depth = np.nanmean(depth_image) 
            
            # rospy.loginfo("Min depth: %.2f m, Max depth: %.2f m, Avg depth: %.2f m", 
            #              min_depth, max_depth, avg_depth)
            
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))
    
    def pcl_callback(self,msg):
        field_names=[field.name for field in msg.fields]
        cloud_data = list(point_cloud2.read_points(msg, skip_nans=True, field_names = field_names))

        drone_pos, drone_quat = self.drone_pose_tracker.get_latest_pose_from_odom_msg()

        R_drone = self.get_rotation_matrix_from_quaternion(drone_quat) 

        R_x = np.array([[1, 0, 0],
                    [0, 0, 1],
                    [0, -1, 0]])
        
        R_z = np.array([[0, 1, 0],
                    [-1, 0, 0],
                    [0, 0, 1]])
        
        R_camera_to_body = np.dot(R_z, R_x)

        R_total = np.dot(R_drone,R_camera_to_body)

        if len(cloud_data)>0:
            self.pcd = o3d.geometry.PointCloud()

            if "rgb" in field_names:
                IDX_RGB_IN_FIELD=3 # x, y, z, rgb

                # Get xyz
                xyz = [(x,y,z) for x,y,z,rgb in cloud_data ] # (why cannot put this line below rgb?)

                # Get rgb
                # Check whether int or float
                if type(cloud_data[0][IDX_RGB_IN_FIELD])==float: # if float (from pcl::toROSMsg)
                    rgb = [convert_rgbFloat_to_tuple(rgb) for x,y,z,rgb in cloud_data ]
                else:
                    rgb = [convert_rgbUint32_to_tuple(rgb) for x,y,z,rgb in cloud_data ]
                
                # Apply rotation to all points
                xyz_transformed = np.dot(xyz, R_total.T)+drone_pos

                # combine
                self.pcd.points = o3d.utility.Vector3dVector(np.array(xyz_transformed))
                self.pcd.colors = o3d.utility.Vector3dVector(np.array(rgb)/255.0)
            else:
                xyz = [(x,y,z) for x,y,z in cloud_data ] # get xyz

                xyz_transformed = np.dot(xyz, R_total.T)+drone_pos

                self.pcd.points = o3d.Vector3dVector(np.array(xyz_transformed))

            self.update_obstacle_bbx(self.pcd)
            self.publish_bounding_boxes()
            self.publish_bbx_marker_array() 
        
    def publish_bounding_boxes(self):
        msg = BoundingBoxList()

        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = MAP_FRAME

        for box in self.bbxs:
            # Create a single bounding box message
            bb_msg = BoundingBox()
            
            # Get min and max bounds
            min_bound = box.get_min_bound()
            max_bound = box.get_max_bound()
            
            # Set min bound
            bb_msg.min_bound = Point32()
            bb_msg.min_bound.x = float(min_bound[0])
            bb_msg.min_bound.y = float(min_bound[1])
            bb_msg.min_bound.z = float(min_bound[2])
            
            # Set max bound
            bb_msg.max_bound = Point32()
            bb_msg.max_bound.x = float(max_bound[0])
            bb_msg.max_bound.y = float(max_bound[1])
            bb_msg.max_bound.z = float(max_bound[2])
            
            # Add to array
            msg.boxes.append(bb_msg)

        self.bbx_pub.publish(msg)
        rospy.loginfo(f"Published {len(msg.boxes)} bounding boxes")
    
    def publish_bbx_marker_array(self):
        marker_array = MarkerArray()
        
        for i, bbox in enumerate(self.bbxs):
            marker = Marker()
            marker.header.frame_id = "world"  # e.g., "map" or "world"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "bboxes"
            marker.id = i
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            
            min_bound = bbox.get_min_bound()
            max_bound = bbox.get_max_bound()

            # Calculate center and dimensions
            center_x = (float(min_bound[0]) + float(max_bound[0])) / 2
            center_y = (float(min_bound[1]) + float(max_bound[1])) / 2
            center_z = (float(min_bound[2]) + float(max_bound[2])) / 2
            
            marker.pose.position.x = center_x
            marker.pose.position.y = center_y
            marker.pose.position.z = center_z
            marker.pose.orientation.w = 1.0  # Neutral orientation
            
            marker.scale.x = abs(float(max_bound[0]) - float(min_bound[0]))
            marker.scale.y = abs(float(max_bound[1]) - float(min_bound[1]))
            marker.scale.z = abs(float(max_bound[2]) - float(min_bound[2]))
            
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 0.5  # Semi-transparent
            marker.lifetime = rospy.Duration(3.0)  # 1 second lifetime
            
            marker_array.markers.append(marker)
        
        self.marker_pub.publish(marker_array)
    
    def update_obstacle_bbx(self,pcd):
        st = time.time()
        self.pcd_down = pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)
        print("Point clouds",self.pcd_down)

        # Segment plane using RANSAC
        plane_model, inliers = self.pcd_down.segment_plane(distance_threshold=0.01, 
                                                    ransac_n=3,
                                                    num_iterations=1000)
        [a, b, c, d] = plane_model
        plane_normal = np.array([a,b,c])
        z_axis = np.array([0, 0, 1])
        cos_theta = np.dot(plane_normal, z_axis) / (np.linalg.norm(plane_normal) * np.linalg.norm(z_axis))
        angle_with_z = np.degrees(np.arccos(cos_theta))

        angle_threshold = 5.0 # in degrees

        position,orientation = self.drone_pose_tracker.get_latest_pose_from_odom_msg()
        drone_altitude = position[2]
        print(f"drone_altitude, max_vertical_view ----> {drone_altitude} , {max_vertical_view}")

        if ((abs(angle_with_z) < angle_threshold) and (drone_altitude < max_vertical_view)): 
            self.ground_plane = self.pcd_down.select_by_index(inliers)
            self.ground_plane.paint_uniform_color([0.0, 1.0, 0.0])  # Paint ground green
            self.obstacles = self.pcd_down.select_by_index(inliers, invert=True)
        else:
            self.ground_plane = o3d.geometry.PointCloud()
            self.obstacles = self.pcd_down

        labels = np.array(self.obstacles.cluster_dbscan(eps=0.2, min_points=10, print_progress=True))
        max_label = labels.max()
        print(f"Point cloud has {max_label + 1} clusters")

        # Color each cluster
        colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
        colors[labels < 0] = 0  # Paint noise points black
        self.obstacles.colors = o3d.utility.Vector3dVector(colors[:, :3])

        # Create a list to store bounding boxes
        bounding_boxes = []

        # Get unique labels excluding noise (-1)
        unique_labels = list(set(labels))
        if -1 in unique_labels:
            unique_labels.remove(-1)

        # Process each cluster
        for label in unique_labels:
            # Get points belonging to the current cluster
            cluster_indices = np.where(labels == label)[0]
            cluster_cloud = self.obstacles.select_by_index(cluster_indices)
            
            # Set minimum number of points for meaningful bounding box
            if len(cluster_indices) < 10:
                continue
                
            # Create an oriented bounding box
            obb = cluster_cloud.get_oriented_bounding_box()
            obb.color = [1.0, 0.0, 0.0]  # Red color for bounding boxes
            
            # Alternative: use axis-aligned bounding box
            # aabb = cluster_cloud.get_axis_aligned_bounding_box()
            # aabb.color = [1.0, 0.0, 0.0]
            
            bounding_boxes.append(obb)
        position,orientation = self.drone_pose_tracker.get_latest_pose_from_odom_msg()
        # self.bbxs = self.transform_bounding_boxes_to_map_frame(bounding_boxes,position,orientation)
        self.bbxs=bounding_boxes
        rospy.loginfo(f"BBX computation done in : {time.time()-st}s")
        for bbx in self.bbxs:
            print(f"Bounding Box : {bbx.get_max_bound()} , {bbx.get_min_bound()}")

        self.update_drone_bbx()

    def update_drone_bbx(self):
        # Create a drone bounding box
        position, orientation = self.drone_pose_tracker.get_latest_pose_from_odom_msg()
        drone_bbx0 = o3d.geometry.OrientedBoundingBox(
            R=o3d.geometry.get_rotation_matrix_from_xyz([0, 0, 0]),
            extent=[DRONE_DIM['x'][1]-DRONE_DIM['x'][0],
                    DRONE_DIM['y'][1]-DRONE_DIM['y'][0],
                    DRONE_DIM['z'][1]-DRONE_DIM['z'][0]],
            center=[0, 0, 0]
        )
        self.drone_bbx = self.transform_bounding_boxes_to_map_frame([drone_bbx0],position,orientation)[0]
        self.drone_bbx.color = [0.0, 0.0, 1.0]
        if self.drone_bbx:
            print(f"Drone Bounding Box : {self.drone_bbx.get_max_bound()} , {self.drone_bbx.get_min_bound()}")

    def transform_bounding_boxes_to_map_frame(self,bounding_boxes, drone_position, drone_orientation):
        # Create rotation matrix
        if drone_orientation.shape == (4,):  # Quaternion
            # Convert quaternion to rotation matrix
            qx, qy, qz, qw = drone_orientation
            rotation_matrix = np.array([
                [1-2*qy*qy-2*qz*qz, 2*qx*qy-2*qz*qw, 2*qx*qz+2*qy*qw],
                [2*qx*qy+2*qz*qw, 1-2*qx*qx-2*qz*qz, 2*qy*qz-2*qx*qw],
                [2*qx*qz-2*qy*qw, 2*qy*qz+2*qx*qw, 1-2*qx*qx-2*qy*qy]
            ])
        else:  # Rotation matrix
            rotation_matrix = drone_orientation
        
        # Transform each bounding box to map frame
        map_frame_boxes = []
        for box in bounding_boxes:
            # Create a new box with the same parameters
            center = np.asarray(box.center)
            R = np.asarray(box.R)
            extent = np.asarray(box.extent)
            
            # Apply rotation: R_map = R_drone * R_box
            new_R = np.matmul(rotation_matrix, R)
            
            # Apply translation: center_map = R_drone * center_drone + position_drone
            new_center = np.matmul(rotation_matrix, center) + drone_position
            
            # Create new oriented bounding box in map frame
            map_box = o3d.geometry.OrientedBoundingBox(
                center=new_center,
                R=new_R,
                extent=extent
            )
            
            # Copy color if it exists
            if hasattr(box, 'color'):
                map_box.color = box.color
            
            map_frame_boxes.append(map_box)
        
        return map_frame_boxes 

    def update_voxel_map(self, event):
        self.voxel_map.update_visualisation([self.ground_plane,self.obstacles]+[bbx for bbx in self.bbxs]+[self.drone_bbx])

    def update_pcd_visualisation(self, event):
        self.pcd_vis.update_visualisation([self.pcd]+[bbx for bbx in self.bbxs]+[self.drone_bbx])

    def get_rotation_matrix_from_quaternion(self,quaternion):
        qx, qy, qz, qw = quaternion
        rotation_matrix = np.array([
            [1-2*qy*qy-2*qz*qz, 2*qx*qy-2*qz*qw, 2*qx*qz+2*qy*qw],
            [2*qx*qy+2*qz*qw, 1-2*qx*qx-2*qz*qz, 2*qy*qz-2*qx*qw],
            [2*qx*qz-2*qy*qw, 2*qy*qz+2*qx*qw, 1-2*qx*qx-2*qy*qy]
        ])
        return rotation_matrix

if __name__ == '__main__':
    try:
        ObstacleDetectionNode()
    except rospy.ROSInterruptException:
        pass
