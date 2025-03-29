import open3d as o3d
from threading import Thread 
import rospy

class Visualizer:
    def __init__(self,name):
        self.name = name
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name=name, width=1920, height=1080)
        self.current_geometries = []
        self.update_required = False
        self.vis_thread = Thread(target=self._vis_loop)
        self.vis_thread.daemon = True
        self.vis_thread.start()

    def _vis_loop(self):
        while not rospy.is_shutdown():
            if self.update_required:
                self.vis.clear_geometries()
                for geometry in self.current_geometries:
                    if geometry:
                        self.vis.add_geometry(geometry)
                self.update_required = False
            self.vis.poll_events()
            self.vis.update_renderer()
            rospy.sleep(0.01)  # Reduce CPU usage

    def update_voxel_grid(self, geometries):
        self.current_geometries = geometries
        self.update_required = True
        rospy.loginfo(f'{self.name} visualisation updated')

# class PointCloudVis:
#     def __init__(self):

#         self.vis = o3d.visualization.Visualizer()
#         self.vis.create_window(window_name='Point Cloud', width=1920, height=1080)
#         self.current_pcd = None
#         self.current_bbxs = None
#         self.update_required = False
#         self.vis_thread = Thread(target=self._vis_loop)
#         self.vis_thread.daemon = True
#         self.vis_thread.start()

#     def _vis_loop(self):
#         while not rospy.is_shutdown():
#             if self.update_required:
#                 self.vis.clear_geometries()
#                 if self.current_pcd:
#                     self.vis.add_geometry(self.current_pcd)
#                 if self.current_bbxs:
#                     for bbx in self.current_bbxs:
#                         self.vis.add_geometry(bbx)
#                 self.update_required = False
#             self.vis.poll_events()
#             self.vis.update_renderer()
#             rospy.sleep(0.01)  # Reduce CPU usage

#     def update_visualisation(self, pcd,bbxs):
#         if len(pcd.points) > 0:
#             self.current_pcd = pcd
#             self.current_bbxs = bbxs
#             self.update_required = True
#             rospy.loginfo('Point cloud updated')