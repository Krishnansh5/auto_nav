import open3d as o3d
from threading import Thread 
import rospy

class Visualizer:
    def __init__(self,name):
        self.name = name
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name=name, width=1920, height=1080)
        self.coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
        self.current_geometries = []
        self.update_required = False
        self.camera_params = None
        self.vis_thread = Thread(target=self._vis_loop)
        self.vis_thread.daemon = True
        self.vis_thread.start()
        self.first_update=True

    def _vis_loop(self):
        while not rospy.is_shutdown():
            if self.update_required:
                
                # self.camera_params = self.vis.get_view_control().convert_to_pinhole_camera_parameters()

                self.vis.clear_geometries()
                if not self.first_update:
                    for geometry in self.current_geometries:
                        if geometry:
                            self.vis.add_geometry(geometry,reset_bounding_box=False)
                            
                    self.vis.add_geometry(self.coordinate_frame,reset_bounding_box=False) 
                else:
                    for geometry in self.current_geometries:
                        if geometry:
                            self.vis.add_geometry(geometry)
                            
                    self.vis.add_geometry(self.coordinate_frame) 
                    self.first_update=False

                # if self.camera_params is not None:
                #     self.vis.get_view_control().convert_from_pinhole_camera_parameters(self.camera_params)

                self.update_required = False
            self.vis.poll_events()
            self.vis.update_renderer()
            rospy.sleep(0.01)  # Reduce CPU usage

    def update_visualisation(self, geometries):
        self.current_geometries = geometries
        self.update_required = True
        rospy.loginfo(f'{self.name} visualisation updated')