import json
import matplotlib.pyplot as plt
import numpy as np
import h5py as h5

from ..core.io import load_geometry_json, load_mapping_json
from ..core.center_geometry import center_geometry
from ..core.get_cam_basis import get_cam_basis
from ..core.calculate_led_angles import calculate_led_angles
from ..core.calculate_gaze_angles import calculate_gaze_angles

class Trajectory:
    def __init__(self, h5_file, geometry_file, mapping_file, reference_frame=0):
        self.h5_file = h5.File(h5_file, 'r')

        self.geometry = load_geometry_json(geometry_file)
        self.mapping = load_mapping_json(mapping_file)
        self.centered_geometry = center_geometry(self.geometry, self.mapping)

        self.cam_basis = get_cam_basis(self.centered_geometry)
        

        self.led_ids = list(self.centered_geometry['leds'].keys())
        self.led_angles = {}

        for led_id in self.led_ids:
            print(led_id)
            led_centered_coordinates = self.centered_geometry['leds'][led_id]
            led_el, led_az = calculate_led_angles(led_centered_coordinates, self.cam_basis)
            self.led_angles[led_id] = [led_el, led_az]
        
        self.plot_cam_basis()
        # self.best_offset = self.find_best_offset(reference_frame)

    def plot_cam_basis(self, basis_check='ne'):
        axis = plt.axes(projection='3d')
        axis.plot(*self.centered_geometry['camera'], 'ro')
        axis.plot(*self.centered_geometry['observer'], 'go')

        for k,v in self.centered_geometry['leds'].items():
            axis.plot(*v, 'ko')

        axis.set_xlabel('X', fontsize=20)
        axis.set_ylabel('Y', fontsize=20)
        axis.set_zlabel('Z', fontsize=20)

        # %% 

        x = [[0, val] for val in self.cam_basis[0]]
        x = np.array(x) * 30
        y = [[0, val] for val in self.cam_basis[1]]
        y = np.array(y) * 30
        z = [[0, val] for val in self.cam_basis[2]]
        z = np.array(z) * 30
        
        axis.set(xlim=(-30, 30), ylim=(-30, 30), zlim=(-30, 30))
        axis.plot(*x, 'r')
        axis.plot(*y, 'g')
        axis.plot(*z, 'b')
        axis.plot(*self.cam_basis[0], 'ko')
        axis.plot(*self.cam_basis[1], 'ko')
        axis.plot(*self.cam_basis[2], 'ko')

        axis.set_aspect('auto')
        
        for led_id in self.led_ids:
            t, p = self.led_angles[led_id] 
            print(led_id, np.rad2deg([t, p]))
            r = 20
            temp = np.array([r*np.sin(p) * np.cos(t), r*np.sin(t), r*np.cos(t) * np.cos(p)])

            # print(led_id, t, p, temp)
            temp = np.matmul(temp, self.cam_basis)


            axis.plot([0, temp[0]], [0, temp[1]], [0, temp[2]], 'y', linewidth=1, alpha=0.5)


        plt.show()

         

    # def find_best_offset(self, reference_frame, num_offsets=50):
    #     distance_mtx = np.zeros((num_offsets*2, num_offsets*2))
    #     for i in np.arange(-num_offsets,num_offsets):
    #         for j in np.arange(-num_offsets,num_offsets):
    #             offset = [i,j]
    #             for predict_led in ['sw', 'se']:
    #                 for reference_led in ['sw', 'se']:
    #                     if predict_led == reference_led:
    #                         continue

    #                     predicted_led_co = self.h5_file['fiducial_coordinates'][predict_led][reference_frame]
    #                     fiducial_co = self.h5_file['fiducial_coordinates'][reference_led][reference_frame]
    #                     camera_co = self.h5_file['fiducial_coordinates']['cam'][reference_frame]


    #                     p_elevation, p_azimuth = calculate_gaze_angles(predicted_led_co, fiducial_co, camera_co, self.led_angles[reference_led], offset=offset)

    #                     true = np.rad2deg(self.led_angles[predict_led])
    #                     estimate = np.rad2deg([p_elevation, p_azimuth])*2
                        
# #                         # Compute euclidean distance
    #                     distance = np.sqrt(np.sum((true - estimate)**2))
    #                     if distance == np.nan:
    #                         print(i,j)
    #                     distance_mtx[i+num_offsets,j+num_offsets] = distance

    #     plt.imshow(distance_mtx)
    #     plt.colorbar()
    #     plt.show()

    #     w = np.where(distance_mtx == np.nanmin(distance_mtx))
# #         print(w)
# #         xoffsets = np.arange(-num_offsets,num_offsets)
# #         yoffsets = np.arange(-num_offsets,num_offsets)

# #         best_offset = [xoffsets[w[0][0]], yoffsets[w[1][0]]]
# #         print(best_offset)

                

                


