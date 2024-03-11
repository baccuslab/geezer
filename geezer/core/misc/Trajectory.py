import json
import tqdm
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
            led_centered_coordinates = self.centered_geometry['leds'][led_id]
            led_el, led_az = calculate_led_angles(led_centered_coordinates, self.cam_basis)
            self.led_angles[led_id] = [led_el, led_az]
        
        self.plot_cam_basis()
        self.offsets= {}
        for led_id in self.led_ids:
            self.offsets[led_id] = self.coordinate_grid_search(2200, led_id)


    def coordinate_grid_search(self, reference_frame, reference_led='se', num_offsets=150):
        num_predicted_leds = len(self.led_ids) - 1

        distances = {}

        for predict_led in self.led_ids:
            if predict_led == reference_led:
                continue
            distances[predict_led] = np.zeros((num_offsets*2, num_offsets*2))
        
        x_offsets = np.arange(-num_offsets,num_offsets)
        y_offsets = np.arange(-num_offsets,num_offsets)
        

        for xi in tqdm.tqdm(x_offsets):
            for yi in y_offsets:
                offset = [xi,yi]
                
                for predict_led in self.led_ids:
                        if predict_led == reference_led:
                            continue

                        predicted_led_co = self.h5_file['fiducial_coordinates'][predict_led][reference_frame]
                        fiducial_co = self.h5_file['fiducial_coordinates'][reference_led][reference_frame]
                        camera_co = self.h5_file['fiducial_coordinates']['cam'][reference_frame]


                        p_elevation, p_azimuth = calculate_gaze_angles(predicted_led_co, fiducial_co, camera_co, self.led_angles[reference_led], offset=offset)

                        t_elevation, t_azimuth = self.led_angles[predict_led] 
                        elevation_distance = circular_mmse(p_elevation, t_elevation)
                        azimuth_distance = circular_mmse(p_azimuth, t_azimuth)


                        distances[predict_led][xi+num_offsets,yi+num_offsets] = azimuth_distance
        
        best_offsets = {}


        for k,v in distances.items():
            # print(k, np.nanmin(v))
            plt.imshow(v)
            plt.title(k)
            plt.colorbar()
            plt.show()

            w = np.where(v == np.nanmin(v))
            best_offsets[k] = [x_offsets[w[0][0]], y_offsets[w[1][0]]]

            info = 'Reference: {}\nPredicted: {}\nOffsets: {}\nDistance: {}'.format(reference_led, k, best_offsets[k], np.nanmin(v)) 
            
            print(info)
            print('-------------------')
        
        return best_offsets
                

                





    def plot_cam_basis(self, basis_check='ne'):
        axis = plt.axes(projection='3d')
        axis.plot(*self.centered_geometry['camera'], 'ro')
        axis.plot(*self.centered_geometry['observer'], 'go')



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
        


        axis.set_aspect('auto')
        
        for led_id in self.led_ids:
            t, p = self.led_angles[led_id] 
            print(led_id, np.rad2deg([t, p]))
            r = 20
            temp = np.array([r*np.sin(p) * np.cos(t), r*np.sin(t), r*np.cos(t) * np.cos(p)])

            # print(led_id, t, p, temp)
            temp = np.matmul(temp, self.cam_basis)


            axis.plot([0, temp[0]], [0, temp[1]], [0, temp[2]], 'y', linewidth=1)

        axis.set(xlim=(-30, 30), ylim=(-30, 30), zlim=(-30, 30))
        axis.plot(*x, 'r')
        axis.plot(*y, 'g')
        axis.plot(*z, 'b')
        axis.plot(*self.cam_basis[0], 'ko')
        axis.plot(*self.cam_basis[1], 'ko')
        axis.plot(*self.cam_basis[2], 'ko')

        for k,v in self.centered_geometry['leds'].items():
            axis.plot(*v, 'ro', markersize=10)

        plt.show()

    
def circular_mmse(a,b):
    diff = np.angle(np.exp(1j * (a - b)))
    mse = np.mean(np.square(diff))
    return mse


