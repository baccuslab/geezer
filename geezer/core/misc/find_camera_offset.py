import numpy as np
from .calc_gaze_angle import calc_gaze_angle

def find_camera_offset(pupil_co, 
NUM_OFFSETS = 150
distance_mtx = np.zeros((NUM_OFFSETS*2, NUM_OFFSETS*2))
for i in np.arange(-NUM_OFFSETS,NUM_OFFSETS):
    for j in np.arange(-NUM_OFFSETS,NUM_OFFSETS):
        offset = [i,j]
        for predict_led in ['sw', 'se']:
            for reference_led in ['sw', 'se']:
                if predict_led == reference_led:
                    continue
                predicted_led_co = file['fiducial_coordinates'][predict_led][reference_frame]
                fiducial_co = file['fiducial_coordinates'][reference_led][reference_frame]
                camera_co = file['fiducial_coordinates']['cam'][reference_frame]


                p_elevation, p_azimuth = geezer.calc_gaze_angle(predicted_led_co, fiducial_co, camera_co, led_angles[reference_led], offset=offset)

                true = np.rad2deg(led_angles[predict_led])
                estimate = np.rad2deg([p_elevation, p_azimuth])*2

                # Compute euclidean distance
                distance = np.sqrt(np.sum((true - estimate)**2))
                if distance == np.nan:
                    print(i,j)
                distance_mtx[i+NUM_OFFSETS,j+NUM_OFFSETS] = distance

plt.imshow(distance_mtx)
plt.colorbar()
plt.show()
