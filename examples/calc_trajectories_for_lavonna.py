import tqdm
import cv2
import pickle
import geezer
from numpy import pi
import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
import csv
import numpy as np


cap = cv2.VideoCapture('/home/jbmelander/Lavonna/cam_23211830.mp4')
frame_number = 146484

r,f = cap.read()
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

print(width, height)
def parse_csv(file_path):
    """
    Parse a csv file into a dictionary of numpy arrays
    """
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        data = list(reader)
    
    mock_file = {}
    data_dict = {}
    for i in tqdm.tqdm(range(22)):
        label = data[1][i]
        if label not in data_dict:
            data_dict[label] = {}
        data_type = data[2][i]
    
        data_dict[label][data_type] = np.array([d[i] for d in data[3:]]).astype(float)
    
    mock_file['fiducial_coordinates'] = {}
    for k,v in data_dict.items():
        if 'x' in v.keys() and 'y' in v.keys():
            mock_file['fiducial_coordinates'][k] = np.array([v['x'], height-v['y']]).T

    return mock_file 

def quick_pupil_centroid_estimator(mock_file):
    xs = []
    ys = []

    for name in ['Left', 'Right', 'Top', 'Bottom']:
        xs.append(mock_file['fiducial_coordinates'][name][:,0])
        ys.append(mock_file['fiducial_coordinates'][name][:,1])
        
    xs= np.array(xs)
    ys= np.array(ys)
    xs = xs.mean(axis=0)
    ys = ys.mean(axis=0)

    mock_file['pup_co'] = np.array([xs, ys]).T

    return mock_file


csv_file = '/home/jbmelander/Lavonna/cam_23211830DLC_resnet50_e1_eyeApr12shuffle1_1030000.csv'
raw_data = parse_csv(csv_file)
data = quick_pupil_centroid_estimator(raw_data)

plt.plot(data['pup_co'][:5000,0], data['pup_co'][:5000,1], 'ko', alpha=0.1)
plt.plot(data['fiducial_coordinates']['LED1'][:500,0], data['fiducial_coordinates']['LED1'][:500,1], 'ro', alpha=0.1)
plt.plot(data['fiducial_coordinates']['LED2'][:500,0], data['fiducial_coordinates']['LED2'][:500,1], 'ro', alpha=0.1)
plt.plot(data['fiducial_coordinates']['LED3'][:500,0], data['fiducial_coordinates']['LED3'][:500,1], 'ro', alpha=0.1)
plt.show()


mock_file = data

r,f = cap.read()

plt.imshow(f)
plt.show()

colors = ['bo', 'go', 'co']
count = 0
for i in range(1,4):
    plt.plot(data['fiducial_coordinates']['LED{}'.format(i)][frame_number,0], data['fiducial_coordinates']['LED{}'.format(i)][frame_number,1], colors[count], label=i)
    count +=1

plt.plot(data['pup_co'][frame_number,0], data['pup_co'][frame_number,1], 'go')
plt.legend()

plt.show()

mapping = {
        'leds':
        {'LED1': 'LED1',
         'LED3': 'LED3'},
        'camera': {'cam': 'LED2'},
        'pupil': 'headbar'}

geometry = geezer.load_geometry_json('/home/jbmelander/Lavonna/lav/lav_geometray.json')
centered_geometry = geezer.center_geometry(geometry, mapping)

basis = geezer.get_cam_basis(centered_geometry)
led_angles = {}
for led in ['LED1', 'LED3']:
    led_elevation, led_azimuth = geezer.calculate_led_angles(centered_geometry['leds'][led], basis)
    led_angles[led] = (led_elevation, led_azimuth)
geezer.plot_cam_basis(centered_geometry, basis, led_angles)
# # %%



REFERENCE_FRAME = 0
dist, offsets = geezer.estimate_camera_led_offset(data, ['LED1', 'LED3'], led_angles, REFERENCE_FRAME, 50, verbose=True, cam_name='LED2')


trajectories = {}
for LED in ['LED1', 'LED3']:
    cam_offset = mock_file['fiducial_coordinates']['LED2'][REFERENCE_FRAME] - mock_file['fiducial_coordinates'][LED][REFERENCE_FRAME]

    num_frames = mock_file['fiducial_coordinates'][LED].shape[0]
    pupil_coordinates = mock_file['pup_co'][:]
    led_coordinates = mock_file['fiducial_coordinates'][LED][:]

    camera_coordinates = mock_file['fiducial_coordinates'][LED][:] + cam_offset

    trajectory = geezer.calculate_gaze_trajectory(pupil_coordinates, led_coordinates, camera_coordinates, led_angles[LED], offset=offsets[0])


    elevations = trajectory[:,0]
    azimuths = trajectory[:,1]

    trajectories[LED] = trajectory


# # %%
geezer.plot_cam_basis(centered_geometry, basis, led_angles)
# # %%
table_trajectories = {}
for LED in ['LED1', 'LED3']:
    sanitized_trajectory = geezer.sanitize_trajectory(trajectories[LED], [10, 30], 1.5, 10, window_mean_proc=False)
    table_trajectory = geezer.basis_transform_to_table(trajectory, basis)

    relative_trajectory = np.zeros_like(table_trajectory)
    relative_trajectory[:,0] = geezer.get_relative_angle(table_trajectory[:,0])
    relative_trajectory[:,1] = geezer.get_relative_angle(table_trajectory[:,1])

    table_trajectories[LED] =relative_trajectory 


for LED in ['LED1', 'LED3']:
    plt.plot(np.rad2deg(table_trajectories[LED][:,0]-table_trajectories[LED][0,0]), label='elevation {}'.format(LED), color='k')
    plt.plot(np.rad2deg(table_trajectories[LED][:,1]-table_trajectories[LED][0,1]), label='azimuth {}'.format(LED), color='r')
    plt.ylabel('$\Delta \Theta$')
    plt.legend()
    plt.show()
