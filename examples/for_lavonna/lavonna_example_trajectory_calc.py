import tqdm

import geezer
from numpy import pi
import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
import csv

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
            mock_file['fiducial_coordinates'][k] = np.array([v['x'], v['y']]).T

    return mock_file 

def quick_pupil_centroid_estimator(mock_file):
    xs = []
    ys = []

    for name in ['Left', 'Right', 'Top', 'Bottom']:
        xs.append(mock_file['fiducial_coordinates'][name][:,0])
        ys.append(mock_file['fiducial_coordinates'][name][:,1])
        
    xs= np.array(xs).T
    ys= np.array(ys).T
    xs = xs.mean(axis=1)
    ys = ys.mean(axis=1)

    mock_file['pup_co'] = np.array([xs, ys]).T

    return mock_file['pup_co']


csv_file = '/home/jbmelander/Code/Lavonna/cam_23211830DLC_resnet50_e1_eyeApr12shuffle1_1030000.csv'
raw_data = parse_csv(csv_file)
data = quick_pupil_centroid_estimator(raw_data)

# mapping = {
#         'leds':
#         {'A': 'A',
#          'B': 'B',
#          'C': 'C',
#          'D': 'D',
#          'E': 'E'},
#         'camera': {'cam': 'lens'},
#         'pupil': 'headbar'}

# geometry = geezer.load_geometry_json('/home/jbmelander/Code/geezer/geometries/lavonna_may_2024.json')
# centered_geometry = geezer.center_geometry(geometry, mapping)

# basis = geezer.get_cam_basis(centered_geometry)
# led_angles = {}
# for led in ['A', 'B', 'C', 'D']:
#     led_elevation, led_azimuth = geezer.calculate_led_angles(centered_geometry['leds'][led], basis)
#     led_angles[led] = (led_elevation, led_azimuth)
# geezer.plot_cam_basis(centered_geometry, basis, led_angles)
# # # %%



# # REFERENCE_FRAME = 19764
# # # # # %%
# # dist, offsets = geezer.estimate_camera_led_offset(mock_file, ['nw', 'sw'], led_angles, REFERENCE_FRAME, 100, verbose=True)


# # # # # %% 
# # keys = list(mock_file['fiducial_coordinates'].keys())

# # for k in keys:
# #     co = mock_file['fiducial_coordinates'][k][:]
# #     plt.plot(co[REFERENCE_FRAME,0], -1*co[REFERENCE_FRAME,1], 'o', label=k)
# # plt.show()

# # # # # %% 
# # # # t= np.arange(0, co[:,0].shape[0])/30

# # # # plt.plot(t, co[:,0])
# # # # plt.show()

# # # # fig,ax = plt.subplots(2, sharex=True)
# # # # ax[0].plot(co[:,0])
# # # # ax[1].plot(np.diff(co[:,0]))
# # # # ax[1].axhline(y=30, color='r')
# # # # ax[1].axhline(y=-30, color='r')
# # # # plt.show()
# # # # # # %%
# # # trajectories = {}

# # xs = []
# # ys = []


# # pup_co = np.array([xs, ys]).T
# # mock_file['pup_co'] = pup_co

# # trajectories = {}
# # for LED in ['nw', 'sw']:
# #     cam_offset = mock_file['fiducial_coordinates']['camera'][REFERENCE_FRAME] - mock_file['fiducial_coordinates'][LED][REFERENCE_FRAME]

# #     num_frames = mock_file['fiducial_coordinates'][LED].shape[0]
# #     pupil_coordinates = mock_file['pup_co'][:]
# #     led_coordinates = mock_file['fiducial_coordinates'][LED][:]

# #     camera_coordinates = mock_file['fiducial_coordinates'][LED][:] + cam_offset

# #     trajectory = geezer.calculate_gaze_trajectory(pupil_coordinates, led_coordinates, camera_coordinates, led_angles[LED], offset=offsets[0])

# #     elevations = trajectory[:,0]
# #     azimuths = trajectory[:,1]

# #     trajectories[LED] = trajectory


# # # # %%
# # trajectory = np.zeros((1000,2)) 
# # trajectory[:,0] = np.zeros_like(trajectory[:,0]) 
# # trajectory[:,1] = np.linspace(0, 2*pi, trajectory.shape[0])

# # plt.plot(np.linspace(0, 2*pi, trajectory.shape[0]), trajectory[:,1], label='camera azimuth (el=0)', color='k')

# # table_trajectory = geezer.basis_transform_to_table(trajectory, basis)

# # plt.plot(np.linspace(0, 2*pi, trajectory.shape[0]), table_trajectory[:,0], label='table elevation', color='c')
# # plt.axvline(x=pi/2, color='r')
# # plt.axhline(y=0, color='r')
# # plt.title('camera elevation at 0, azimuth at pi/2 should have table elevation at ')
# # plt.legend()
# # plt.show()

# # # # %%
# # geezer.plot_cam_basis(centered_geometry, basis, led_angles)
# # # # %%
# # table_trajectories = {}
# # for LED in ['nw', 'sw']:
# #     # trajectory = geezer.sanitize_trajectory(trajectories[LED], [10, 30], 1.5, 10)
# #     trajectory = geezer.sanitize_trajectory(trajectories[LED], [10, 30], 1.5, 10, window_mean_proc=False)

# #     # plt.plot(trajectory[:,0], label='sanitized')
# #     # plt.plot(nw_trajectory[:,0], label='no window')
# #     # plt.legend()
# #     # plt.show()
# #     table_trajectory = geezer.basis_transform_to_table(trajectory, basis)

# #     plt.plot(trajectories[LED][:,1], label='camera azimuth', color='k')
# #     plt.plot(table_trajectory[:,1], label='table azimuth', color='c')
# #     plt.legend()
# #     plt.show()

# #     relative_trajectory = np.zeros_like(table_trajectory)
# #     relative_trajectory[:,0] = geezer.get_relative_angle(table_trajectory[:,0])
# #     relative_trajectory[:,1] = geezer.get_relative_angle(table_trajectory[:,1])

# #     table_trajectories[LED] =relative_trajectory 
# #     plt.plot(relative_trajectory[:,1], label='table elevation', color='c')
# #     plt.show()

# #     # fig, ax = plt.subplots(6, sharex=True)
# #     # ax[0].set_title('camera elevation')
# #     # ax[0].plot(trajectory[:,0], label='camera', color='k')
# #     # ax[1].set_title('table elevation')
# #     # ax[1].plot(table_trajectory[:,0], label='table', color='c')
# #     # ax[2].set_title('camera azimuth')
# #     # ax[2].plot(trajectory[:,1], color='k')
# #     # ax[3].plot(table_trajectory[:,1], color='c')
# #     # ax[3].set_title('table azimuth')

# #     # ax[4].plot(geezer.get_relative_angle(table_trajectory[:,1]))
# #     # ax[5].plot(geezer.get_relative_angle(table_trajectory[:,0]))

# #     # plt.legend()
# #     # plt.show()

# # # %%

# # print(np.rad2deg(table_trajectories['sw'][:,0].max()-table_trajectories['sw'][:,0].min()))
# # print(np.rad2deg(table_trajectories['sw'][:,1].max()-table_trajectories['sw'][:,1].min()))

# # plt.hist(np.rad2deg(table_trajectories['sw'][:,1]), bins=100)
# # plt.show()

# # # %%
      
# # plt.plot(np.rad2deg(table_trajectories['sw'][:,0]))
# # plt.plot(np.rad2deg(table_trajectories['nw'][:,0]))
# # plt.show()

# # plt.plot(np.rad2deg(table_trajectories['sw'][:,1]))
# # plt.plot(np.rad2deg(table_trajectories['nw'][:,1]))
# # plt.show()
# # # %% 
# # # plt.plot(file['interp_trajectories']['sw'][:,0])
# # # plt.plot(table_trajectories['sw'][:,0])
# # # plt.plot(table_trajectories['se'][:,0])
# # # plt.show()

# # # %% 
# # file.close()

# # with h5.File(GEEZER_H5_FILEPATH, 'a') as file:
# #     try:
# #         del file['raw_trajectories']
# #         del file['interp_table_trajectories']
# #     except:
# #         pass
# #     # file.create_dataset('raw_trajectories/se', data=trajectories['se'])
# #     # file.create_dataset('raw_trajectories/sw', data=trajectories['sw'])

# #     file.create_dataset('interp_table_trajectories_no_window/se', data=table_trajectories['se'])
# #     file.create_dataset('interp_table_trajectories_no_window/sw', data=table_trajectories['sw'])

# # # %% 
# # file.close()
# # # %%
# # # fig,ax = plt.subplots(2,2, sharex=True)
# # # for led in ['se', 'sw']:
# # #     trajectories[led] = []
# # #     for frame in range(num_frames):
# # #         pup_co = file['pup_co'][frame]
# # #         fiducial_co = file['fiducial_coordinates'][led][frame]
# # #         camera_co = file['fiducial_coordinates']['sw'][frame] + cam_offset

# # #         _e, _a= geezer.calculate_gaze_angles(pup_co, fiducial_co, camera_co, led_angles[led], offset=best_offset)

# # #         trajectories[led].append([_e, _a])

# # #     trajectories[led] = np.array(trajectories[led])

# # #     if led == 'se':
# # #         color = 'r'
# # #     else:
# # #         color = 'k'

# # #     ax[0,0].plot(np.rad2deg(trajectories[led][:,0]), color)
# # #     ax[1,0].plot(np.rad2deg(trajectories[led][:,1]), color)
# # #     ax[0,1].plot(np.rad2deg(trajectories[led][:,0] - trajectories[led][0,0]), color)
# # #     ax[1,1].plot(np.rad2deg(trajectories[led][:,1] - trajectories[led][0,1]), color)
# # # plt.show()


# # # file.close()
# # # with h5.File('/home/dennis/Data/data_1/GolDRoger/goldroger_geezer_round_2.h5', 'a') as file:
# # #     file.create_dataset('new_trajectories/se', data=trajectories['se'])
# # #     file.create_dataset('new_trajectories/sw', data=trajectories['sw'])


# # # # Which one should we use for the camera offset - test by using it to predict others and taking the best
