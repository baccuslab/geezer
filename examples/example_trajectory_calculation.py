
# %%
import tqdm
import geezer
from numpy import pi
import h5py as h5
import numpy as np
import matplotlib.pyplot as plt

reference_frame = 11572

geometry_file = '/home/dennis/Code/geezer/geometries/jan_2024_geometry.json'
h5_filepath= '/home/dennis/Data/data_2/kizaru/kizaru_geezer_output.h5'

file = h5.File(h5_filepath, 'r')

mapping = {
        'leds':
        {'ne': 'ne',
         'se': 'se',
         'sw': 'sw'},
        'camera': {'cam': 'lens'},
        'pupil': 'headbar'}

geometry = geezer.load_geometry_json(geometry_file)
centered_geometry = geezer.center_geometry(geometry, mapping)

# %%
basis = geezer.get_cam_basis(centered_geometry)
led_angles = {}
for led in ['ne', 'se', 'sw']:
    led_elevation, led_azimuth = geezer.calculate_led_angles(centered_geometry['leds'][led], basis)
    led_angles[led] = (led_elevation, led_azimuth)

# %% 

# %%
dist, offsets = geezer.estimate_camera_led_offset(file, ['se', 'sw'], led_angles, reference_frame, 100, verbose=True)


# %% 
keys = list(file['fiducial_coordinates'].keys())

for k in keys:
    co = file['fiducial_coordinates'][k][:]
    plt.plot(co[reference_frame,0], -1*co[reference_frame,1], 'o', label=k)
plt.show()

# %% 
t= np.arange(0, co[:,0].shape[0])/30

plt.plot(t, co[:,0])
plt.show()

fig,ax = plt.subplots(2, sharex=True)
ax[0].plot(co[:,0])
ax[1].plot(np.diff(co[:,0]))
ax[1].axhline(y=30, color='r')
ax[1].axhline(y=-30, color='r')
plt.show()
# %%
trajectories = {}

for LED in ['se', 'sw']:
    cam_offset = file['fiducial_coordinates']['cam'][reference_frame] - file['fiducial_coordinates'][LED][reference_frame]

    num_frames = file['fiducial_coordinates'][LED].shape[0]
    pupil_coordinates = file['pup_co'][:]
    led_coordinates = file['fiducial_coordinates'][LED][:]

    camera_coordinates = file['fiducial_coordinates'][LED][:] + cam_offset


    trajectory = geezer.calculate_gaze_trajectory(pupil_coordinates, led_coordinates, camera_coordinates, led_angles[LED], offset=offsets[0])

    elevations = trajectory[:,0]
    azimuths = trajectory[:,1]

    trajectories[LED] = trajectory

# %% 
trajectories['se'].shape
plt.plot(trajectories['se'][:,0])
plt.show()

# %% 
trajectories.keys()

# %%
trajectory = np.zeros((1000,2)) 
trajectory[:,0] = np.zeros_like(trajectory[:,0]) 
trajectory[:,1] = np.linspace(0, 2*pi, trajectory.shape[0])

plt.plot(np.linspace(0, 2*pi, trajectory.shape[0]), trajectory[:,1], label='camera azimuth (el=0)', color='k')

table_trajectory = geezer.basis_transform_to_table(trajectory, basis)

plt.plot(np.linspace(0, 2*pi, trajectory.shape[0]), table_trajectory[:,0], label='table elevation', color='c')
plt.axvline(x=pi/2, color='r')
plt.axhline(y=0, color='r')
plt.title('camera elevation at 0, azimuth at pi/2 should have table elevation at ')
plt.legend()
plt.show()

# %%
geezer.plot_cam_basis(centered_geometry, basis, led_angles)
# %%
table_trajectories = {}
for LED in ['se', 'sw']:
    trajectory = geezer.sanitize_trajectory(trajectories[LED], [10, 30], 1.5, 10)
    table_trajectory = geezer.basis_transform_to_table(trajectory, basis)

    relative_trajectory = np.zeros_like(table_trajectory)
    relative_trajectory[:,0] = geezer.get_relative_angle(table_trajectory[:,0])
    relative_trajectory[:,1] = geezer.get_relative_angle(table_trajectory[:,1])

    table_trajectories[LED] =relative_trajectory 

    fig, ax = plt.subplots(6, sharex=True)
    ax[0].set_title('camera elevation')
    ax[0].plot(trajectory[:,0], label='camera', color='k')
    ax[1].set_title('table elevation')
    ax[1].plot(table_trajectory[:,0], label='table', color='c')
    ax[2].set_title('camera azimuth')
    ax[2].plot(trajectory[:,1], color='k')
    ax[3].plot(table_trajectory[:,1], color='c')
    ax[3].set_title('table azimuth')

    ax[4].plot(geezer.get_relative_angle(table_trajectory[:,1]))
    ax[5].plot(geezer.get_relative_angle(table_trajectory[:,0]))

    plt.legend()
    plt.show()



# %% 
# plt.plot(file['interp_trajectories']['sw'][:,0])
# plt.plot(table_trajectories['sw'][:,0])
# plt.plot(table_trajectories['se'][:,0])
# plt.show()

# %% 
file.close()

with h5.File(h5_filepath, 'a') as file:
    try:
        del file['raw_trajectories']
        del file['interp_table_trajectories']
    except:
        pass
    file.create_dataset('raw_trajectories/se', data=trajectories['se'])
    file.create_dataset('raw_trajectories/sw', data=trajectories['sw'])

    file.create_dataset('interp_table_trajectories/se', data=table_trajectories['se'])
    file.create_dataset('interp_table_trajectories/sw', data=table_trajectories['sw'])

# %% 

# %%
# fig,ax = plt.subplots(2,2, sharex=True)
# for led in ['se', 'sw']:
#     trajectories[led] = []
#     for frame in range(num_frames):
#         pup_co = file['pup_co'][frame]
#         fiducial_co = file['fiducial_coordinates'][led][frame]
#         camera_co = file['fiducial_coordinates']['sw'][frame] + cam_offset

#         _e, _a= geezer.calculate_gaze_angles(pup_co, fiducial_co, camera_co, led_angles[led], offset=best_offset)

#         trajectories[led].append([_e, _a])

#     trajectories[led] = np.array(trajectories[led])

#     if led == 'se':
#         color = 'r'
#     else:
#         color = 'k'

#     ax[0,0].plot(np.rad2deg(trajectories[led][:,0]), color)
#     ax[1,0].plot(np.rad2deg(trajectories[led][:,1]), color)
#     ax[0,1].plot(np.rad2deg(trajectories[led][:,0] - trajectories[led][0,0]), color)
#     ax[1,1].plot(np.rad2deg(trajectories[led][:,1] - trajectories[led][0,1]), color)
# plt.show()


# file.close()
# with h5.File('/home/dennis/Data/data_1/GolDRoger/goldroger_geezer_round_2.h5', 'a') as file:
#     file.create_dataset('new_trajectories/se', data=trajectories['se'])
#     file.create_dataset('new_trajectories/sw', data=trajectories['sw'])


# # Which one should we use for the camera offset - test by using it to predict others and taking the best
