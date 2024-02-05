%load_ext autoreload
%autoreload 2

# %%
import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append('/home/dennis/Code/geezer/src/')
import utils

geezer_file = '/home/dennis/Data/data_3/imu/imu_geezer_output.h5'

# %%
fid_co = {}
with h5.File(geezer_file,'r') as f:
    fiducials = f['fiducial_coordinates']
    reference_frame = 2717

    for k,v in fiducials.items():
        print(k, v.shape)
        plt.plot(v[reference_frame,0], 500-v[reference_frame,1], 'o', label=k)
    plt.legend()
    plt.show()

    for k,v in fiducials.items():
        fid_co[k] = v[:]
        print(k, v.shape)
        plt.plot(v[:,0] - v[0,0], label=k)
    plt.legend()

    plt.show()


    reference_positions = {}
    for k,v in fiducials.items():
        reference_positions[k] = v[reference_frame, :]
    
    # print(f['meta'].keys()
    # print(f.keys())
    # print(f.attrs.keys())
# %% 
cam_estimates = {}
leds_to_use = ['se', 'sw']
for led in leds_to_use:
    diff = reference_positions['cam'] - reference_positions[led]
    cam_estimates[led] = diff
se_estimate = reference_positions['se'] - reference_positions['sw']
sw_estimate = reference_positions['sw'] - reference_positions['se']

geometry_data = utils.load_geometry_json('/home/dennis/Code/geezer/geometries/jan_2024_geometry.json')

mapping = {'led':
        {'ne': 'ne',
         'se': 'se',
         'sw': 'sw'},
        'camera': {'cam': 'lens'},
        'pupil': 'headbar'}

# %%% center the geometry
def get_centered_geometry(geometry_data, mapping):
    observer = mapping['pupil']
    observer = geometry_data[observer]
    observer_geometry= np.array([observer['x'], observer['y'], observer['z']])


    temp = list(mapping['camera'].keys())
    assert len(temp) == 1
    camera = mapping['camera'][temp[0]]
    camera = geometry_data[camera]
    camera_geometry = np.array([camera['x'], camera['y'], camera['z']])

    leds = list(mapping['led'].keys())
    led_geometry = {}
    for led_name in leds:
        led = mapping['led'][led_name]
        led = geometry_data[led]
        led = [led['x'], led['y'], led['z']]
        led_geometry[led_name] = np.array(led)


    centered_coordinates = {}
    centered_coordinates['observer'] = observer_geometry - observer_geometry
    centered_coordinates['camera'] = camera_geometry - observer_geometry
    centered_coordinates['leds'] = {}

    for led_name in leds:
        centered_coordinates['leds'][led_name] = led_geometry[led_name] - observer_geometry

    return centered_coordinates

# # %% 
centered_geometry = get_centered_geometry(geometry_data, mapping)
basis = utils.get_basis(centered_geometry['camera'])

# # %% 
with h5.File(geezer_file, 'r') as f:
    frame_idxs = f['frame_idxs'][:]  
    sidx = np.argsort(frame_idxs)
    
    # Check if frame idxs are sequential
    if np.prod(np.diff(frame_idxs)) != 1:
        print('Error: Frame idxs are not sequential. Please check.')
        raise ValueError
    
    # led_pix_co = {}
    pup_pix_co = f['pup_co'][:][sidx]

    temp = list(mapping['camera'].keys())[0]
    cam_pix_co = f['fiducial_coordinates'][temp][:][sidx]

    for k,v in f['fiducial_coordinates'].items():
        if k in mapping['led'].keys():
            led_pix_co[k] = v[:][sidx]

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a-b)**2, axis=1))

def windowed_mean(x, window_size):
    return np.convolve(x, np.ones(window_size)/window_size, mode='same')

se_co = led_pix_co['se']
sw_co = led_pix_co['sw']
se_prediction = led_pix_co['sw']+se_estimate
sw_prediction = led_pix_co['se']+sw_estimate
residuals = euclidean_distance(se_co, se_prediction)

avg_se_x = windowed_mean(se_co[:,0], 10)
avg_se_y = windowed_mean(se_co[:,1], 10)
avg_sw_x = windowed_mean(sw_co[:,0], 10)
avg_sw_y = windowed_mean(sw_co[:,1], 10)

avg_se_co = np.column_stack((avg_se_x, avg_se_y))
avg_sw_co = np.column_stack((avg_sw_x, avg_sw_y))
print(avg_se_co.shape)


plt.hist(residuals, bins=100)
plt.show()

low = residuals < 40 
high = residuals > 110
mid = np.logical_and(residuals > 40, residuals < 100)

_low = np.where(low, se_co[:,0], np.nan)
_high = np.where(high, se_co[:,0], np.nan)
_mid = np.where(mid, se_co[:,0], np.nan)

fig, ax = plt.subplots(4,1, sharex=True)
ax[0].plot(se_co[:,0], 'y', lw=0.7)
ax[0].plot(_low, 'o', markersize=1.5, color='r')
ax[0].plot(_mid, 'o', markersize=1.5, color='c')
ax[0].plot(_high, 'o', markersize=1.5, color='k')

ax[1].plot(se_prediction[:,0], 'r', lw=0.7)
ax[1].plot(se_co[:,0], 'y', lw=0.7)
ax[1].plot(avg_se_co[:,0], 'k', lw=0.7)

ax[2].plot(sw_co[:,0], 'y', lw=0.7)
ax[2].plot(sw_prediction[:,0], 'r', lw=0.7)
ax[2].plot(avg_sw_co[:,0], 'k', lw=0.7)

lo = euclidean_distance(avg_se_co, se_co)[:]
ax[3].plot(lo, 'k', lw=0.7)

lolo = euclidean_distance(avg_sw_co, sw_co)[:]
ax[3].plot(lolo, 'r', lw=0.7)

lololo = euclidean_distance(avg_se_co, se_prediction)[:]
ax[3].plot(lololo, 'c', lw=0.7)

lolololo = euclidean_distance(avg_sw_co, sw_prediction)[:]
ax[3].plot(lolololo, 'y', lw=0.7)
plt.show()


# se_co = led_pix_co['se']
# se_prediction = led_pix_co['sw']+se_estimate

# sw_co = led_pix_co['sw']
# sw_prediction = led_pix_co['se']+sw_estimate

# def euclidean_distance(a, b):
#     return np.sqrt(np.sum((a-b)**2, axis=1))


# fig, ax = plt.subplots(2,1, sharex=True)

# ax[0].plot(euclidean_distance(se_co, se_prediction))
# ax[1].plot(se_co[:,0])
# ax[1].plot(se_prediction[:,0])
# plt.show()
# z = windowed_mean(se_co[:,0], 10)
# # plt.plot(z)
# fig, ax = plt.subplots(2,1, sharex=True)

# ax[0].plot(se_co[:,0])
# z = windowed_mean(se_co[:,0], 10)
# ax[0].plot(z)
# ax[1].plot(np.diff(z))
# plt.show()

# fig,ax = plt.subplots(2,2, sharex=True)
# ax[0,0].plot(se_co[:,0], 'k', label='se')
# ax[0,0].plot(se_prediction[:,0], 'r', label='se_prediction')
# ax[1,0].plot(euclidean_distance(se_co, se_prediction), 'k', label='se_residual')
# ax[0,1].plot(windowed_mean(se_co), 'k', label='sw')
# ax[0,1].plot(windowed_mean[:,0], 'r', label='sw_prediction')
# ax[1,2].plot(euclidean_distance(sw_co, sw_prediction), 'k', label='sw_residual')

# plt.show()
# # se_residual = euclidean_distance(se_co, se_prediction)
# sw_residual = euclidean_distance(sw_co, sw_prediction)
# plt.plot(se_residual, label='se_residual')
# plt.plot(sw_residual, label='sw_residual')
# plt.legend()
# plt.show()
# fig, ax = plt.subplots(2,2, sharex=True)
# ax[0, 0].plot(led_pix_co['se'][:,0], 'k', label='se')
# ax[0, 0].plot(se_prediction[:,0], 'r', label='se_prediction')
# ax[0, 0].legend()
# ax[0, 0].set_title('se')

# ax[1, 0].plot(led_pix_co['sw'][:,0], 'k', label='sw')
# ax[1, 0].plot(sw_prediction[:,0], 'r', label='sw_prediction')
# ax[1, 0].legend()
# ax[1, 0].set_title('sw')

# se_residual = led_pix_co['se'] - se_prediction
# sw_residual = led_pix_co['sw'] - sw_prediction

# ax[0, 1].plot(se_residual[:,0], 'k', label='se')
# ax[0, 1].set_title('se_residual')

# ax[1, 1].plot(sw_residual[:,0], 'k', label='sw')
# ax[1, 1].set_title('sw_residual')
# plt.show()

