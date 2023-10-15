import numpy as np
import h5py as h5
import matplotlib.pyplot as plt
from utils import cartesian_to_spherical
import utils

map = {}
map['observer'] = 'headbar'
map['camera'] = 'camera'
map['leds'] = ['ne', 'nw', 'se', 'sw']
geometry_filename = '/home/dennis/july2023.json'
geometry_data = utils.load_geometry_json(geometry_filename)

rig_geometry = {}
rig_geometry['leds'] = []

observer = map['observer']  
observer = geometry_data[observer]
observer = [observer['x'], observer['y'], observer['z']]

camera = map['camera']  
camera = geometry_data[camera]
camera = [camera['x'], camera['y'], camera['z']]

leds = map['leds']
for led in leds:
    led = geometry_data[led]
    led = [led['x'], led['y'], led['z']]
    rig_geometry['leds'].append(led)

rig_geometry['observer'] = observer
rig_geometry['camera'] = camera

centered_rig_geometry = {}
centered_rig_geometry['leds'] = []

# %%

for led in rig_geometry['leds']:
    centered_rig_geometry['leds'].append(np.array(led) - np.array(rig_geometry['observer']))
centered_rig_geometry['observer'] = np.array(rig_geometry['observer']) - np.array(rig_geometry['observer'])
centered_rig_geometry['camera'] = np.array(rig_geometry['camera']) - np.array(rig_geometry['observer'])
# %%

basis = utils.get_basis(centered_rig_geometry)

print(basis)
# %%
coord_filename = '/home/dennis/July6_2023_geezer.h5'
with h5.File(coord_filename, 'r') as f:
    frame_idxs = f['frame_idxs'][:]  
    sidx = np.argsort(frame_idxs)
    frame_idxs = frame_idxs[sidx]
    
    led_co = [] 
    pup_co = f['pup_co'][:][sidx]
    for i,(k,v) in enumerate(f['fids_co'].items()):
        if i==0:
            cam_co = v[:][sidx]
        else:
            led_co.append(v[:][sidx])
# %%

cam_preds = {}
for led_idx in range(len(led_co)):
    dxdy = cam_co-led_co[led_idx]
    dxdy = np.mean(dxdy,axis=0)
    cam_preds[led_idx] = dxdy

fig, ax = plt.subplots(1)
ax.plot(led_co[0][0,0], led_co[0][0,1],'ro')
ax.plot(led_co[1][0,0], led_co[1][0,1],'go')
ax.plot(led_co[2][0,0], led_co[2][0,1],'bo')
ax.plot(led_co[3][0,0], led_co[3][0,1],'yo')
ax.plot(cam_co[0,0], cam_co[0,1],'ko')
plt.show()
# %%

for l in range(len(led_co)):
    c = led_co[l][:] + cam_preds[l]
    plt.plot(c[:,0])
plt.show()
plt.close()

ts = []
ps = []
for i in range(4):
    el, az =  utils.get_led_angle(centered_rig_geometry['leds'][i], basis)
    num_frames = frame_idxs.shape[0]
    _thetas = []
    _phis = []
    for frame in tqdm.tqdm(range(num_frames)):
        idx = frame_idxs[frame]
        led_pix = led_co[i][idx]
        pup_pix = pup_co[idx]
        cam_pix = led_co[i][idx] + cam_preds[i]
        cam_pix[1] -= 10

        e,a = utils.calc_gaze_angle(pup_pix, led_pix, cam_pix, [el, az])
        t,p = utils.ray_trace(centered_rig_geometry, a, e) 
        _thetas.append(np.pi/2 - t)
        _phis.append(p)

    ts.append(_thetas)
    ps.append(_phis)


