# import coltrane
# import tqdm

# import numpy as np
# import matplotlib.pyplot as plt
# import h5py as h5

# leds = ['cam','ne','nw','se','sw']
# with h5.File('/home/dennis/July6_2023_geezer.h5','r') as f:
#     print(f.keys())
#     frame_idxs = f['frame_idxs'][:]  
#     sidx = np.argsort(frame_idxs)
#     frame_idxs = frame_idxs[sidx]
#     plt.plot(np.diff(frame_idxs))
#     plt.show()

#     led_co = {}
#     pup_co = f['pup_co'][:][sidx]
#     for i,(k,v) in enumerate(f['fids_co'].items()):
#         print(k)
#         print(v.shape)
#         led_co[leds[i]] = v[:][sidx]


    
# def euclidean_distance(x1, x2):
#     # x1 and x2 are a n by 2 arrays 
#     return np.sqrt(np.sum((x1-x2)**2, axis=1))
# # %%

# coordinates = coltrane.load_measurements('/home/dennis/Code/geezer/configs/d222_calibration_jbm109.json')
# centered = coltrane.center_coordinates(coordinates)
# basis = coltrane.get_basis(centered)

# # simple MLE estimate
# # %%
# cam_preds = {}
# for l in ['ne','nw','se','sw']:
#     dxdy = led_co['cam']-led_co[l]
#     dxdy = np.mean(dxdy,axis=0)
#     cam_preds[l] = dxdy

# for l in ['ne','nw','se','sw']:
#     c = led_co[l][:] + cam_preds[l]
#     plt.plot(c[:,0])
# plt.show()
# # %%    


# #     plt.plot(led_co['cam'][:,1]-led_co[l][:,1])
# #     plt.axhline(y=(np.mean(led_co['cam'][:,1]-led_co[l][:,1])), color='r', lw=2)
# #     plt.show()
     
# leds = ['ne','se','sw','nw']
# thetas = {}
# phis = {}
# for led in leds:
#     el, az =  coltrane.get_led_angle(centered[led], basis)

#     thetas[led] =[] 
#     phis[led] =[] 
    
#     num_frames = frame_idxs.shape[0]
#     for frame in tqdm.tqdm(range(num_frames)):
#         idx = frame_idxs[frame]
#         led_pix = led_co[led][idx]
#         pup_pix = pup_co[idx]
#         cam_pix = led_co[led][idx] + cam_preds[led]
#         cam_pix[1] -= 10

#         e,a = coltrane.calc_gaze_angle(pup_pix, led_pix, cam_pix, [el, az])
#         t,p = coltrane.ray_trace(centered, a, e) 
#         thetas[led].append(np.pi/2 - t)
#         phis[led].append(p)

#     thetas[led] = np.array(thetas[led])
#     phis[led] = np.array(phis[led])

# import numpy as np

# # %%
# def nan_helper(y):
#     """Helper to handle indices and logical indices of NaNs.

#     Input:
#         - y, 1d numpy array with possible NaNs
#     Output:
#         - nans, logical indices of NaNs
#         - index, a function, with signature indices= index(logical_indices),
#           to convert logical indices of NaNs to 'equivalent' indices
#     Example:
#         >>> # linear interpolation of NaNs
#         >>> nans, x= nan_helper(y)
#         >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
#     """

#     return np.isnan(y), lambda z: z.nonzero()[0]


# # %%
# def euclidean_distance(x1, x2):
#     # x1 and x2 are a n by 2 arrays 
#     return np.sqrt(np.sum((x1-x2)**2, axis=1))

# e = euclidean_distance(led_co['ne'],led_co['sw'])
# e = np.abs(e - e[0])
# e = e >50 
# ee = euclidean_distance(led_co['se'],led_co['nw'])
# ee = np.abs(ee - ee[0])
# ee = ee >50 
# # plt.plot(e)
# # plt.show()
# # se = (thetas['se'])
# cleaned_up_phis = {}
# cleaned_up_thetas = {}
# for led in ['se','sw', 'ne', 'nw']:
#     sw = phis[led] 
#     sw = np.rad2deg(sw)
#     sw = sw - sw[20000]
#     # plt.plot(sw,'r')
#     # plt.show()

#     sw[e] = np.nan
#     sw[ee] = np.nan

#     # w = np.diff(sw) > 3 
#     # w = np.concatenate([w,[False]])
#     # sw[w] = np.nan
#     # y = sw
#     sw[e] = np.nan
#     sw[ee] = np.nan
#     nans, x = nan_helper(sw)
#     # # interpolate with scipy

#     # w = np.diff(sw) > 5
#     # sw[:-1][w] = np.nan
#     # nans, x = nan_helper(sw)
#     sw[nans]= np.interp(x(nans), x(~nans), sw[~nans])
#     # plt.plot(sw,'r')
#     # plt.show()

#     w = np.abs(np.diff(sw)) > 1 
#     sw[1:][w] = np.nan

#     sw[sw > 20] = np.nan
#     sw[sw < -20] = np.nan
#     nans, x = nan_helper(sw)
#     sw[nans]= np.interp(x(nans), x(~nans), sw[~nans])
#     # plt.plot(sw, 'k', lw=0.5)
#     # plt.show()
#     cleaned_up_phis[led] = sw

#     sw = thetas[led] 
#     sw = np.rad2deg(sw)
#     sw = sw - sw[20000]
#     # plt.plot(sw,'r')
#     # plt.show()

#     sw[e] = np.nan
#     sw[ee] = np.nan
    

#     # w = np.diff(sw) > 3 
#     # w = np.concatenate([w,[False]])
#     # sw[w] = np.nan
#     # y = sw
#     sw[e] = np.nan
#     sw[ee] = np.nan
import coltrane
import tqdm

import numpy as np
import matplotlib.pyplot as plt
import h5py as h5

leds = ['cam','ne','nw','se','sw']
with h5.File('/home/dennis/July6_2023_geezer.h5', 'r') as f:
    print(f.keys())
    frame_idxs = f['frame_idxs'][:]  
    sidx = np.argsort(frame_idxs)
    frame_idxs = frame_idxs[sidx]
    plt.plot(np.diff(frame_idxs))
    plt.show()

    led_co = {}
    pup_co = f['pup_co'][:][sidx]
    for i,(k,v) in enumerate(f['fids_co'].items()):
        print(k)
        print(v.shape)
        led_co[leds[i]] = v[:][sidx]
# %%

    
def euclidean_distance(x1, x2):
    # x1 and x2 are a n by 2 arrays 
    return np.sqrt(np.sum((x1-x2)**2, axis=1))

coordinates = coltrane.load_measurements('/home/dennis/Code/geezer/configs/d222_calibration_jbm109.json')
centered = coltrane.center_coordinates(coordinates)
basis = coltrane.get_basis(centered)

# simple MLE estimate
cam_preds = {}
for l in ['ne','nw','se','sw']:
    dxdy = led_co['cam']-led_co[l]
    dxdy = np.mean(dxdy,axis=0)
    cam_preds[l] = dxdy

for l in ['ne','nw','se','sw']:
    c = led_co[l][:] + cam_preds[l]
    plt.plot(c[:,0])
plt.show()
    


#     plt.plot(led_co['cam'][:,1]-led_co[l][:,1])
#     plt.axhline(y=(np.mean(led_co['cam'][:,1]-led_co[l][:,1])), color='r', lw=2)
#     plt.show()
     
leds = ['ne','se','sw','nw']
thetas = {}
phis = {}
for led in leds:
    el, az =  coltrane.get_led_angle(centered[led], basis)
    print(led, np.rad2deg(el), np.rad2deg(az))

    thetas[led] =[] 
    phis[led] =[] 
    
    num_frames = frame_idxs.shape[0]
    for frame in tqdm.tqdm(range(num_frames)):
        idx = frame_idxs[frame]
        led_pix = led_co[led][idx]
        pup_pix = pup_co[idx]
        cam_pix = led_co[led][idx] + cam_preds[led]
        cam_pix[1] -= 10

        e,a = coltrane.calc_gaze_angle(pup_pix, led_pix, cam_pix, [el, az])
        t,p = coltrane.ray_trace(centered, a, e) 
        thetas[led].append(np.pi/2 - t)
        phis[led].append(p)

    thetas[led] = np.array(thetas[led])
    phis[led] = np.array(phis[led])

import numpy as np

# %%
def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]


# %%
def euclidean_distance(x1, x2):
    # x1 and x2 are a n by 2 arrays 
    return np.sqrt(np.sum((x1-x2)**2, axis=1))

e = euclidean_distance(led_co['ne'],led_co['sw'])
e = np.abs(e - e[0])
e = e >50 
ee = euclidean_distance(led_co['se'],led_co['nw'])
ee = np.abs(ee - ee[0])
ee = ee >50 
# plt.plot(e)
# plt.show()
# se = (thetas['se'])
cleaned_up_phis = {}
cleaned_up_thetas = {}
for led in ['se','sw', 'ne', 'nw']:
    sw = phis[led] 
    sw = np.rad2deg(sw)
    sw = sw - sw[20000]
    # plt.plot(sw,'r')
    # plt.show()

    sw[e] = np.nan
    sw[ee] = np.nan

    # w = np.diff(sw) > 3 
    # w = np.concatenate([w,[False]])
    # sw[w] = np.nan
    # y = sw
    sw[e] = np.nan
    sw[ee] = np.nan
    nans, x = nan_helper(sw)
    # # interpolate with scipy

    # w = np.diff(sw) > 5
    # sw[:-1][w] = np.nan
    # nans, x = nan_helper(sw)
    sw[nans]= np.interp(x(nans), x(~nans), sw[~nans])
    # plt.plot(sw,'r')
    # plt.show()

    w = np.abs(np.diff(sw)) > 1 
    sw[1:][w] = np.nan

    sw[sw > 20] = np.nan
    sw[sw < -20] = np.nan
    nans, x = nan_helper(sw)
    sw[nans]= np.interp(x(nans), x(~nans), sw[~nans])
    # plt.plot(sw, 'k', lw=0.5)
    # plt.show()
    cleaned_up_phis[led] = sw

    sw = thetas[led] 
    sw = np.rad2deg(sw)
    sw = sw - sw[20000]
    # plt.plot(sw,'r')
    # plt.show()

    sw[e] = np.nan
    sw[ee] = np.nan

    # w = np.diff(sw) > 3 
    # w = np.concatenate([w,[False]])
    # sw[w] = np.nan
    # y = sw
    sw[e] = np.nan
    sw[ee] = np.nan
    nans, x = nan_helper(sw)
    # # interpolate with scipy

    # w = np.diff(sw) > 5
    # sw[:-1][w] = np.nan
    # nans, x = nan_helper(sw)\
    sw[nans]= np.interp(x(nans), x(~nans), sw[~nans])
    # plt.plot(sw,'r')
    # plt.show()

    w = np.abs(np.diff(sw)) > 1 
    sw[1:][w] = np.nan

    sw[sw > 5] = np.nan
    sw[sw < -5] = np.nan
    nans, x = nan_helper(sw)
    sw[nans]= np.interp(x(nans), x(~nans), sw[~nans])
    # plt.plot(sw, 'k', lw=0.5)
    # plt.show()
    cleaned_up_thetas[led] = sw


tax = np.arange(0,cleaned_up_thetas['se'].shape[0])/30
plt.plot(tax,cleaned_up_phis['se'], 'k', label='se')
plt.plot(tax,cleaned_up_phis['ne'], 'r', label='ne')
plt.plot(tax,cleaned_up_phis['sw']*-1, 'g', label='sw')
plt.xlabel('time (s)')
plt.ylabel('phi (deg)')
plt.legend()
plt.show()
import pickle
pickle.dump(cleaned_up_phis, open('j6_cleaned_up_phis.pkl','wb'))
pickle.dump(cleaned_up_thetas, open('j6_cleaned_up_thetas.pkl','wb'))



# %%
plt.rcParams['axes.axisbelow'] = False
plt.rcParams['xtick.bottom'] = False
plt.rcParams['xtick.labelbottom'] = False
plt.rcParams['ytick.left'] = False
plt.rcParams['ytick.labelleft'] = False
# plt.rcParams['axes.spines.left'] = False
# plt.rcParams['axes.spines.bottom'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.right'] = False

# # %% 
# import cv2
# print(sw.shape)
# cap = cv2.VideoCapture('/run/media/jerome/datum/cured/c007/June262023/cam_22248110.mp4')
# print(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     nans, x = nan_helper(sw)
#     # # interpolate with scipy

#     # w = np.diff(sw) > 5
#     # sw[:-1][w] = np.nan
#     # nans, x = nan_helper(sw)
#     sw[nans]= np.interp(x(nans), x(~nans), sw[~nans])
#     # plt.plot(sw,'r')
#     # plt.show()

#     w = np.abs(np.diff(sw)) > 1 
#     sw[1:][w] = np.nan

#     sw[sw > 5] = np.nan
#     sw[sw < -5] = np.nan
#     nans, x = nan_helper(sw)
#     sw[nans]= np.interp(x(nans), x(~nans), sw[~nans])
#     # plt.plot(sw, 'k', lw=0.5)
#     # plt.show()
#     cleaned_up_thetas[led] = sw


# plt.plot(cleaned_up_phis['se'], 'k')
# plt.plot(cleaned_up_phis['se'], 'r')
# plt.show()
# import pickle
# pickle.dump(cleaned_up_phis, open('cleaned_up_phis.pkl','wb'))
# pickle.dump(cleaned_up_thetas, open('cleaned_up_thetas.pkl','wb'))





# # # %% 
# # import cv2
# # print(sw.shape)
# # cap = cv2.VideoCapture('/run/media/jerome/datum/cured/c007/June262023/cam_22248110.mp4')
# # print(cap.get(cv2.CAP_PROP_FRAME_COUNT))
