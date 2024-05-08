import numpy as np
import math
import json

import matplotlib.pyplot as plt

import threading
import sys 
import cv2
import h5py as h5
import os

import time
from multiprocessing import Pool

import numpy as np
import skimage.filters
import numpy as np
from scipy.spatial.distance import cdist
from skimage.feature import blob_log
from skimage import measure
from scipy.spatial.distance import cdist
import IPython
from scipy.ndimage import rotate


def process_frame_fids(frame, fids, fid_params):
    ff, ft = dogs_fids(frame, fid_params)
    final_fid_xys = {} 
    for k,v in fids.items():
        final_fid_xys[k] = find_closest_centroid(ft, 100, v)
# fids_parameters =  {'exp': 1, 'small': 3, 'large': 11, 'thresh': 65}
    return final_fid_xys

def dogs_fids(og_frame, fid_params):
    exp, gs, gl, thresh = fid_params
    gauss_small = gs
    gauss_large = gl
    dog = og_frame 
    dog = dog - np.min(dog)
    dog = dog / np.max(dog)
    dog = dog ** exp
    dog = skimage.filters.gaussian(dog, sigma=gauss_small) - skimage.filters.gaussian(dog, sigma=gauss_large)
    dog = dog - np.min(dog)
    dog = dog / np.max(dog)
    dog = dog * 255
    fid_filtered = dog.astype(np.uint8)

    _, thresh = cv2.threshold(dog, thresh, 255, cv2.THRESH_BINARY)
    fid_thresh = thresh.astype(np.uint8)

    return fid_filtered, fid_thresh

def ground_truth_ellipse_adjustment(frame, pup_co, width, height, phi,fraction=0.01, scale=2):
    """
    Function: takes ellipse parameters, rotates the frame usign scipy.ndimage,
    adds the width and height parameters, and does some scaling and fractional thersholding
    to take a more realistic and better fit pupil major and minor axes.
    This is done by normalizing the image by the minimum pixel vaue of the image, and then taking all other values a certain
    fraction of the minimum value or below to be the included width or height parameters.
    Arguments:
    - frame: image frame of some shape (m,n,3)
    - pup_co: pupil coordinates
    - width: major axis of pupil
    - height: minor axis of pupileight[frame_idx]
    - scale: scale to increase width and height to find new adjusted width and height
    Returns:
    - masked frame with ellipse
    - adjusted hieght
    - adjusted width
    """
    x, y = pup_co
    ellipse_width = width
    ellipse_height = height
    angle = phi

    frame_rotated, new_pups = rot(frame, pup_co, angle)
    new_width = int(ellipse_width*scale)
    new_height = int(ellipse_height*scale)

    xmin = int(new_pups[0]-(new_width/2))
    xmax = int(new_pups[0]+(new_width/2))
    ymin = int(new_pups[1]-(new_height/2))
    ymax = int(new_pups[1]+(new_height/2))
    origxmin = int(new_pups[0]-(ellipse_width/2))
    origxmax = int(new_pups[0]+(ellipse_width/2))
    origymin = int(new_pups[1]-(ellipse_height/2))
    origymax = int(new_pups[1]+(ellipse_height/2))

    ytime = np.arange(ymin,ymax)
    ytime_norm = np.arange(ymin-ymin,ymax-ymin)
    origytime = np.arange(origymin,origymax)
    xtime = np.arange(xmin,xmax)
    # print(xtime)
    # print(ytime)
    # print(frame_rotated.shape)
    xtime_norm = np.arange(xmin-xmin,xmax-xmin)
    origxtime = np.arange(origxmin,origxmax)

    htime = np.zeros(ytime.shape)
    for i in range(origytime.shape[0]):
        htime[np.where(ytime == origytime[i])[0]] = 1
    wtime = np.zeros(xtime.shape)
    for k in range(origxtime.shape[0]):
        wtime[np.where(xtime == origxtime[k])[0]] = 1
    
    metric_y = np.min(frame_rotated[origytime,int(new_pups[0])])
    metric_frame_y = (frame_rotated[ytime,int(new_pups[0])] - metric_y) / metric_y
    htime_new = np.zeros(ytime.shape)
    htime_new = htime
    htime_new[np.where(metric_frame_y <= 0.01)[0]] = 1
    where_r = np.where(htime_new == 1)[0]
    where_r_diff = np.diff(where_r)
    # print(where_r_diff)
    where_r_diff_idx = np.where(where_r_diff > 1)[0]
    # print(where_r_diff_idx)
    for r in range(where_r_diff_idx.shape[0]):
        idx = where_r_diff_idx[r]
        htime_new[where_r[idx]:where_r[idx+1]] = 1

    metric_x = np.min(frame_rotated[int(new_pups[1]),origxtime])
    metric_frame_x = (frame_rotated[int(new_pups[1]),xtime] - metric_x) / metric_x
    wtime_new = np.zeros(xtime.shape)
    wtime_new = wtime
    wtime_new[np.where(metric_frame_x <= 0.01)[0]] = 1
    where_r = np.where(wtime_new == 1)[0]
    where_r_diff = np.diff(where_r)
    # print(where_r_diff)
    where_r_diff_idx = np.where(where_r_diff > 1)[0]
    # print(where_r_diff_idx)
    for r in range(where_r_diff_idx.shape[0]):
        idx = where_r_diff_idx[r]
        wtime_new[where_r[idx]:where_r[idx+1]] = 1

    adjusted_width = np.sum(wtime_new)
    adjusted_height = np.sum(htime_new)


    mask = np.zeros_like(frame)
    cv2.ellipse(mask, (int(x), int(y)), (int(adjusted_width), int(adjusted_height)), angle, 0, 360, (255, 255, 255), 1)
    result = cv2.addWeighted(frame, 1, mask, 0.5, 0)
    return result, adjusted_width, adjusted_height


def ground_truth_ellipse_adjustment_draw(frame, pup_co, width, height, phi,fraction=0.01, scale=2):
    """
    Function: takes ellipse parameters, rotates the frame usign scipy.ndimage,
    adds the width and height parameters, and does some scaling and fractional thersholding
    to take a more realistic and better fit pupil major and minor axes.
    This is done by normalizing the image by the minimum pixel vaue of the image, and then taking all other values a certain
    fraction of the minimum value or below to be the included width or height parameters.
    Arguments:
    - frame: image frame of some shape (m,n,3)
    - pup_co: pupil coordinates
    - width: major axis of pupil
    - height: minor axis of pupileight[frame_idx]
    - scale: scale to increase width and height to find new adjusted width and height
    Returns:
    - masked frame with ellipse
    - adjusted hieght
    - adjusted width
    """
    x, y = pup_co
    ellipse_width = width
    ellipse_height = height
    angle = phi

    frame_rotated, new_pups = rot(frame, pup_co, angle)
    new_width = int(ellipse_width*scale)
    new_height = int(ellipse_height*scale)

    xmin = int(new_pups[0]-(new_width/2))
    xmax = int(new_pups[0]+(new_width/2))
    ymin = int(new_pups[1]-(new_height/2))
    ymax = int(new_pups[1]+(new_height/2))
    origxmin = int(new_pups[0]-(ellipse_width/2))
    origxmax = int(new_pups[0]+(ellipse_width/2))
    origymin = int(new_pups[1]-(ellipse_height/2))
    origymax = int(new_pups[1]+(ellipse_height/2))

    ytime = np.arange(ymin,ymax)
    ytime_norm = np.arange(ymin-ymin,ymax-ymin)
    origytime = np.arange(origymin,origymax)
    xtime = np.arange(xmin,xmax)
    # print(xtime)
    # print(ytime)
    # print(frame_rotated.shape)
    xtime_norm = np.arange(xmin-xmin,xmax-xmin)
    origxtime = np.arange(origxmin,origxmax)

    htime = np.zeros(ytime.shape)
    for i in range(origytime.shape[0]):
        htime[np.where(ytime == origytime[i])[0]] = 1
    wtime = np.zeros(xtime.shape)
    for k in range(origxtime.shape[0]):
        wtime[np.where(xtime == origxtime[k])[0]] = 1
    
    metric_y = np.min(frame_rotated[origytime,int(new_pups[0])])
    metric_frame_y = (frame_rotated[ytime,int(new_pups[0])] - metric_y) / metric_y
    htime_new = np.zeros(ytime.shape)
    htime_new = htime
    htime_new[np.where(metric_frame_y <= fraction)[0]] = 1
    where_r = np.where(htime_new == 1)[0]
    where_r_diff = np.diff(where_r)
    # print(where_r_diff)
    where_r_diff_idx = np.where(where_r_diff > 1)[0]
    # print(where_r_diff_idx)
    for r in range(where_r_diff_idx.shape[0]):
        idx = where_r_diff_idx[r]
        htime_new[where_r[idx]:where_r[idx+1]] = 1

    metric_x = np.min(frame_rotated[int(new_pups[1]),origxtime])
    metric_frame_x = (frame_rotated[int(new_pups[1]),xtime] - metric_x) / metric_x
    wtime_new = np.zeros(xtime.shape)
    wtime_new = wtime
    wtime_new[np.where(metric_frame_x <= fraction)[0]] = 1
    where_r = np.where(wtime_new == 1)[0]
    where_r_diff = np.diff(where_r)
    # print(where_r_diff)
    where_r_diff_idx = np.where(where_r_diff > 1)[0]
    # print(where_r_diff_idx)
    for r in range(where_r_diff_idx.shape[0]):
        idx = where_r_diff_idx[r]
        wtime_new[where_r[idx]:where_r[idx+1]] = 1

    adjusted_width = np.sum(wtime_new)
    adjusted_height = np.sum(htime_new)


    mask1 = np.zeros_like(frame)
    cv2.ellipse(mask1, (int(x), int(y)), (int(adjusted_width), int(adjusted_height)), angle, 0, 360, (255, 255, 255), 1)
    mask2 = np.zeros_like(frame)
    cv2.ellipse(mask2, (int(x), int(y)), (int(ellipse_width), int(ellipse_height)), angle, 0, 360, (80, 80, 80), 1)
    result2 = cv2.addWeighted(frame, 1, mask1, 0.5, 0)
    result = cv2.addWeighted(result2, 1, mask2, 0.5, 0)
    return result, adjusted_width, adjusted_height




def rot(image, xy, angle):
    im_rot = rotate(image,angle) 
    org_center = (np.array(image.shape[:2][::-1])-1)/2.
    rot_center = (np.array(im_rot.shape[:2][::-1])-1)/2.
    org = xy-org_center
    a = np.deg2rad(angle)
    new = np.array([org[0]*np.cos(a) + org[1]*np.sin(a),
            -org[0]*np.sin(a) + org[1]*np.cos(a) ])
    return im_rot, new+rot_center


def euclidean_distance(a, b):
    return np.sqrt(np.sum((a-b)**2, axis=1))

def windowed_mean(x, window_size):
    return np.convolve(x, np.ones(window_size)/window_size, mode='same')

def sanitize_fiducial_coordinates(og_led_pix_co, og_cam_pix_co, pix_disp_thresh=15):
    '''
    This function takes the fiducial coordinates and removes any outliers, 
    filling in missing values (beyond some threshold) with the mean of the
    estimate from other LEDs.

    Use only the LEDs that you wish to use for this estimation in the dict

    Args:
        og_led_pix_co (dict): Dictionary of LED pixel coordinates
        og_cam_pix_co (list): List of camera pixel coordinates

    Returns:
        led_pix_co (dict): Dictionary of LED pixel coordinates
        cam_pix_co (list): List of camera pixel coordinates
    '''

    led_pix_co = {}
    cam_pix_co = {}
    
    # Set all times when the LED displacement is greater than some threshold to NaN
    for k, v in og_led_pix_co.items():
        xs = v[:,0]
        ys = v[:,1]

        median_x = np.median(v[:,0])
        median_y = np.median(v[:,1])

        a = np.where(xs > median_x+pix_disp_thresh)[0]
        b = np.where(xs < median_x-pix_disp_thresh)[0]
        c = np.where(ys > median_y+pix_disp_thresh)[0]
        d = np.where(ys < median_y-pix_disp_thresh)[0]

        print(a)
        
        v[a,:] = np.nan
        v[b,:] = np.nan
        v[c,:] = np.nan
        v[d,:] = np.nan

        led_pix_co[k] = v



    # Same with camera
    xs = og_cam_pix_co[:,0]
    ys = og_cam_pix_co[:,1]

    median_x = np.median(og_cam_pix_co[:,0])
    median_y = np.median(og_cam_pix_co[:,1])

    a = np.where(xs > median_x+pix_disp_thresh)[0]
    b = np.where(xs < median_x-pix_disp_thresh)[0]
    c = np.where(ys > median_y+pix_disp_thresh)[0]
    d = np.where(ys < median_y-pix_disp_thresh)[0]

    og_cam_pix_co[a,:] = np.nan
    og_cam_pix_co[b,:] = np.nan
    og_cam_pix_co[c,:] = np.nan
    og_cam_pix_co[d,:] = np.nan

    cam_pix_co = og_cam_pix_co
    
    # Find times when all LEDs are not NaN
    # Could do this pairwise, probably should for cam
    prod = np.ones_like(cam_pix_co)

    for k, v in led_pix_co.items():
        prod = np.multiply(prod, ~np.isnan(v))

    prod = np.multiply(prod, ~np.isnan(cam_pix_co))
    all_fiducial_idxs = np.where(prod == True)[0]
    
    cam_offsets = {}
    full_cam_preds = {}
    filled_cam_preds = {}

    for led in led_pix_co.keys():
        cam_offsets[led] = cam_pix_co[all_fiducial_idxs, :] - led_pix_co[led][all_fiducial_idxs,:] 
        
        full_cam_preds[led]= led_pix_co[led][:,:] + np.median(cam_offsets[led], axis=0)
        
        filled_cam_preds[led] = np.where(np.isnan(cam_pix_co), full_cam_preds[led], cam_pix_co)
    
    # Average
    cam_pix_co = np.mean([filled_cam_preds['sw'], filled_cam_preds['se']], axis=0)


    # Mechanism for "filling in" missing values using EM, the way we do for cam, but taking
    # into account multiple expectations and doing pairwise

    # Find times when blinks occur




    return led_pix_co, cam_pix_co

def find_closest_centroid(image, threshold_value, point):
    # Threshold the image
    thresholded_image = np.where(image > threshold_value, 255, 0)

    # Find connected components above the threshold
    labels = measure.label(thresholded_image)
    regions = measure.regionprops(labels) 

    # Find the centroid coordinates of each connected component
    centroids = []
    for region in regions:
        centroids.append(region.centroid)

    # Find the closest centroid to the given point
    closest_centroid = None
    closest_distance = None
    for centroid in centroids:
        distance = np.sqrt((centroid[0] - point[1])**2 + (centroid[1] - point[0])**2)
        if closest_centroid is None or distance < closest_distance:
            closest_centroid = centroid
            closest_distance = distance

    return (closest_centroid[1], closest_centroid[0])

def dogs(og_frame, pupil_params, fid_params):
    print(pupil_params)
    exp, gs, gl, thresh = pupil_params

    gauss_small = gs
    gauss_large = gl

    dog = og_frame
    #thresshold the large amplitude to for detecting the pupil 
    dog = dog - np.min(dog)
    dog = dog / np.max(dog)
    dog = np.max(dog) - dog
    dog = dog ** exp
    dog = skimage.filters.gaussian(dog, sigma=gauss_small) - skimage.filters.gaussian(dog, sigma=gauss_large)
    dog = dog - np.min(dog)
    dog = dog / np.max(dog)
    dog = dog * 255
    dog = dog.astype(np.uint8)

    pup_filtered = dog
    
    _, thresh = cv2.threshold(dog, thresh, 255, cv2.THRESH_BINARY)
    pup_thresh = thresh
    
    exp, gs, gl, thresh = fid_params
    gauss_small = gs
    gauss_large = gl
    dog = og_frame 
    dog = dog - np.min(dog)
    dog = dog / np.max(dog)
    dog = dog ** exp
    dog = skimage.filters.gaussian(dog, sigma=gauss_small) - skimage.filters.gaussian(dog, sigma=gauss_large)
    dog = dog - np.min(dog)
    dog = dog / np.max(dog)
    dog = dog * 255
    fid_filtered = dog.astype(np.uint8)

    _, thresh = cv2.threshold(dog, thresh, 255, cv2.THRESH_BINARY)
    fid_thresh = thresh.astype(np.uint8)

    return pup_filtered, pup_thresh, fid_filtered, fid_thresh

def load_geometry_json(json_path):
    objects = {}
    with open(json_path, 'r') as f:
        data = json.load(f)
    for di in data:
        objects[di['name']] = {}
        objects[di['name']]['x'] = di['x']
        objects[di['name']]['y'] = di['y']
        objects[di['name']]['z'] = di['z']
    return objects


def get_basis(centered_camera_coordinates):
    _z = centered_camera_coordinates
    _x = np.cross([_z[0], _z[1], 0], [0, 0, -1])
    _x[2] = 0
    _y = np.cross(_x, _z)
    _y *= -1
    
    _x_norm = np.linalg.norm(_x)
    _y_norm = np.linalg.norm(_y)
    _z_norm = np.linalg.norm(_z)

    _z_unit = np.array(_z / _z_norm)
    _x_unit = np.array(_x / _x_norm)
    _y_unit = np.array(_y / _y_norm)

    v_basis = np.array([_x_unit, _y_unit, _z_unit]).T

    return v_basis
def get_led_angle(led_co, basis):
    tx,ty,tz = np.linalg.inv(basis) @ led_co.T
    el = np.arctan(ty / np.sqrt(tx**2 + tz**2))
    az = np.arctan(tx / tz)
    
    return el, az

def calc_gaze_angle(pupil_co, led_co, cam_co, led_angles, offset=[0,0]):
    px = pupil_co[0]
    py = pupil_co[1]

    cx = cam_co[0]
    cy = cam_co[1]

    fx = led_co[0]
    fy = led_co[1]
    led_el = led_angles[0]
    led_az = led_angles[1]

    pd = py - cy
    fd = fy - cy
    el = np.arcsin((pd/fd)*(np.sin(led_el/2)))

    pd = px - cx
    fd = fx - cx
    numerator = pd/np.cos(el)
    demoninator = fd / np.cos(np.sin(led_el)/2)
    az = np.arcsin((numerator/demoninator)*(np.sin(led_az/2)))
    return el,az
def ray_trace(coor,azimuth,elevation):
    cam_co = coor["camera"]
    eye_co = coor["observer"]
    # mon_co = coor['left_mon']


    _z = cam_co - eye_co
    _x = np.cross([_z[0], _z[1], 0], [0, 0, -1])
    _x[2] = 0
    _y = np.cross(_x, _z)*-1  
    _x_norm = np.linalg.norm(_x)
    _y_norm = np.linalg.norm(_y)
    _z_norm = np.linalg.norm(_z)
    _z_unit = np.array(_z / _z_norm)
    _x_unit = np.array(_x / _x_norm)
    _y_unit = np.array(_y / _y_norm)

    # basis transformation matrix from camera coordinates to table coordinates
    v_basis = np.array([_x_unit, _y_unit, _z_unit]).T
    # P: cartesian camera coordinates of pupil
    P = np.array(([np.cos(elevation)*np.sin(azimuth),np.sin(elevation),np.cos(elevation)*np.cos(azimuth)]))*10
    P_table = np.dot(v_basis,P)+eye_co

    #convert cartesian to spherical coordinates
    # r = np.linalg.norm(P_table)

    # theta = np.arccos(P_table[2]/r)

    # phi = np.arctan(P_table[1]/P_table[0])
    theta, phi = cartesian_to_spherical(P_table)
    return theta, phi

def cartesian_to_spherical(xyz):
    x = xyz[0]
    y = xyz[1]
    z = xyz[2]
    r = math.sqrt(x**2 + y**2 + z**2)
    theta = math.acos(z / r)
    phi = math.atan2(y, x)
    
    return theta, phi

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


def process_frame(frame, pup, fids, pupil_params, fid_params, ellipse=False):
    print(pupil_params)
    print('Pupil')
    print(fid_params)
    print('Fiducials')
    if ellipse:
        # THIS IS CURRENTLY BRROKEN
        pf, pt, ff, ft = dogs(frame, pupil_params, fid_params)
        center = find_closest_centroid(pt, 100, pup)

        pup_loc = pupil_locator(pt, center) #finds edges
        pxy, width, height, phi = fit(pup_loc) #fits ellipse

        final_fid_xys = {} 
        for k,v in fids.items():
            final_fid_xys[k] = find_closest_centroid(ft, 100, v)

        return pxy, final_fid_xys, width, height, phi
    else:
        pf, pt, ff, ft = dogs(frame, pupil_params, fid_params)
        pxy= find_closest_centroid(pt, 100, pup)
        final_fid_xys = {} 
        for k,v in fids.items():
            final_fid_xys[k] = find_closest_centroid(ft, 100, v)

        
        width = None
        height = None
        phi = None

        return pxy, final_fid_xys, width, height, phi

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

