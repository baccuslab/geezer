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

