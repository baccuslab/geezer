import numpy as np
import math
import json

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

def process_frame(frame, pup, fids, pupil_params, fid_params):
    t = time.time()
    pf, pt, ff, ft = dogs(frame, pupil_params, fid_params)
    pup_xy = find_closest_centroid(pt, 100, pup)
    fid_xys = [find_closest_centroid(ft, 100, fid) for fid in fids]
    print(time.time() - t)

    return pup_xy, fid_xys

def dogs(og_frame, pupil_params, fid_params):
    print(pupil_params)
    exp, gs, gl, thresh = pupil_params

    gauss_small = gs
    gauss_large = gl

    dog = og_frame 
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


def get_basis(eye_co):

    _z = eye_co['camera']
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

def calc_gaze_angle(pupil_co, led_co, cam_co, led_angles):
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


