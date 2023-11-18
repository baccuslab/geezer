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
import IPython


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

angular_iter = 24
angular_range = np.arange(angular_iter, dtype=np.int8)
point_source = np.zeros(angular_iter, dtype=np.float64)
step_list_source = np.zeros(angular_iter, dtype=np.int8)

diagonal_size = 2**10

step_size = np.deg2rad(360 / angular_iter)
limit = np.arange(250)  # max size of shape; normalize qqqq
cos_sin_steps = np.array([(np.cos(i * step_size), np.sin(i * step_size)) for i in angular_range], dtype=np.float64)

kernel = np.ones((1, 1), np.uint8)

main_diagonal = np.eye(diagonal_size, diagonal_size, dtype=bool)

half_diagonal = np.full((diagonal_size, diagonal_size), False, dtype=bool)
fourth_diagonal = half_diagonal.copy()
third_diagonal = half_diagonal.copy()
onefourth = 1/4
onethird = 1/3

invhalf_diagonal = half_diagonal.copy()
invfourth_diagonal = half_diagonal.copy()
invthird_diagonal = half_diagonal.copy()

for i, _ in enumerate(half_diagonal):
    half_diagonal[int(i/2), i] = True
    fourth_diagonal[int(i/4), i] = True
    third_diagonal[int(i/3), i] = True

    invhalf_diagonal[i, int(i/2)] = True
    invfourth_diagonal[i, int(i/4)] = True
    invthird_diagonal[i, int(i/3)] = True


rr_stock = np.zeros((32), dtype=np.float64)

rr_2d = np.zeros((32, 2), dtype=np.float64)
rr_2d_cr = np.zeros((4, 2), dtype=np.float64)

rx_multiply = np.ones((32), dtype=np.float64)
ry_multiply = rx_multiply.copy()

crop_stock = np.zeros((32), dtype=int)
crop_stock_cr = np.zeros((4), dtype=int)
center_shape = (2, 31)


onehalf_ry_add = [8,10,12,14]
onehalf_rx_add = [8,11,12,15]
onehalf_rx_subtract = [9,10,13,14]
onehalf_ry_subtract = [9,11,13,15]
onehalf_ry_multiplier = [8,9,10,11]
onehalf_rx_multiplier = [12,13,14,15]


onefourth_ry_add = [16,19,20,21]
onefourth_rx_add = [16,17,20,23]
onefourth_rx_subtract = [18,19,21,22]
onefourth_ry_subtract = [17,18,22,23]
onefourth_ry_multiplier = [16,17,18,19]
onefourth_rx_multiplier = [20,21,22,23]


onethird_ry_add = [24,25,28,29]
onethird_rx_add = [24,27,28,31]
onethird_rx_subtract = [25,26,29,30]
onethird_ry_subtract = [26,27,30,31]
onethird_ry_multiplier = [24,25,26,27]
onethird_rx_multiplier = [28,29,30,31]


rx_multiplied = np.array(np.concatenate((onehalf_rx_multiplier, onefourth_rx_multiplier, onethird_rx_multiplier)), dtype=int)
ry_multiplied = np.array(np.concatenate((onehalf_ry_multiplier, onefourth_ry_multiplier, onethird_ry_multiplier)), dtype=int)
ones_ = np.ones(4, dtype=np.float64)
rx_multiply = np.array(np.concatenate((ones_ * .5, ones_ * onefourth, ones_*onethird)))

ry_multiply = np.array(np.concatenate((ones_ * .5, ones_ * onefourth, ones_*onethird)))

ry_add = np.array(np.concatenate(([0, 2, 4],onehalf_ry_add,onefourth_ry_add,onethird_ry_add)),dtype=int)
rx_add = np.array(np.concatenate(([1, 2, 5],onehalf_rx_add,onefourth_rx_add,onethird_rx_add)), dtype=int)


ry_subtract = np.array(np.concatenate(([3, 5, 7],onehalf_ry_subtract,onefourth_ry_subtract,onethird_ry_subtract )))


rx_subtract = np.array(np.concatenate(([3, 4, 6],onehalf_rx_subtract,onefourth_rx_subtract,onethird_rx_subtract)))


black = [35, 35, 35]

angle_dev = -22.5

min_radius = 2
max_radius = 200

threshold = len(crop_stock) * min_radius * 1.05

def pupil_locator(frame, center, min_radius=2, max_radius=400, threshold=threshold):
    # try:
    #     cent = np.round(center).astype(int)
    # except:
    #     return

    center = np.round(center).astype(int)

    canvas = np.array(frame, dtype=int)
    canvas[-1,:] = canvas[:, -1] = canvas[0,:] = canvas[:, 0] = 0

    r = np.zeros((32,2),dtype=np.float64)
    crop_list = np.zeros((32), dtype=int)

    canvas_ = canvas[center[1]:, center[0]:]
    canv_shape0, canv_shape1 = canvas_.shape
    crop_canvas = np.flip(canvas[:center[1], :center[0]])
    crop_canv_shape0, crop_canv_shape1 = crop_canvas.shape

    crop_canvas2 = np.fliplr(canvas[center[1]:, :center[0]])
    crop_canv2_shape0, crop_canv2_shape1 = crop_canvas2.shape

    crop_canvas3 = np.flipud(canvas[:center[1], center[0]:])
    crop_canv3_shape0, crop_canv3_shape1 = crop_canvas3.shape

    canvas2 = np.flip(canvas)

    canvas2 = np.flip(canvas)

    crop_list=np.array([
    np.argmax(canvas_[:, 0][min_radius:max_radius] == 0), np.argmax(canvas_[0, :][min_radius:max_radius] == 0), np.argmax(canvas_[main_diagonal[:canv_shape0, :canv_shape1]][min_radius:max_radius] == 0),
    np.argmax(crop_canvas[main_diagonal[:crop_canv_shape0, :crop_canv_shape1]][min_radius:max_radius] == 0), np.argmax(crop_canvas2[main_diagonal[:crop_canv2_shape0, :crop_canv2_shape1]][min_radius:max_radius] == 0),
    np.argmax(crop_canvas3[main_diagonal[:crop_canv3_shape0, :crop_canv3_shape1]][min_radius:max_radius] == 0), np.argmax(canvas2[-center[1], -center[0]:][min_radius:max_radius] == 0), np.argmax(canvas2[-center[1]:, -center[0]][min_radius:max_radius] == 0),
    np.argmax(canvas_[ half_diagonal[:canv_shape0, :canv_shape1]][min_radius:max_radius] == 0), np.argmax(crop_canvas[half_diagonal[:crop_canv_shape0, :crop_canv_shape1]][min_radius:max_radius] == 0), np.argmax(crop_canvas2[half_diagonal[:crop_canv2_shape0, :crop_canv2_shape1]][min_radius:max_radius] == 0),
    np.argmax(crop_canvas3[half_diagonal[:crop_canv3_shape0, :crop_canv3_shape1]][min_radius:max_radius] == 0), np.argmax(canvas_[invhalf_diagonal[:canv_shape0, :canv_shape1]][min_radius:max_radius] == 0),
    np.argmax(crop_canvas[invhalf_diagonal[:crop_canv_shape0, :crop_canv_shape1]][min_radius:max_radius] == 0), np.argmax(crop_canvas2[invhalf_diagonal[:crop_canv2_shape0, :crop_canv2_shape1]][min_radius:max_radius] == 0),
    np.argmax(crop_canvas3[invhalf_diagonal[:crop_canv3_shape0, :crop_canv3_shape1]][min_radius:max_radius] == 0), np.argmax(canvas_[fourth_diagonal[:canv_shape0, :canv_shape1]][min_radius:max_radius] == 0), np.argmax(crop_canvas3[fourth_diagonal[:crop_canv3_shape0, :crop_canv3_shape1]][min_radius:max_radius] == 0),
    np.argmax(crop_canvas[fourth_diagonal[:crop_canv_shape0, :crop_canv_shape1]][min_radius:max_radius] == 0), np.argmax(crop_canvas2[fourth_diagonal[:crop_canv2_shape0, :crop_canv2_shape1]][min_radius:max_radius] == 0), np.argmax(canvas_[invfourth_diagonal[:canv_shape0, :canv_shape1]][min_radius:max_radius] == 0),
    np.argmax(crop_canvas2[invfourth_diagonal[:crop_canv2_shape0, :crop_canv2_shape1]][min_radius:max_radius] == 0), np.argmax(crop_canvas[invfourth_diagonal[:crop_canv_shape0, :crop_canv_shape1]][min_radius:max_radius] == 0), np.argmax(crop_canvas3[invfourth_diagonal[:crop_canv3_shape0, :crop_canv3_shape1]][min_radius:max_radius] == 0),
    np.argmax(canvas_[third_diagonal[:canv_shape0, :canv_shape1]][min_radius:max_radius] == 0), np.argmax(crop_canvas2[third_diagonal[:crop_canv2_shape0, :crop_canv2_shape1]][min_radius:max_radius] == 0), np.argmax(crop_canvas[third_diagonal[:crop_canv_shape0, :crop_canv_shape1]][min_radius:max_radius] == 0),
    np.argmax(crop_canvas3[third_diagonal[:crop_canv3_shape0, :crop_canv3_shape1]][min_radius:max_radius] == 0), np.argmax(canvas_[invthird_diagonal[:canv_shape0, :canv_shape1]][min_radius:max_radius] == 0), np.argmax(crop_canvas2[invthird_diagonal[:crop_canv2_shape0, :crop_canv2_shape1]][min_radius:max_radius] == 0),
    np.argmax(crop_canvas[invthird_diagonal[:crop_canv_shape0, :crop_canv_shape1]][min_radius:max_radius] == 0), np.argmax(crop_canvas3[invthird_diagonal[:crop_canv3_shape0, :crop_canv3_shape1]][min_radius:max_radius] == 0)
    ], dtype=int) + min_radius



    if np.sum(crop_list) < threshold:
        #origin inside corneal reflection?
        offset_list = np.array([
        np.argmax(canvas_[:, 0][1:] == 255), np.argmax(canvas_[0, :][1:] == 255), np.argmax(canvas_[main_diagonal[:canv_shape0, :canv_shape1]][1:] == 255),
        np.argmax(crop_canvas[main_diagonal[:crop_canv_shape0, :crop_canv_shape1]][1:] == 255), np.argmax(crop_canvas2[main_diagonal[:crop_canv2_shape0, :crop_canv2_shape1]][1:] == 255),
        np.argmax(crop_canvas3[main_diagonal[:crop_canv3_shape0, :crop_canv3_shape1]][1:] == 255), np.argmax(canvas2[-center[1], -center[0]:][1:] == 255), np.argmax(canvas2[-center[1]:, -center[0]][1:] == 255),
        np.argmax(canvas_[ half_diagonal[:canv_shape0, :canv_shape1]][1:] == 255), np.argmax(crop_canvas[half_diagonal[:crop_canv_shape0, :crop_canv_shape1]][1:] == 255), np.argmax(crop_canvas2[half_diagonal[:crop_canv2_shape0, :crop_canv2_shape1]][1:] == 255),
        np.argmax(crop_canvas3[half_diagonal[:crop_canv3_shape0, :crop_canv3_shape1]][1:] == 255), np.argmax(canvas_[invhalf_diagonal[:canv_shape0, :canv_shape1]][1:] == 255),
        np.argmax(crop_canvas[invhalf_diagonal[:crop_canv_shape0, :crop_canv_shape1]][1:] == 255), np.argmax(crop_canvas2[invhalf_diagonal[:crop_canv2_shape0, :crop_canv2_shape1]][1:] == 255),
        np.argmax(crop_canvas3[invhalf_diagonal[:crop_canv3_shape0, :crop_canv3_shape1]][1:] == 255), np.argmax(canvas_[fourth_diagonal[:canv_shape0, :canv_shape1]][1:] == 255), np.argmax(crop_canvas3[fourth_diagonal[:crop_canv3_shape0, :crop_canv3_shape1]][1:] == 255),
        np.argmax(crop_canvas[fourth_diagonal[:crop_canv_shape0, :crop_canv_shape1]][1:] == 255), np.argmax(crop_canvas2[fourth_diagonal[:crop_canv2_shape0, :crop_canv2_shape1]][1:] == 255), np.argmax(canvas_[invfourth_diagonal[:canv_shape0, :canv_shape1]][1:] == 255),
        np.argmax(crop_canvas2[invfourth_diagonal[:crop_canv2_shape0, :crop_canv2_shape1]][1:] == 255), np.argmax(crop_canvas[invfourth_diagonal[:crop_canv_shape0, :crop_canv_shape1]][1:] == 255), np.argmax(crop_canvas3[invfourth_diagonal[:crop_canv3_shape0, :crop_canv3_shape1]][1:] == 255),
        np.argmax(canvas_[third_diagonal[:canv_shape0, :canv_shape1]][1:] == 255), np.argmax(crop_canvas2[third_diagonal[:crop_canv2_shape0, :crop_canv2_shape1]][1:] == 255), np.argmax(crop_canvas[third_diagonal[:crop_canv_shape0, :crop_canv_shape1]][1:] == 255),
        np.argmax(crop_canvas3[third_diagonal[:crop_canv3_shape0, :crop_canv3_shape1]][1:] == 255), np.argmax(canvas_[invthird_diagonal[:canv_shape0, :canv_shape1]][1:] == 255), np.argmax(crop_canvas2[invthird_diagonal[:crop_canv2_shape0, :crop_canv2_shape1]][1:] == 255),
        np.argmax(crop_canvas[invthird_diagonal[:crop_canv_shape0, :crop_canv_shape1]][1:] == 255), np.argmax(crop_canvas3[invthird_diagonal[:crop_canv3_shape0, :crop_canv3_shape1]][1:] == 255)
        ], dtype=int) + 1

        crop_list=np.array([
        np.argmax(canvas_[:, 0][offset_list[0]:] == 0), np.argmax(canvas_[0, :][offset_list[1]:] == 0), np.argmax(canvas_[main_diagonal[:canv_shape0, :canv_shape1]][offset_list[2]:] == 0),
        np.argmax(crop_canvas[main_diagonal[:crop_canv_shape0, :crop_canv_shape1]][offset_list[3]:] == 0), np.argmax(crop_canvas2[main_diagonal[:crop_canv2_shape0, :crop_canv2_shape1]][offset_list[4]:] == 0),
        np.argmax(crop_canvas3[main_diagonal[:crop_canv3_shape0, :crop_canv3_shape1]][offset_list[5]:] == 0), np.argmax(canvas2[-center[1], -center[0]:][offset_list[6]:] == 0), np.argmax(canvas2[-center[1]:, -center[0]][offset_list[7]:] == 0),
        np.argmax(canvas_[ half_diagonal[:canv_shape0, :canv_shape1]][offset_list[8]:] == 0), np.argmax(crop_canvas[half_diagonal[:crop_canv_shape0, :crop_canv_shape1]][offset_list[9]:] == 0), np.argmax(crop_canvas2[half_diagonal[:crop_canv2_shape0, :crop_canv2_shape1]][offset_list[10]:] == 0),
        np.argmax(crop_canvas3[half_diagonal[:crop_canv3_shape0, :crop_canv3_shape1]][offset_list[11]:] == 0), np.argmax(canvas_[invhalf_diagonal[:canv_shape0, :canv_shape1]][offset_list[12]:] == 0),
        np.argmax(crop_canvas[invhalf_diagonal[:crop_canv_shape0, :crop_canv_shape1]][offset_list[13]:] == 0), np.argmax(crop_canvas2[invhalf_diagonal[:crop_canv2_shape0, :crop_canv2_shape1]][offset_list[14]:] == 0),
        np.argmax(crop_canvas3[invhalf_diagonal[:crop_canv3_shape0, :crop_canv3_shape1]][offset_list[15]:] == 0), np.argmax(canvas_[fourth_diagonal[:canv_shape0, :canv_shape1]][offset_list[16]:] == 0), np.argmax(crop_canvas3[fourth_diagonal[:crop_canv3_shape0, :crop_canv3_shape1]][offset_list[17]:] == 0),
        np.argmax(crop_canvas[fourth_diagonal[:crop_canv_shape0, :crop_canv_shape1]][offset_list[18]:] == 0), np.argmax(crop_canvas2[fourth_diagonal[:crop_canv2_shape0, :crop_canv2_shape1]][offset_list[19]:] == 0), np.argmax(canvas_[invfourth_diagonal[:canv_shape0, :canv_shape1]][offset_list[20]:] == 0),
        np.argmax(crop_canvas2[invfourth_diagonal[:crop_canv2_shape0, :crop_canv2_shape1]][offset_list[21]:] == 0), np.argmax(crop_canvas[invfourth_diagonal[:crop_canv_shape0, :crop_canv_shape1]][offset_list[22]:] == 0), np.argmax(crop_canvas3[invfourth_diagonal[:crop_canv3_shape0, :crop_canv3_shape1]][offset_list[23]:] == 0),
        np.argmax(canvas_[third_diagonal[:canv_shape0, :canv_shape1]][offset_list[24]:] == 0), np.argmax(crop_canvas2[third_diagonal[:crop_canv2_shape0, :crop_canv2_shape1]][offset_list[25]:] == 0), np.argmax(crop_canvas[third_diagonal[:crop_canv_shape0, :crop_canv_shape1]][offset_list[26]:] == 0),
        np.argmax(crop_canvas3[third_diagonal[:crop_canv3_shape0, :crop_canv3_shape1]][offset_list[27]:] == 0), np.argmax(canvas_[invthird_diagonal[:canv_shape0, :canv_shape1]][offset_list[28]:] == 0), np.argmax(crop_canvas2[invthird_diagonal[:crop_canv2_shape0, :crop_canv2_shape1]][offset_list[29]:] == 0),
        np.argmax(crop_canvas[invthird_diagonal[:crop_canv_shape0, :crop_canv_shape1]][offset_list[30]:] == 0), np.argmax(crop_canvas3[invthird_diagonal[:crop_canv3_shape0, :crop_canv3_shape1]][offset_list[31]:] == 0)
        ], dtype=int) + offset_list

        if np.sum(crop_list) < threshold:
            raise IndexError("Lost track, do reset")

    r[:8,:] = center
    r[ry_add, 1] += crop_list[ry_add]
    r[rx_add, 0] += crop_list[rx_add]
    r[ry_subtract, 1] -= crop_list[ry_subtract] #
    r[rx_subtract, 0] -= crop_list[rx_subtract]
    r[rx_multiplied, 0] *= rx_multiply
    r[ry_multiplied, 1] *= ry_multiply
    r[8:,:] += center

    return cond(r, crop_list)

def cond(r, crop_list):

    ### return the r within the normalized difference ###
    
    dists = np.linalg.norm(np.mean(r, axis = 0, dtype=np.float64) - r, axis = 1)

    mean_ = np.mean(dists)
    std_ = np.std(dists)
    lower, upper = mean_ - std_, mean_ + std_ *.8
    cond_ = np.logical_and(np.greater_equal(dists, lower),np.less(dists, upper))

    return r[cond_]

def draw_ellpse_on_frame(frame, pxy, width, height, phi):
    # Assuming you have your image as a NumPy array
    # image = np.array(...)  # Replace with your actual NumPy array

    # Extract the ellipse parameters
    center_x, center_y = pxy  # Center coordinates
    ellipse_width = width
    ellipse_height = height
    angle = phi  # Angle of rotation

    # Create an empty mask to draw the ellipse on
    mask = np.zeros_like(frame)

    # Calculate the rotation matrix for the ellipse
    rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), angle, 1)

    # Draw the rotated ellipse on the mask
    cv2.ellipse(mask, (int(center_x), int(center_y)), (int(ellipse_width), int(ellipse_height)), angle, 0, 360, (255, 255, 255), 1)

    # Apply the mask to the original image
    result = cv2.addWeighted(frame, 1, mask, 0.5, 0)

    # Display or save the result
    cv2.imshow('Ellipse on Image', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save the result to a file
    cv2.imwrite('output_image.jpg', result)

params = None

def fit(r):

    # This function takes a list of values fed forward from the pupil locator function.

    """Least Squares fitting algor6ithm
    Theory taken from (*)
    Solving equation Sa=lCa. with a = |a b c d f g> and a1 = |a b c>
        a2 = |d f g>
    Args
    ----
    data (list:list:float): list of two lists containing the x and y data of the
        ellipse. of the form [[x1, x2, ..., xi],[y1, y2, ..., yi]]
    Returns
    ------
    coef (list): list of the coefficients describing an ellipse
        [a,b,c,d,f,g] corresponding to ax**2+2bxy+cy**2+2dx+2fy+g
    """

    # if r is not None:

    x, y = r[:,0], r[:,1]
    

    # Quadratic part of design matrix [eqn. 15] from (*)

    D1 = np.mat(np.vstack([x ** 2, x * y, y ** 2])).T
    # Linear part of design matrix [eqn. 16] from (*)
    D2 = np.mat(np.vstack([x, y, np.ones(len(x))])).T

    # forming scatter matrix [eqn. 17] from (*)
    S1 = D1.T * D1
    S2 = D1.T * D2
    S3 = D2.T * D2

    # Constraint matrix [eqn. 18]
    C1 = np.mat('0. 0. 2.; 0. -1. 0.; 2. 0. 0.')

    # Reduced scatter matrix [eqn. 29]
    M = C1.I * (S1 - S2 * S3.I * S2.T)

    # M*|a b c >=l|a b c >. Find eigenvalues and eigenvectors from this equation [eqn. 28]
    eval, evec = np.linalg.eig(M)

    # eigenvector must meet constraint 4ac - b^2 to be valid.
    cond = 4 * np.multiply(evec[0, :], evec[2, :]) - np.power(evec[1, :], 2)
    a1 = evec[:, np.nonzero(cond.A > 0)[1]]
    # self.fitscore=eval[np.nonzero(cond.A > 0)[1]]

    # |d f g> = -S3^(-1)*S2^(T)*|a b c> [eqn. 24]
    #a2 = -S3.I * S2.T * a1

    # eigenvectors |a b c d f g>
    coef = np.vstack([a1, -S3.I * S2.T * a1])


    """finds the important parameters of the fitted ellipse

    Theory taken form http://mathworld.wolfram
    Args
    -----
    coef (list): list of the coefficients describing an ellipse
        [a,b,c,d,f,g] corresponding to ax**2+2bxy+cy**2+2dx+2fy+g
    Returns
    _______
    center (List): of the form [x0, y0]
    width (float): major axis
    height (float): minor axis
    phi (float): rotation of major axis form the x-axis in radians
    """

    # eigenvectors are the coefficients of an ellipse in general form
    # a*x^2 + 2*b*x*y + c*y^2 + 2*d*x + 2*f*y + g = 0 [eqn. 15) from (**) or (***)
    a = coef[0, 0]
    b = coef[1, 0] / 2.
    c = coef[2, 0]
    d = coef[3, 0] / 2.
    f = coef[4, 0] / 2.
    g = coef[5, 0]

#    if (a - c) == 0:
#        return True

    # finding center of ellipse [eqn.19 and 20] from (**)
    af = a * f
    cd = c * d
    bd = b * d
    ac = a * c

    b_sq = b ** 2.
    z_ = (b_sq - ac)
    x0 = (cd - b * f) / z_#(b ** 2. - a * c)
    y0 = (af - bd) / z_#(b ** 2. - a * c)

    # Find the semi-axes lengths [eqn. 21 and 22] from (**)
    ac_subtr = a - c
    numerator = 2 * (af * f + cd * d + g * b_sq - 2 * bd * f - ac * g)
    denom = ac_subtr * np.sqrt(1 + 4 * b_sq / ac_subtr**2)
    denominator1, denominator2 = (np.array([-denom, denom], dtype=np.float64) - c - a) * z_

    width = np.sqrt(numerator / denominator1)
    height = np.sqrt(numerator / denominator2)

    phi = .5 * np.arctan((2. * b) / ac_subtr)
    params = ((x0, y0), width, height, np.rad2deg(phi) % 360)

    #self.center, self.width, self.height, self.angle = self.params
    return params

def distance(a, b):
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def center_loc(bin_img, center):

    circles = cv2.HoughCircles(bin_img, cv2.HOUGH_GRADIENT, 1, 10, param1=200, param2=17, minRadius=min_radius, maxRadius=max_radius)

    if circles is None:
        return
    else:
        smallest = -1
        current = -1

        for circle in circles[0, :]:
            #print(circle[:2])

            score = distance(circle[:2], center) + np.mean(bin_img[int(circle[1])-min_radius:int(circle[1])+min_radius, int(circle[0]-min_radius):int(circle[0]+min_radius)])

            bin_img[int(circle[1]), int(circle[0])] = 100
            ## To plot the image of where the pupil is being located##
            # cv2.imshow("kk", bin_img)
            # key = cv2.waitKey(0)

            # if key == ord('q') or key == 27:  # 27 is the ASCII code for the 'esc' key
            #     cv2.destroyAllWindows()
            ############################## 
            if smallest == -1:
                smallest = score
                current = circle[:2]
            elif score < smallest:
                smallest = score
                current = circle[:2]

        center = tuple(current)

    return center

def process_ellipse(frame, pup, fids, pupil_params, fid_params):
    pf, pt, ff, ft = dogs(frame, pupil_params, fid_params)
    center = find_closest_centroid(pt, 100, pup)
    pup_loc = pupil_locator(pt, center) #finds edges
    pxy, width, height, phi = fit(pup_loc) #fits ellipse
    fid_xys = [find_closest_centroid(ft, 100, fid) for fid in fids] 

    return pxy, fid_xys, width, height, phi