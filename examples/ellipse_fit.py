import numpy as np 
import numpy as np
import math
import json
import tqdm

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



"""
Goal for this file:
- Try out ellipse fitting function of geezer
- Make an improvement so it doesn't break
- Implement in existing code
"""

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
    #threshold the large amplitude to for detecting the pupil 
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

def cond(r, crop_list):

    ### return the r within the normalized difference ###

    dists = np.linalg.norm(np.mean(r, axis = 0, dtype=np.float64) - r, axis = 1)

    mean_ = np.mean(dists)
    std_ = np.std(dists)
    lower, upper = mean_ - std_, mean_ + std_ *.8
    cond_ = np.logical_and(np.greater_equal(dists, lower),np.less(dists, upper))

    return r[cond_]


def pupil_locator(frame, center, min_radius=2, max_radius=100, threshold=64): #threshold=64 usually
    # try:
    #     cent = np.round(center).astype(int)
    # except:
    #     return
    diagonal_size = 2**10
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

    #rx_multiply[onethird_rx_multiplier] = onethird
    #rx_multiply[onefourth_rx_multiplier] = onefourth
    #rx_multiply[onehalf_rx_multiplier] = .5

    #ry_multiply[onethird_ry_multiplier] = onethird
    #ry_multiply[onefourth_ry_multiplier] = onefourth
    #ry_multiply[onehalf_ry_multiplier] = .5


    ry_add = np.array(np.concatenate(([0, 2, 4],onehalf_ry_add,onefourth_ry_add,onethird_ry_add)),dtype=int)
    rx_add = np.array(np.concatenate(([1, 2, 5],onehalf_rx_add,onefourth_rx_add,onethird_rx_add)), dtype=int)


    ry_subtract = np.array(np.concatenate(([3, 5, 7],onehalf_ry_subtract,onefourth_ry_subtract,onethird_ry_subtract )))


    rx_subtract = np.array(np.concatenate(([3, 4, 6],onehalf_rx_subtract,onefourth_rx_subtract,onethird_rx_subtract)))



    black = [35, 35, 35]

    angle_dev = -22.5

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
    np.argmax(canvas_[:, 0][min_radius:max_radius] == 0), 
    np.argmax(canvas_[0, :][min_radius:max_radius] == 0), 
    np.argmax(canvas_[main_diagonal[:canv_shape0, :canv_shape1]][min_radius:max_radius] == 0),
    np.argmax(crop_canvas[main_diagonal[:crop_canv_shape0, :crop_canv_shape1]][min_radius:max_radius] == 0),
    np.argmax(crop_canvas2[main_diagonal[:crop_canv2_shape0, :crop_canv2_shape1]][min_radius:max_radius] == 0),
    np.argmax(crop_canvas3[main_diagonal[:crop_canv3_shape0, :crop_canv3_shape1]][min_radius:max_radius] == 0),
    np.argmax(canvas2[-center[1], -center[0]:][min_radius:max_radius] == 0),
    np.argmax(canvas2[-center[1]:, -center[0]][min_radius:max_radius] == 0),
    np.argmax(canvas_[half_diagonal[:canv_shape0, :canv_shape1]][min_radius:max_radius] == 0),
    np.argmax(crop_canvas[half_diagonal[:crop_canv_shape0, :crop_canv_shape1]][min_radius:max_radius] == 0),
    np.argmax(crop_canvas2[half_diagonal[:crop_canv2_shape0, :crop_canv2_shape1]][min_radius:max_radius] == 0),
    np.argmax(crop_canvas3[half_diagonal[:crop_canv3_shape0, :crop_canv3_shape1]][min_radius:max_radius] == 0),
    np.argmax(canvas_[invhalf_diagonal[:canv_shape0, :canv_shape1]][min_radius:max_radius] == 0),
    np.argmax(crop_canvas[invhalf_diagonal[:crop_canv_shape0, :crop_canv_shape1]][min_radius:max_radius] == 0),
    np.argmax(crop_canvas2[invhalf_diagonal[:crop_canv2_shape0, :crop_canv2_shape1]][min_radius:max_radius] == 0),
    np.argmax(crop_canvas3[invhalf_diagonal[:crop_canv3_shape0, :crop_canv3_shape1]][min_radius:max_radius] == 0),
    np.argmax(canvas_[fourth_diagonal[:canv_shape0, :canv_shape1]][min_radius:max_radius] == 0),
    np.argmax(crop_canvas3[fourth_diagonal[:crop_canv3_shape0, :crop_canv3_shape1]][min_radius:max_radius] == 0),
    np.argmax(crop_canvas[fourth_diagonal[:crop_canv_shape0, :crop_canv_shape1]][min_radius:max_radius] == 0),
    np.argmax(crop_canvas2[fourth_diagonal[:crop_canv2_shape0, :crop_canv2_shape1]][min_radius:max_radius] == 0),
    np.argmax(canvas_[invfourth_diagonal[:canv_shape0, :canv_shape1]][min_radius:max_radius] == 0),
    np.argmax(crop_canvas2[invfourth_diagonal[:crop_canv2_shape0, :crop_canv2_shape1]][min_radius:max_radius] == 0),
    np.argmax(crop_canvas[invfourth_diagonal[:crop_canv_shape0, :crop_canv_shape1]][min_radius:max_radius] == 0),
    np.argmax(crop_canvas3[invfourth_diagonal[:crop_canv3_shape0, :crop_canv3_shape1]][min_radius:max_radius] == 0),
    np.argmax(canvas_[third_diagonal[:canv_shape0, :canv_shape1]][min_radius:max_radius] == 0),
    np.argmax(crop_canvas2[third_diagonal[:crop_canv2_shape0, :crop_canv2_shape1]][min_radius:max_radius] == 0),
    np.argmax(crop_canvas[third_diagonal[:crop_canv_shape0, :crop_canv_shape1]][min_radius:max_radius] == 0),
    np.argmax(crop_canvas3[third_diagonal[:crop_canv3_shape0, :crop_canv3_shape1]][min_radius:max_radius] == 0),
    np.argmax(canvas_[invthird_diagonal[:canv_shape0, :canv_shape1]][min_radius:max_radius] == 0),
    np.argmax(crop_canvas2[invthird_diagonal[:crop_canv2_shape0, :crop_canv2_shape1]][min_radius:max_radius] == 0),
    np.argmax(crop_canvas[invthird_diagonal[:crop_canv_shape0, :crop_canv_shape1]][min_radius:max_radius] == 0),
    np.argmax(crop_canvas3[invthird_diagonal[:crop_canv3_shape0, :crop_canv3_shape1]][min_radius:max_radius] == 0)
    ], dtype=int) + min_radius


    '''
    This is how the workflow for the image_processing begins. 
    -Load the video
    -Load the frames of interest: start_frame:end_frame
    -Mean of the frame
    -THEN USE PROCESS FRAME FUNCTION
    Focus on how to use this and find where the issue is in the process_image function.
    '''
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
    
def process_frame(frame, pup, fids, pupil_params, fid_params, ellipse=True):
    if ellipse:
        # THIS IS CURRENTLY BRROKEN
        pf, pt, ff, ft = dogs(frame, pupil_params, fid_params)
        center = find_closest_centroid(pt, 100, pup)

        pup_loc = pupil_locator(pt, center) #finds edges
        # print(pup_loc)
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

        
        width = 0
        height = 0
        phi = 0
# fids_parameters =  {'exp': 1, 'small': 3, 'large': 11, 'thresh': 65}
        return pxy, final_fid_xys, width, height, phi

def ground_truth_ellipse(frame, pup_co, width, height, phi):
    x, y = pup_co
    ellipse_width = width
    ellipse_height = height
    angle = phi

    mask = np.zeros_like(frame)
    cv2.ellipse(mask, (int(x), int(y)), (int(ellipse_width), int(ellipse_height)), angle, 0, 360, (255, 255, 255), 1)
    result = cv2.addWeighted(frame, 1, mask, 0.5, 0)

    return result

# pup_parameters = [3, 20, 50, 200]
# fids_parameters =  [1, 3, 11, 65]
# fids_parameters =  [1, 1, 40, 65] # for 4/10/2024 params 89k 91k
# fids_parameters =  [1, 3, 11, 80] # for 4/11/2024 params 89k5 91k

# pup_parameters = [5, 20, 50, 220] # for 4/22/2024 params 89k5 91k new 
# fids_parameters =  [1, 2, 20, 80] # for 4/22/2024 params 89k5 91k new 
pup_parameters = [5, 11, 60, 220] # for 5/8/2024 params 3k 167k new 
fids_parameters =  [1, 2, 20, 80] # for 5/8/2024 params 3k 167k new 

# pup_parameters = {'exp': 3, 'small': 20, 'large': 50, 'thresh': 200}
# fids_parameters =  {'exp': 1, 'small': 3, 'large': 11, 'thresh': 65}

# pup_parameters = {'exp': 3, 'small': 20, 'large': 50, 'thresh': 200}
# fids_parameters =  {'exp': 1, 'small': 3, 'large': 11, 'thresh': 65}

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

from scipy.ndimage import rotate

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

        # Define a worker function for each process
def worker(mp4_filename, image_folder, start_frame, end_frame, pxy, fxys, proc_fid_params, proc_pup_params):  
    '''
    This is how the workflow for the image_processing begins. 
    -Load the video
    -Load the frames of interest: start_frame:end_frame
    -Mean of the frame
    -THEN USE PROCESS FRAME FUNCTION
    Focus on how to use this and find where the issue is in the process_image function.
    4/3/2024: Found it in the pupil_locator file that I imported.
    It will take a  while to understand it but so far it throws up an error = 1, even without the ellipse.
    '''
    result_list = []
    video = cv2.VideoCapture(mp4_filename)
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # Get video properties (width, height, frames per second, etc.)
    # width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    # height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # fps = video.get(cv2.CAP_PROP_FPS)
    # num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    # print(video.dtype)
    # Define the codec and create a VideoWriter object
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # fourcc = cv2.VideoWriter_fourcc('F','M','P','4')
    # print(width)
    # print(height)
    # out = cv2.VideoWriter(new_mp4_name,fourcc,fps,(height,width))
    print("Started worker")
    local_results_sw = np.zeros((end_frame-start_frame,2))
    local_results_se = np.zeros((end_frame-start_frame,2))
    os.chdir(image_folder)
    g = h5.File('results.h5', 'w')
    new_results_group = g.create_group('new_results')
    for frame_idx in tqdm.tqdm(range(start_frame, end_frame)):
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = video.read()
        frame = frame.mean(axis=2)
        if not ret:
            break

        # try:
        # print(frame.shape)
        # # frame = np.uint8(frame.mean(axis=2))
        # print(frame.shape)
        # if frame_idx == start_frame:
        #     _processed_frame = process_frame_fids(
        #         frame, pxy, fxys, proc_pup_params, proc_fid_params,
        #     ellipse=False)
        # else:
        #     _processed_frame = process_frame(
        #         frame, _processed_frame[0], _processed_frame[1], proc_pup_params, proc_fid_params,
        #     ellipse=False)
        
        # local_results[frame_idx-start_frame,:] = _processed_frame[0]
        _processed_frame = process_frame_fids(frame, fxys, proc_fid_params)
        
        local_results_sw[frame_idx-start_frame,:] = _processed_frame['sw']
        local_results_se[frame_idx-start_frame,:] = _processed_frame['se']


    new_results_sw = blink_identification_zscore(local_results_sw,threshold=0.5)
    new_results_se = blink_identification_zscore(local_results_se,threshold=0.5)
    new_results_group.create_dataset('se', data=new_results_se)
    new_results_group.create_dataset('sw', data=new_results_sw)
    new_loc_results_group = g.create_group('new_loc_results')
    new_loc_results_group.create_dataset('sw', data=local_results_sw)
    new_loc_results_group.create_dataset('se', data=local_results_sw)
    pup_xy_group = g.create_group('pup_xy')
    width_height_group_old = g.create_group('w_h_old')
    width_height_group_new = g.create_group('w_h_new')
    phis_group = g.create_group('phis')
    phi_s = np.zeros(end_frame-start_frame)
    pups_xys = np.zeros((end_frame-start_frame,2))
    ws_hs_old = np.zeros((end_frame-start_frame,2))
    ws_hs_new = np.zeros((end_frame-start_frame,2))
    # newloc_results = np.zeros(end_frame-start_frame, dtype=object)
    newpath1 = image_folder + 'gt_ellipse' 
    if not os.path.exists(newpath1):
        os.mkdir(newpath1)
    newpath2 = image_folder + 'gt_ellipse_adj' 
    if not os.path.exists(newpath2):
        os.mkdir(newpath2)
    for frame_idx in tqdm.tqdm(range(start_frame, end_frame)):
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = video.read()
        frame = frame.mean(axis=2)
        if not ret:
            break
        norm_idx = frame_idx-start_frame
        print(norm_idx)
        if new_results_se[norm_idx,1] == 0 and new_results_sw[norm_idx,1] == 0: #use index 1 as there is usually a lot of horizontal eye movements
            _processed_frame = process_frame(
                frame, pxy, fxys, proc_pup_params, proc_fid_params,
            ellipse=True)
            # result = ground_truth_ellipse(frame, _processed_frame[0], _processed_frame[2], _processed_frame[3], _processed_frame[4])
            ws_hs_old[norm_idx,0] = _processed_frame[2]
            ws_hs_old[norm_idx,1] = _processed_frame[3]
            phi_s[norm_idx] = _processed_frame[4]
            pups_xys[norm_idx,:] = _processed_frame[0]
            # result = np.uint8(result)
            # output_file1 = os.path.join(newpath1, f"frame_{frame_idx}.png")
            # cv2.imwrite(output_file1, result)


            result, ad_width, ad_height = ground_truth_ellipse_adjustment_draw(frame, _processed_frame[0], _processed_frame[2], _processed_frame[3], _processed_frame[4],fraction=0.008)
            result = np.uint8(result)
            ws_hs_new[norm_idx,0] = ad_width
            ws_hs_new[norm_idx,1] = ad_height
            output_file1 = os.path.join(newpath1, f"frame_{frame_idx}.png")
            cv2.imwrite(output_file1, result)
        
        # Save the frame as an image
        elif new_results_se[norm_idx,1] == 1 or new_results_sw[norm_idx,1] == 1: 
            _processed_frame = process_frame(
                frame, pxy, fxys, proc_pup_params, proc_fid_params,
            ellipse=False)
            result = np.uint8(frame)
            output_file1 = os.path.join(newpath1, f"frame_{frame_idx}_blink.png")
            # output_file2 = os.path.join(newpath2, f"frame_{frame_idx}_blink.png")
            cv2.imwrite(output_file1, result)
            # cv2.imwrite(output_file2, result)
    #     newloc_results[norm_idx] = _processed_frame
    #     print(result.shape)
    #     # if frame_idx == 5149:
    #     #     plt.figure()
    #     #     plt.imshow(result)
    #     #     fname = 'new_ellip_' + str(frame_idx) + '.png'
    #         # plt.savefig(fname)
        # Construct the output file path
        # output_file = os.path.join(image_folder, f"frame_{frame_idx}.png")
        
        # Save the frame as an image
        # cv2.imwrite(output_file, result)
    
        # if not success:
        #     print("Error: Failed to write frame to output video")
        # error_flag=0 # no error
        # processed_frame = [frame_idx, _processed_frame, error_flag]
        # local_results.append(processed_frame)
        # except: 
        #     error_flag=1 # processing error
        #     processed_frame = [frame_idx, False, error_flag]
        #     local_results.append(processed_frame)
    # pass
    pup_xy_group.create_dataset('data', data=pups_xys)
    width_height_group_old.create_dataset('data', data=ws_hs_old)
    width_height_group_new.create_dataset('data', data=ws_hs_new)
    phis_group.create_dataset('data', data=phi_s)
        # else:
        #     break
            # input("Frame not found {}".format(frame_idx))
            # print('Frame not found {}'.format(frame_idx))
            # error_flag=2 # frame not found
            # processed_frame = [frame_idx, False, error_flag]
            # local_results.append(processed_frame)

    # out.release()
    video.release()
    g.close()
    # Append local results to the shared list
    # results = [new_results, newloc_results]# [new_results]
    # result_list.extend(local_results)
    # results = list(result_list)
    # return results

def save_to_h5(h5_filename,video_name):
    # Open the existing HDF5 file in write mode
    h5_file = h5.File(h5_filename, 'a')

    # Open the MP4 video file
    video_capture = cv2.VideoCapture(video_name)

    # Initialize variables
    frame_count = 0

    # Iterate through the frames of the video
    while True:
        # Read a frame from the video
        ret, frame = video_capture.read()
        if not ret:
            break
    
        # Convert the frame to grayscale if needed
        # gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
        # Save the frame to the HDF5 file
        h5_file['frames/frame{}'.format(frame_count)] = frame
    
        # Increment frame count
        frame_count += 1

    # Close the video capture and HDF5 file
    video_capture.release()
    h5_file.close()

def images_to_video(input_dir, output_video_path, fps):
    # Get a list of image files in the input directory
    image_files = [f for f in os.listdir(input_dir) if f.endswith('.png')]

    # Sort the files by their names
    image_files.sort()

    # Read the first image to get dimensions
    first_image = cv2.imread(os.path.join(input_dir, image_files[0]))
    height, width, _ = first_image.shape

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Write each image to the video
    for image_file in image_files:
        image_path = os.path.join(input_dir, image_file)
        frame = cv2.imread(image_path)
        out.write(frame)

    # Release the VideoWriter# %%
    out.release()

def blink_identification(led_coords, frame_rate=30, threshold=30, min_duration=100, max_duration=400):
    """
    Identify blinks for each frame of LED coordinates
    Arguments: 
    led_coords: shape (n_frames,2)
    threshold: value to classify changes as a blink 
    min_duration: minimum value of duration as a blink (default 100ms)
    max_duration: maximum value of duration as a blink (default 400ms)
    Returns:
    led_blinks: np.array of shape (n_frames,2) where 1 == blink and 0 == open
    """
    led_blinks = np.zeros(led_coords.shape)
    min_d = int(min_duration*frame_rate/1000) #framerate
    max_d = int(max_duration*frame_rate/1000)
    for i in range(led_coords.shape[1]):
        diff = np.abs(np.diff(led_coords[:,i]))
        diffind = np.where(diff > threshold)[0]
        diffind += 1
        led_blinks[diffind,i] = 1
        diffindind = np.diff(diffind)
        inds = np.where((diffindind < max_d) & (diffindind >= min_d))[0]
        for j in range(inds.shape[0]):
            led_blinks[diffind[inds[j]]:diffind[inds[j]+1],i] = 1
    return led_blinks

def blink_identification_zscore(led_coords, frame_rate=30, threshold=0.4):
    """
    Identify blinks for each frame of LED coordivideo.release()
    g.close()
    frame_rate: Hz
    threshold: value to classify changes as a blink in # pf standard deviations
    Returns:
    led_blinks: np.array of shape (n_frames,2) where 1 == blink and 0 == open
    """
    led_blinks = np.zeros(led_coords.shape)
    for i in range(led_coords.shape[1]):
        led_co_zscore = np.abs((led_coords[:,i] - np.mean(led_coords[:,i])) / np.std(led_coords[:,i]))
        led_co_inds = np.where(led_co_zscore >= threshold)[0]
        for j in range(len(led_co_inds)):
            if led_co_inds[j] == led_coords.shape[0]-1:
                pass
            else:
                led_co_inds[j] += 1
        led_blinks[led_co_inds,i] = 1
        led_blinks[led_coords.shape[0]-1,i] = 0
    return led_blinks

# 4/11/2024
# Now, I have the ellipse fitter working and now the dea is to find where the failure cases are/
# What I have so found is two cases:
# 1) When the pupil shifts its location, the ellipse fitter is then confused.
# Solution1: Find the pupil location, fit it to the model, then input that into the ellipse fitter function
# 2) When there a blink is in progress, the ellipse fitter acts wonky and delivers an error
# Solution2: Use the blink ID function to skip frames where a blink occurs.

# 4/22/2024
# I have played with the LED and Pupil params in geezer as much as possible to find the perfect file
# however, there are many frames

# 4/25/2024
# So far, I have improved the blink_id function.
# The pupil tracking seems to be working fine, the only issue is the blinks are not accurately detected.
# One potential issue to help with this is to use the LED position, which shoud not change in theory.
# Then, by normalizing and thesholding, we can then use this as a better measure for blinks.
# After this, the next issue is to see where there are failure cases for the pupil fitting.

# 4/26/2024
# SB: Before or after pupil fitting, go and take line profile of major axis + find where the gray to black ratio starts to change,
# SB: then, we can finally fit the pupil in a normal shape.
# To do: After pupil_xys, width, and height are calculated, save values and image in results h5 file.
# Then, take a few cases for the line profile for both width and height, and see where the transition occurs.
# Find some threshold for a fraction of maximum/minimum of the value, and generate a new width and height for the ellipse. 
# This will ensure no underfitting of the ellipse.

# 4/29/2024 - 4/30/2024
# Adding the pup_xy and widths + heights to the h5 files, then phis.
# Afterward, I tried to find the phis, then tried to change to accomodate phis and thetas to the major and minor axes of the eye.

# 5/8/2024
# I have now added the two circles to compared the earleir result of the pupil to the other one, and it seems that it works, but 
# it keeps flickering very rapidly, so I wonder how it could change. I will run this version through the entire video and see how it turns out.


if __name__ == '__main__':
    os.chdir('/data/cortex/raw/GolDRoger/jackfish')
    # old one 5050:5150
    # pupp = np.array([263.10875262, 145.67872117])
    # fidd = { 'cam': [253.67774351, 144.02126387], 'ne' : [253.67774351, 144.02126387], 'nw' : [253.67774351, 144.02126387], 'se' : [382.06504065, 140.82384824], 'sw' : [ 62.59235669, 211.14012739]}

    # for 2k - 19k goldroger
    # pupp = np.array([403.3343436,  331.14531823])
    # fidd = { 'cam': [330.37254902, 198.79271709], 'ne' : [415.74676259, 185.00143885], 'nw' : [289.975, 221.13611111], 'se' : [566.50411523, 467.42592593], 'sw' : [210.08602151, 410.48387097]}

    # for 3k to 9k goldroger
    # pupp = np.array([418.89221557, 323.22253452])
    # fidd = { 'nw' : [288.11336032, 221.33603239], 'se' : [288.11336032, 221.33603239], 'sw' : [288.11336032, 221.33603239]}
    # for 89k to 91k goldroger
    # pupp = np.array([473.07111597, 313.96170678])
    # fidd = { 'se' : [559.0, 282.0], 'sw' : [200.10043668, 391.99344978]}

    # # for 89k5 to 91k goldroger
    # pupp = np.array([428.26662144, 333.69063772])
    # fidd = { 'se' : [552.10762332, 305.80269058], 'sw' : [202.67213115, 397.7147541]}

    # for 3k to 167k goldroger
    pupp = np.array([418.0694424, 324.72649698])
    fidd = { 'se' : [554.43661972, 312.79577465], 'sw' : [207.31304348, 409.03478261]}

    worker('/data/cortex/raw/GolDRoger/jackfish/cam_22248110_crop.mp4', '/home/yfaragal/07062023/July062023jf/goldroger_3k-167k/', 3000,167000,pupp,fidd,fids_parameters,pup_parameters)
    images_to_video('/home/yfaragal/07062023/July062023jf/goldroger_3k-167k/gt_ellipse','/home/yfaragal/07062023/July062023jf/goldroger_3k-167k.mp4',30)
    # images_to_video('/home/yfaragal/07062023/July062023jf/goldroger_89k-91k-zscore-two-leds-ellipse-dgt_ellipse_adj','/home/yfaragal/07062023/July062023jf/goldroger_crop9-zscore-gt-ellipse-adj.mp4',30)
    # print(rls)
    # for i in range(100):
    #     print(rls[i][2])
    # rls_list = [rls]
    # np.save('ellipse_results.npy', rls_list, allow_pickle=True)


    # JBM: crop function reduces image quality, can do indexing instead