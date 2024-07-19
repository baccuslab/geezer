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

    D1 = np.asmatrix(np.vstack([x ** 2, x * y, y ** 2])).T
    # Linear part of design matrix [eqn. 16] from (*)
    D2 = np.asmatrix(np.vstack([x, y, np.ones(len(x))])).T

    # forming scatter matrix [eqn. 17] from (*)
    S1 = D1.T * D1
    S2 = D1.T * D2
    S3 = D2.T * D2

    # Constraint matrix [eqn. 18]
    C1 = np.asmatrix('0. 0. 2.; 0. -1. 0.; 2. 0. 0.')

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
    # print(phi)
    params = ((x0, y0), width, height, np.rad2deg(np.real(phi)) % 360) #change to remove imaginary components (7/15/2024)

    #self.center, self.width, self.height, self.angle = self.params
    return params

def pupil_detect_blink(pups_xy, blinks, std_threshold=3.0):
    """
    Use distribution of pupil x,y coordinates to determine if there is a blink.
    This is a quality control after blinks are detected to remove any odd edge cases
    and classify them as blinks
    Input:
    - pups_xy: pupil x and y locations (n,2)
    - blinks: array of blinks, = blink, 0 = n bl;ink, (n)
    - std_threshold: number of standard deviations to classify as a blink
    Output:
    - new_pups_xy: adjusted pupil coordinates
    - new_blinks: adjusted blinks
    """
    new_pups_xy = pups_xy
    new_blinks = blinks
    for i in range(pups_xy.shape[1]):
        nonzero_idx = np.where(pups_xy[:,i] != 0)[0]
        mean_pups = np.mean(pups_xy[nonzero_idx,i])
        std_pups = np.std(pups_xy[nonzero_idx,i])
        edge_low = mean_pups - (std_threshold*std_pups)
        edge_high = mean_pups + (std_threshold*std_pups)
        idx_low = np.where(pups_xy[:,i] <= edge_low)[0]
        idx_high = np.where(pups_xy[:,i] >= edge_high)[0]
        new_pups_xy[idx_low,i] = 0
        new_pups_xy[idx_high,i] = 0
        new_blinks[idx_low] = 1
        new_blinks[idx_high] = 1
    zero_idx_0 = np.where(new_pups_xy[:,0] == 0)[0]
    zero_idx_1 = np.where(new_pups_xy[:,1] == 0)[0]
    new_pups_xy[zero_idx_0,1] = 0
    new_pups_xy[zero_idx_1,0] = 0
    return new_pups_xy, new_blinks

def ellipse_scaling(frame, pup_co, width, height, phi,min_scale=0.6,max_scale=2.0,step_size=0.05, threshold=0.011):
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
    - min_scale: minimum scale to increase/decrease width and height to find new adjusted width and height
    - max_scale: maximum scale to increase/decrease width and height to find new adjusted width and height
    - threshold: threshold for difference in z-scored values
    Returns:
    - adjusted hieght
    - adjusted width
    """
    x, y = pup_co
    ellipse_width = width
    ellipse_height = height
    angle = phi

    scales = np.flip(np.arange(min_scale,max_scale,step_size))
    # print(scales)
    scale_pixel_mu = np.zeros(scales.shape)
    for k in range(scales.shape[0]):
        frame2 = None
        frame3 = None
        # img1_bg = None
        masks = np.zeros_like(frame)
        cv2.ellipse(masks, (int(x), int(y)), (int(ellipse_width*scales[k]), int(ellipse_height*scales[k])), int(angle), 0, 360, (128, 128, 128), -1)
        mask_inv = cv2.bitwise_not(masks)
        # plt.imshow(mask_inv)
        # plt.show()
        frame2 = frame
        frame3 = frame
        cv2.ellipse(frame2, (int(x), int(y)), (int(ellipse_width*scales[k]), int(ellipse_height*scales[k])), int(angle), 0, 360, (0, 0, 0), 1)
        # masked = cv2.bitwise_and(masks, frame, mask=mask_inv)
        img2_bg = cv2.bitwise_and(frame3, frame2,mask = masks)
        # plt.figure()
        # plt.imshow(img2_bg)
        # print(img2_bg)
        # plt.show()
        scale_pixel_mu[k] = np.mean(frame[np.where(img2_bg != 0)]) #- pixel_mu_1) /pixel_mu_1)
        # dst = cv2.add(masked, img1_bg)
    # print(scales)
    # print(scale_pixel_mu)
    frame1 = None
    frame0 = None
    img1_bg = None
    masks_1 = np.zeros_like(frame)
    frame1 = frame
    frame0 = frame
    cv2.ellipse(masks_1, (int(x), int(y)), (int(ellipse_width), int(ellipse_height)), int(angle), 0, 360, (128, 128, 128), -1)
    mask_inv_1 = cv2.bitwise_not(masks_1)
    # plt.imshow(mask_inv)
    # plt.show()
    cv2.ellipse(frame1, (int(x), int(y)), (int(ellipse_width), int(ellipse_height)), int(angle), 0, 360, (0, 0, 0), 1)
    # masked = cv2.bitwise_and(masks, frame, mask=mask_inv)
    img1_bg = cv2.bitwise_and(frame0, frame1,mask = masks_1)
    # print(img1_bg)
    # plt.imshow(img1_bg)
    pixel_mu_1 = np.mean(frame0[np.where(img1_bg != 0)])
    # print(pixel_mu_1)
    scale_pixel_mu  = (scale_pixel_mu - pixel_mu_1) / pixel_mu_1
    # print(scale_pixel_mu)
    # plt.figure()
    # plt.plot(np.flip(scales[1:]),np.diff(np.flip(scale_pixel_mu)/step_size))
    # print(np.where(scale_pixel_mu <= threshold))
    final_scale = scales[np.min(np.where(scale_pixel_mu <= threshold))]

    adjusted_width = ellipse_width*final_scale
    adjusted_height = ellipse_height*final_scale
    return adjusted_width, adjusted_height

def ellipse_smoothing(current_height, current_width, previous_height, previous_width, a=0.8):
    """
    Function: smooth ellipse parameters (specifically width and height) by 
    the previous ellipse parameter, unless if there was no ellipse
    D(t*) = a*D(t) + (1-a)*D(t-1)
    Where D is the tuple of the width and height at frame t, and a is a value between 0 and 1.
    Input:
    - current_height: D(t) 
    - current_width: D(t)
    - previous_height: D(t-1)
    - previous_width: D(t-1)
    - a: weigthing value
    Output:
    - new_current_width
    - new_current_height
    """
    w_t = current_width
    h_t = current_height
    w_tm1 = previous_width
    h_tm1 = previous_height
    new_current_width = (a*w_t) + ((1-a)*w_tm1)
    new_current_height = (a*h_t) + ((1-a)*h_tm1)
    return new_current_height, new_current_width

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

def blink_identification_zscore(led_coords, frame_rate=30, threshold=0.4,cleanup=1, min_duration=100, max_duration=120):
    """
    Identify blinks for each frame of LED coordinates
    Arguments:
    - led_coords: coordinates in the shape of (n,2) for x,y
    - frame_rate: Hz
    - cleanup: cleanup boolean value for cleaning up fiducials that are smaller than a usual blink
    - min_duration: a minimum duration of a blink in milliseconds
    - threshold: value to classify changes as a blink in # of standard deviations
    Returns:
    - led_blinks: np.array of shape (n_frames,2) where 1 == blink and 0 == open
    """
    led_blinks = np.zeros(led_coords.shape)
    min_d = int(min_duration*frame_rate/1000)
    max_d = int(max_duration*frame_rate/1000)
    # baseline_leds = sanitize_fiducial_coordinates_yf(led_cords,)
    for i in range(led_coords.shape[1]):
        led_co_zscore = np.abs((led_coords[:,i] - np.mean(led_coords[:,i])) / np.std(led_coords[:,i]))
        led_co_inds = np.where(led_co_zscore >= threshold)[0]
        for j in range(len(led_co_inds)):
            if led_co_inds[j] == led_coords.shape[0]-1:
                continue
            else:
                led_co_inds[j] += 1
        led_blinks[led_co_inds,i] = 1
        led_blinks[led_coords.shape[0]-1,i] = 0
        new_led_co_inds = np.where(led_blinks[:,i] == 1)[0]
        for m in range(new_led_co_inds.shape[0]):
            if new_led_co_inds[m] == 0 or new_led_co_inds[m] == 1 or new_led_co_inds[m] == led_coords.shape[0]-1 or new_led_co_inds[m] == led_coords.shape[0]-2:
                continue
            else:
                if led_blinks[new_led_co_inds[m]-2,i] == 1 and led_blinks[new_led_co_inds[m]-1,i] == 0:
                    led_blinks[new_led_co_inds[m]-1,i] = 1
                else:
                    continue    
        new_led_co_inds = np.where(led_blinks[:,i] == 1)[0]
        for m in range(new_led_co_inds.shape[0]):
            if new_led_co_inds[m] == 0 or new_led_co_inds[m] == led_coords.shape[0]-1:
                continue
            else:
                if led_blinks[new_led_co_inds[m]-1,i] == 0 and led_blinks[new_led_co_inds[m]+1,i] == 0:
                    led_blinks[new_led_co_inds[m],i] = 0
                else:
                    continue
        led_co_inds = np.where(led_blinks[:,i] == 1)[0]
        if cleanup == 1:
            diff_inds = np.diff(led_co_inds)
            # last = diff_inds[-1] - diff_inds[-2]
            # diff_inds = np.insert(blink_duration,len(blink_duration),last)
            first = diff_inds[0] - 0
            diff_inds = np.insert(diff_inds,0,first)
            space_in_between = np.where(diff_inds >= min_d)[0]
            space_in_between = space_in_between[space_in_between < max_d]
            for q in range(space_in_between.shape[0]):
                led_blinks[led_co_inds[space_in_between[q]-1]:led_co_inds[space_in_between[q]]] = 1
            led_co_inds = np.where(led_blinks[:,i] == 1)[0]
            diff_inds = np.diff(led_co_inds)
            nzero_diff_blink_inds = np.where(diff_inds != 1)[0]
            blink_duration = np.diff(nzero_diff_blink_inds)
            # diff_blink_duration = np.diff(blink_duration)
            for k in range(blink_duration.shape[0]):
                if blink_duration[k] <= min_d:
                    idxs = np.arange(led_co_inds[nzero_diff_blink_inds[k]+1],led_co_inds[nzero_diff_blink_inds[k]+1]-1+blink_duration[k],1)
                    led_blinks[idxs,i] = 0 #detects if there is a sequence longer than a certain value
                else:
                    continue
                
    return led_blinks


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

