import numpy as np 
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




        # Define a worker function for each process
def worker(start_frame, end_frame, pxy, fxys):
'''
This is how the workflow for the image_processing begins. 
-Load the video
-Load the frames of interest: start_frame:end_frame
-Mean of the frame
-THEN USE PROCESS FRAME FUNCTION
Focus on how to use this and find where the issue is in the process_image function.
'''
    video = cv2.VideoCapture(self.mp4_filename)
    print("Started worker")
    local_results = []
    for frame_idx in tqdm.tqdm(range(start_frame, end_frame)):
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = video.read()
        # frame = frame.mean(axis=2)
        if ret:

            try:
                frame = frame.mean(axis=2)
                _processed_frame = process_frame(
                    frame, pxy, fxys, proc_pup_params, proc_fid_params
                )
                
                error_flag=0 # no error
                processed_frame = [frame_idx, _processed_frame, error_flag]
                local_results.append(processed_frame)
            except:
                error_flag=1 # processing error
                processed_frame = [frame_idx, False, error_flag]
                local_results.append(processed_frame)
            # # pass

        else:
            # input("Frame not found {}".format(frame_idx))
            # print('Frame not found {}'.format(frame_idx))
            error_flag=2 # frame not found
            processed_frame = [frame_idx, False, error_flag]
            local_results.append(processed_frame)

    # Append local results to the shared list
    result_list.extend(local_results)
    results = list(result_list)

