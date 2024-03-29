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
                _processed_frame = geezer.process_frame(
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


