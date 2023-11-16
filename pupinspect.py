import h5py as h5
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
import numpy as np
import cv2

import cv2
import numpy as np
import h5py
import multiprocessing as mp

filename = '/media/grandline/ExtremeSSD/0706/july062023jf/minibatch.h5'
frames = '/media/grandline/ExtremeSSD/0706/july062023jf/super_crop.mp4'#'/media/grandline/ExtremeSSD/0706/july062023jf/cam_crop.mp4'#'/media/grandline/ExtremeSSD/0706/july062023jf/small_crop.mp4'
frame = '/media/grandline/ExtremeSSD/0706/july062023jf/og_frame.npy'

with h5py.File(filename, 'r') as f:
    keys = list(f.keys())
    fids_co = f['fids_co']
    frame_idxs = f['frame_idxs'][:]
    height = f['height'][:]
    phi = f['phi'][:]
    pup_co = f['pup_co'][:]
    width = f['width'][:]

def ground_truth_ellipse(frame, pup_co, width, height, phi):
    x, y = pup_co
    ellipse_width = width
    ellipse_height = height
    angle = phi

    mask = np.zeros_like(frame)
    cv2.ellipse(mask, (int(x), int(y)), (int(ellipse_width), int(ellipse_height)), angle, 0, 360, (255, 255, 255), 1)
    result = cv2.addWeighted(frame, 1, mask, 0.5, 0)

    return result

cap = cv2.VideoCapture(frames)
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use mp4v codec for .mp4

out = cv2.VideoWriter('sample_overlay_4.mp4', fourcc, fps, (frame_width, frame_height))

for idx in frame_idxs: #range(2000, 2000 + len(frame_idxs)):
    if idx >= len(pup_co):
        continue  # Skip frames for which there's no data

    # Set the current frame position in the video capture
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)

    ret, frame = cap.read()

    if ret:
        frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        result = ground_truth_ellipse(frame, pup_co[frame_idx-1], width[frame_idx-1], height[frame_idx-1], phi[frame_idx-1]) #ground_truth_ellipse(frame, pup_co[frame_idx-2000], width[frame_idx-2000], height[frame_idx-2000], phi[frame_idx-2000])
        out.write(result)

out.release()
cap.release()


# def process_frame(frame_idx, pup_co, width, height, phi, cap, out):
#     cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
#     ret, frame = cap.read()

#     if ret:
#         result = ground_truth_ellipse(frame, pup_co[frame_idx-1], width[frame_idx-1], height[frame_idx-1], phi[frame_idx-1])
#         out.write(result)
    
# if __name__ == '__main__':
#     filename = '/media/grandline/ExtremeSSD/0706/july062023jf/minibatch.h5'
#     frames = '/media/grandline/ExtremeSSD/0706/july062023jf/super_crop.mp4'#'/media/grandline/ExtremeSSD/0706/july062023jf/cam_crop.mp4'#'/media/grandline/ExtremeSSD/0706/july062023jf/small_crop.mp4'

#     with h5py.File(filename, 'r') as f:
#         keys = list(f.keys())
#         fids_co = f['fids_co']
#         frame_idxs = f['frame_idxs'][:]
#         height = f['height'][:]
#         phi = f['phi'][:]
#         pup_co = f['pup_co'][:]
#         width = f['width'][:]
    
#     cap = cv2.VideoCapture(frames)
#     fps = int(cap.get(cv2.CAP_PROP_FPS))
#     frame_width = int(cap.get(3))
#     frame_height = int(cap.get(4))
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use mp4v codec for .mp4
#     out = cv2.VideoWriter('sample_overlay.mp4', fourcc, fps, (frame_width, frame_height))

#     num_processes = 100 # mp.cpu_count()

#     processes = []
#     for idx in frame_idxs:
#         p = mp.Process(target=process_frame, args=(idx, pup_co, width, height, phi, cap, out))
#         processes.append(p)
#         p.start()
    
#     for process in processes:
#         process.join()
    
#     out.release()
#     cap.release()