import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from utils import dogs, pupil_locator, fit
import IPython


frames = '/media/grandline/ExtremeSSD/EyeTracking20231116/F/cropped.mp4'
video = cv2.VideoCapture(frames)

video.set(cv2.CAP_PROP_POS_FRAMES, 180000 - 1)

ret, frame = video.read()

if ret:

    frame_array = np.array(frame)

    np.save('test_array_1.npy', frame_array)

video.release()
# IPython.embed()

def ground_truth_ellipse(frame, pup_co, width, height, phi):
    x, y = pup_co
    ellipse_width = width
    ellipse_height = height
    angle = phi

    mask = np.zeros_like(frame)
    cv2.ellipse(mask, (int(x), int(y)), (int(ellipse_width), int(ellipse_height)), angle, 0, 360, (255, 255, 255), 1)
    result = cv2.addWeighted(frame, 1, mask, 0.5, 0)

    return result


pupil_params = [10, 5, 150, 120]#[4, 25, 150, 160] #[3, 20, 120, 200]
fid_params = [1,3,11,90]#[1, 3, 11, 100]
pup = [338.0906788247214, 201.07244174265455]
fids = [[183.8809523809524, 46.35309017223915],[147.59321175278623, 65.968085106383],[244.6874366767984, 51.747213779128685],[67.17173252279635, 217.49392097264442],[385.9154002026343, 148.84143870314085]]

frame = '/home/grandline/geezer/test_array_1.npy'
frame = np.load(frame)
frame = np.mean(frame, axis=2)

pf, pt, ff, ft = dogs(frame, pupil_params, fid_params)

surround = pupil_locator(pt, pup)
center, width, height, phi = fit(surround)
ellipse = ground_truth_ellipse(frame, center, width, height, phi)
plt.imshow(pt)
plt.scatter(surround[:, 0], surround[:, 1], c='r')
plt.imshow(ellipse)
plt.show() 
