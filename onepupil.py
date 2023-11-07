import numpy as np
import utils
import matplotlib.pyplot as plt
import IPython

import cv2
import nutils

import numpy as np
import scipy.signal

frame_file = '/home/grandline/geezer/og_frame.npy'
frame = np.load(frame_file)
pupil_params = [7,12,90,150]
fid_params = [1,2,11,90]#[1,3,11,90]#[1, 3, 11, 100]
pup = [275.0906788247214, 137.07244174265455]
fids = [[183.8809523809524, 46.35309017223915],[147.59321175278623, 65.968085106383],[244.6874366767984, 51.747213779128685],[67.17173252279635, 217.49392097264442],[385.9154002026343, 148.84143870314085]]
# random_vec = np.random.randint(0, 255, (32, 2))

file = '/media/grandline/ExtremeSSD/rot_10ms.h5'

import h5py as h5 

with h5.File(file, 'r') as f:
    keys = list(f.keys())
    frame_idx =  f['series_001/epoch_001/frame_idxs'][:]

print(frame_idx)


################################################################
################################################################
################################################################

# def find_min_left_right_max_with_threshold(line_values, threshold):
#     # Find the minimum value and its index
#     min_value = min(line_values)
#     min_index = line_values.index(min_value)

#     left_max_index = None
#     right_max_index = None

#     # Search for the nearest local maximum to the left of the minimum
#     for i in range(min_index - 1, -1, -1):
#         if line_values[i] > line_values[i + 1]:
#             left_max_index = i
#             if line_values[left_max_index] >= threshold:
#                 break
#             else:
#                 left_max_index = None

#     # Search for the nearest local maximum to the right of the minimum
#     for i in range(min_index + 1, len(line_values)):
#         if line_values[i] > line_values[i - 1]:
#             right_max_index = i
#             if line_values[right_max_index] > threshold:
#                 break
#             else:
#                 right_max_index = None

#     # Check if left and right maximum points were found
#     if left_max_index is not None and right_max_index is not None:
#         left_max_value = line_values[left_max_index]
#         right_max_value = line_values[right_max_index]

#         return {
#             "minimum": (min_index, min_value),
#             "left_maximum": (left_max_index, left_max_value),
#             "right_maximum": (right_max_index, right_max_value)
#         }
#     else:
#         return None

################################################################
################################################################
################################################################

# pxy, fid_xys, width, height, phi = nutils.process_ellipse(frame, pup, fids, pupil_params, fid_params)

# print(pxy, fid_xys, width, height, phi)

# coords = nutils.pupil_locator(frame, pup)

# elips = nutils.fit(coords)

# fig, axes= plt.subplots(1, 4, figsize=(20, 5))

# pf, pt, ff, ft =  utils.dogs(frame, pupil_params, fid_params)

# coords = nutils.pupil_locator(pt, pup)

# coords = np.array(coords)

# x = coords[:,0]
# y = coords[:,1]


# axes[0].imshow(frame)
# axes[1].scatter(x,y)
# axes[2].imshow(pt)
# axes[3].scatter(x,y)
# axes[3].imshow(frame)
# axes[3].imshow(pt, alpha=0.5)
# plt.show()

################################################################
################################################################
################################################################

# center, width, height, angle = nutils.fit(coords)

# # # # Ellipse parameters
# # # center = (2, 3)  # (x, y)
# # # height = 4
# # # width = 2
# # # angle = 125  # Angle in degrees (counter-clockwise)

# # Calculate major and minor axis lengths
# major_axis_length = 2 * max(height, width)
# minor_axis_length = 2 * min(height, width)

# print("major axis length",major_axis_length, "minor axis legnth", minor_axis_length)

# # Calculate the endpoints of the major axis
# angle_rad = np.deg2rad(angle)  # Convert angle to radians
# x1_major = center[0] - 0.5 * major_axis_length * np.cos(angle_rad)
# y1_major = center[1] - 0.5 * major_axis_length * np.sin(angle_rad)
# x2_major = center[0] + 0.5 * major_axis_length * np.cos(angle_rad)
# y2_major = center[1] + 0.5 * major_axis_length * np.sin(angle_rad)

# # Calculate the endpoints of the minor axis
# x1_minor = center[0] - 0.5 * minor_axis_length * np.cos(angle_rad + np.pi/2)
# y1_minor = center[1] - 0.5 * minor_axis_length * np.sin(angle_rad + np.pi/2)
# x2_minor = center[0] + 0.5 * minor_axis_length * np.cos(angle_rad + np.pi/2)
# y2_minor = center[1] + 0.5 * minor_axis_length * np.sin(angle_rad + np.pi/2)

# # # Plot the ellipse, major axis, and minor axis
# # fig, ax = plt.subplots()
# # # ellipse = Ellipse(center, width, height, angle=angle, fill=False, color='b', linestyle='--', label='Ellipse')
# # major_axis = plt.Line2D([x1_major, x2_major], [y1_major, y2_major], color='r', label='Major Axis')
# # minor_axis = plt.Line2D([x1_minor, x2_minor], [y1_minor, y2_minor], color='g', label='Minor Axis')
# # frames = plt.imshow(frame)
# # filler = plt.imshow(pt, alpha=0.5)

# # ax.add_patch(ellipse)
# # ax.add_line(major_axis)
# # ax.add_line(minor_axis)
# # ax.imshow(frame)
# # ax.scatter(x,y)
# # ax.imshow(pt, alpha=0.5)

# # Set axis limits and labels
# # ax.set_xlim(center[0] - height, center[0] + height)
# # ax.set_ylim(center[1] - height, center[1] + height)
# # ax.set_aspect('equal', 'box')
# # plt.xlabel('X-axis')
# # plt.ylabel('Y-axis')

# # # Show the plot with both major and minor axes
# # plt.legend()
# # plt.grid()
# # plt.show()
# # plt.show()

# import numpy as np
# import matplotlib.pyplot as plt

# # Assuming you have your image data in the 'frame' variable
# image_data = frame

# # Define the coordinates of the major axis endpoints (x1, y1) and (x2, y2)
# x1, y1 = x1_major.astype(int), y1_major.astype(int)  # Replace with your coordinates
# x2, y2 = x2_major.astype(int), y2_major.astype(int)  # Replace with your coordinates

# # Calculate the slope and direction of the line
# slope = (y2 - y1) / (x2 - x1)

# # Extend the line by a certain length (e.g., 10 pixels) in both directions
# extension_length = 100

# # Calculate the extended endpoints
# x1_extended = x1 - extension_length
# y1_extended = y1 - int(extension_length * slope)
# x2_extended = x2 + extension_length
# y2_extended = y2 + int(extension_length * slope)

# # Calculate the range of X values along the extended line
# x_values_extended = np.arange(x1_extended, x2_extended + 1)

# # Calculate the corresponding Y values along the extended line using the slope
# y_values_extended = (y1_extended + (x_values_extended - x1_extended) * slope).astype(int)

# # Ensure the extended line stays within the image boundaries
# x_values_extended = np.clip(x_values_extended, 0, image_data.shape[1] - 1)
# y_values_extended = np.clip(y_values_extended, 0, image_data.shape[0] - 1)

# # Extract pixel values along the extended line from the image
# line_pixel_values_extended = [image_data[y, x] for x, y in zip(x_values_extended, y_values_extended)]

# threshold = 104  # Adjust the threshold as needed
# # Example usage with a threshold
# result = find_min_left_right_max_with_threshold(line_pixel_values_extended, threshold)
# print(result)
# pupil_area = (result['right_maximum'][0] - result['left_maximum'][0]) * np.pi
# print('pupil area',pupil_area)

# # Plot the image with the major axis and the extended line
# plt.figure(figsize=(12, 5))

# plt.subplot(1, 2, 1)
# plt.imshow(image_data, cmap='gray')
# # plt.plot([x1, x2], [y1, y2], color='red', linewidth=2, label='Major Axis')
# plt.plot(x_values_extended, y_values_extended, color='blue', linewidth=2, label='Extended Line through Major Axis')
# plt.title('Image with Major Axis and Extended Line')

# # Display the pixel values along the extended line
# plt.subplot(1, 2, 2)
# plt.plot(line_pixel_values_extended, label='Pixel Values')
# plt.axvline(result['left_maximum'][0],color='red')
# plt.axvline(result['right_maximum'][0],color='red')
# plt.axvline(result['minimum'][0],color='green')
# plt.xlabel('Pixel Position')
# plt.ylabel('Pixel Value')
# plt.title('Pixel Values along the Extended Line')
# plt.legend()

# plt.tight_layout()
# plt.show()


# # Example usage
# # line_values = [10, 20, 30, 15, 5, 25, 20, 35, 30]

################################################################
################################################################
################################################################

