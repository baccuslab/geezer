import numpy as np
import matplotlib.pyplot as plt

top = [340.1, 79.9]
bottom = [206.4, 219.5]

frame = '/home/grandline/geezer/og_frame.npy'
frame = np.load(frame)

print(int(round(top[0])), int(round(top[1])))

top_x_int = int(round(top[0]))
top_y_int = int(round(top[1]))

# Calculate the midpoint coordinates
midpoint = [(top[0] + bottom[0]) / 2, (top[1] + bottom[1]) / 2]
midpoint = [round(midpoint[0]), round(midpoint[1])]
print(top)
print(midpoint)
print(bottom)
midpoint_x = int(midpoint[0])
midpoint_y = int(midpoint[1])

bottom_x_int = int(round(bottom[0]))
bottom_y_int = int(round(bottom[1]))
# Check if the coordinates are within the valid range
if 0 <= top_y_int < frame.shape[0] and 0 <= top_x_int < frame.shape[1]:
    pixel_intensity = frame[top_y_int, top_x_int]
    print(f"Pixel intensity at top point (x={top_x_int}, y={top_y_int}) is {pixel_intensity}")
else:
    print("Coordinates are out of bounds")

if 0 <= midpoint_y < frame.shape[0] and 0 <= midpoint_x < frame.shape[1]:
    pixel_intensity = frame[midpoint_y, midpoint_x]
    print(f"Pixel intensity at midpoint (x={midpoint_x}, y={midpoint_y}) is {pixel_intensity}")
else:
    print("Coordinates are out of bounds")

if 0 <= bottom_y_int < frame.shape[0] and 0 <= bottom_x_int < frame.shape[1]:
    pixel_intensity = frame[bottom_y_int, bottom_x_int]
    print(f"Pixel intensity at bottom point (x={bottom_x_int}, y={bottom_y_int}) is {pixel_intensity}")
else:
    print("Coordinates are out of bounds")

plt.imshow(frame)
plt.scatter(top[0], top[1], c='r')
plt.scatter(bottom[0], bottom[1], c='r')
plt.plot([top[0], bottom[0]], [top[1], bottom[1]])
plt.scatter(midpoint[0], midpoint[1], c='r')
plt.show()