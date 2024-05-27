import matplotlib.pyplot as plt
import numpy as np


def point_on_line(start, end, N):

    x1,y1 = start
    x2,y2 = end
    # Calculate the total length of the line segment
    length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    # Calculate the ratio of N to the total length
    ratio = N / length
    
    # Calculate the coordinates of the point at distance N from the start
    x = x1 + ratio * (x2 - x1)
    y = y1 + ratio * (y2 - y1)
    
    return x, y

# Test the function
monitor_start = (-13.5, -7.6)
monitor_end = (0, 8.75)
N = 21.25-6.25


point_x, point_y = point_on_line(monitor_start, monitor_end, N)

# Visualization
x1,x2 = monitor_start[0], monitor_end[0]
y1,y2 = monitor_start[1], monitor_end[1]
plt.figure(figsize=(8, 6))
plt.plot([x1, x2], [y1, y2], 'b-', label='Line segment')
plt.plot(point_x, point_y, 'ro', label=f'Point {N} inches from start')
plt.text(point_x, point_y, f'({point_x:.2f}, {point_y:.2f})', fontsize=12, ha='right')
plt.scatter([x1, x2], [y1, y2], c='g', label='Start and End Points')
plt.legend()
plt.xlabel('X coordinate (inches)')
plt.ylabel('Y coordinate (inches)')
plt.title('Point on Line Segment')
plt.grid(True)
plt.axis('equal')
plt.show()

