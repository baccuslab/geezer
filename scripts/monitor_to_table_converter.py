import sys
from numpy import sqrt
import numpy as np

def point_on_line(A, B, d):
    x1, y1 = A
    x2, y2 = B
    distance_AB = sqrt((x2 - x1)**2 + (y2 - y1)**2)
    ratio = d / distance_AB
    x = x1 + ratio * (x2 - x1)
    y = y1 + ratio * (y2 - y1)
    return x, y

A = (32.25, 3)
B = (19, 19.5)

left_monitor_coordinate = (-13.875, -6.929)
right_monitor_coordinate = (0, 9)

relative_to_monitor = {}

relative_to_monitor['nw'] = 100
relative_to_monitor['ne'] = 21+1/8-3.335
relative_to_monitor['sw'] = 2.8175
relative_to_monitor['se'] = 21+1/8-2.066

for key in relative_to_monitor: 
    distance = relative_to_monitor[key]
    x, y = point_on_line(left_monitor_coordinate, right_monitor_coordinate, distance)

    x = np.round(x, 6)
    y = np.round(y, 6)
    print(key, x, y)

def point_on_line(A, B, added):
    x1, y1, z1 = A
    x2, y2, z2 = B
    distance_AB = sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)

    d = distance_AB + added

    ratio = d / distance_AB
    x = x1 + ratio * (x2 - x1)
    y = y1 + ratio * (y2 - y1)
    z = z1 + ratio * (z2 - z1)

    return x, y, z

A = (0, 0, 14.375)
B = (-3.776, 4.664, 24.375)

print(point_on_line(A, B, 10.5))
