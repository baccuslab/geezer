import sys
from numpy import sqrt
import numpy as np

def point_on_line(A, B, d):
    x1, y1 = A
    x2, y2 = B
    distance_AB = sqrt((x2 - x1)**2 + (y2 - y1)**2)
    if d > distance_AB:
        raise ValueError("Distance d is greater than the length of the line segment AB.")
    ratio = d / distance_AB
    x = x1 + ratio * (x2 - x1)
    y = y1 + ratio * (y2 - y1)
    return x, y

A = (32.25, 3)
B = (19, 19.5)

print(point_on_line(A,B, 2+7/8))
