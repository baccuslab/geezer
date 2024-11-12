import numpy as np
import math

def get_length(L, R):
    # Calculate the direction vector from L to R
    direction_vector = [R[0] - L[0], R[1] - L[1]]
    
    # Calculate the length of the direction vector
    length = math.sqrt(direction_vector[0] ** 2 + direction_vector[1] ** 2)
    print(f"Length of direction vector: {length}")
    return length

def camera_position(L, R, offset_from_left):
    direction_vector = [R[0] - L[0], R[1] - L[1]]
    
    # Calculate the length of the direction vector
    length = math.sqrt(direction_vector[0] ** 2 + direction_vector[1] ** 2)
    
    # Normalize the direction vector
    unit_vector = [direction_vector[0] / length, direction_vector[1] / length]
    
    # Calculate the camera position by moving `offset_from_left` units from L along the direction vector
    camera_x = L[0] + unit_vector[0] * offset_from_left
    camera_y = L[1] + unit_vector[1] * offset_from_left
    
    return (camera_x, camera_y)

# Example usage
L = [-13.5, 0.5]
R = [0, 17]
offset_from_left = get_length(L,R) - 6.25
#
camera_coordinates = camera_position(L, R, offset_from_left)
print(f"Camera coordinates: {camera_coordinates}")

