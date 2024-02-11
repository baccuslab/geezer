import json
import numpy as np


def get_basis(centered_camera_coordinates):
    
    _z = centered_camera_coordinates
    _x = np.cross([_z[0], _z[1], 0], [0, 0, -1])
    _x[2] = 0
    _y = np.cross(_x, _z)
    _y *= -1

    _x_norm = np.linalg.norm(_x)
    _y_norm = np.linalg.norm(_y)
    _z_norm = np.linalg.norm(_z)

    _z_unit = np.array(_z / _z_norm)
    _x_unit = np.array(_x / _x_norm)
    _y_unit = np.array(_y / _y_norm)

    v_basis = np.array([_x_unit, _y_unit, _z_unit])
    return v_basis

def get_led_angle(led_co, basis):
    tx,ty,tz = led_co @ np.linalg.inv(basis)


    el = np.arctan(ty / np.sqrt(tx**2 + tz**2))
    az = np.arctan(tx / tz)
    
    el = np.sign(ty) * np.abs(el)
    az = np.sign(tx) * np.abs(az)
    return el, az

def calc_gaze_angle(pupil_co, led_co, cam_co, led_angles, offset=[0,0]):
    px = pupil_co[0]
    py = pupil_co[1]

    cx = cam_co[0] + offset[0]
    cy = cam_co[1] + offset[1]

    fx = led_co[0]
    fy = led_co[1]

    led_el = led_angles[0]
    led_az = led_angles[1]

    pd = py - cy
    fd = fy - cy
    el = np.arcsin((pd/fd)*(np.sin(led_el/2)))

    pd = px - cx
    fd = fx - cx
    numerator = pd/np.cos(el)
    demoninator = fd / np.cos(led_el/2)
    az = np.arcsin((numerator/demoninator)*(np.sin(led_az/2)))
    return el,az

def ray_trace(coor,angle):
    elevation = angle[0]
    azimuth = angle[1]

    cam_co = coor["camera"]
    eye_co = coor["observer"]


    _z = cam_co - eye_co
    _x = np.cross([_z[0], _z[1], 0], [0, 0, -1])
    _x[2] = 0
    _y = np.cross(_x, _z)*-1  
    _x_norm = np.linalg.norm(_x)
    _y_norm = np.linalg.norm(_y)
    _z_norm = np.linalg.norm(_z)
    _z_unit = np.array(_z / _z_norm)
    _x_unit = np.array(_x / _x_norm)
    _y_unit = np.array(_y / _y_norm)

    # basis transformation matrix from camera coordinates to table coordinates
    v_basis = np.array([_x_unit, _y_unit, _z_unit]).T
    # P: cartesian camera coordinates of pupil
    P = np.array(([np.cos(elevation)*np.sin(azimuth),np.sin(elevation),np.cos(elevation)*np.cos(azimuth)]))*10
    P_table = np.dot(v_basis,P)+eye_co

    #convert cartesian to spherical coordinates
    # r = np.linalg.norm(P_table)

    # theta = np.arccos(P_table[2]/r)

    # phi = np.arctan(P_table[1]/P_table[0])
    theta, phi = cartesian_to_spherical(P_table)
    return theta, phi

