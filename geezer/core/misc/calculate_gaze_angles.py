import numpy as np

def calculate_gaze_angles(pupil_co, led_co, cam_co, led_angles, offset=[0,0]):
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

