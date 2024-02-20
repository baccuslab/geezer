import numpy as np
import math
import tqdm
import matplotlib.pyplot as plt

def plot_cam_basis(centered_geometry, cam_basis, led_angles):
    axis = plt.axes(projection='3d')
    axis.plot(*centered_geometry['camera'], 'ro')
    axis.plot(*centered_geometry['observer'], 'go')

    led_ids = list(centered_geometry['leds'].keys())



    axis.set_xlabel('X', fontsize=20)
    axis.set_ylabel('Y', fontsize=20)
    axis.set_zlabel('Z', fontsize=20)

    # %% 

    x = [[0, val] for val in cam_basis[0]]
    x = np.array(x) * 30
    y = [[0, val] for val in cam_basis[1]]
    y = np.array(y) * 30
    z = [[0, val] for val in cam_basis[2]]
    z = np.array(z) * 30
    


    axis.set_aspect('auto')
    
    for led_id in led_ids:
        t, p = led_angles[led_id] 
        print(led_id, np.rad2deg([t, p]))
        r = 20
        temp = np.array([r*np.sin(p) * np.cos(t), r*np.sin(t), r*np.cos(t) * np.cos(p)])

        # print(led_id, t, p, temp)
        temp = np.matmul(temp, cam_basis)


        axis.plot([0, temp[0]], [0, temp[1]], [0, temp[2]], 'y', linewidth=1)

    axis.set(xlim=(-30, 30), ylim=(-30, 30), zlim=(-30, 30))
    axis.plot(*x, 'r')
    axis.plot(*y, 'g')
    axis.plot(*z, 'b')
    axis.plot(*cam_basis[0], 'ko')
    axis.plot(*cam_basis[1], 'ko')
    axis.plot(*cam_basis[2], 'ko')

    for k,v in centered_geometry['leds'].items():
        axis.plot(*v, 'ro', markersize=10)

    plt.show()
def get_cam_basis(centered_coordinates):
    centered_camera_coordinates = centered_coordinates['camera']

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


def center_geometry(geometry_data, mapping):
    observer = mapping['pupil']
    observer = geometry_data[observer]
    observer_geometry= np.array([observer['x'], observer['y'], observer['z']])


    temp = list(mapping['camera'].keys())
    assert len(temp) == 1
    camera = mapping['camera'][temp[0]]
    camera = geometry_data[camera]
    camera_geometry = np.array([camera['x'], camera['y'], camera['z']])

    leds = list(mapping['leds'].keys())
    led_geometry = {}
    for led_name in leds:
        led = mapping['leds'][led_name]
        led = geometry_data[led]
        led = [led['x'], led['y'], led['z']]
        led_geometry[led_name] = np.array(led)


    centered_coordinates = {}
    centered_coordinates['observer'] = observer_geometry - observer_geometry
    centered_coordinates['camera'] = camera_geometry - observer_geometry
    centered_coordinates['leds'] = {}

    for led_name in leds:
        centered_coordinates['leds'][led_name] = led_geometry[led_name] - observer_geometry

    return centered_coordinates

def calculate_led_angles(led_co, basis):
    tx,ty,tz = led_co @ np.linalg.inv(basis)
    
    el = np.arctan2(ty,np.sqrt(tx**2 + tz**2)) 
    az = np.arctan2(tx,tz)
    
    return el, az

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

def calculate_gaze_trajectory(pupil_co, led_co, cam_co, led_angles, offset=[0,0]):
    num_frames = pupil_co.shape[0]
    gaze_angles = np.zeros((num_frames, 2))

    for i in range(num_frames):
        el, az = calculate_gaze_angles(pupil_co[i], led_co[i], cam_co[i], led_angles, offset)
        gaze_angles[i] = [el, az]

    return gaze_angles


def thresholded_nan(signal, pix_disp_thresh=15):
    '''
    This function takes the fiducial coordinates and removes any outliers, 
    filling in missing values (beyond some threshold) with the mean of the
    estimate from other LEDs.

    Use only the LEDs that you wish to use for this estimation in the dict

    Args:
        og_led_pix_co (dict): Dictionary of LED pixel coordinates
        og_cam_pix_co (list): List of camera pixel coordinates

    Returns:
        led_pix_co (dict): Dictionary of LED pixel coordinates
        cam_pix_co (list): List of camera pixel coordinates
    '''

    
    # Set all times when the LED displacement is greater than some threshold to NaN
    v = np.copy(signal)

    median = np.nanmedian(v)

    a = np.where(v> median+pix_disp_thresh)[0]
    b = np.where(v< median-pix_disp_thresh)[0]
        
    v[a] = np.nan
    v[b] = np.nan

    return v

def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]


def basis_transform_to_table(trajectory, basis):
    elevation = trajectory[:,0]
    azimuth = trajectory[:,1]
    P = np.array(([np.cos(elevation)*np.sin(azimuth),np.sin(elevation),np.cos(elevation)*np.cos(azimuth)]))*10
    table_coordinates = P.T @ basis
    
    table_trajectory = []
    for xyz in table_coordinates:
        theta, phi = cartesian_to_spherical(xyz)
        table_trajectory.append([theta, phi])
    return np.array(table_trajectory)

def cartesian_to_spherical(xyz):
    x = xyz[0]
    y = xyz[1]
    z = xyz[2] # r = math.sqrt(x**2 + y**2 + z**2)

    theta = np.arctan2(z, math.sqrt(x**2 + y**2))
    phi = np.arctan2(y, x)
    
    return theta, phi

