import json
import numpy as np

def load_geometry_json(json_path):
    objects = {}
    with open(json_path, 'r') as f:
        data = json.load(f)
    for di in data:
        objects[di['name']] = {}
        objects[di['name']]['x'] = di['x']
        objects[di['name']]['y'] = di['y']
        objects[di['name']]['z'] = di['z']
    return objects

def get_centered_geometry(geometry_data, mapping):
    observer = mapping['pupil']
    observer = geometry_data[observer]
    observer_geometry= np.array([observer['x'], observer['y'], observer['z']])


    temp = list(mapping['camera'].keys())
    assert len(temp) == 1
    camera = mapping['camera'][temp[0]]
    camera = geometry_data[camera]
    camera_geometry = np.array([camera['x'], camera['y'], camera['z']])

    leds = list(mapping['led'].keys())
    led_geometry = {}
    for led_name in leds:
        led = mapping['led'][led_name]
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
def get_basis(centered_camera_coordinates):
    _z = centered_camera_coordinates
    _x = np.cross([_z[0], _z[1], 0], [0, 0, -1])
    _x[2] = 0
    _y = np.cross(_x, _z)
    _y *= -1

    print(_x, _y, _z)
    
    _x_norm = np.linalg.norm(_x)
    _y_norm = np.linalg.norm(_y)
    _z_norm = np.linalg.norm(_z)

    _z_unit = np.array(_z / _z_norm)
    _x_unit = np.array(_x / _x_norm)
    _y_unit = np.array(_y / _y_norm)

    v_basis = np.array([_x_unit, _y_unit, _z_unit]).T
    return v_basis

def get_led_angle(led_co, basis):
    tx,ty,tz = np.linalg.inv(basis) @ led_co.T
    el = np.arctan(ty / np.sqrt(tx**2 + tz**2))
    az = np.arctan(tx / tz)
    
    return el, az

