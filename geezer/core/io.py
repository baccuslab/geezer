import numpy as np
import json

def load_mapping_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

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

def center_geometry(geometry_data, mapping):
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
