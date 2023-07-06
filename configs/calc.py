import json
import numpy as np

fname = '/home/jerome/geezer/configs/d222_calibration_jbm109.json'
with open(fname) as f:
    config = json.load(f)

mon = config[0]

sub = np.array(mon['left_co']) - np.array(mon['right_co'])

factor = 3.5/np.sqrt(np.sum(sub**2))

print('nw')
print(mon['left_co'] - factor*sub)

factor = 3/np.sqrt(np.sum(sub**2))

print('sw')
print(mon['left_co'] - factor*sub)

factor = 3/np.sqrt(np.sum(sub**2))
print('ne')
print(mon['right_co'] + factor*sub)

print('se')
factor = 0.6/np.sqrt(np.sum(sub**2))
print(mon['right_co'] + factor*sub)

