import numpy as np

def calculate_led_angles(led_co, basis):
    tx,ty,tz = led_co @ np.linalg.inv(basis)
    
    el = np.arctan2(ty,np.sqrt(tx**2 + tz**2)) 
    az = np.arctan2(tx,tz)
    
    return el, az
