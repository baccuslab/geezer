import h5py as h5
import sys
sys.path.append('/home/dennis/Code/geezer/src/')
import utils
import numpy as np
import matplotlib.pyplot as plt

with h5.File('/home/dennis/goldroger_geezer_output_debugged.h5', 'r') as file:

            
    fig, ax = plt.subplots(2,1, sharex=True) 
    ax[0].set_title('thetas')
    for key in list(file['raw_trajectories'].keys()):
        out = file['raw_trajectories'][key][:]

        ax[0].plot(np.rad2deg(out[:,0] - out[30000,0]), label=key)
    ax[0].legend()
    print(out.shape)
    ax[1].set_title('phis')
    for key in list(file['raw_trajectories'].keys()):
        out = file['raw_trajectories'][key][:]
        
        zi = out[:,1] - out[30000,1]
        if key == 'sw':
            zi *= -1
            # zi = zi - 2*np.pi
        ax[1].plot(np.rad2deg(zi)-np.rad2deg(zi[30000]), label=key)
    ax[1].legend()

    plt.show()
    # print(out.shape)

    dist = utils.euclidean_distance(led_pix_co['sw'], led_pix_co['cam'])
    dist = np.abs(dist - dist[0])

    dist2 = utils.euclidean_distance(led_pix_co['se'], led_pix_co['cam'])
    dist2 = np.abs(dist2 - dist2[0])

    st = dist > 50
    st2 = dist2 > 50
    
    raw_phis = f['raw_trajectories']['sw'][:,1]
    raw_phis = np.rad2deg(raw_phis)

    raw_thetas = f['raw_trajectories']['sw'][:,0]
    raw_thetas = np.rad2deg(raw_thetas)

    raw_phis[st] = np.nan
    raw_phis[st2] = np.nan
    raw_thetas[st] = np.nan
    raw_thetas[st2] = np.nan
    
    w = np.diff(raw_phis) > 3
    w = np.concatenate([w,[False]])
    raw_phis[w] = np.nan

    nans, x = utils.nan_helper(raw_phis)

    raw_phis[nans]= np.interp(x(nans), x(~nans), raw_phis[~nans])

