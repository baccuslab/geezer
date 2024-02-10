import h5py as h5
import sys
sys.path.append('/home/melandrocyte/Code/geezer/src/')
import utils
import numpy as np
import matplotlib.pyplot as plt

plot_raw_trajectories = True
clean = True


with h5.File('/home/melandrocyte/Code/goldroger_geezer.h5', 'r') as file:
# with h5.File('/home/melandrocyte/imu_geezer_output_round_2.h5', 'r') as file:
    # plt.plot(file['interp_trajectories']['sw'][:,0])
    # plt.show()
    if plot_raw_trajectories:
        # fig, ax = plt.subplots(3,1, sharex=True) 


        # ax[0].set_title('thetas')
        for key in list(file['raw_trajectories'].keys()):
            out = file['raw_trajectories'][key][:]
            plt.plot(np.rad2deg(out[:,0]), label=key)
        print(out.shape)
        plt.show()

        # ax[1].set_title('phis')
        for key in list(file['raw_trajectories'].keys()):
            out = file['raw_trajectories'][key][:]
            
            zi = out[:,1] 
            if key == 'sw':
                zi *= -1
            plt.plot(np.rad2deg(zi), label=key)
            
        plt.show()
        for key in list(file['raw_trajectories'].keys()):
            out = file['interp_trajectories'][key][:]
            
            zi = out[:,1]
            if key == 'sw':
                zi *= -1
                # zi = zi - 2*np.pi
            plt.plot(np.rad2deg(zi), label=key)


        plt.show()
        

        # plt.plot(file['interp_trajectories']['ne'][:,0], 'ko-')
        
        
        # fig = plt.gcf()
        # plt.plot(file['raw_trajectories']['ne'][:,0], 'ro-', alpha=0.5)
        # plt.show()
    
    # if clean:
        # # First, find places where LEDs go haywire
        # led_pix_co = {}

        # led_pix_co['sw'] = file['fiducial_coordinates']['sw'][:]
        # led_pix_co['se'] = file['fiducial_coordinates']['se'][:]
        # led_pix_co['ne'] = file['fiducial_coordinates']['ne'][:]
        # cam_pix_co = file['fiducial_coordinates']['cam'][:]
        
        
        # led_pix_co, cam_pix_co = utils.sanitize_fiducial_coordinates(led_pix_co, cam_pix_co)
