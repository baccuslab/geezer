import h5py as h5
import sys
sys.path.append('/home/dennis/Code/geezer/src/')
import utils
import numpy as np
import matplotlib.pyplot as plt

plot_raw_trajectories = True
clean = True


# with h5.File('/home/dennis/Data/data_1/GolDRoger/goldroger_geezer.h5', 'r') as file:
with h5.File('/home/dennis/Data/data_3/imu/imu_geezer_output.h5', 'r') as file:
    print(file.keys())
    # if plot_raw_trajectories:
    #     fig, ax = plt.subplots(3,1, sharex=True) 


    #     ax[0].set_title('thetas')
    #     for key in list(file['raw_trajectories'].keys()):
    #         out = file['raw_trajectories'][key][:]
    #         ax[0].plot(np.rad2deg(out[:,0] - out[120000,0]), label=key)
    #     ax[0].legend()
    #     print(out.shape)
    #     ax[1].set_title('phis')
    #     for key in list(file['raw_trajectories'].keys()):
    #         out = file['raw_trajectories'][key][:]
            
    #         zi = out[:,1] - out[30000,1]
    #         if key == 'sw':
    #             zi *= -1
    #             # zi = zi - 2*np.pi
    #         ax[1].plot(np.rad2deg(zi)-np.rad2deg(zi[120000]), label=key)
    #     ax[1].legend()
    #     ax[2].set_title('phis')
    #     for key in list(file['raw_trajectories'].keys()):
    #         out = file['interp_trajectories'][key][:]
            
    #         zi = out[:,1] - out[30000,1]
    #         if key == 'sw':
    #             zi *= -1
    #             # zi = zi - 2*np.pi
    #         ax[2].plot(np.rad2deg(zi)-np.rad2deg(zi[120000]), label=key)
    #     ax[2].legend()


    #     plt.show()
        

    #     # plt.plot(file['interp_trajectories']['ne'][:,0], 'ko-')
        
        
    #     # fig = plt.gcf()
    #     # plt.plot(file['raw_trajectories']['ne'][:,0], 'ro-', alpha=0.5)
    #     # plt.show()
    
    # # if clean:
    #     # # First, find places where LEDs go haywire
    #     # led_pix_co = {}

    #     # led_pix_co['sw'] = file['fiducial_coordinates']['sw'][:]
    #     # led_pix_co['se'] = file['fiducial_coordinates']['se'][:]
    #     # led_pix_co['ne'] = file['fiducial_coordinates']['ne'][:]
    #     # cam_pix_co = file['fiducial_coordinates']['cam'][:]
        
        
    #     # led_pix_co, cam_pix_co = utils.sanitize_fiducial_coordinates(led_pix_co, cam_pix_co)
