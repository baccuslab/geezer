from geezer import Trajectory

h5_file = '/home/dennis/Data/data_3/imu/imu_geezer_output_round_2.h5'
geometry_file = '/home/dennis/Code/geezer/geometries/jan_2024_geometry.json'
mapping_file = '/home/dennis/Code/geezer/geometries/jan_2024_mapping.json'

reference_frame = 2200 
t = Trajectory(h5_file, geometry_file, mapping_file, reference_frame)



