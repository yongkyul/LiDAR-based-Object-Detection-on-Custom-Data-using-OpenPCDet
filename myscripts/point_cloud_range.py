import numpy as np
import os

# Define the path to your .bin files directory
bin_files_path = './data/custom/points'

# Initialize min and max values for x, y, z
min_x, min_y, min_z = float('inf'), float('inf'), float('inf')
max_x, max_y, max_z = float('-inf'), float('-inf'), float('-inf')

# Iterate over all .bin files to calculate the overall point cloud range
for file_name in os.listdir(bin_files_path):
    if file_name.endswith('.bin'):
        file_path = os.path.join(bin_files_path, file_name)
        point_cloud_data = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)

        # Update min and max values
        min_vals = np.min(point_cloud_data[:, :3], axis=0)
        max_vals = np.max(point_cloud_data[:, :3], axis=0)

        min_x, min_y, min_z = min(min_x, min_vals[0]), min(min_y, min_vals[1]), min(min_z, min_vals[2])
        max_x, max_y, max_z = max(max_x, max_vals[0]), max(max_y, max_vals[1]), max(max_z, max_vals[2])

# Combine into point cloud range
point_cloud_range = [min_x, min_y, min_z, max_x, max_y, max_z]

print("POINT_CLOUD_RANGE:", point_cloud_range)