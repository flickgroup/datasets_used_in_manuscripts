import numpy as np
import dpdata
import glob
import os
import re

converged_folders = np.load("converged_folders.npy").astype(int)

# print(converged_folders)
unconverged_folders = np.load("unconverged_folders.npy")


d_cps = dpdata.MultiSystems()

file_pattern = ''
#
# Get a list of files matching the pattern
# file_list = glob.glob(file_pattern+'*')

# Get all folders in current directory
all_dirs = [d for d in glob.glob("*") if os.path.isdir(d)]

# Filter those whose name is a number
file_list = [d for d in all_dirs if re.fullmatch(r"\d+", d)]

# Sort the folders numerically
file_list = sorted(file_list)
# Print the sorted list
directories = []
for item in file_list:
    if os.path.isdir(item):
        directories.append(item)
directories = sorted(directories, key=lambda x: int(x[len(file_pattern):]))
directories = np.array(directories)
# print(directories)
directories = directories[converged_folders]
file_count = len(directories)

for idx in range(file_count):
# #for idx in range(100):
    d_cp = dpdata.LabeledSystem(os.path.join(directories[idx], "output"), fmt="cp2k/output")
    d_cps.append(d_cp)

# print(d_cps)
d_cps.to_deepmd_raw("deepmd_data")
