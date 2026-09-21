import os
import re
import numpy as np

target_string = "Localization  for spin   1 converged"
converged_folders = []
unconverged_folders = []

# Check each numeric-named folder
for name in os.listdir():
    if os.path.isdir(name) and re.fullmatch(r"\d+", name):
        if int(name) <= 500:
            output_path = os.path.join(name, "output")
            found = False
            if os.path.isfile(output_path):
                with open(output_path, 'r', errors='ignore') as f:
                    for line in f:
                        if target_string in line:
                            converged_folders.append(name)
                            found = True
                            break
            if not found:
                unconverged_folders.append(name)

# Sort both lists numerically
converged_folders = sorted(converged_folders, key=int)
unconverged_folders = sorted(unconverged_folders, key=int)

# Print results
# print("✅ Converged folders:", converged_folders)
# print("❌ Unconverged folders:", unconverged_folders)
print(len(converged_folders), len(unconverged_folders))

np.save("converged_folders.npy", converged_folders)
np.save("unconverged_folders.npy", unconverged_folders)
