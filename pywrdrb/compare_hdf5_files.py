import os
import h5py
import numpy as np

def compare_hdf5(file1, file2):
    def compare_datasets(dset1, dset2, path):
        differences = []
        # Check if the shapes are different
        if dset1.shape != dset2.shape:
            differences.append(f"Shape mismatch at {path}: {dset1.shape} vs {dset2.shape}")
        
        # Check if the data types are different
        if dset1.dtype != dset2.dtype:
            differences.append(f"Data type mismatch at {path}: {dset1.dtype} vs {dset2.dtype}")
        
        # Check if the values are different
        if not np.array_equal(dset1[()], dset2[()]):
            diff_values = np.where(dset1[()] != dset2[()])
            differences.append(f"Value mismatch at {path}, differences at indices: {diff_values}")
        
        return differences
    
    def compare_groups(group1, group2, path="/"):
        differences = []
        
        # Check if the keys (i.e., datasets or groups) in the groups are the same
        keys1 = set(group1.keys())
        keys2 = set(group2.keys())
        
        # Find missing datasets or groups
        missing_in_file2 = keys1 - keys2
        missing_in_file1 = keys2 - keys1
        
        if missing_in_file2:
            differences.append(f"Missing in {file2}: {missing_in_file2} at path {path}")
        if missing_in_file1:
            differences.append(f"Missing in {file1}: {missing_in_file1} at path {path}")
        
        # Compare common keys
        for key in keys1.intersection(keys2):
            item1 = group1[key]
            item2 = group2[key]
            new_path = path + key + "/"
            
            if isinstance(item1, h5py.Dataset) and isinstance(item2, h5py.Dataset):
                differences.extend(compare_datasets(item1, item2, new_path))
            elif isinstance(item1, h5py.Group) and isinstance(item2, h5py.Group):
                differences.extend(compare_groups(item1, item2, new_path))
            else:
                differences.append(f"Type mismatch at {new_path}: one is a group, the other is a dataset")
        
        return differences
    
    # Open both HDF5 files
    with h5py.File(file1, 'r') as f1, h5py.File(file2, 'r') as f2:
        differences = compare_groups(f1, f2)
        
        if differences:
            return differences
            # for diff in differences:
            #     print(diff)
        else:
            print("The HDF5 files are the same.")
            return None

# Example usage

inflow_list = ["nhmv10_withObsScaled", "nwmv21_withObsScaled", "nhmv10", "nwmv21"]

wd = r"C:\Users\cl2769\Documents\GitHub\Pywr-DRB\output_data"

diff = {}
for inflow in inflow_list:
    file1 = os.path.join(wd, f'drb_output_{inflow}.hdf5')
    file2 = os.path.join(wd, f'drb_output_{inflow}_cl.hdf5')
    
    print(inflow)
    diff[inflow] = compare_hdf5(file1, file2)

# Output
# nhmv10_withObsScaled
# The HDF5 files are the same.
# nwmv21_withObsScaled
# The HDF5 files are the same.
# nhmv10
# The HDF5 files are the same.
# nwmv21
# The HDF5 files are the same.

