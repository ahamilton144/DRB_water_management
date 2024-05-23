import h5py
import pandas as pd
import numpy as np
import warnings

from .pywr_drb_node_data import obs_site_matches

pywrdrb_all_nodes = list(obs_site_matches.keys())

def combine_batched_hdf5_outputs(batch_files, combined_output_file):
    """
    Aggregate multiple HDF5 files into a single HDF5 file.
    
    Args:
        batch_files (list): List of HDF5 files to combine.
        combined_output_file (str): Full output file path & name to write combined HDF5.
    
    Returns:
        None
    """
    with h5py.File(combined_output_file, 'w') as hf_out:
        # Since keys are same in all files, we just take keys from the first file
        with h5py.File(batch_files[0], 'r') as hf_in:
            keys = list(hf_in.keys())
            
            time_key = None
            datetime_key_opts = ['time', 'date', 'datetime']
            for dt_key in datetime_key_opts:
                if dt_key in keys:
                    time_key = dt_key
                    break
            if time_key is None:
                err_msg = f'No time key found in HDF5 file {batch_files[0]}.'
                err_msg += f' Expected keys: {datetime_key_opts}'
                raise ValueError(err_msg)
            time_array=hf_in[time_key][:]
            
        for key in keys:
            add_key_to_output = True

            # Skip datetime keys and scenarios    
            if key in datetime_key_opts + ['scenarios']:
                continue
    
            # Accumulate data from all files for a specific key
            data_for_key = []
            for file in batch_files:
                with h5py.File(file, 'r') as hf_in:
                    if key not in hf_in:
                        add_key_to_output = False

                    if add_key_to_output:
                        data_for_key.append(hf_in[key][:,:])
            
            if add_key_to_output:
                # Concatenate along the scenarios axis
                combined_data = np.concatenate(data_for_key, axis=1)

                # Write combined data to the output file
                hf_out.create_dataset(key, data=combined_data)
        
        # Time needs to be handled differently since 1D
        hf_out.create_dataset(time_key, data=time_array)
    return


def export_ensemble_to_hdf5(dict, 
                            output_file):
    """
    Export a dictionary of ensemble data to an HDF5 file.
    Data is stored in the dictionary as {realization number (int): pd.DataFrame}.
    
    Args:
        dict (dict): A dictionary of ensemble data.
        output_file (str): Full output file path & name to write HDF5.
        
    Returns:
        None    
    """
    
    dict_keys = list(dict.keys())
    N = len(dict)
    T, M = dict[dict_keys[0]].shape
    column_labels = dict[dict_keys[0]].columns.to_list()
    
    with h5py.File(output_file, 'w') as f:
        for key in dict_keys:
            data = dict[key]
            datetime = data.index.astype(str).tolist() #.strftime('%Y-%m-%d').tolist()
            
            grp = f.create_group(key)
                    
            # Store column labels as an attribute
            grp.attrs['column_labels'] = column_labels

            # Create dataset for dates
            grp.create_dataset('date', data=datetime)
            
            # Create datasets for each array subset from the group
            for j in range(M):
                dataset = grp.create_dataset(column_labels[j], 
                                             data=data[column_labels[j]].to_list())
    return


def get_hdf5_realization_numbers(filename):
    """
    Checks the contents of an hdf5 file, and returns a list 
    of the realization ID numbers contained.
    Realizations have key 'realization_i' in the HDF5.

    Args:
        filename (str): The HDF5 file of interest

    Returns:
        list: Containing realizations ID numbers; realizations have key 'realization_i' in the HDF5.
    """
    realization_numbers = []
    with h5py.File(filename, 'r') as file:
        # Get the keys in the HDF5 file
        keys = list(file.keys())

        # Get the df using a specific node key
        node_data = file[keys[0]]
        column_labels = node_data.attrs['column_labels']
        
        # Iterate over the columns and extract the realization numbers
        for col in column_labels:
            
            # handle different types of column labels
            if type(col) == str:
                if col.startswith('realization_'):
                    # Extract the realization number from the key
                    realization_numbers.append(int(col.split('_')[1]))
                else:
                    realization_numbers.append(col)
            elif type(col) == int:
                realization_numbers.append(col)
            else:
                err_msg = f'Unexpected type {type(col)} for column label {col}.'
                err_msg +=  f'in HDF5 file {filename}'
                raise ValueError(err_msg)
    return realization_numbers


def extract_realization_from_hdf5(hdf5_file, 
                                  realization,
                                  stored_by_node=False):
    """
    Pull a single inflow realization from an HDF5 file of inflows. 

    Args:
        hdf5_file (str): The filename for the hdf5 file
        realization (int): Integer realization index
        stored_by_node (bool): Whether the data is stored with node name as key.

    Returns:
        pandas.DataFrame: A DataFrame containing the realization
    """
    
    with h5py.File(hdf5_file, 'r') as f:
        if stored_by_node:
            # Extract timeseries data from realization for each node
            data = {}
                
            for node in pywrdrb_all_nodes:
                node_data = f[node]
                column_labels = node_data.attrs['column_labels']
                
                err_msg = f'The specified realization {realization} is not available in the HDF file.'
                assert(realization in column_labels), err_msg + f' Realizations available: {column_labels}'
                data[node] = node_data[realization][:]
            
            dates = node_data['date'][:].tolist()
            
        else:
            realization_group = f[realization]
            
            # Extract column labels
            column_labels = realization_group.attrs['column_labels']
            # Extract timeseries data for each location
            data = {}
            for label in column_labels:
                dataset = realization_group[label]
                data[label] = dataset[:]
            
            # Get date indices
            dates = realization_group['date'][:].tolist()
        data['datetime'] = dates
        
    # Combine into dataframe
    df = pd.DataFrame(data, index = dates)
    df.index = pd.to_datetime(df.index.astype(str))
    return df