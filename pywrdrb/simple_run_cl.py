"""
This script is used to run a Pywr-DRB simulation.

Usage:
python3 drb_run_sim.py <inflow_type>

"""
import sys
import os
import math
import time
from pywr.model import Model
from pywr.recorders import TablesRecorder

os.chdir(r"C:\Users\CL\Documents\GitHub\Pywr-DRB")
sys.path.insert(0, os.path.abspath("./"))
sys.path.insert(0, os.path.abspath("../"))

import pywrdrb.parameters.general
import pywrdrb.parameters.ffmp
import pywrdrb.parameters.starfit
import pywrdrb.parameters.lower_basin_ffmp
import pywrdrb.parameters.temperature

from pywrdrb.make_model import make_model
from pywrdrb.utils.directories import output_dir, model_data_dir, input_dir
from pywrdrb.utils.hdf5 import (
    get_hdf5_realization_numbers,
    combine_batched_hdf5_outputs,
)
from pywrdrb.utils.options import inflow_type_options

t0 = time.time()

### specify inflow type from command line args
# nhmv10_withObsScaled nwmv21_withObsScaled nhmv10 nwmv21
#inflow_type = sys.argv[1]
inflow_type = "nhmv10"

assert (
    inflow_type in inflow_type_options
), f"Invalid inflow_type specified. Options: {inflow_type_options}"

### assume we want to run the full range for each dataset
if (
    inflow_type in ("nwmv21", "nhmv10", "WEAP_29June2023_gridmet")
    or "withObsScaled" in inflow_type
):
    start_date = "1983-10-01"
    end_date = "2016-12-31"
elif "syn_obs_pub" in inflow_type:
    start_date = "1945-01-01"
    end_date = "2021-12-31"
elif "obs_pub" in inflow_type:
    start_date = "1945-01-01"
    end_date = "2022-12-31"

#%%
from pywrdrb.model_builder import PywrdrbModelBuilder
# Set the filename based on inflow type
model_filename = f"{model_data_dir}drb_model_full_{inflow_type}_test_temp.json"


output_filename = f"{output_dir}drb_output_{inflow_type}_test_temp.hdf5"
model_filename = f"{model_data_dir}drb_model_full_{inflow_type}_test_temp.json"

### make model json files
#make_model(inflow_type, model_filename, start_date, end_date)
mb = PywrdrbModelBuilder(
    inflow_type, 
    start_date, 
    end_date,
    options={
        "inflow_ensemble_indices":None,
        "use_hist_NycNjDeliveries":True, 
        "predict_temperature":True, 
        "predict_salinity":False, 
        })
mb.make_model()
mb.write_model(model_filename)

#%%
### Load the model
model = Model.load(model_filename)

### Add a storage recorder
TablesRecorder(
    model, output_filename, parameters=[p for p in model.parameters if p.name]
)

#%%
### Run the model
stats = model.run()
stats_df = stats.to_dataframe()

#%%
import h5py

def hdf5_to_dict(file_path):
    def recursive_dict(group):
        d = {}
        for key, item in group.items():
            if isinstance(item, h5py._hl.dataset.Dataset):
                d[key] = item[()]
            elif isinstance(item, h5py._hl.group.Group):
                d[key] = recursive_dict(item)
        return d

    with h5py.File(file_path, 'r') as f:
        data_dict = recursive_dict(f)

    return data_dict

hdf5_data = hdf5_to_dict(output_filename)





