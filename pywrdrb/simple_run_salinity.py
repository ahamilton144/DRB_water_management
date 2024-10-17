"""
This script is used to run a Pywr-DRB simulation.

Usage:
python3 drb_run_sim.py <inflow_type>

"""
import os
import sys
import time
import h5py
import pandas as pd
import matplotlib.pyplot as plt
from pywr.model import Model
from pywr.recorders import TablesRecorder
from pathnavigator import PathNavigator

root_dir = r"C:\Users\CL\Documents\GitHub\Pywr-DRB"
pn = PathNavigator(root_dir)
pn.chdir()
pn.mkdir("temp_lstm_exp")
pn.temp_lstm_exp.mkdir("outputs")
pn.temp_lstm_exp.mkdir("figures")

#pn.add_to_sys_path()
#pn.pywrdrb.chdir()
#sys.path.insert(0, os.path.abspath("./"))
#sys.path.insert(0, os.path.abspath("../"))

import pywrdrb.parameters.general
import pywrdrb.parameters.ffmp
import pywrdrb.parameters.starfit
import pywrdrb.parameters.lower_basin_ffmp
from pywrdrb.utils.options import inflow_type_options
from pywrdrb.model_builder import PywrdrbModelBuilder

class Timer:
    def __init__(self):
        self.start_time = None
        self.end_time = None
    
    def start(self):
        """Start the timer."""
        self.start_time = time.time()
        self.end_time = None
        print("Timer started.")
    
    def stop(self):
        """Stop the timer."""
        if self.start_time is None:
            raise Exception("Timer has not been started yet!")
        self.end_time = time.time()
        print("Timer stopped.")
    
    def elapsed_time(self):
        """Get the elapsed time in HH:mm:ss format."""
        if self.start_time is None:
            raise Exception("Timer has not been started yet!")
        
        # If the timer is still running, calculate elapsed time until now
        if self.end_time is None:
            elapsed_seconds = time.time() - self.start_time
        else:
            elapsed_seconds = self.end_time - self.start_time
        
        # Convert the elapsed time into hours, minutes, and seconds
        hours, remainder = divmod(elapsed_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"
    
    def elapsed_seconds(self):
        """Get the elapsed time in seconds."""
        if self.start_time is None:
            raise Exception("Timer has not been started yet!")
        
        # If the timer is still running, calculate elapsed time until now
        if self.end_time is None:
            elapsed_seconds = time.time() - self.start_time
        else:
            elapsed_seconds = self.end_time - self.start_time
        return elapsed_seconds
    
    def reset(self):
        """Reset the timer."""
        self.start_time = None
        self.end_time = None
        print("Timer reset.")

def run_model(inflow_type, predict_salinity, timer, torch_seed=4, new_end_date=None):
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
    
    if new_end_date is not None:
        end_date = new_end_date
        
    if predict_salinity:
        sname = "with_salinity_lstm"
    else:
        sname = "no_salinity_lstm"
        
    # Set the filename based on inflow type
    model_filename = rf"{pn.pywrdrb.model_data.dir()}/drb_model_full_{inflow_type}_{sname}_torchseed{torch_seed}.json"
    output_filename = rf"{pn.temp_lstm_exp.outputs.dir()}/drb_output_{inflow_type}_{sname}_torchseed{torch_seed}.hdf5"
    
    print(f"Running drb_model_full_{inflow_type}_{sname}_torchseed{torch_seed}.json")
    ### make model json files
    mb = PywrdrbModelBuilder(
        inflow_type, 
        start_date, 
        end_date,
        options={
            "inflow_ensemble_indices":None,
            "use_hist_NycNjDeliveries":True, 
            "predict_temperature":False, 
            "temperature_torch_seed": torch_seed,
            "predict_salinity":predict_salinity, 
            "salinity_torch_seed": torch_seed
            })
    mb.make_model()
    mb.write_model(model_filename)

    #timer.start()
    ### Load the model
    model = Model.load(model_filename)
    
    ### Add a storage recorder
    TablesRecorder(
        model, output_filename, parameters=[p for p in model.parameters if p.name]
    )
    
    timer.start()
    ### Run the model
    stats = model.run()
    #stats_df = stats.to_dataframe()
    timer.stop()
    print(f"Time used from loading model to complete the simulation: {timer.elapsed_time()}")
    return timer.elapsed_seconds()

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

inflow_type = 'nhmv10_withObsScaled'
predict_salinity = True
torch_seed = 8
if predict_salinity:
    sname = "with_salinity_lstm"
else:
    sname = "no_salinity_lstm"
new_end_date = "2016-12-31"#"2003-10-01"

run_model(
    inflow_type=inflow_type, 
    predict_salinity=predict_salinity, 
    timer=Timer(), 
    torch_seed=torch_seed, 
    new_end_date=new_end_date
    )

output_filename = rf"{pn.temp_lstm_exp.outputs.dir()}/drb_output_{inflow_type}_{sname}_torchseed{torch_seed}.hdf5"
hdf5_data = hdf5_to_dict(output_filename)

#%%
var_list = [
    "drought_level_agg_nyc",
    "salt_front_adjust_factor_delMontague",
    "salt_front_adjust_factor_delTrenton",
    "mrf_target_delMontague",
    "mrf_target_delTrenton",
    "salt_front_river_mile",
    ]
name_list = [
    "level",
    "salt_front_factor_Montague",
    "salt_front_factor_Trenton",
    "mrf_target_Montague",
    "mrf_target_Trenton",
    "salt_front_river_mile",
    ]
df = pd.DataFrame()
for name, var in zip(name_list, var_list):
    df[name] = hdf5_data[var].flatten()
df.index = pd.date_range(start="1983-10-01", end=new_end_date, freq="D")

ax = df["level"].plot()
ax.set_ylabel("Level index")
plt.show()
