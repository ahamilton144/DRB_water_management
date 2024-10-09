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
from pathnavigator import PathNavigator

root_dir = r"C:\Users\CL\Documents\GitHub\Pywr-DRB"
pn = PathNavigator(root_dir)
pn.chdir()
pn.mkdir("temp_lstm_exp")
pn.temp_lstm_exp.mkdir("models")
pn.temp_lstm_exp.mkdir("outputs")
pn.temp_lstm_exp.mkdir("figures")

#sys.path.insert(0, os.path.abspath("./"))
#sys.path.insert(0, os.path.abspath("../"))

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

def run_model(inflow_type, predict_temperature, timer):
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
        
    if predict_temperature:
        tname = "with_temp_lstm"
    else:
        tname = "no_temp_lstm"
        
    # Set the filename based on inflow type
    model_filename = rf"{pn.temp_lstm_exp.models.dir()}/drb_model_full_{inflow_type}_{tname}.json"
    output_filename = rf"{pn.temp_lstm_exp.outputs.dir()}/drb_output_{inflow_type}_{tname}.hdf5"
    
    print(f"Running drb_model_full_{inflow_type}_{tname}.json")
    ### make model json files
    mb = PywrdrbModelBuilder(
        inflow_type, 
        start_date, 
        end_date,
        options={
            "inflow_ensemble_indices":None,
            "use_hist_NycNjDeliveries":True, 
            "predict_temperature":predict_temperature, 
            "predict_salinity":False, 
            })
    mb.make_model()
    mb.write_model(model_filename)

    timer.start()
    ### Load the model
    model = Model.load(model_filename)
    
    ### Add a storage recorder
    TablesRecorder(
        model, output_filename, parameters=[p for p in model.parameters if p.name]
    )
    
    ### Run the model
    stats = model.run()
    #stats_df = stats.to_dataframe()
    timer.stop()
    print(f"Time used from loading model to complete the simulation: {timer.elapsed_time()}")
    return timer.elapsed_seconds()

time_used = {}
for inflow_type in ['nhmv10_withObsScaled', 'nwmv21_withObsScaled', 'nhmv10', 'nwmv21']:
    for predict_temperature in [True, False]:
        seconds = run_model(inflow_type, predict_temperature, Timer())
        time_used[(inflow_type, predict_temperature)] = seconds
#%%
import h5py
import matplotlib.pyplot as plt
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
#%%
import pandas as pd

var_list = [
    "total_thermal_release_requirement",
    "predicted_max_temperature_at_lordsville_without_thermal_release_mu",
    "predicted_max_temperature_at_lordsville_without_thermal_release_sd",
    "downstream_add_thermal_release_to_target_cannonsville",
    "downstream_add_thermal_release_to_target_pepacton",
    "thermal_release_cannonsville",
    "thermal_release_pepacton",
    "predicted_max_temperature_at_lordsville_mu",
    "predicted_max_temperature_at_lordsville_sd"
    ]
name_list = [
    "thermal_release_req",
    "predicted_max_temp_mu_no_thermal_release",
    "predicted_max_temp_sd_no_thermal_release",
    "downstream_release_cannonsville",
    "downstream_release_pepacton",
    "thermal_release_cannonsville",
    "thermal_release_pepacton",
    "predicted_max_temp_mu",
    "predicted_max_temp_sd"
    ]

df = pd.DataFrame()
for name, var in zip(name_list, var_list):
    df[name] = hdf5_data[var].flatten()
    
times = hdf5_data["time"]
dates = pd.to_datetime({'year': times['year'], 'month': times['month'], 'day': times['day']})
df.index = dates

df["thermal_release"] = df["thermal_release_cannonsville"] + df["thermal_release_pepacton"] 
#%%
# Assuming 'df' has been properly defined and subsetted as in your provided code.
dff = df["2001-6-1": "2001-8-31"]  # Use your desired date range.
dff = df["2005-6-1": "2005-8-31"]

fig, ax = plt.subplots()
x = dff.index
# Plot mean temperature with and without control
ax.plot(x, dff["predicted_max_temp_mu_no_thermal_release"], label="no control", zorder=10)
ax.plot(x, dff["predicted_max_temp_mu"], lw=1, label="with control", zorder=20)

# Adding the uncertainty bands around the mean predictions
ax.fill_between(x, 
                dff["predicted_max_temp_mu_no_thermal_release"] - dff["predicted_max_temp_sd_no_thermal_release"], 
                dff["predicted_max_temp_mu_no_thermal_release"] + dff["predicted_max_temp_sd_no_thermal_release"], 
                color='blue', alpha=0.3, label='+/- 1 sd', zorder=9)

ax.fill_between(x, 
                dff["predicted_max_temp_mu"] - dff["predicted_max_temp_sd"], 
                dff["predicted_max_temp_mu"] + dff["predicted_max_temp_sd"], 
                color='orange', alpha=0.3, label='+/- 1 sd', zorder=9)

# Horizontal line for the temperature threshold
ax.axhline(23.89, color="r", ls="-", label="threshold", lw=1, zorder=100)

# Highlight days with significant thermal releases
for i, v in enumerate(dff["thermal_release"]):
    if v > 10:
        ax.axvline(x[i], zorder=1, lw=3, c="lightgrey")

# Set labels, limits, and legend
ax.set_ylabel("Max daily temperature (degree C)")
ax.set_xlabel("Date")
ax.set_ylim((20, 30))
ax.legend(ncol=3, loc='upper right', bbox_to_anchor=(1, 1.14), frameon=False)
plt.xticks(rotation=45)
plt.show()

# Count days above the threshold
count_above_threshold_no_control = (dff["predicted_max_temp_mu_no_thermal_release"] > 23.89).sum()
count_above_threshold_with_control = (dff["predicted_max_temp_mu"] > 23.89).sum()
print("Number of days without control above 23.89 degree C:", count_above_threshold_no_control)
print("Number of days with control above 23.89 degree C:", count_above_threshold_with_control)

#%%
dff = df["2001-6-1": "2001-8-31"]
#dff = df["2005-6-1": "2005-8-31"]
fig, ax = plt.subplots()
x = dff.index
ax.plot(x, dff["predicted_max_temp_mu_no_thermal_release"], label="no control", zorder=10)
ax.plot(x, dff["predicted_max_temp_mu"], lw=1, label="with control", zorder=20)
ax.axhline(23.89, color="r", ls="-", label="threshold") # => 75F = 23.89

for i, v in enumerate(dff["thermal_release"]):
    if v > 10:
        ax.axvline(x[i], zorder=1, lw=3, c="lightgrey")

ax.set_ylabel("Max daily temperature (degree C)")
ax.set_xlabel("Date")
ax.set_ylim((20, 30))
ax.legend(ncol=3, loc='upper right', bbox_to_anchor=(1, 1.1), frameon=False)
plt.xticks(rotation=45)
plt.show()

count_above_threshold = (df["predicted_max_temp_mu_no_thermal_release"] > 23.89).sum()
print("Number of days without control above 23.89 degree C:", count_above_threshold)

count_above_threshold = (df["predicted_max_temp_mu"] > 23.89).sum()
print("Number of days with control above 23.89 degree C:", count_above_threshold)

