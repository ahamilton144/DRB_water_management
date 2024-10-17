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
from tqdm import tqdm

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

def run_model(inflow_type, predict_temperature, timer, torch_seed=4, new_end_date=None):
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
        
    if predict_temperature:
        tname = "with_temp_lstm"
    else:
        tname = "no_temp_lstm"
        
    # Set the filename based on inflow type
    model_filename = rf"{pn.pywrdrb.model_data.dir()}/drb_model_full_{inflow_type}_{tname}_torchseed{torch_seed}.json"
    output_filename = rf"{pn.temp_lstm_exp.outputs.dir()}/drb_output_{inflow_type}_{tname}_torchseed{torch_seed}.hdf5"
    
    print(f"Running drb_model_full_{inflow_type}_{tname}_torchseed{torch_seed}.json")
    ### make model json files
    mb = PywrdrbModelBuilder(
        inflow_type, 
        start_date, 
        end_date,
        options={
            "inflow_ensemble_indices":None,
            "use_hist_NycNjDeliveries":True, 
            "predict_temperature":predict_temperature, 
            "temperature_torch_seed": torch_seed,
            "predict_salinity":False, 
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
predict_temperature = True
torch_seed = 8
df_list = []
for torch_seed in tqdm(range(10)):
    if predict_temperature:
        tname = "with_temp_lstm"
    else:
        tname = "no_temp_lstm"
    new_end_date = "2016-12-31"#"1993-10-01"
    
    run_model(
        inflow_type=inflow_type, 
        predict_temperature=predict_temperature, 
        timer=Timer(), 
        torch_seed=torch_seed, 
        new_end_date=new_end_date
        )
    
    output_filename = rf"{pn.temp_lstm_exp.outputs.dir()}/drb_output_{inflow_type}_{tname}_torchseed{torch_seed}.hdf5"
    hdf5_data = hdf5_to_dict(output_filename)
    var_list = [
        "total_thermal_release_requirement",
        "predicted_max_temperature_at_lordville_without_thermal_release_mu",
        "predicted_max_temperature_at_lordville_without_thermal_release_sd",
        "downstream_add_thermal_release_to_target_cannonsville",
        "downstream_add_thermal_release_to_target_pepacton",
        "thermal_release_cannonsville",
        "thermal_release_pepacton",
        "predicted_max_temperature_at_lordville_mu",
        "predicted_max_temperature_at_lordville_sd"
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
    df["thermal_release"] = df["thermal_release_cannonsville"] + df["thermal_release_pepacton"] 
    df["total_release"] = df["downstream_release_cannonsville"] + df["downstream_release_pepacton"] 
    df.index = pd.date_range(start="1983-10-01", end=new_end_date, freq="D")
    df_list.append(df)
#%%
time_used = {}
for inflow_type in ['nhmv10_withObsScaled', 'nwmv21_withObsScaled', 'nhmv10', 'nwmv21']:
    for predict_temperature in [True, False]:
        seconds = run_model(inflow_type, predict_temperature, Timer())
        time_used[(inflow_type, predict_temperature)] = seconds
#%%
df = pd.DataFrame(
    list(time_used.values()),
    index=pd.MultiIndex.from_tuples(time_used.keys(), names=["Inflow type", "Coupled"]),
    columns=["Value"]
    )
df = df.unstack(level=-1)['Value'].rename(columns={True: "coupled LSTM", False: "no LSTM"})
ax = df.plot(kind='bar', figsize=(8, 5), rot=0)
ax.set_xlabel('Inflow type', fontsize=12)
ax.set_ylabel('Runtime (seconds)', fontsize=12)
ax.set_ylim([0, 180])

differences = df['coupled LSTM'] - df['no LSTM']
for i, diff in enumerate(differences):
    ax.text(i, max(df.iloc[i])+5, f'Diff: {diff:.2f}', ha='center', fontsize=10)
    
plt.legend(ncol=2, loc='upper right', bbox_to_anchor=(1, 1.1), frameon=False)
plt.show()
#%%
res = {}
date_range = pd.date_range(start="1983-10-01", end="2016-12-31", freq="D")
for inflow_type in ['nhmv10_withObsScaled', 'nwmv21_withObsScaled', 'nhmv10', 'nwmv21']:
    for m in ['coupled LSTM', 'no LSTM']:
        if m == 'coupled LSTM':
            tname = "with_temp_lstm"
            output_filename = rf"{pn.temp_lstm_exp.outputs.dir()}/drb_output_{inflow_type}_{tname}.hdf5"
            hdf5_data = hdf5_to_dict(output_filename)
            var_list = [
                "total_thermal_release_requirement",
                "predicted_max_temperature_at_lordville_without_thermal_release_mu",
                "predicted_max_temperature_at_lordville_without_thermal_release_sd",
                "downstream_add_thermal_release_to_target_cannonsville",
                "downstream_add_thermal_release_to_target_pepacton",
                "thermal_release_cannonsville",
                "thermal_release_pepacton",
                "predicted_max_temperature_at_lordville_mu",
                "predicted_max_temperature_at_lordville_sd"
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
            df["thermal_release"] = df["thermal_release_cannonsville"] + df["thermal_release_pepacton"] 
            df["total_release"] = df["downstream_release_cannonsville"] + df["downstream_release_pepacton"] 
        else:
            tname = "no_temp_lstm"
            output_filename = rf"{pn.temp_lstm_exp.outputs.dir()}/drb_output_{inflow_type}_{tname}.hdf5"
            hdf5_data = hdf5_to_dict(output_filename)
            var_list = [
                "downstream_release_target_cannonsville",
                "downstream_release_target_pepacton",
                ]
            name_list = [
                "downstream_release_cannonsville",
                "downstream_release_pepacton",
                ]
            df = pd.DataFrame()
            for name, var in zip(name_list, var_list):
                df[name] = hdf5_data[var].flatten()
            df["total_release"] = df["downstream_release_cannonsville"] + df["downstream_release_pepacton"] 
        
        df.index = date_range
        res[(inflow_type, m)] = df
        
#%%
def plot_temp(df, dates=["1983-10-01", "2016-12-31"], add_band=False, inflow_type="", ylim=(20, 30), threshold=23.89, df_obv=None, df_drivers= None, ylim2=(-30, 35)):
    
    
    fig, ax = plt.subplots(figsize=(10,7))
    
    # Plot mean temperature with and without control
    if isinstance(df, list):
        for i, dff in enumerate(df):
            dfff = dff[dates[0]: dates[1]]
            x = dfff.index
            #ax.plot(x, dfff["predicted_max_temp_mu_no_thermal_release"], label="before ctrl", zorder=10)
            ax.plot(x, dfff["predicted_max_temp_mu"], lw=1, label=f"seed{i}", zorder=20)
    else:
        dff = df[dates[0]: dates[1]]
        x = dff.index
        ax.plot(x, dff["predicted_max_temp_mu_no_thermal_release"], label="before ctrl", zorder=10)
        ax.plot(x, dff["predicted_max_temp_mu"], lw=1, label="after ctrl", zorder=20)
    
        # Adding the uncertainty bands around the mean predictions
        if add_band:
            ax.fill_between(x, 
                            dff["predicted_max_temp_mu_no_thermal_release"] - dff["predicted_max_temp_sd_no_thermal_release"], 
                            dff["predicted_max_temp_mu_no_thermal_release"] + dff["predicted_max_temp_sd_no_thermal_release"], 
                            color='blue', alpha=0.3, label='+/- 1 sd', zorder=9)
            
            ax.fill_between(x, 
                            dff["predicted_max_temp_mu"] - dff["predicted_max_temp_sd"], 
                            dff["predicted_max_temp_mu"] + dff["predicted_max_temp_sd"], 
                            color='orange', ls=":", alpha=0.3, label='+/- 1 sd', zorder=9)
    
        # Highlight days with significant thermal releases
        for i, v in enumerate(dff["thermal_release"]):
            if v > 10:
                ax.axvline(x[i], zorder=1, lw=3, c="lightgrey")
        
        
        # Add violation day count
        count_above_threshold_no_control = (dff["predicted_max_temp_mu_no_thermal_release"] > threshold).sum()
        count_above_threshold_with_control = (dff["predicted_max_temp_mu"] > threshold).sum()
        
        ax.text(x=1.02, y=0.3, s=f"#violations: {count_above_threshold_no_control}", color='blue', transform=ax.transAxes)
        ax.text(x=1.02, y=0.25, s=f"#violations: {count_above_threshold_with_control}", color='orange', transform=ax.transAxes)
    
    # Horizontal line for the temperature threshold
    ax.axhline(threshold, color="r", ls="-", label="threshold", lw=1, zorder=100)
    if df_obv is not None:
        df_obv_ = df_obv[dates[0]: dates[1]]
        ax.scatter(df_obv_.index, df_obv_['obv'], color='black', marker='x', s=20, zorder=99)
        count_above_threshold_obv = (df_obv_['obv'] > threshold).sum()
        ax.text(x=1.02, y=0.2, s=f"#violations: {count_above_threshold_obv}", color='k', transform=ax.transAxes)
    
    if df_drivers is not None:
        ax2 = ax.twinx()
        df_drivers[dates[0]: dates[1]].plot(ax=ax2)
        ax2.set_ylim(ylim2)
        ax2.legend()
    # Set labels, limits, and legend
    ax.set_title(inflow_type)
    ax.set_ylabel("Max daily temperature (degree C)")
    ax.set_xlabel("Date")
    ax.set_ylim(ylim)
    ax.legend(ncol=1, loc='center right', bbox_to_anchor=(1.18, 0.5), frameon=False)
    plt.xticks(rotation=45)
    plt.tight_layout()
    # plt.savefig(os.path.join(
    #     pn.temp_lstm_exp.figures.dir(), 
    #     f"temp_ts_{inflow_type}_{dates[0]}_{dates[1]}_{add_band}.png"
    #     ), bbox_inches='tight')
    plt.show()

df_obv = pd.read_csv(os.path.join(pn.temp_lstm_exp.dir(), "stream_temp_obs.csv"), parse_dates=True, index_col=[0])
df_drivers = pd.read_csv(os.path.join(pn.temp_lstm_exp.dir(), "drivers.csv"), parse_dates=True, index_col=[0])
df_drivers = df_drivers[['tmin', 'tmax', 'srad', 'reservoir_release', 'dwallin_mean_temp',
       'prcp', 'ws', 'rhmean', 'rhmin', 'rhmax']]

df_drivers["1993-6-01": "1993-9-30"].plot()
plt.show()

plot_temp(df_list, dates=["1993-6-01", "1993-9-30"], add_band=False, inflow_type="nhmv10_withObsScaled", ylim=(15, 30), threshold=23.89, df_obv=df_obv)

df = df_list[4]
plot_temp(df, dates=["1993-6-01", "1993-9-30"], add_band=False, inflow_type="nhmv10_withObsScaled", ylim=(15, 30), threshold=23.89, df_obv=df_obv, df_drivers=df_drivers[['tmin', 'tmax']])

plot_temp(df, dates=["1993-6-01", "1993-9-30"], add_band=False, inflow_type="nhmv10_withObsScaled", ylim=(15, 30), threshold=23.89, df_obv=df_obv, df_drivers=df_drivers[['srad']], ylim2=(0, 350))
# 
# for inflow_type in ['nhmv10_withObsScaled', 'nwmv21_withObsScaled', 'nhmv10', 'nwmv21']:
#     df = res[(inflow_type, 'coupled LSTM')]
#     plot_temp(df, dates=["1983-10-01", "2016-12-31"], add_band=False, inflow_type=inflow_type, ylim=(22, 30))
    
#     plot_temp(df, dates=["2001-6-1", "2001-8-31"], add_band=False, inflow_type=inflow_type, ylim=(22, 30), df_obv=df_obv)
#     plot_temp(df, dates=["2005-6-1", "2005-8-31"], add_band=False, inflow_type=inflow_type, ylim=(22, 30), df_obv=df_obv)
    
#     plot_temp(df, dates=["2001-6-1", "2001-8-31"], add_band=True, inflow_type=inflow_type, ylim=(22, 30), df_obv=df_obv)
#     plot_temp(df, dates=["2005-6-1", "2005-8-31"], add_band=True, inflow_type=inflow_type, ylim=(22, 30), df_obv=df_obv)

#%%

def plot_total_releases(res, dates=["1983-10-01", "2016-12-31"], inflow_type="", ylim=(180, 700)):
    df_with_lstm = res[(inflow_type, 'coupled LSTM')][dates[0]: dates[1]]
    df_no_lstm = res[(inflow_type, 'no LSTM')][dates[0]: dates[1]]
    
    dff = pd.DataFrame(index=df_with_lstm.index)
    dff['no lstm'] = df_no_lstm['total_release']
    dff['before ctrl'] = df_with_lstm['total_release'] - df_with_lstm["thermal_release"]
    dff['after ctrl'] = df_with_lstm['total_release']
    x = df_with_lstm.index
    fig, ax = plt.subplots(figsize=(6,5))
    
    ax.plot(x, df_no_lstm['total_release'], lw=1, ls="--", c="k", zorder=100, label='no lstm')
    ax.plot(x, df_with_lstm['total_release']- df_with_lstm["thermal_release"], lw=1, ls="-", c="blue", zorder=100, label='before ctrl')
    ax.plot(x, df_with_lstm['total_release'], lw=1, ls="-.", c="orange", zorder=100, label='after ctrl')
    
    # Set labels, limits, and legend
    ax.set_title(inflow_type)
    ax.set_ylabel("Total release from\nCannonsville and Pepacton (MGD)")
    ax.set_xlabel("Date")
    ax.set_ylim(ylim)
    ax.legend(ncol=1, loc='upper left', frameon=False)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(
        pn.temp_lstm_exp.figures.dir(), 
        f"total_release_ts_{inflow_type}_{dates[0]}_{dates[1]}.png"
        ), bbox_inches='tight')
    plt.show()


for inflow_type in ['nhmv10_withObsScaled', 'nwmv21_withObsScaled', 'nhmv10', 'nwmv21']:
    plot_total_releases(res, dates=["2001-6-1", "2001-8-31"], inflow_type=inflow_type)
    plot_total_releases(res, dates=["2005-6-1", "2005-8-31"], inflow_type=inflow_type)




#%%
# # Count days above the threshold
# count_above_threshold_no_control = (dff["predicted_max_temp_mu_no_thermal_release"] > 23.89).sum()
# count_above_threshold_with_control = (dff["predicted_max_temp_mu"] > 23.89).sum()
# print("Number of days without control above 23.89 degree C:", count_above_threshold_no_control)
# print("Number of days with control above 23.89 degree C:", count_above_threshold_with_control)

# #%%
# import pandas as pd

# var_list = [
#     "total_thermal_release_requirement",
#     "predicted_max_temperature_at_lordville_without_thermal_release_mu",
#     "predicted_max_temperature_at_lordville_without_thermal_release_sd",
#     "downstream_add_thermal_release_to_target_cannonsville",
#     "downstream_add_thermal_release_to_target_pepacton",
#     "thermal_release_cannonsville",
#     "thermal_release_pepacton",
#     "predicted_max_temperature_at_lordville_mu",
#     "predicted_max_temperature_at_lordville_sd"
#     ]
# name_list = [
#     "thermal_release_req",
#     "predicted_max_temp_mu_no_thermal_release",
#     "predicted_max_temp_sd_no_thermal_release",
#     "downstream_release_cannonsville",
#     "downstream_release_pepacton",
#     "thermal_release_cannonsville",
#     "thermal_release_pepacton",
#     "predicted_max_temp_mu",
#     "predicted_max_temp_sd"
#     ]

# df = pd.DataFrame()
# for name, var in zip(name_list, var_list):
#     df[name] = hdf5_data[var].flatten()
    
# times = hdf5_data["time"]
# dates = pd.to_datetime({'year': times['year'], 'month': times['month'], 'day': times['day']})
# df.index = dates

# df["thermal_release"] = df["thermal_release_cannonsville"] + df["thermal_release_pepacton"] 
# #%%
# # Assuming 'df' has been properly defined and subsetted as in your provided code.
# dff = df["2001-6-1": "2001-8-31"]  # Use your desired date range.
# dff = df["2005-6-1": "2005-8-31"]

# fig, ax = plt.subplots()
# x = dff.index
# # Plot mean temperature with and without control
# ax.plot(x, dff["predicted_max_temp_mu_no_thermal_release"], label="no control", zorder=10)
# ax.plot(x, dff["predicted_max_temp_mu"], lw=1, label="with control", zorder=20)

# # Adding the uncertainty bands around the mean predictions
# ax.fill_between(x, 
#                 dff["predicted_max_temp_mu_no_thermal_release"] - dff["predicted_max_temp_sd_no_thermal_release"], 
#                 dff["predicted_max_temp_mu_no_thermal_release"] + dff["predicted_max_temp_sd_no_thermal_release"], 
#                 color='blue', alpha=0.3, label='+/- 1 sd', zorder=9)

# ax.fill_between(x, 
#                 dff["predicted_max_temp_mu"] - dff["predicted_max_temp_sd"], 
#                 dff["predicted_max_temp_mu"] + dff["predicted_max_temp_sd"], 
#                 color='orange', alpha=0.3, label='+/- 1 sd', zorder=9)

# # Horizontal line for the temperature threshold
# ax.axhline(23.89, color="r", ls="-", label="threshold", lw=1, zorder=100)

# # Highlight days with significant thermal releases
# for i, v in enumerate(dff["thermal_release"]):
#     if v > 10:
#         ax.axvline(x[i], zorder=1, lw=3, c="lightgrey")

# # Set labels, limits, and legend
# ax.set_ylabel("Max daily temperature (degree C)")
# ax.set_xlabel("Date")
# ax.set_ylim((20, 30))
# ax.legend(ncol=3, loc='upper right', bbox_to_anchor=(1, 1.14), frameon=False)
# plt.xticks(rotation=45)
# plt.show()

# # Count days above the threshold
# count_above_threshold_no_control = (dff["predicted_max_temp_mu_no_thermal_release"] > 23.89).sum()
# count_above_threshold_with_control = (dff["predicted_max_temp_mu"] > 23.89).sum()
# print("Number of days without control above 23.89 degree C:", count_above_threshold_no_control)
# print("Number of days with control above 23.89 degree C:", count_above_threshold_with_control)

# #%%
# dff = df["2001-6-1": "2001-8-31"]
# #dff = df["2005-6-1": "2005-8-31"]
# fig, ax = plt.subplots()
# x = dff.index
# ax.plot(x, dff["predicted_max_temp_mu_no_thermal_release"], label="no control", zorder=10)
# ax.plot(x, dff["predicted_max_temp_mu"], lw=1, label="with control", zorder=20)
# ax.axhline(23.89, color="r", ls="-", label="threshold") # => 75F = 23.89

# for i, v in enumerate(dff["thermal_release"]):
#     if v > 10:
#         ax.axvline(x[i], zorder=1, lw=3, c="lightgrey")

# ax.set_ylabel("Max daily temperature (degree C)")
# ax.set_xlabel("Date")
# ax.set_ylim((20, 30))
# ax.legend(ncol=3, loc='upper right', bbox_to_anchor=(1, 1.1), frameon=False)
# plt.xticks(rotation=45)
# plt.show()

# count_above_threshold = (df["predicted_max_temp_mu_no_thermal_release"] > 23.89).sum()
# print("Number of days without control above 23.89 degree C:", count_above_threshold)

# count_above_threshold = (df["predicted_max_temp_mu"] > 23.89).sum()
# print("Number of days with control above 23.89 degree C:", count_above_threshold)

