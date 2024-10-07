"""
Contains the custom parameter classes which use the LSTM model from Zwart et al. (2023)
to predict mean water temperature at Lordville each timestep.

Classes:
- TemperaturePrediction

LSTM model reference:
Zwart, J. A., Oliver, S. K., Watkins, W. D., Sadler, J. M., Appling, A. P., Corson‐Dosch,
H. R., ... & Read, J. S. (2023). Near‐term forecasts of stream temperature using deep learning
and data assimilation in support of management decisions.
JAWRA Journal of the American Water Resources Association, 59(2), 317-337.
"""

import pandas as pd
from pywr.parameters import Parameter, load_parameter

from pywrdrb.utils.constants import cms_to_mgd
from pywrdrb.utils.dates import temp_pred_date_range
from pywrdrb.utils.directories import ROOT_DIR

# Adding BMI LSTM model dir to the path
import sys

bmi_temp_model_path = f"{ROOT_DIR}/../../bmi-stream-temp"

sys.path.insert(1, f"{bmi_temp_model_path}/3_model_train/src")
sys.path.insert(1, f"{bmi_temp_model_path}/3_model_train/in")
sys.path.insert(1, f"{bmi_temp_model_path}/4_model_forecast/src")

# BMI class for running the LSTM model
#import torch_bmi

# Schema is needed for different scenarios
class TemperatureLSTM():
    def __init__(self) -> None:
        self.timestep = None
        self.scenario_index = None
        self.balancing_factors = {"cannonsville": 0.5, "pepacton": 0.5}
        self.predicted_temp = None

    def get_temp(self, reservoir, release_vol, timestep, scenario_index):
        if self.predicted_temp is None:
            self.predicted_temp = self.predict_temp(release_vol)
            self.timestep = timestep
            self.scenario_index = scenario_index
        elif self.timestep != timestep or self.scenario_index != scenario_index:
            self.predicted_temp = self.predict_temp(release_vol)
            self.timestep = timestep
            self.scenario_index = scenario_index
        return self.predicted_temp * self.balancing_factors[reservoir]

    def predict_temp(self, release_vol):
        # LSTM model prediction

        return 0.1

temperature_lstm = TemperatureLSTM()


class TotalThermalReleaseRequirement(Parameter):
    def __init__(self, model, cannonsville_release, pepacton_release, **kwargs):
        super().__init__(model, **kwargs)
        self.cannonsville_release = cannonsville_release
        self.pepacton_release = pepacton_release
        self._total_thermal_release = 0.0
    
    def value(self, timestep, scenario_index):
        # Total release from both reservoirs
        total_reservoir_release = (
            self.cannonsville_release.get_value(scenario_index) 
            + self.pepacton_release.get_value(scenario_index)
        )

        self._total_thermal_release = 0.01 #"DPS"
        return self._total_thermal_release
    
    @classmethod
    def load(cls, model, data):
        cannonsville_release = load_parameter(
            model, "max_flow_delivery_nyc_cannonsville"
        )
        pepacton_release = load_parameter(model, "max_flow_delivery_nyc_pepacton")      

        return cls(model, cannonsville_release, pepacton_release, **data) 
TotalThermalReleaseRequirement.register()

class AllocateThermalReleaseRequirement(Parameter):
    def __init__(self, model, reservoir, thermal_release_requirement, **kwargs):
        super().__init__(model, **kwargs)
        self.reservoir = reservoir
        self.thermal_release_requirement = thermal_release_requirement

        # hardcoded allocation factor for now
        self.allocation_factor = {"cannonsville": 0.5, "pepacton": 0.5}

    def value(self, timestep, scenario_index):
        # Total release from both reservoirs
        thermal_release_requirement = self.thermal_release_requirement.get_value(scenario_index)
        thermal_release = self.allocation_factor[self.reservoir] * thermal_release_requirement

        #!!!! Will need to consider the extreme case where no additional water to release
        # track the annual thermal release  

        return thermal_release
    
    @classmethod
    def load(cls, model, data):
        assert "reservoir" in data.keys()
        assert data["reservoir"] in ["cannonsville", "pepacton"]
        reservoir = data.pop("reservoir")

        thermal_release_requirement = load_parameter(model, "thermal_release_requirement")

        return cls(model, reservoir, thermal_release_requirement, **data)    

AllocateThermalReleaseRequirement.register()


class PredictedMaxTemperatureAtLordsville(Parameter):
    def __init__(self, model, cannonsville_release, pepacton_release, additional_thermal_release_cannonsville, additional_thermal_release_pepacton, **kwargs):
        super().__init__(model, **kwargs)
        self.cannonsville_release = cannonsville_release
        self.pepacton_release = pepacton_release
        self.additional_thermal_release_cannonsville = additional_thermal_release_cannonsville
        self.additional_thermal_release_pepacton = additional_thermal_release_pepacton

        # INitialize the LSTM model here
        def temperature_lstm(total_reservoir_release):
            return (72, 3)
        self.temperature_lstm = temperature_lstm
        self.mu, self.sig = 0, 0
    
    def value(self, timestep, scenario_index):
        # Total release from both reservoirs
        total_reservoir_release = (
            self.cannonsville_release.get_value(scenario_index) 
            + self.pepacton_release.get_value(scenario_index)
            + self.additional_thermal_release_cannonsville.get_value(scenario_index)
            + self.additional_thermal_release_pepacton.get_value(scenario_index)
        )

        #self.mu, self.sig = self.temperature_lstm(total_reservoir_release)
        self.mu += 0.1
        self.sig += 0.1
        fake_value = -99
        return fake_value
    
    @classmethod
    def load(cls, model, data):
        cannonsville_release = load_parameter(
            model, "max_flow_delivery_nyc_cannonsville"
        )
        pepacton_release = load_parameter(model, "max_flow_delivery_nyc_pepacton")      

        additional_thermal_release_cannonsville = load_parameter(model, "additional_thermal_release_cannonsville")
        additional_thermal_release_pepacton = load_parameter(model, "additional_thermal_release_pepacton")
        return cls(model, cannonsville_release, pepacton_release, additional_thermal_release_cannonsville, additional_thermal_release_pepacton, **data) 
PredictedMaxTemperatureAtLordsville.register()

class GetTemperatureLSTMValue(Parameter):
    def __init__(self, model, variable, predicted_max_temperature_at_lordsville_run_lstm, **kwargs):
        super().__init__(model, **kwargs)
        self.variable = variable
        self.predicted_max_temperature_at_lordsville_run_lstm = predicted_max_temperature_at_lordsville_run_lstm

    def value(self, timestep, scenario_index):
        fake_value = self.predicted_max_temperature_at_lordsville_run_lstm.get_value(scenario_index)
        if self.variable == "mu":
            return self.predicted_max_temperature_at_lordsville_run_lstm.mu
        elif self.variable == "sig":
            return self.predicted_max_temperature_at_lordsville_run_lstm.sig
        else:
            raise ValueError("Invalid variable. Must be 'mu' or 'sig'.")
        
    @classmethod
    def load(cls, model, data):
        assert "variable" in data.keys()
        variable = data.pop("variable")
        predicted_max_temperature_at_lordsville_run_lstm = load_parameter(model, "predicted_max_temperature_at_lordsville_run_lstm")
        return cls(model, variable, predicted_max_temperature_at_lordsville_run_lstm, **data)
GetTemperatureLSTMValue.register()

# class TemperaturePrediction(Parameter):
#     """
#     Uses the LSTM model to predict mean water temperature at Lordville each timestep.

#     Attributes:
#         model: Pywr model object
#         cannonsville_release: Pywr parameter object for Cannonsville release
#         pepacton_release: Pywr parameter object for Pepacton release
#         lstm: BMI LSTM model object

#     Methods:
#         value(timestep, scenario_index):
#         load(model, data): loads the parameter from the model dictionary
#     """

#     def __init__(self, model, cannonsville_release, pepacton_release, **kwargs):
#         super().__init__(model, **kwargs)
#         self.cannonsville_release = cannonsville_release
#         self.pepacton_release = pepacton_release

#         # LSTM valid prediction date range
#         self.lstm_date_range = pd.date_range(
#             start=temp_pred_date_range[0], end=temp_pred_date_range[1]
#         )

#         # location of the BMI configuration file
#         bmi_cfg_file = f"{bmi_temp_model_path}/model_config.yml"
#         self.bmi_cfg_file = bmi_cfg_file

#         # creating an instance of an LSTM model
#         print("Creating an instance of an BMI_LSTM model object")
#         self.lstm = torch_bmi.bmi_lstm()

#         # Initializing the BMI for LSTM
#         print("Initializing the temperature prediction LSTM\n")
#         self.lstm.initialize(bmi_cfg_file=bmi_cfg_file)

#         ### Manully advance the LSTM to the pywrdrb simulation start
#         # LSTM is set up to start on 1982-04-03
#         simulation_start = model.timestepper.start
#         days_to_advance = (simulation_start - self.lstm_date_range[0]).days

#         # Advance the LSTM model to the simulation start date
#         for ti in range(days_to_advance):
#             unscaled_data = (
#                 self.lstm.x[
#                     0,
#                     int(self.lstm.t),
#                 ]
#                 * (self.lstm.input_std + 1e-10)
#                 + self.lstm.input_mean
#             )
#             for i in range(len(unscaled_data)):
#                 self.lstm.set_value(self.lstm.x_vars[i], unscaled_data[i])
#             self.lstm.update()
#         print(
#             f"Advanced LSTM temperature prediction model {days_to_advance} to start of pywrdrb simulation."
#         )
#         print(f"LSTM model is now at date {self.lstm.dates[int(self.lstm.t)]}.")

#     def value(self, timestep, scenario_index):
#         """
#         Returns the mean prediction of the maximum water temperature
#         predicted by the LSTM model for the current timestep and scenario index.

#         Args:
#             timestep (int): current timestep
#             scenario_index (int): current scenario index

#         Returns:
#             float:
#         """

#         # Get reservoir releases for the current timestep
#         total_reservoir_release = sum(
#             [
#                 self.cannonsville_release.get_value(scenario_index),
#                 self.pepacton_release.get_value(scenario_index),
#             ]
#         )

#         # convert from MGD to m3 s-1
#         total_reservoir_release = total_reservoir_release * (1.0 / cms_to_mgd)

#         # Unscaling the driver data stored in x for the current time step
#         # data are already scaled so need to unscale prior to putting in the model
#         unscaled_data = (
#             self.lstm.x[
#                 0,
#                 int(self.lstm.t),
#             ]
#             * (self.lstm.input_std + 1e-10)
#             + self.lstm.input_mean
#         )

#         # Setting the unscaled data values for the current time step in the BMI model
#         for i in range(len(unscaled_data)):
#             var_name = self.lstm.x_vars[i]
#             if var_name == "reservoir_release":
#                 self.lstm.set_value(var_name, total_reservoir_release)
#             else:
#                 self.lstm.set_value(var_name, unscaled_data[i])

#         # run the BMI model with the update() function
#         self.lstm.update()

#         # retrieving the main prediction output from the BMI LSTM model
#         # the predicted mean of max water temperature is stored in CSDMS naming convention
#         t_pred = []
#         self.lstm.get_value(
#             "channel_water_surface_water__mu_max_of_temperature", t_pred
#         )

#         err_msg = (
#             "LSTM model should only return one value for the predicted temperature"
#         )
#         err_msg += f" but returned {len(t_pred)} values."
#         assert len(t_pred) == 1, err_msg
#         return t_pred[0]

#     @classmethod
#     def load(cls, model, data):
#         """Setup the parameter."""

#         cannonsville_release = load_parameter(
#             model, "max_flow_delivery_nyc_cannonsville"
#         )
#         pepacton_release = load_parameter(model, "max_flow_delivery_nyc_pepacton")

#         return cls(model, cannonsville_release, pepacton_release, **data)


# # register the custom parameter so Pywr recognizes it
# TemperaturePrediction.register()



# class archiveTemperaturePrediction(Parameter):
#     """
#     Uses the LSTM model to predict mean water temperature at Lordville each timestep.

#     Attributes:
#         model: Pywr model object
#         cannonsville_release: Pywr parameter object for Cannonsville release
#         pepacton_release: Pywr parameter object for Pepacton release
#         lstm: BMI LSTM model object

#     Methods:
#         value(timestep, scenario_index):
#         load(model, data): loads the parameter from the model dictionary
#     """

#     def __init__(self, model, cannonsville_release, pepacton_release, **kwargs):
#         super().__init__(model, **kwargs)
#         self.cannonsville_release = cannonsville_release
#         self.pepacton_release = pepacton_release

#         # LSTM valid prediction date range
#         self.lstm_date_range = pd.date_range(
#             start=temp_pred_date_range[0], end=temp_pred_date_range[1]
#         )

#         # location of the BMI configuration file
#         bmi_cfg_file = f"{bmi_temp_model_path}/model_config.yml"
#         self.bmi_cfg_file = bmi_cfg_file

#         # creating an instance of an LSTM model
#         print("Creating an instance of an BMI_LSTM model object")
#         self.lstm = torch_bmi.bmi_lstm()

#         # Initializing the BMI for LSTM
#         print("Initializing the temperature prediction LSTM\n")
#         self.lstm.initialize(bmi_cfg_file=bmi_cfg_file)

#         ### Manully advance the LSTM to the pywrdrb simulation start
#         # LSTM is set up to start on 1982-04-03
#         simulation_start = model.timestepper.start
#         days_to_advance = (simulation_start - self.lstm_date_range[0]).days

#         # Advance the LSTM model to the simulation start date
#         for ti in range(days_to_advance):
#             unscaled_data = (
#                 self.lstm.x[
#                     0,
#                     int(self.lstm.t),
#                 ]
#                 * (self.lstm.input_std + 1e-10)
#                 + self.lstm.input_mean
#             )
#             for i in range(len(unscaled_data)):
#                 self.lstm.set_value(self.lstm.x_vars[i], unscaled_data[i])
#             self.lstm.update()
#         print(
#             f"Advanced LSTM temperature prediction model {days_to_advance} to start of pywrdrb simulation."
#         )
#         print(f"LSTM model is now at date {self.lstm.dates[int(self.lstm.t)]}.")

#     def value(self, timestep, scenario_index):
#         """
#         Returns the mean prediction of the maximum water temperature
#         predicted by the LSTM model for the current timestep and scenario index.

#         Args:
#             timestep (int): current timestep
#             scenario_index (int): current scenario index

#         Returns:
#             float:
#         """

#         # Get reservoir releases for the current timestep
#         total_reservoir_release = sum(
#             [
#                 self.cannonsville_release.get_value(scenario_index),
#                 self.pepacton_release.get_value(scenario_index),
#             ]
#         )

#         # convert from MGD to m3 s-1
#         total_reservoir_release = total_reservoir_release * (1.0 / cms_to_mgd)

#         # Unscaling the driver data stored in x for the current time step
#         # data are already scaled so need to unscale prior to putting in the model
#         unscaled_data = (
#             self.lstm.x[
#                 0,
#                 int(self.lstm.t),
#             ]
#             * (self.lstm.input_std + 1e-10)
#             + self.lstm.input_mean
#         )

#         # Setting the unscaled data values for the current time step in the BMI model
#         for i in range(len(unscaled_data)):
#             var_name = self.lstm.x_vars[i]
#             if var_name == "reservoir_release":
#                 self.lstm.set_value(var_name, total_reservoir_release)
#             else:
#                 self.lstm.set_value(var_name, unscaled_data[i])

#         # run the BMI model with the update() function
#         self.lstm.update()

#         # retrieving the main prediction output from the BMI LSTM model
#         # the predicted mean of max water temperature is stored in CSDMS naming convention
#         t_pred = []
#         self.lstm.get_value(
#             "channel_water_surface_water__mu_max_of_temperature", t_pred
#         )

#         err_msg = (
#             "LSTM model should only return one value for the predicted temperature"
#         )
#         err_msg += f" but returned {len(t_pred)} values."
#         assert len(t_pred) == 1, err_msg
#         return t_pred[0]

#     @classmethod
#     def load(cls, model, data):
#         """Setup the parameter."""

#         cannonsville_release = load_parameter(
#             model, "max_flow_delivery_nyc_cannonsville"
#         )
#         pepacton_release = load_parameter(model, "max_flow_delivery_nyc_pepacton")

#         return cls(model, cannonsville_release, pepacton_release, **data)


# register the custom parameter so Pywr recognizes it
#TemperaturePrediction.register()


"""
### EXAMPLE - NOT NECESSARY TO FOLLW THIS.

class CalculateRequiredReleasesForSalinity(Parameter):
    def __init__(self):

        self.lstm = torch_bmi.initalize()
        self.target_temp = 75 #degree-F
        pass

    def value(self, timestep, scenario_index):

        nyc_drought_level = None
        is_drought_emergency = False

        if is_drought_emergency:
            # Use LSTM to determine release that successfully
            # keeps temperature below threshold

            violation = True

            while violation:
                candidate_release_vol = 5

                # predict temp with candidate release
                temp_prediction = lstm.predict_temp(candidate_release_vol)

                if temp_prediction > self.target_temp:
                    violation = True
                    candidate_release_vol += 10

        else:
            releases_for_temperature = 0.0

        # Use LSTM to determine release that successfully
        # keeps temperature below threshold

        return releases_for_temperature
"""
