"""
This file contains different class objects which are used to construct custom Pywr parameters.

The parameters created here are used to implement the flexible flow management program (FFMP)
for the three NYC reservoirs.
"""

import numpy as np
import pandas as pd
from pywr.parameters import Parameter, load_parameter


class FfmpNycRunningAvgParameter(Parameter):
    """
    Enforces the constraint on NYC deliveries from the FFMP, based on a running average.

    Args:
        model (Model): The Pywr model instance.
        node (Node): The node associated with the parameter.
        max_avg_delivery (ConstantParameter): The maximum average delivery constant parameter.

    Attributes:
        node (Node): The node associated with the parameter.
        max_avg_delivery (float): The maximum average delivery value.
        max_delivery (ndarray): An array to hold the parameter state.

    Methods:
        setup(): Allocates an array to hold the parameter state.
        reset(): Resets the amount remaining in all states to the initial value.
        value(timestep, scenario_index): Returns the current volume remaining for the scenario.
        after(): Updates the parameter requirement based on running average and updates the date for tomorrow.

    Class Methods:
        load(model, data): Loads the parameter from model and data dictionary.

    """
    def __init__(self, model, node, max_avg_delivery, **kwargs):
        super().__init__(model, **kwargs)
        self.node = node
        self.max_avg_delivery = max_avg_delivery.get_constant_value()
        self.children.add(max_avg_delivery)

    def setup(self):
        """Allocates an array to hold the parameter state."""
        super().setup()
        num_scenarios = len(self.model.scenarios.combinations)
        self.max_delivery = np.empty([num_scenarios], np.float64)

    def reset(self):
        """Resets the amount remaining in all states to the initial value."""
        self.timestep = self.model.timestepper.delta
        self.datetime = pd.to_datetime(self.model.timestepper.start)
        self.max_delivery[...] = self.max_avg_delivery * self.timestep

    def value(self, timestep, scenario_index):
        """
        Returns the current volume remaining for the scenario.

        Args:
            timestep (Timestep): The current timestep.
            scenario_index (ScenarioIndex): The scenario index.

        Returns:
            float: The current volume remaining for the scenario.
        """
        return self.max_delivery[scenario_index.global_id]

    def after(self):
        """Updates the parameter requirement based on running average and updates the date for tomorrow.
        """
        ### if it is may 31, reset max delivery to original value (800)
        if self.datetime.month == 5 and self.datetime.day == 31:
            self.max_delivery[...] = self.max_avg_delivery * self.timestep
        ### else update the requirement based on running average: maxdel_t = maxdel_{t-1} - flow_{t-1} + max_avg_del
        else:
            self.max_delivery += (self.max_avg_delivery - self.node.flow) * self.timestep
            self.max_delivery[self.max_delivery < 0] = 0  # max delivery cannot be less than zero
        ### update date for tomorrow
        self.datetime += pd.Timedelta(1, 'd')

    @classmethod
    def load(cls, model, data):
        """
        Loads the parameter from model and data dictionary.

        Args:
            model (Model): The Pywr model instance.
            data (dict): The data dictionary containing the parameter information.

        Returns:
            FfmpNycRunningAvgParameter: The loaded parameter instance.
        """
        node = model.nodes[data.pop("node")]
        max_avg_delivery = load_parameter(model, data.pop("max_avg_delivery"))
        return cls(model, node, max_avg_delivery, **data)



class FfmpNjRunningAvgParameter(Parameter):
    """
    Enforces the constraint on NJ deliveries from the FFMP, based on a running average.

    Args:
        model (Model): The Pywr model instance.
        node (Node): The node associated with the parameter.

    Attributes:
        node (Node): The node associated with the parameter.
        max_avg_delivery (float): The maximum average delivery value.
        max_daily_delivery (float): The maximum daily delivery value.
        drought_factor (Parameter): The drought factor parameter.
        max_delivery (ndarray): An array to hold the parameter state.
        current_drought_factor (ndarray): An array to hold the current drought factor.
        previous_drought_factor (ndarray): An array to hold the previous drought factor.

    Methods:
        setup(): Allocates arrays to hold the parameter state and drought factors.
        reset(): Resets the amount remaining in all states to the initial value.
        value(timestep, scenario_index): Returns the current volume remaining for the scenario.
        after(): Updates the parameter requirement based on running average and updates the date for tomorrow.

    Class Methods:
        load(model, data): Loads the parameter from model and data dictionary.

    """
    def __init__(self, model, node, max_avg_delivery, max_daily_delivery, drought_factor, **kwargs):
        super().__init__(model, **kwargs)
        self.node = node
        self.max_avg_delivery = max_avg_delivery.get_constant_value()
        self.max_daily_delivery = max_daily_delivery.get_constant_value()
        self.drought_factor = drought_factor
        self.children.add(max_avg_delivery)
        self.children.add(max_daily_delivery)
        self.children.add(drought_factor)

    def setup(self):
        """Allocate an array to hold the parameter state."""
        super().setup()
        num_scenarios = len(self.model.scenarios.combinations)
        self.max_delivery = np.empty([num_scenarios], np.float64)
        self.current_drought_factor = np.empty([num_scenarios], np.float64)
        self.previous_drought_factor = np.empty([num_scenarios], np.float64)

    def reset(self):
        """Resets the amount remaining in all states to the initial value."""
        self.timestep = self.model.timestepper.delta
        self.datetime = pd.to_datetime(self.model.timestepper.start)
        self.max_delivery[...] = self.max_avg_delivery * self.timestep
        self.current_drought_factor[...] = 1.0
        self.previous_drought_factor[...] = 1.0

    def value(self, timestep, scenario_index):
        """
        Returns the current volume remaining for the scenario.

        Args:
            timestep (Timestep): The current timestep.
            scenario_index (ScenarioIndex): The scenario index.

        Returns:
            float: The current volume remaining for the scenario.
        
        """
        return self.max_delivery[scenario_index.global_id]

    def after(self):
        """Updates the parameter requirement based on running average and updates the date for tomorrow."""
        self.current_drought_factor[...] = self.drought_factor.get_all_values()
        ### loop over scenarios
        for s, factor in enumerate(self.current_drought_factor):
            ### first check if today's drought_factor is same as yesterday's
            if factor == self.previous_drought_factor[s]:
                ### if today is same drought factor as yesterday, and factor is 1.0, we reset running avg on first day of each month
                if (self.datetime.day == 1) and (factor == 1.0):
                    self.max_delivery[s] = self.max_avg_delivery * factor * self.timestep
                ### else, update requirement based on running average: maxdel_t = maxdel_{t-1} - flow_{t-1} + max_avg_del
                else:
                    self.max_delivery[s] += (self.max_avg_delivery * factor - self.node.flow[s]) * self.timestep
            ### if today's drought factor is different from yesterday, we always reset running avg
            else:
                self.max_delivery[s] = self.max_avg_delivery * factor * self.timestep

        ### max delivery cannot be less than zero
        self.max_delivery[self.max_delivery < 0] = 0
        ### max delivery cannot be larger than daily limit
        self.max_delivery[self.max_delivery > self.max_daily_delivery] = self.max_daily_delivery
        ### update date & previous factor for tomorrow
        self.datetime += pd.Timedelta(1, 'd')
        self.previous_drought_factor[...] = self.current_drought_factor[...]

    @classmethod
    def load(cls, model, data):
        node = model.nodes[data.pop("node")]
        max_avg_delivery = load_parameter(model, data.pop("max_avg_delivery"))
        max_daily_delivery = load_parameter(model, data.pop("max_daily_delivery"))
        drought_factor = load_parameter(model, data.pop('drought_factor'))
        return cls(model, node, max_avg_delivery, max_daily_delivery, drought_factor, **data)



###


class VolBalanceNYCDemandTarget(Parameter):
    """
    Updates the contribution to NYC deliveries made by each of the NYC
    reservoirs, in such a way as to balance the relative storages across the three reservoirs.
    See comments on this GitHub issue for the equations & logic:
    https://github.com/users/ahamilton144/projects/1/views/1?pane=issue&itemId=7840442

    Args:
        model (Model): The Pywr model instance.
        node (Node): The node associated with the parameter.

    Attributes:
        node (Node): The node associated with the parameter.
        max_volume_agg_nyc (Parameter): The maximum volume aggregate NYC parameter.
        volume_agg_nyc (Parameter): The volume aggregate NYC parameter.
        max_flow_delivery_nyc (Parameter): The maximum flow delivery NYC parameter.
        flow_agg_nyc (Parameter): The flow aggregate NYC parameter.
        max_vol_reservoir (Parameter): The maximum volume reservoir parameter.
        vol_reservoir (Parameter): The volume reservoir parameter.
        flow_reservoir (Parameter): The flow reservoir parameter.

    Methods:
        value(timestep, scenario_index): Returns the target NYC delivery for this reservoir to balance storages across reservoirs.

    Class Methods:
        load(model, data): Loads the parameter from model and data dictionary.

    """
    def __init__(self, model, node, max_volume_agg_nyc, volume_agg_nyc, max_flow_delivery_nyc, flow_agg_nyc,
                 max_vol_reservoir, vol_reservoir, flow_reservoir, **kwargs):
        super().__init__(model, **kwargs)
        self.node = node
        self.max_volume_agg_nyc = max_volume_agg_nyc
        self.volume_agg_nyc = volume_agg_nyc
        self.max_flow_delivery_nyc = max_flow_delivery_nyc
        self.flow_agg_nyc = flow_agg_nyc
        self.max_vol_reservoir = max_vol_reservoir
        self.vol_reservoir = vol_reservoir
        self.flow_reservoir = flow_reservoir

        self.children.add(max_volume_agg_nyc)
        self.children.add(volume_agg_nyc)
        self.children.add(max_flow_delivery_nyc)
        self.children.add(flow_agg_nyc)
        self.children.add(max_vol_reservoir)
        self.children.add(vol_reservoir)
        self.children.add(flow_reservoir)


    def value(self, timestep, scenario_index):
        ### return the target NYC delivery for this reservoir to balance storages across reservoirs
        target = self.vol_reservoir.get_value(scenario_index) + self.flow_reservoir.get_value(scenario_index) - \
               (self.max_vol_reservoir.get_value(scenario_index) / self.max_volume_agg_nyc.get_value(scenario_index)) * \
               (self.volume_agg_nyc.get_value(scenario_index) + self.flow_agg_nyc.get_value(scenario_index) - self.max_flow_delivery_nyc.get_value(scenario_index))
        return max(target, 0.)


    @classmethod
    def load(cls, model, data):
        reservoir = data.pop("node")
        node = model.nodes[reservoir]
        reservoir = reservoir.split('_')[1]
        max_volume_agg_nyc = load_parameter(model, 'max_volume_agg_nyc')
        volume_agg_nyc = load_parameter(model, 'volume_agg_nyc')
        max_flow_delivery_nyc = load_parameter(model, 'max_flow_delivery_nyc')
        flow_agg_nyc = load_parameter(model, 'flow_agg_nyc')
        max_vol_reservoir = load_parameter(model, f'max_volume_{reservoir}')
        vol_reservoir = load_parameter(model, f'volume_{reservoir}')
        flow_reservoir = load_parameter(model, f'flow_{reservoir}')
        return cls(model, node, max_volume_agg_nyc, volume_agg_nyc, max_flow_delivery_nyc, flow_agg_nyc,
                   max_vol_reservoir, vol_reservoir, flow_reservoir, **data)




class VolBalanceNYCDemandFinal(Parameter):
    """
    This is the second step in updating the contribution to NYC deliveries made by each of the NYC
    reservoirs, in such a way as to balance the relative storages across the three reservoirs.
    Step one is VolBalanceNYCDemandTarget above.
    In this second step, each reservoir's contributions are rescaled to ensure their sum is equal to demand.
    See comments on this GitHub issue for the equations & logic:
    https://github.com/users/ahamilton144/projects/1/views/1?pane=issue&itemId=7840442

    Args:
        model (Model): The Pywr model instance.
        node (Node): The node associated with the parameter.
        max_flow_delivery_nyc (Parameter): The maximum flow NYC delivery parameter.
        volbalance_target_max_flow_delivery_nyc_reservoir (Parameter): The volume balance target maximum flow delivery NYC reservoir parameter.
        volbalance_target_max_flow_delivery_agg_nyc (Parameter): The volume balance target maximum flow delivery aggregate NYC parameter.

    Keyword Args:
        Additional keyword arguments.

    Attributes:
        node (Node): The node associated with the parameter.
        max_flow_delivery_nyc (Parameter): The maximum flow NYC delivery parameter.
        volbalance_target_max_flow_delivery_nyc_reservoir (Parameter): The volume balance target maximum flow delivery NYC reservoir parameter.
        volbalance_target_max_flow_delivery_agg_nyc (Parameter): The volume balance target maximum flow delivery aggregate NYC parameter.

    Methods:
        value(timestep, scenario_index): Returns the rescaled max flow NYC delivery for this reservoir.

    Class Methods:
        load(model, data): Loads the parameter from model and data dictionary.

    """
    def __init__(self, model, node, max_flow_delivery_nyc, volbalance_target_max_flow_delivery_nyc_reservoir,
                 volbalance_target_max_flow_delivery_agg_nyc, **kwargs):
        super().__init__(model, **kwargs)
        self.node = node
        self.max_flow_delivery_nyc = max_flow_delivery_nyc
        self.volbalance_target_max_flow_delivery_nyc_reservoir = volbalance_target_max_flow_delivery_nyc_reservoir
        self.volbalance_target_max_flow_delivery_agg_nyc = volbalance_target_max_flow_delivery_agg_nyc
        self.children.add(max_flow_delivery_nyc)
        self.children.add(volbalance_target_max_flow_delivery_nyc_reservoir)
        self.children.add(volbalance_target_max_flow_delivery_agg_nyc)

    def value(self, timestep, scenario_index):
        """
        Returns the rescaled max flow NYC delivery for this reservoir, after zeroing out if any reservoir had negative target

        Args:
            timestep (Timestep): The current timestep.
            scenario_index (ScenarioIndex): The scenario index.

        Returns:
            float: The rescaled max flow NYC delivery for this reservoir.
        """
        return self.max_flow_delivery_nyc.get_value(scenario_index) * \
               self.volbalance_target_max_flow_delivery_nyc_reservoir.get_value(scenario_index) / \
               self.volbalance_target_max_flow_delivery_agg_nyc.get_value(scenario_index)

    @classmethod
    def load(cls, model, data):
        """Loads the parameter from model and data dictionary.

        Args:
            model (Model): The Pywr model instance.
            data (dict): The data dictionary containing the parameter information.

        Returns:
            VolBalanceNYCDemandFinal: The loaded parameter instance.
        """
        reservoir = data.pop("node")
        node = model.nodes[reservoir]
        reservoir = reservoir.split('_')[1]
        max_flow_delivery_nyc = load_parameter(model, 'max_flow_delivery_nyc')
        volbalance_target_max_flow_delivery_nyc_reservoir = load_parameter(model, f'volbalance_target_max_flow_delivery_nyc_{reservoir}')
        volbalance_target_max_flow_delivery_agg_nyc = load_parameter(model, f'volbalance_target_max_flow_delivery_agg_nyc')
        return cls(model, node, max_flow_delivery_nyc, volbalance_target_max_flow_delivery_nyc_reservoir,
                   volbalance_target_max_flow_delivery_agg_nyc, **data)



class VolBalanceNYCDownstreamMRFTargetAgg(Parameter):
    """
    Calculates the total releases from NYC reservoirs needed to meet the Montague and Trenton flow targets,
    after subtracting out flows from the rest of the basin, and adding max deliveries to NJ.

    Args:
        model (Model): The Pywr model instance.

    Attributes:
        volbalance_flow_agg_nonnyc_delMontague (Parameter): The volume balance flow aggregate non-NYC Montague delivery parameter.
        mrf_target_delMontague (Parameter): The MRF target Montague delivery parameter.
        volbalance_flow_agg_nonnyc_delTrenton (Parameter): The volume balance flow aggregate non-NYC Trenton delivery parameter.
        max_flow_delivery_nj (Parameter): The maximum flow delivery to NJ parameter.
        mrf_target_delTrenton (Parameter): The MRF target Trenton delivery parameter.

    Methods:
        value(timestep, scenario_index): Returns the total flow needed from NYC reservoirs to meet Montague and Trenton targets.

    Class Methods:
        load(model, data): Loads the parameter from model and data dictionary.

    """
    def __init__(self, model, volbalance_flow_agg_nonnyc_delMontague, mrf_target_delMontague,\
                   volbalance_flow_agg_nonnyc_delTrenton, max_flow_delivery_nj, mrf_target_delTrenton, **kwargs):
        super().__init__(model, **kwargs)
        self.volbalance_flow_agg_nonnyc_delMontague = volbalance_flow_agg_nonnyc_delMontague
        self.mrf_target_delMontague = mrf_target_delMontague
        self.volbalance_flow_agg_nonnyc_delTrenton = volbalance_flow_agg_nonnyc_delTrenton
        self.max_flow_delivery_nj = max_flow_delivery_nj
        self.mrf_target_delTrenton = mrf_target_delTrenton
        self.children.add(volbalance_flow_agg_nonnyc_delMontague)
        self.children.add(mrf_target_delMontague)
        self.children.add(volbalance_flow_agg_nonnyc_delTrenton)
        self.children.add(max_flow_delivery_nj)
        self.children.add(mrf_target_delTrenton)

    def value(self, timestep, scenario_index):
        """Returns the total flow needed from NYC reservoirs to meet Montague and Trenton targets.

        Args:
            timestep (Timestep): The current timestep.
            scenario_index (ScenarioIndex): The scenario index.

        Returns:
            float: The total flow needed from NYC reservoirs to meet Montague and Trenton targets.
        """
        req_delMontague = max(self.mrf_target_delMontague.get_value(scenario_index) -
                              self.volbalance_flow_agg_nonnyc_delMontague.get_value(scenario_index),
                              0.)
        req_delTrenton = max(self.mrf_target_delTrenton.get_value(scenario_index) -
                             self.volbalance_flow_agg_nonnyc_delTrenton.get_value(scenario_index) +
                             self.max_flow_delivery_nj.get_value(scenario_index),
                             0.)
        return max(req_delMontague, req_delTrenton)

    @classmethod
    def load(cls, model, data):
        """
        Loads the parameter from model and data dictionary.

        Args:
            model (Model): The Pywr model instance.
            data (dict): The data dictionary containing the parameter information.

        Returns:
            VolBalanceNYCDemandTarget: The loaded parameter instance.
        """
        volbalance_flow_agg_nonnyc_delMontague = load_parameter(model, 'volbalance_flow_agg_nonnyc_delMontague')
        mrf_target_delMontague = load_parameter(model, 'mrf_target_delMontague')
        volbalance_flow_agg_nonnyc_delTrenton = load_parameter(model, 'volbalance_flow_agg_nonnyc_delTrenton')
        max_flow_delivery_nj = load_parameter(model, 'max_flow_delivery_nj')
        mrf_target_delTrenton = load_parameter(model, 'mrf_target_delTrenton')

        return cls(model, volbalance_flow_agg_nonnyc_delMontague, mrf_target_delMontague,\
                   volbalance_flow_agg_nonnyc_delTrenton, max_flow_delivery_nj, mrf_target_delTrenton, **data)




class VolBalanceNYCDownstreamMRFTarget(Parameter):
    """
    Custom Pywr parameter class. This parameter updates the contribution to meeting Montague & Trenton flow targets,
    by each of the NYC reservoirs, in such a way as to balance the relative storages across the three reservoirs.
    See comments on this GitHub issue for the equations & logic:
    https://github.com/users/ahamilton144/projects/1/views/1?pane=issue&itemId=7840442

    Args:
        model (Model): The Pywr model instance.
        node (Node): The node associated with the parameter.

    Attributes:
        node (Node): The node associated with the parameter.
        max_volume_agg_nyc (Parameter): The maximum volume aggregate NYC parameter.
        volume_agg_nyc (Parameter): The volume aggregate NYC parameter.
        volbalance_relative_mrf_montagueTrenton (Parameter): The volbalance relative MRF Montague & Trenton parameter.
        flow_agg_nyc (Parameter): The flow aggregate NYC parameter.
        max_vol_reservoir (Parameter): The maximum volume reservoir parameter.
        vol_reservoir (Parameter): The volume reservoir parameter.
        flow_reservoir (Parameter): The flow reservoir parameter.

    Methods:
        value(timestep, scenario_index): Returns the target downstream MRF release for this reservoir to balance storages across reservoirs.

    Class Methods:
        load(model, data): Loads the parameter from model and data dictionary.
    
    """
    def __init__(self, model, node, max_volume_agg_nyc, volume_agg_nyc, volbalance_relative_mrf_montagueTrenton, flow_agg_nyc,
                 max_vol_reservoir, vol_reservoir, flow_reservoir, **kwargs):
        super().__init__(model, **kwargs)
        self.node = node
        self.max_volume_agg_nyc = max_volume_agg_nyc
        self.volume_agg_nyc = volume_agg_nyc
        self.volbalance_relative_mrf_montagueTrenton = volbalance_relative_mrf_montagueTrenton
        self.flow_agg_nyc = flow_agg_nyc
        self.max_vol_reservoir = max_vol_reservoir
        self.vol_reservoir = vol_reservoir
        self.flow_reservoir = flow_reservoir

        self.children.add(max_volume_agg_nyc)
        self.children.add(volume_agg_nyc)
        self.children.add(volbalance_relative_mrf_montagueTrenton)
        self.children.add(flow_agg_nyc)
        self.children.add(max_vol_reservoir)
        self.children.add(vol_reservoir)
        self.children.add(flow_reservoir)


    def value(self, timestep, scenario_index):
        """
        Returns the target downstream MRF release for this reservoir to balance storages across reservoirs.

        Args:
            timestep (Timestep): The current timestep.
            scenario_index (ScenarioIndex): The scenario index.

        Returns:
            float: The target downstream MRF release for this reservoir.
        """
        target = self.vol_reservoir.get_value(scenario_index) + self.flow_reservoir.get_value(scenario_index) - \
               (self.max_vol_reservoir.get_value(scenario_index) / self.max_volume_agg_nyc.get_value(scenario_index)) * \
               (self.volume_agg_nyc.get_value(scenario_index) + self.flow_agg_nyc.get_value(scenario_index) - \
                self.volbalance_relative_mrf_montagueTrenton.get_value(scenario_index))
        return max(target, 0.)


    @classmethod
    def load(cls, model, data):
        """Loads the parameter from model and data dictionary.

        Args:
            model (Model): The Pywr model instance.
            data (dict): The data dictionary containing the parameter information.

        Returns:
            VolBalanceNYCDownstreamMRFTarget: The loaded parameter instance.
        """
        reservoir = data.pop("node")
        node = model.nodes[reservoir]
        reservoir = reservoir.split('_')[1]
        max_volume_agg_nyc = load_parameter(model, 'max_volume_agg_nyc')
        volume_agg_nyc = load_parameter(model, 'volume_agg_nyc')
        volbalance_relative_mrf_montagueTrenton = load_parameter(model, 'volbalance_relative_mrf_montagueTrenton')
        flow_agg_nyc = load_parameter(model, 'flow_agg_nyc')
        max_vol_reservoir = load_parameter(model, f'max_volume_{reservoir}')
        vol_reservoir = load_parameter(model, f'volume_{reservoir}')
        flow_reservoir = load_parameter(model, f'flow_{reservoir}')
        return cls(model, node, max_volume_agg_nyc, volume_agg_nyc, volbalance_relative_mrf_montagueTrenton, flow_agg_nyc,
                   max_vol_reservoir, vol_reservoir, flow_reservoir, **data)






class VolBalanceNYCDownstreamMRFFinal(Parameter):
    """
    Second step in updating the contribution to meeting Montague & Trenton flow
    targets, by each of the NYC reservoirs, in such a way as to balance the relative storages across the three reservoirs.
    Step one is VolBalanceNYCDownstreamMRFTargetAgg above.
    In this second step, each reservoir's contributions are rescaled to ensure their sum is equal to demand.
    See comments on this GitHub issue for the equations & logic:
    https://github.com/users/ahamilton144/projects/1/views/1?pane=issue&itemId=7840442

    Args:
        model (Model): The Pywr model instance.
        node (Node): The node associated with the parameter.

    Attributes:
        node (Node): The node associated with the parameter.
        volbalance_relative_mrf_montagueTrenton (Parameter): The volbalance relative MRF Montague & Trenton parameter.
        volbalance_target_max_flow_montagueTrenton_reservoir (Parameter): The volbalance target max flow Montague & Trenton reservoir parameter.
        volbalance_target_max_flow_montagueTrenton_agg_nyc (Parameter): The volbalance target max flow Montague & Trenton aggregate NYC parameter.

    Methods:
        value(timestep, scenario_index): Returns the rescaled max flow NYC delivery for this reservoir.

    Class Methods:
        load(model, data): Loads the parameter from model and data dictionary.

    """
    def __init__(self, model, node, volbalance_relative_mrf_montagueTrenton, volbalance_target_max_flow_montagueTrenton_reservoir,
                 volbalance_target_max_flow_montagueTrenton_agg_nyc, **kwargs):
        super().__init__(model, **kwargs)
        self.node = node
        self.volbalance_relative_mrf_montagueTrenton = volbalance_relative_mrf_montagueTrenton
        self.volbalance_target_max_flow_montagueTrenton_reservoir = volbalance_target_max_flow_montagueTrenton_reservoir
        self.volbalance_target_max_flow_montagueTrenton_agg_nyc = volbalance_target_max_flow_montagueTrenton_agg_nyc
        self.children.add(volbalance_relative_mrf_montagueTrenton)
        self.children.add(volbalance_target_max_flow_montagueTrenton_reservoir)
        self.children.add(volbalance_target_max_flow_montagueTrenton_agg_nyc)

    def value(self, timestep, scenario_index):
        """
        Returns the rescaled max flow NYC delivery for this reservoir.

        Args:
            timestep (Timestep): The current timestep.
            scenario_index (ScenarioIndex): The scenario index.

        Returns:
            float: The rescaled max flow NYC delivery for this reservoir.
        """
        ### rescale the max flow NYC delivery for this reservoir after zeroing out if any reservoir had negative target
        return self.volbalance_relative_mrf_montagueTrenton.get_value(scenario_index) * \
               self.volbalance_target_max_flow_montagueTrenton_reservoir.get_value(scenario_index) / \
               self.volbalance_target_max_flow_montagueTrenton_agg_nyc.get_value(scenario_index)

    @classmethod
    def load(cls, model, data):
        """
        Loads the parameter from model and data dictionary.

        Args:
            model (Model): The Pywr model instance.
            data (dict): The data dictionary containing the parameter information.

        Returns:
            VolBalanceNYCDownstreamMRFFinal: The loaded parameter instance.
        """
        reservoir = data.pop("node")
        node = model.nodes[reservoir]
        reservoir = reservoir.split('_')[1]
        volbalance_relative_mrf_montagueTrenton = load_parameter(model, 'volbalance_relative_mrf_montagueTrenton')
        volbalance_target_max_flow_montagueTrenton_reservoir = load_parameter(model, f'volbalance_target_max_flow_montagueTrenton_{reservoir}')
        volbalance_target_max_flow_montagueTrenton_agg_nyc = load_parameter(model, f'volbalance_target_max_flow_montagueTrenton_agg_nyc')
        return cls(model, node, volbalance_relative_mrf_montagueTrenton, volbalance_target_max_flow_montagueTrenton_reservoir,
                   volbalance_target_max_flow_montagueTrenton_agg_nyc, **data)




class NYCCombinedReleaseFactor(Parameter):
    """
    Decides whether an NYC reservoir's release is dictated by its own
    storage (in the case of flood operations) or the aggregate storage across the three NYC reservoirs
    (in the case of normal or drought operations). It returns the "factor" which is a multiplier to baseline release
    value for the reservoir.
    See 8/30/2022 comment on this GitHub issue for the equation & logic:
    https://github.com/users/ahamilton144/projects/1/views/1?pane=issue&itemId=7839486

    Args:
        model (Model): The Pywr model instance.
        node (Node): The node associated with the parameter.

    Attributes:
        node (Node): The node associated with the parameter.
        drought_level_agg_nyc (Parameter): The drought level aggregate NYC parameter.
        mrf_drought_factor_agg_reservoir (Parameter): The MRF drought factor aggregate reservoir parameter.
        mrf_drought_factor_individual_reservoir (Parameter): The MRF drought factor individual reservoir parameter.

    Methods:
        value(timestep, scenario_index): Returns the overall release factor for the NYC reservoir.

    Class Methods:
        load(model, data): Loads the parameter from model and data dictionary.

    """
    def __init__(self, model, node, drought_level_agg_nyc, mrf_drought_factor_agg_reservoir,
                 mrf_drought_factor_individual_reservoir, **kwargs):
        super().__init__(model, **kwargs)
        self.node = node
        self.drought_level_agg_nyc = drought_level_agg_nyc
        self.mrf_drought_factor_agg_reservoir = mrf_drought_factor_agg_reservoir
        self.mrf_drought_factor_individual_reservoir = mrf_drought_factor_individual_reservoir
        self.children.add(drought_level_agg_nyc)
        self.children.add(mrf_drought_factor_agg_reservoir)
        self.children.add(mrf_drought_factor_individual_reservoir)

    def value(self, timestep, scenario_index):
        """
        Returns the overall release factor for the NYC reservoir, depending on whether it is flood stage 
        (in which case we use the reservoirs individual storage) or normal/drought stage 
        (in which case we use aggregate storage across the NYC reservoirs).

        Args:
            timestep (Timestep): The current timestep.
            scenario_index (ScenarioIndex): The scenario index.

        Returns:
            float: The overall release factor for the NYC reservoir.
        """
        ### $$ factor_{combined-cannonsville} = \min(\max(levelindex_{aggregated} - 2, 0), 1) * factor_{cannonsville}[levelindex_{aggregated}] +
        ###                                     \min(\max(3 - levelindex_{aggregated}, 0), 1) * factor_{cannonsville}[levelindex_{cannonsville}] $$

        return min(max(self.drought_level_agg_nyc.get_value(scenario_index) - 2, 0), 1) * \
                    self.mrf_drought_factor_agg_reservoir.get_value(scenario_index) + \
               min(max(3 - self.drought_level_agg_nyc.get_value(scenario_index), 0), 1) * \
                    self.mrf_drought_factor_individual_reservoir.get_value(scenario_index)

    @classmethod
    def load(cls, model, data):
        """
        Loads the parameter from model and data dictionary.

        Args:
            model (Model): The Pywr model instance.
            data (dict): The data dictionary containing the parameter information.

        Returns:
            NYCCombinedReleaseFactor: The loaded parameter instance.
        """
        reservoir = data.pop("node")
        node = model.nodes[reservoir]
        reservoir = reservoir.split('_')[1]
        drought_level_agg_nyc = load_parameter(model, f'drought_level_agg_nyc')
        mrf_drought_factor_agg_reservoir = load_parameter(model, f'mrf_drought_factor_agg_{reservoir}')
        mrf_drought_factor_individual_reservoir = load_parameter(model, f'mrf_drought_factor_individual_{reservoir}')
        return cls(model, node, drought_level_agg_nyc, mrf_drought_factor_agg_reservoir,
                   mrf_drought_factor_individual_reservoir, **data)


FfmpNjRunningAvgParameter.register()
VolBalanceNYCDemandTarget.register()
VolBalanceNYCDemandFinal.register()
VolBalanceNYCDownstreamMRFTargetAgg.register()
VolBalanceNYCDownstreamMRFTarget.register()
VolBalanceNYCDownstreamMRFFinal.register()
NYCCombinedReleaseFactor.register()


