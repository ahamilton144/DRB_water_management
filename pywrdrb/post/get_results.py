import numpy as np
import pandas as pd
import h5py

from utils.lists import reservoir_list, reservoir_list_nyc, majorflow_list, reservoir_link_pairs
from utils.lists import drbc_lower_basin_reservoirs
from utils.constants import cms_to_mgd, cfs_to_mgd, cm_to_mg
from utils.hdf5 import get_hdf5_realization_numbers

### Contains functions used to process Pywr-DRB data.  

def get_pywr_results(output_dir, model, results_set='all', scenario=0, datetime_index=None):
    """
    Gathers simulation results from Pywr model run and returns a pd.DataFrame.

    Args:
        output_dir (str): The output directory.
        model (str): The model datatype name (e.g., "nhmv10").
        results_set (str, optional): The results set to return. Can be one of the following:
            - "all": Return all results.
            - "reservoir_downstream_gage": Return downstream gage flow below reservoir.
            - "res_storage": Return reservoir storages.
            - "major_flow": Return flow at major flow points of interest.
            - "inflow": Return the inflow at each catchment.
            (Default: 'all')
        scenario (int, optional): The scenario index number. (Default: 0)
        datetime_index (Pandas datetime_index): Creating the dates are slow: if this isn't our first data retrieval, we can provide the dates from a previous results dataframe.

    Returns:
        pd.DataFrame: The simulation results with datetime index.
    """
    with h5py.File(f'{output_dir}drb_output_{model}.hdf5', 'r') as f:
        keys = list(f.keys())
        results = pd.DataFrame()
        if results_set == 'all':
            for k in keys:
                results[k] = f[k][:, scenario]
        elif results_set == 'reservoir_downstream_gage':
            ## Need to pull flow data for link_ downstream of reservoirs instead of simulated outflows
            keys_with_link = [k for k in keys if k.split('_')[0] == 'link' and k.split('_')[1] in reservoir_link_pairs.values()]
            # print(keys_with_link)
            for k in keys_with_link:
                res_name = [res for res, link in reservoir_link_pairs.items() if link == k.split('_')[1]][0]
                results[res_name] = f[k][:, scenario]
            # Now pull simulated relases from un-observed reservoirs
            keys_without_link = [k for k in keys if k.split('_')[0] == 'outflow' and k.split('_')[1] in reservoir_list and k.split('_')[1] not in reservoir_link_pairs.keys()]
            for k in keys_without_link:
                results[k.split('_')[1]] = f[k][:, scenario]
        elif results_set == 'res_storage':
            keys = [k for k in keys if k.split('_')[0] == 'reservoir' and k.split('_')[1] in reservoir_list]
            for k in keys:
                results[k.split('_')[1]] = f[k][:, scenario]
        elif results_set == 'major_flow':
            keys = [k for k in keys if k.split('_')[0] == 'link' and k.split('_')[1] in majorflow_list]
            for k in keys:
                results[k.split('_')[1]] = f[k][:, scenario]
        elif results_set == 'res_release':
            ### reservoir releases are "outflow" plus "spill". Not all reservoirs have spill.
            keys_outflow = [f'outflow_{r}' for r in reservoir_list]
            for k in keys_outflow:
                results[k.split('_')[1]] = f[k][:, scenario]
            keys_spill = [f'spill_{r}' for r in reservoir_list]
            for k in keys_spill:
                results[k.split('_')[1]] += f[k][:, scenario]
        elif results_set == 'downstream_release_target':
            for reservoir in reservoir_list_nyc:
                results[reservoir] = f[f'{results_set}_{reservoir}'][:, scenario]
        elif results_set == 'inflow':
            keys = [k for k in keys if k.split('_')[0] == 'catchment']
            for k in keys:
                results[k.split('_')[1]] = f[k][:, scenario]
        elif results_set == 'catchment_withdrawal':
            keys = [k for k in keys if k.split('_')[0] == 'catchmentWithdrawal']
            for k in keys:
                results[k.split('_')[1]] = f[k][:, scenario]
        elif results_set == 'catchment_consumption':
            keys = [k for k in keys if k.split('_')[0] == 'catchmentConsumption']
            for k in keys:
                results[k.split('_')[1]] = f[k][:, scenario]
        elif results_set in ('prev_flow_catchmentWithdrawal', 'max_flow_catchmentWithdrawal', 'max_flow_catchmentConsumption'):
            keys = [k for k in keys if results_set in k]
            for k in keys:
                results[k.split('_')[-1]] = f[k][:, scenario]
        elif results_set in ('res_level'):
            keys = [k for k in keys if 'drought_level' in k]
            for k in keys:
                results[k.split('_')[-1]] = f[k][:, scenario]
        elif results_set == 'ffmp_level_boundaries':
            keys = [f'level{l}' for l in ['1b','1c','2','3','4','5']]
            for k in keys:
                results[k] = f[k][:, scenario]
        elif results_set == 'mrf_target':
            keys = [k for k in keys if results_set in k]
            for k in keys:
                results[k.split('mrf_target_')[1]] = f[k][:, scenario]
        elif results_set == 'nyc_release_components':
            keys = [f'mrf_target_individual_{reservoir}' for reservoir in reservoir_list_nyc] + \
                    [f'flood_release_{reservoir}' for reservoir in reservoir_list_nyc] + \
                    [f'mrf_montagueTrenton_{reservoir}' for reservoir in reservoir_list_nyc] + \
                    [f'spill_{reservoir}' for reservoir in reservoir_list_nyc]
            for k in keys:
                results[k] = f[k][:, scenario]
        elif results_set == 'lower_basin_mrf_contributions':
            keys = [f'mrf_trenton_{reservoir}' for reservoir in drbc_lower_basin_reservoirs]
            for k in keys:
                results[k] = f[k][:, scenario]
        elif results_set == 'ibt_demands':
            keys = ['demand_nyc','demand_nj']
            for k in keys:
                results[k] = f[k][:, scenario]
        elif results_set == 'ibt_diversions':
            keys = ['delivery_nyc','delivery_nj']
            for k in keys:
                results[k] = f[k][:, scenario]
        elif results_set == 'mrf_targets':
            keys = ['mrf_target_delMontague','mrf_target_delTrenton']
            for k in keys:
                results[k] = f[k][:, scenario]
        elif results_set == 'all_mrf':
            keys = [k for k in keys if 'mrf' in k]
            for k in keys:
                results[k] = f[k][:, scenario]

        else:
            print('Invalid results_set specified.')
            return

        if datetime_index is not None:
            if len(datetime_index) == len(f['time']):
                results.index = datetime_index
                reuse_datetime_index = True
            else:
                reuse_datetime_index = False
        else:
            reuse_datetime_index = False

        if not reuse_datetime_index:
            # Format datetime index
            day = [f['time'][i][0] for i in range(len(f['time']))]
            month = [f['time'][i][2] for i in range(len(f['time']))]
            year = [f['time'][i][3] for i in range(len(f['time']))]
            date = [f'{y}-{m}-{d}' for y, m, d in zip(year, month, day)]
            datetime_index = pd.to_datetime(date)
            results.index = datetime_index

        return results, datetime_index


### load flow estimates from raw input datasets
def get_base_results(input_dir, model, datetime_index=None, results_set='all', ensemble_scenario=None):
    """
    Function for retrieving and organizing results from non-pywr streamflows (NHM, NWM, WEAP).

    Args:
        input_dir (str): The input data directory.
        model (str): The model datatype name (e.g., "nhmv10").
        datetime_index: The datetime index.
        results_set (str, optional): The results set to return. Can be one of the following:
            - "all": Return all results.
            - "reservoir_downstream_gage": Return downstream gage flow below reservoir.
            - "major_flow": Return flow at major flow points of interest.
            (Default: 'all')

    Returns:
        pd.DataFrame: The retrieved and organized results with datetime index.
    """
    if ensemble_scenario is None:
        gage_flow = pd.read_csv(f'{input_dir}gage_flow_{model}.csv')
        gage_flow.index = pd.DatetimeIndex(gage_flow['datetime'])
        gage_flow = gage_flow.drop('datetime', axis=1)
    else:
        with h5py.File(f'{input_dir}gage_flow_{model}.hdf5', 'r') as f:
            nodes = list(f.keys())
            gage_flow = pd.DataFrame()
            for node in nodes:
                gage_flow[node] = f[f'{node}/realization_{ensemble_scenario}']

            if datetime_index is not None:
                if len(datetime_index) == len(f[nodes[0]]['date']):
                    gage_flow.index = datetime_index
                    reuse_datetime_index = True
                else:
                    reuse_datetime_index = False
            else:
                reuse_datetime_index = False

            if not reuse_datetime_index:
                # Format datetime index
                # day = [f[nodes[0]]['date'][i][0] for i in range(len(f[nodes[0]]['date']))]
                # month = [f[nodes[0]]['date'][i][2] for i in range(len(f['time']))]
                # year = [f['time'][i][3] for i in range(len(f['time']))]
                # date = [f'{y}-{m}-{d}' for y, m, d in zip(year, month, day)]
                datetime = [str(d, 'utf-8') for d in f[nodes[0]]['date']]
                datetime_index = pd.to_datetime(datetime)
                gage_flow.index = datetime_index
    if results_set == 'reservoir_downstream_gage':
        available_release_data = gage_flow.columns.intersection(reservoir_link_pairs.values())
        reservoirs_with_data = [list(filter(lambda x: reservoir_link_pairs[x] == site, reservoir_link_pairs))[0] for
                                site in available_release_data]
        gage_flow = gage_flow.loc[:, available_release_data]
        gage_flow.columns = reservoirs_with_data
    elif results_set == 'major_flow':
        for c in gage_flow.columns:
            if c not in majorflow_list:
                gage_flow = gage_flow.drop(c, axis=1)
    # print(f'Index with notation {gage_flow.index[0]} and type {type(gage_flow.index)}')
    # gage_flow = gage_flow.loc[datetime_index, :]

    return gage_flow, datetime_index




def get_all_historic_reconstruction_pywr_results(output_dir, model_list, 
                                                results_set,
                                                start_date, end_date,
                                                realization_subset = None):
    """Loads all historic reconstruction results, 
    stored in a single dictionary.

    Args:
        output_dir (str): Output folder directory.
        model_list (list): List of model names.
        results_set (str): Types of results to return.
        start_date (str): Start date.
        end_date (str): End date.

    Returns:
        dict: A dictionary of results.
    """
    reults_type_options = ['all', 'reservoir_downstream_gage', 
                           'res_storage', 'major_flow', 'inflow', 'res_release', 
                           'catchment_withdrawal', 'catchment_consumption', 
                           'res_level', 'ffmp_level_boundaries', 'mrf_targets', 'mrf_target', 
                           'nyc_release_components', 'ibt_demands',
                           'lower_basin_mrf_contributions', 'ibt_diversions',
                           'release_needed_mrf_trenton_step1', 
                           'all_mrf']
    
    assert(results_set in reults_type_options), f'results_set must be one of {reults_type_options}'
    
    datetime_index = pd.date_range(start_date, end_date, freq='D')
    results = {}
    # Loop through models
    for model in model_list:
        output_filename = f'{output_dir}drb_output_{model}.hdf5'
        
        # Handle ensembles differently
        if 'ensemble' in model:
            input_filename = f'{output_dir}../input_data/historic_ensembles/gage_flow_{model}.hdf5'
            realization_numbers = get_hdf5_realization_numbers(input_filename) if realization_subset is None else realization_subset

            results[f'pywr_{model}'] = {}
            # Loop through ensemble realizations
            for real in realization_numbers:
                results[f'pywr_{model}'][f'realization_{real}'], datetime_index = get_pywr_results(output_dir, model, 
                                                                                                results_set=results_set, 
                                                                                                datetime_index=datetime_index, 
                                                                                                scenario= real)

        else:
            results[f'pywr_{model}'], datetime_index = get_pywr_results(output_dir, model, 
                                                                        results_set=results_set, 
                                                                        datetime_index=datetime_index) 
    return results