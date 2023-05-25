import json
import pandas as pd
import sys
sys.path.append('..')

from pywrdrb.utils.directories import input_dir, model_data_dir

model_full_file = model_data_dir + 'drb_model_full.json'
model_sheets_start = model_data_dir + 'drb_model_'

EPS = 1e-8

### function for writing all relevant parameters to simulate starfit reservoir
def create_starfit_params(d, r):
    ### d = param dictionary, r = reservoir name

    ### parameters associated with STARFIT rule type
    starfit_remove_Rmax = True
    starfit_linear_below_NOR = True

    ### first get starfit const params for this reservoir
    for s in ['NORhi_alpha', 'NORhi_beta', 'NORhi_max', 'NORhi_min', 'NORhi_mu',
              'NORlo_alpha', 'NORlo_beta', 'NORlo_max', 'NORlo_min', 'NORlo_mu',
              'Release_alpha1', 'Release_alpha2', 'Release_beta1', 'Release_beta2',
              'Release_c', 'Release_max', 'Release_min', 'Release_p1', 'Release_p2',
              'Adjusted_CAP_MG', 'GRanD_MEANFLOW_MGD']:
        name = 'starfit_' + s + '_' + r
        d[name] = {}
        d[name]['type'] = 'constant'
        d[name]['url'] = 'drb_model_istarf_conus.csv'
        d[name]['column'] = s
        d[name]['index_col'] = 'reservoir'
        d[name]['index'] = r

    ### aggregated params - each needs agg function and list of params to agg
    agg_param_list = [('NORhi_sin', 'product', ['sin_weekly', 'NORhi_alpha']),
                      ('NORhi_cos', 'product', ['cos_weekly', 'NORhi_beta']),
                      ('NORhi_sum', 'sum', ['NORhi_mu', 'NORhi_sin', 'NORhi_cos']),
                      ('NORhi_minbound', 'max', ['NORhi_sum', 'NORhi_min']),
                      ('NORhi_maxbound', 'min', ['NORhi_minbound', 'NORhi_max']),
                      ('NORhi_final', 'product', ['NORhi_maxbound', 0.01]),
                      ('NORlo_sin', 'product', ['sin_weekly', 'NORlo_alpha']),
                      ('NORlo_cos', 'product', ['cos_weekly', 'NORlo_beta']),
                      ('NORlo_sum', 'sum', ['NORlo_mu', 'NORlo_sin', 'NORlo_cos']),
                      ('NORlo_minbound', 'max', ['NORlo_sum', 'NORlo_min']),
                      ('NORlo_maxbound', 'min', ['NORlo_minbound', 'NORlo_max']),
                      ('NORlo_final', 'product', ['NORlo_maxbound', 0.01]),
                      ('NORlo_final_unnorm', 'product', ['NORlo_final', 'Adjusted_CAP_MG']),
                      ('neg_NORhi_final_unnorm', 'product', ['neg_NORhi_final', 'Adjusted_CAP_MG']),
                      ('aboveNOR_sum', 'sum', ['volume', 'neg_NORhi_final_unnorm', 'flow_weekly']),
                      ('aboveNOR_final', 'product', ['aboveNOR_sum', 1 / 7]),
                      ('inNOR_sin', 'product', ['sin_weekly', 'Release_alpha1']),
                      ('inNOR_cos', 'product', ['cos_weekly', 'Release_beta1']),
                      ('inNOR_sin2x', 'product', ['sin2x_weekly', 'Release_alpha2']),
                      ('inNOR_cos2x', 'product', ['cos2x_weekly', 'Release_beta2']),
                      ('inNOR_p1a_num', 'sum', ['inNOR_fracvol', 'neg_NORlo_final']),
                      ('inNOR_p1a_denom', 'sum', ['NORhi_final', 'neg_NORlo_final']),
                      ('inNOR_p1a_final', 'product', ['inNOR_p1a_div', 'Release_p1']),
                      ('inNOR_inorm_pt1', 'sum', ['flow', 'neg_GRanD_MEANFLOW_MGD']),
                      ('inNOR_p2i', 'product', ['inNOR_inorm_final', 'Release_p2']),
                      ('inNOR_norm', 'sum',
                       ['inNOR_sin', 'inNOR_cos', 'inNOR_sin2x', 'inNOR_cos2x', 'Release_c', 'inNOR_p1a_final',
                        'inNOR_p2i', 1]),
                      ('inNOR_final', 'product', ['inNOR_norm', 'GRanD_MEANFLOW_MGD']),
                      ('Release_min_norm', 'sum', ['Release_min', 1]),
                      ('Release_min_final', 'product', ['Release_min_norm', 'GRanD_MEANFLOW_MGD'])]

    ### adjust params depending on whether we want to follow starfit strictly (starfit_linear_below_NOR = False),
    ###     or use a smoother linearly declining release policy below NOR
    if starfit_linear_below_NOR == False:
        agg_param_list += [('belowNOR_final', 'product', ['Release_min_final'])]
    else:
        agg_param_list += [('belowNOR_pt1', 'product', ['inNOR_final', 'belowNOR_frac_NORlo']),
                           ('belowNOR_final', 'max', ['belowNOR_pt1', 'Release_min_final'])]

    ### adjust params depending on whether we want to follow starfit directly (starfit_remove_Rmax = False),
    ###     or remove the max release param to allow for more realistic high flows
    if starfit_remove_Rmax == False:
        agg_param_list += [('Release_max_norm', 'sum', ['Release_max', 1])]
    else:
        agg_param_list += [('Release_max_norm', 'sum', ['Release_max', 999999])]

    ### now rest of aggregated params
    agg_param_list += [('Release_max_final', 'product', ['Release_max_norm', 'GRanD_MEANFLOW_MGD']),
                       ('target_pt2', 'max', ['target_pt1', 'Release_min_final']),
                       ('target_final', 'min', ['target_pt2', 'Release_max_final']),
                       ('release_pt1', 'sum', ['flow', 'volume']),
                       ('release_pt2', 'min', ['release_pt1', 'target_final']),
                       ('release_pt3', 'sum', ['release_pt1', 'neg_Adjusted_CAP_MG']),
                       ('release_final', 'max', ['release_pt2', 'release_pt3'])]

    ### loop over agg params, add to pywr dictionary/json
    for s, f, lp in agg_param_list:
        name = 'starfit_' + s + '_' + r
        d[name] = {}
        d[name]['type'] = 'aggregated'
        d[name]['agg_func'] = f
        d[name]['parameters'] = []
        for p in lp:
            if type(p) is int:
                param = p
            elif type(p) is float:
                param = p
            elif type(p) is str:
                if p.split('_')[0] in ('sin', 'cos', 'sin2x', 'cos2x'):
                    param = p
                elif p.split('_')[0] in ('volume', 'flow'):
                    param = p + '_' + r
                else:
                    param = 'starfit_' + p + '_' + r
            else:
                print('unsupported type in parameter list, ', p)
            d[name]['parameters'].append(param)

    ### negative params
    for s in ['NORhi_final', 'NORlo_final', 'GRanD_MEANFLOW_MGD', 'Adjusted_CAP_MG']:
        name = 'starfit_neg_' + s + '_' + r
        d[name] = {}
        d[name]['type'] = 'negative'
        d[name]['parameter'] = 'starfit_' + s + '_' + r

    ### division params
    for s, num, denom in [('inNOR_fracvol', 'volume', 'Adjusted_CAP_MG'),
                          ('inNOR_p1a_div', 'inNOR_p1a_num', 'inNOR_p1a_denom'),
                          ('inNOR_inorm_final', 'inNOR_inorm_pt1', 'GRanD_MEANFLOW_MGD'),
                          ('belowNOR_frac_NORlo', 'volume', 'NORlo_final_unnorm')]:
        name = 'starfit_' + s + '_' + r
        d[name] = {}
        d[name]['type'] = 'division'
        if num.split('_')[0] in ('sin', 'cos', 'sin2x', 'cos2x'):
            d[name]['numerator'] = num
        elif num.split('_')[0] in ('volume', 'flow'):
            d[name]['numerator'] = num + '_' + r
        else:
            d[name]['numerator'] = 'starfit_' + num + '_' + r
        if denom.split('_')[0] in ('sin', 'cos', 'sin2x', 'cos2x'):
            d[name]['denominator'] = denom
        elif denom.split('_')[0] in ('volume', 'flow'):
            d[name]['denominator'] = denom + '_' + r
        else:
            d[name]['denominator'] = 'starfit_' + denom + '_' + r

    ### other params
    other = {'starfit_level_' + r: {'type': 'controlcurveindex',
                                    'storage_node': 'reservoir_' + r,
                                    'control_curves': ['starfit_NORhi_final_' + r,
                                                       'starfit_NORlo_final_' + r]},
             'flow_weekly_' + r: {'type': 'aggregated', 'agg_func': 'product', 'parameters': ['flow_' + r, 7]},
             'volume_' + r: {'type': 'interpolatedvolume',
                             'values': [-EPS, 1000000],
                             'node': 'reservoir_' + r,
                             'volumes': [-EPS, 1000000]},
             'starfit_target_pt1_' + r: {'type': 'indexedarray',
                                         'index_parameter': 'starfit_level_' + r,
                                         'params': ['starfit_aboveNOR_final_' + r,
                                                    'starfit_inNOR_final_' + r,
                                                    'starfit_belowNOR_final_' + r]}}
    for name, params in other.items():
        d[name] = {}
        for k, v in params.items():
            d[name][k] = v

    return d


### create standard model node structures
def add_major_node(model, name, node_type, inflow_type, backup_inflow_type=None, outflow_type=None, downstream_node=None,
                   downstream_lag=0, capacity=None, initial_volume_frac=None, variable_cost=None):
    '''
    Add a major node to the model. Major nodes types include reservoir & river.
    This function will add the major node and all standard minor nodes that belong to each major node
    ( i.e., catchment, withdrawal, consumption, outflow), along with their standard parameters and edges.
    All nodes, edges, and parameters are added to the model dict, which is then returned
    :param model: the dict holding all model elements, which will be written to JSON file at completion.
    :param name: name of major node
    :param node_type: type of major node - either 'reservoir' or 'river'
    :param inflow_type: 'nhmv10', etc
    :param backup_inflow_type: 'nhmv10', etc. only active if inflow_type is a WEAP series - backup used to fill inflows for non-WEAP reservoirs.
    :param outflow_type: define what type of outflow node to use (if any) - either 'starfit' or 'regulatory'
    :param downstream_node: name of node directly downstream, for writing edge network.
    :param downstream_lag: travel time (in days) between flow leaving a node and reaching its downstream node
    :param capacity: (reservoirs only) capacity of reservoir in MG.
    :param initial_volume_frac: (reservoirs only) fraction full for reservoir initially
                           (note this is fraction, not percent, despite that it must be named "initial_volume_pc" for pywr json by convention)
    :param variable_cost: (reservoirs only) If False, cost is fixed throughout simulation.
                           If True, it varies according to state-dependent parameter.
    :return: model
    '''

    ### NYC reservoirs are a bit more complex, leave some of model creation in csv files for now
    is_NYC_reservoir = name in ['cannonsville', 'pepacton', 'neversink']
    ### does it have explicit outflow node for starfit or regulatory behavior?
    has_outflow_node = outflow_type in ['starfit', 'regulatory']
    ### list of river nodes
    river_nodes = ['delLordville', 'delMontague', 'delTrenton', 'outletAssunpink', 'outletSchuylkill',
                   'outletChristina']

    ### first add major node to dict
    if node_type == 'reservoir':
        node_name = f'reservoir_{name}'
        initial_volume = capacity * initial_volume_frac
        reservoir = {
            'name': node_name,
            'type': 'storage',
            'max_volume': f'max_volume_{name}',
            'initial_volume': initial_volume,
            'initial_volume_pc': initial_volume_frac,
            'cost': -10.0 if not variable_cost else f'storage_cost_{name}'
        }
        model['nodes'].append(reservoir)
    elif node_type == 'river':
        node_name = f'link_{name}'
        river = {
            'name': node_name,
            'type': 'link'
        }
        model['nodes'].append(river)

    ### add catchment node that sends inflows to reservoir
    catchment = {
        'name': f'catchment_{name}',
        'type': 'catchment',
        'flow': f'flow_{name}'
    }
    model['nodes'].append(catchment)

    ### add withdrawal node that withdraws flow from catchment for human use
    withdrawal = {
        'name': f'catchmentWithdrawal_{name}',
        'type': 'link',
        'cost': -15.0,
        'max_flow': f'max_flow_catchmentWithdrawal_{name}'
    }
    model['nodes'].append(withdrawal)

    ### add consumption node that removes flow from model - the rest of withdrawals return back to reservoir
    consumption = {
        'name': f'catchmentConsumption_{name}',
        'type': 'output',
        'cost': -200.0,
        'max_flow': f'max_flow_catchmentConsumption_{name}'
    }
    model['nodes'].append(consumption)

    ### add outflow node (if any), either using STARFIT rules or regulatory targets.
    ### Note reservoirs must have either starfit or regulatory, river nodes have either regulatory or None
    if outflow_type == 'starfit':
        outflow = {
            'name': f'outflow_{name}',
            'type': 'link',
            'cost': -500.0,
            'max_flow': f'starfit_release_final_{name}'
        }
        model['nodes'].append(outflow)
    elif outflow_type == 'regulatory':
        outflow = {
            'name': f'outflow_{name}',
            'type': 'rivergauge',
            'mrf': f'mrf_target_{name}',
            'mrf_cost': -1000.0
        }
        model['nodes'].append(outflow)

    ### add Delay node to account for flow travel time between nodes. Lag unit is days.
    if downstream_lag>0:
        delay = {
            'name': f'delay_{name}',
            'type': 'DelayNode',
            'days': downstream_lag
        }
        model['nodes'].append(delay)


    ### now add edges of model flow network
    ### catchment to reservoir
    model['edges'].append([f'catchment_{name}', node_name])
    ### catchment to withdrawal
    model['edges'].append([f'catchment_{name}', f'catchmentWithdrawal_{name}'])
    ### withdrawal to consumption
    model['edges'].append([f'catchmentWithdrawal_{name}', f'catchmentConsumption_{name}'])
    ### withdrawal to reservoir
    model['edges'].append([f'catchmentWithdrawal_{name}', node_name])
    ### reservoir downstream node (via outflow node if one exists)
    if downstream_node in river_nodes:
        downstream_name = f'link_{downstream_node}'
    elif downstream_node[0] == '0':
        downstream_name = f'link_{downstream_node}'
    elif downstream_node == 'output_del':
        downstream_name = downstream_node
    else:
        downstream_name = f'reservoir_{downstream_node}'
    if has_outflow_node:
        model['edges'].append([node_name, f'outflow_{name}'])
        if downstream_lag > 0:
            model['edges'].append([f'outflow_{name}', f'delay_{name}'])
            model['edges'].append([f'delay_{name}', downstream_name])
        else:
            model['edges'].append([f'outflow_{name}', downstream_name])

    else:
        if downstream_lag > 0:
            model['edges'].append([node_name, f'delay_{name}'])
            model['edges'].append([f'delay_{name}', downstream_name])
        else:
            model['edges'].append([node_name, downstream_name])

    ################################################################
    ### now add standard parameters
    ################################################################

    ### inflows to catchment - for now exclude gage nodes, since havent calculated demand yet
    if name[0] == '0':
        model['parameters'][f'flow_{name}'] = {
            'type': 'constant',
            'value': 0.
        }
    else:
        if 'WEAP' not in inflow_type:
            inflow_source = f'{input_dir}catchment_inflow_{inflow_type}.csv'
        else:
            if name in ['cannonsville', 'pepacton', 'neversink', 'wallenpaupack', 'promption',
                        'mongaupeCombined', 'beltzvilleCombined', 'blueMarsh', 'ontelaunee', 'nockamixon', 'assunpink']:
                inflow_source = f'{input_dir}catchment_inflow_{inflow_type}.csv'
            else:
                inflow_source = f'{input_dir}catchment_inflow_{backup_inflow_type}.csv'

        model['parameters'][f'flow_base_{name}'] = {
            'type': 'dataframe',
            'url': inflow_source,
            'column': name,
            'index_col': 'datetime',
            'parse_dates': True
        }
        model['parameters'][f'flow_{name}'] = {
            'type': 'aggregated',
            'agg_func': 'product',
            'parameters': [
                f'flow_base_{name}',
                'flow_factor'
            ]
        }

    ### max volume of reservoir, from GRanD database except where adjusted from other sources (eg NYC)
    if node_type == 'reservoir':
        model['parameters'][f'max_volume_{name}'] = {
            'type': 'constant',
            'url': 'drb_model_istarf_conus.csv',
            'column': 'Adjusted_CAP_MG',
            'index_col': 'reservoir',
            'index': name
        }

    ### for starfit reservoirs, need to add a bunch of starfit specific params
    if outflow_type == 'starfit':
        model['parameters'] = create_starfit_params(model['parameters'], name)

    ### get max flow for catchment withdrawal nodes based on DRBC data - for now exclude gage nodes, since havent calculated demand yet
    if name[0] == '0':
        model['parameters'][f'max_flow_catchmentWithdrawal_{name}'] = {
            'type': 'constant',
            'value': 0.
        }
    else:
        model['parameters'][f'max_flow_catchmentWithdrawal_{name}'] = {
            'type': 'constant',
            'url': f'{input_dir}sw_avg_wateruse_Pywr-DRB_Catchments.csv',
            'column': 'Total_WD_MGD',
            'index_col': 'node',
            'index': node_name
        }

    ### get max flow for catchment consumption nodes based on DRBC data - for now exclude gage nodes, since havent calculated demand yet
    if name[0] == '0':
        model['parameters'][f'max_flow_catchmentConsumption_{name}'] = {
            'type': 'constant',
            'value': 0.
        }
    else:
        ### assume the consumption_t = R * withdrawal_{t-1}, where R is the ratio of avg consumption to withdrawal from DRBC data
        model['parameters'][f'catchmentConsumptionRatio_{name}'] = {
            'type': 'constant',
            'url': f'{input_dir}sw_avg_wateruse_Pywr-DRB_Catchments.csv',
            'column': 'Total_CU_WD_Ratio',
            'index_col': 'node',
            'index': node_name
        }
        model['parameters'][f'prev_flow_catchmentWithdrawal_{name}'] = {
            'type': 'flow',
            'node': f'catchmentWithdrawal_{name}'
        }
        model['parameters'][f'max_flow_catchmentConsumption_{name}'] = {
            'type': 'aggregated',
            'agg_func': 'product',
            'parameters': [
                f'catchmentConsumptionRatio_{name}',
                f'prev_flow_catchmentWithdrawal_{name}'
            ]
        }

    return model


def drb_make_model(inflow_type, backup_inflow_type, start_date, end_date, use_hist_NycNjDeliveries=True):
    '''
    This function creates the JSON file used by Pywr to define the model. THis includes all nodes, edges, and parameters.
    :param inflow_type:
    :param backup_inflow_type:
    :param start_date:
    :param end_date:
    :param use_hist_NycNjDeliveries:
    :return:
    '''

    #######################################################################
    ### Basic model info
    #######################################################################

    ### create dict to hold all model nodes, edges, params, etc, following Pywr protocol. This will be saved to JSON at end.
    model = {
        'metadata': {
            'title': 'DRB',
            'description': 'Pywr DRB representation',
            'minimum_version': '0.4'
        },
        'timestepper': {
            'start': start_date,
            'end': end_date,
            'timestep': 1
        },
        'scenarios': [
            {
                'name': 'inflow',
                'size': 1
            }
        ]
    }

    model['nodes'] = []
    model['edges'] = []
    model['parameters'] = {}


    #######################################################################
    ### add major nodes (e.g., reservoirs) to model, along with corresponding minor nodes (e.g., withdrawals), edges, & parameters
    #######################################################################

    ### get initial reservoir storages as 80% of capacity
    istarf = pd.read_csv(f'{model_data_dir}drb_model_istarf_conus.csv')
    initial_volume_frac = 0.8
    
    def get_reservoir_capacity(reservoir):
        return float(istarf['Adjusted_CAP_MG'].loc[istarf['reservoir'] == reservoir].iloc[0])

    model = add_major_node(model, 'cannonsville', 'reservoir', inflow_type, backup_inflow_type, 'regulatory', '01425000', 0, get_reservoir_capacity('cannonsville'), initial_volume_frac, True)
    model = add_major_node(model, '01425000', 'river', inflow_type, backup_inflow_type, None, 'delLordville', 0)
    model = add_major_node(model, 'pepacton', 'reservoir', inflow_type, backup_inflow_type, 'regulatory', '01417000', 0, get_reservoir_capacity('pepacton'), initial_volume_frac, True)
    model = add_major_node(model, '01417000', 'river', inflow_type, backup_inflow_type, None, 'delLordville', 0)
    model = add_major_node(model, 'delLordville', 'river', inflow_type, backup_inflow_type, None, 'delMontague', 2)
    model = add_major_node(model, 'neversink', 'reservoir', inflow_type, backup_inflow_type, 'regulatory', '01436000', 0, get_reservoir_capacity('neversink'), initial_volume_frac, True)
    model = add_major_node(model, '01436000', 'river', inflow_type, backup_inflow_type, None, 'delMontague', 1)
    model = add_major_node(model, 'wallenpaupack', 'reservoir', inflow_type, backup_inflow_type, 'starfit', 'delMontague', 1, get_reservoir_capacity('wallenpaupack'), initial_volume_frac, False)
    model = add_major_node(model, 'prompton', 'reservoir', inflow_type, backup_inflow_type, 'starfit', 'delMontague', 1, get_reservoir_capacity('prompton'), initial_volume_frac, False)
    model = add_major_node(model, 'shoholaMarsh', 'reservoir', inflow_type, backup_inflow_type, 'starfit', 'delMontague', 1, get_reservoir_capacity('shoholaMarsh'), initial_volume_frac, False)
    model = add_major_node(model, 'mongaupeCombined', 'reservoir', inflow_type, backup_inflow_type, 'starfit', '01433500', 0, get_reservoir_capacity('mongaupeCombined'), initial_volume_frac, False)
    model = add_major_node(model, '01433500', 'river', inflow_type, backup_inflow_type, None, 'delMontague', 0)
    model = add_major_node(model, 'delMontague', 'river', inflow_type, backup_inflow_type, 'regulatory', 'delTrenton', 2)
    model = add_major_node(model, 'beltzvilleCombined', 'reservoir', inflow_type, backup_inflow_type, 'starfit', '01449800', 0, get_reservoir_capacity('beltzvilleCombined'), initial_volume_frac, False)
    model = add_major_node(model, '01449800', 'river', inflow_type, backup_inflow_type, None, 'delTrenton', 2)
    model = add_major_node(model, 'fewalter', 'reservoir', inflow_type, backup_inflow_type, 'starfit', '01447800', 0, get_reservoir_capacity('fewalter'), initial_volume_frac, False)
    model = add_major_node(model, '01447800', 'river', inflow_type, backup_inflow_type, None, 'delTrenton', 2)
    model = add_major_node(model, 'merrillCreek', 'reservoir', inflow_type, backup_inflow_type, 'starfit', 'delTrenton', 1, get_reservoir_capacity('merrillCreek'), initial_volume_frac, False)
    model = add_major_node(model, 'hopatcong', 'reservoir', inflow_type, backup_inflow_type, 'starfit', 'delTrenton', 1, get_reservoir_capacity('hopatcong'), initial_volume_frac, False)
    model = add_major_node(model, 'nockamixon', 'reservoir', inflow_type, backup_inflow_type, 'starfit', 'delTrenton', 0, get_reservoir_capacity('nockamixon'), initial_volume_frac, False)
    model = add_major_node(model, 'delTrenton', 'river', inflow_type, backup_inflow_type, 'regulatory', 'output_del', 1)
    model = add_major_node(model, 'assunpink', 'reservoir', inflow_type, backup_inflow_type, 'starfit', '01463620', 0, get_reservoir_capacity('assunpink'), initial_volume_frac, False)
    model = add_major_node(model, '01463620', 'river', inflow_type, backup_inflow_type, None, 'outletAssunpink', 0)
    model = add_major_node(model, 'outletAssunpink', 'river', inflow_type, backup_inflow_type, None, 'output_del', 0)
    model = add_major_node(model, 'ontelaunee', 'reservoir', inflow_type, backup_inflow_type, 'starfit', 'outletSchuylkill', 2, get_reservoir_capacity('ontelaunee'), initial_volume_frac, False)
    model = add_major_node(model, 'stillCreek', 'reservoir', inflow_type, backup_inflow_type, 'starfit', 'outletSchuylkill', 2, get_reservoir_capacity('stillCreek'), initial_volume_frac, False)
    model = add_major_node(model, 'blueMarsh', 'reservoir', inflow_type, backup_inflow_type, 'starfit', '01470960', 0, get_reservoir_capacity('blueMarsh'), initial_volume_frac, False)
    model = add_major_node(model, '01470960', 'river', inflow_type, backup_inflow_type, None, 'outletSchuylkill', 2)
    model = add_major_node(model, 'greenLane', 'reservoir', inflow_type, backup_inflow_type, 'starfit', 'outletSchuylkill', 1, get_reservoir_capacity('greenLane'), initial_volume_frac, False)
    model = add_major_node(model, 'outletSchuylkill', 'river', inflow_type, backup_inflow_type, None, 'output_del', 0)
    model = add_major_node(model, 'marshCreek', 'reservoir', inflow_type, backup_inflow_type, 'starfit', 'outletChristina', 0, get_reservoir_capacity('marshCreek'), initial_volume_frac, False)
    model = add_major_node(model, 'outletChristina', 'river', inflow_type, backup_inflow_type, None, 'output_del', 0)



    #######################################################################
    ### Add additional nodes beyond those associated with major nodes above
    #######################################################################

    ### Node for NYC aggregated storage across 3 reservoirs
    nyc_reservoirs = ['cannonsville', 'pepacton', 'neversink']
    model['nodes'].append({
            'name': 'reservoir_agg_nyc',
            'type': 'aggregatedstorage',
            'storage_nodes': [f'reservoir_{r}' for r in nyc_reservoirs]
    })

    ### Nodes linking each NYC reservoir to NYC deliveries
    for r in nyc_reservoirs:
        model['nodes'].append({
            'name': f'link_{r}_nyc',
            'type': 'link',
            'cost': -500.0,
            'max_flow': f'volbalance_max_flow_delivery_nyc_{r}'
        })

    ### Nodes for NYC & NJ deliveries
    for d in ['nyc', 'nj']:
        model['nodes'].append({
            'name': f'delivery_{d}',
            'type': 'output',
            'cost': -500.0,
            'max_flow': f'max_flow_delivery_{d}'
        })

    ### Node for final model sink in Delaware Bay
    model['nodes'].append({
            'name': 'output_del',
            'type': 'output'
        })


    #######################################################################
    ### Add additional edges beyond those associated with major nodes above
    #######################################################################

    ### Edges linking each NYC reservoir to NYC deliveries
    for r in nyc_reservoirs:
        model['edges'].append([f'reservoir_{r}', f'link_{r}_nyc'])
        model['edges'].append([f'link_{r}_nyc', 'delivery_nyc'])

    ### Edge linking Delaware River at Trenton to NJ deliveries
    model['edges'].append(['link_delTrenton', 'delivery_nj'])



    #######################################################################
    ### Add additional parameters beyond those associated with major nodes above
    #######################################################################

    ### Define "scenarios" based on flow multiplier -> only run one with 1.0 for now. Could switch scenario to modulate something else.
    model['parameters']['flow_factor'] = {
            'type': 'constantscenario',
            'scenario': 'inflow',
            'values': [1.0]
        }

    ### seasonal sinusoids for use with STARFIT
    for func in ['sin','cos','sin2x','cos2x']:
        for ts in ['weekly','daily']:
            model['parameters'][f'{func}_{ts}'] = {
                'type': f'{ts}profile',
                'url': f'drb_model_{ts}Profiles.csv',
                'index_col': 'profile',
                'index': f'{func}_{ts}'
            }

    ### demand for NYC
    if use_hist_NycNjDeliveries:
        ### if this flag is True, we assume demand is equal to historical deliveries timeseries
        model['parameters'][f'demand_nyc'] = {
            'type': 'dataframe',
            'url': f'{input_dir}deliveryNYC_ODRM_extrapolated.csv',
            'column': 'aggregate',
            'index_col': 'datetime',
            'parse_dates': True
        }
    else:
        ### otherwise, assume demand is equal to max allotment under FFMP
        model['parameters'][f'demand_nyc'] = {
            'type': 'constant',
            'url': 'drb_model_constants.csv',
            'column': 'value',
            'index_col': 'parameter',
            'index': 'max_flow_baseline_delivery_nyc'
        }

    ### repeat for NJ deliveries
    if use_hist_NycNjDeliveries:
        ### if this flag is True, we assume demand is equal to historical deliveries timeseries
        model['parameters'][f'demand_nj'] = {
            'type': 'dataframe',
            'url': f'{input_dir}deliveryNJ_WEAP_23Aug2022_gridmet_extrapolated.csv',
            'index_col': 'datetime',
            'parse_dates': True
        }
    else:
        ### otherwise, assume demand is equal to max allotment under FFMP
        model['parameters'][f'demand_nj'] = {
            'type': 'constant',
            'url': 'drb_model_constants.csv',
            'column': 'value',
            'index_col': 'parameter',
            'index': 'max_flow_baseline_monthlyAvg_delivery_nj'
        }

    ### max allowable delivery to NYC (on moving avg)
    model['parameters']['max_flow_baseline_delivery_nyc'] = {
        'type': 'constant',
        'url': 'drb_model_constants.csv',
        'column': 'value',
        'index_col': 'parameter',
        'index': 'max_flow_baseline_delivery_nyc'
    }

    ### NJ has both a daily limit and monthly average limit
    model['parameters']['max_flow_baseline_daily_delivery_nj'] = {
        'type': 'constant',
        'url': 'drb_model_constants.csv',
        'column': 'value',
        'index_col': 'parameter',
        'index': 'max_flow_baseline_daily_delivery_nj'
    }
    model['parameters']['max_flow_baseline_monthlyAvg_delivery_nj'] = {
        'type': 'constant',
        'url': 'drb_model_constants.csv',
        'column': 'value',
        'index_col': 'parameter',
        'index': 'max_flow_baseline_monthlyAvg_delivery_nj'
    }

    ### levels defining operational regimes for NYC reservoirs base on combined storage: 1a, 1b, 1c, 2, 3, 4, 5.
    ### Note 1a assumed to fill remaining space, doesnt need to be defined here.
    levels = ['1a', '1b', '1c', '2', '3', '4', '5']
    for level in levels[1:]:
        model['parameters'][f'level{level}'] = {
            'type': 'dailyprofile',
            'url': 'drb_model_dailyProfiles.csv',
            'index_col': 'profile',
            'index': f'level{level}'
        }

    ### Control curve index that tells us which level the aggregated NYC storage is currently in
    model['parameters']['drought_level_agg_nyc'] = {
        'type': 'controlcurveindex',
        'storage_node': 'reservoir_agg_nyc',
        'control_curves': [f'level{level}' for level in levels[1:]]
    }

    ### factors defining delivery profiles for NYC and NJ, for each storage level: 1a, 1b, 1c, 2, 3, 4, 5.
    demands = ['nyc', 'nj']
    for demand in demands:
        for level in levels:
            model['parameters'][f'level{level}_factor_delivery_{demand}'] = {
                'type': 'constant',
                'url': 'drb_model_constants.csv',
                'column': 'value',
                'index_col': 'parameter',
                'index': f'level{level}_factor_delivery_{demand}'
            }

    ### Indexed arrays that dictate cutbacks to NYC & NJ deliveries, based on current storage level and DOY
    for demand in demands:
        model['parameters'][f'drought_factor_delivery_{demand}'] = {
                'type': 'indexedarray',
                'index_parameter': 'drought_level_agg_nyc',
                'params': [f'level{level}_factor_delivery_{demand}' for level in levels]
            }

    ### Max allowable daily delivery to NYC & NJ- this will be very large for levels 1&2 (so that moving average limit
    ### in the next parameter is active), but apply daily flow limits for more drought stages.
    model['parameters']['max_flow_drought_delivery_nyc'] = {
            'type': 'aggregated',
            'agg_func': 'product',
            'parameters': [
                'max_flow_baseline_delivery_nyc',
                'drought_factor_delivery_nyc'
            ]
        }

    ### Max allowable delivery to NYC in current time step to maintain moving avg limit. Based on custom Pywr parameter.
    model['parameters']['max_flow_ffmp_delivery_nyc'] = {
            'type': 'FfmpNycRunningAvg',
            'node': 'delivery_nyc',
            'max_avg_delivery': 'max_flow_baseline_delivery_nyc'
        }

    ### Max allowable delivery to NJ in current time step to maintain moving avg limit. Based on custom Pywr parameter.
    model['parameters']['max_flow_ffmp_delivery_nj'] = {
            'type': 'FfmpNjRunningAvg',
            'node': 'delivery_nj',
            'max_avg_delivery': 'max_flow_baseline_monthlyAvg_delivery_nj',
            'max_daily_delivery': 'max_flow_baseline_daily_delivery_nj',
            'drought_factor': 'drought_factor_delivery_nj'
        }

    ### Now actual max flow to NYC delivery node is the min of demand, daily FFMP limit, and daily limit to meet FFMP moving avg limit
    model['parameters']['max_flow_delivery_nyc'] = {
            'type': 'aggregated',
            'agg_func': 'min',
            'parameters': [
                'demand_nyc',
                'max_flow_drought_delivery_nyc',
                'max_flow_ffmp_delivery_nyc'
            ]
        }

    ### Now actual max flow to NJ delivery node is the min of demand and daily limit to meet FFMP moving avg limit
    model['parameters']['max_flow_delivery_nj'] = {
            'type': 'aggregated',
            'agg_func': 'min',
            'parameters': [
                'demand_nj',
                'max_flow_ffmp_delivery_nj'
            ]
        }

    ### baseline release flow rate for each NYC reservoir, dictated by FFMP
    for reservoir in nyc_reservoirs:
        model['parameters'][f'mrf_baseline_{reservoir}'] = {
            'type': 'constant',
            'url': 'drb_model_constants.csv',
            'column': 'value',
            'index_col': 'parameter',
            'index': f'mrf_baseline_{reservoir}'
        }

    ### Control curve index that tells us which level each individual NYC reservoir's storage is currently in
    for reservoir in nyc_reservoirs:
        model['parameters'][f'drought_level_{reservoir}'] = {
            'type': 'controlcurveindex',
            'storage_node': f'reservoir_{reservoir}',
            'control_curves': [f'level{level}' for level in levels[1:]]
        }

    ### Factor governing changing release reqs from NYC reservoirs, based on aggregated storage across 3 reservoirs
    for reservoir in nyc_reservoirs:
        model['parameters'][f'mrf_drought_factor_agg_{reservoir}'] = {
            'type': 'indexedarray',
            'index_parameter': 'drought_level_agg_nyc',
            'params': [f'level{level}_factor_mrf_{reservoir}' for level in levels]
        }

    ### Levels defining operational regimes for individual NYC reservoirs, as opposed to aggregated level across 3 reservoirs
    for reservoir in nyc_reservoirs:
        for level in levels:
            model['parameters'][f'level{level}_factor_mrf_{reservoir}'] = {
                'type': 'dailyprofile',
                'url': 'drb_model_dailyProfiles.csv',
                'index_col': 'profile',
                'index': f'level{level}_factor_mrf_{reservoir}'
            }

    ### Factor governing changing release reqs from NYC reservoirs, based on individual storage for particular reservoir
    for reservoir in nyc_reservoirs:
        model['parameters'][f'mrf_drought_factor_individual_{reservoir}'] = {
            'type': 'indexedarray',
            'index_parameter': f'drought_level_{reservoir}',
            'params': [f'level{level}_factor_mrf_{reservoir}' for level in levels]
        }

    ### Factor governing changing release reqs from NYC reservoirs, depending on whether aggregated or individual storage level is activated
    ### Based on custom Pywr parameter.
    for reservoir in nyc_reservoirs:
        model['parameters'][f'mrf_drought_factor_combined_final_{reservoir}'] = {
            'type': 'NYCCombinedReleaseFactor',
            'node': f'reservoir_{reservoir}'
        }

    ### FFMP mandated releases from NYC reservoirs
    for reservoir in nyc_reservoirs:
        model['parameters'][f'mrf_target_individual_{reservoir}'] = {
            'type': 'aggregated',
            'agg_func': 'product',
            'parameters': [
                f'mrf_baseline_{reservoir}',
                f'mrf_drought_factor_combined_final_{reservoir}'
            ]
        }

    ### variable storage cost for each reservoir, based on its fractional storage
    ### Note: may not need this anymore now that we have volume balancing rules. but maybe makes sense to leave in for extra protection.
    volumes = {'cannonsville': get_reservoir_capacity('cannonsville'),
               'pepacton':  get_reservoir_capacity('pepacton'),
               'neversink': get_reservoir_capacity('neversink')}
    for reservoir in nyc_reservoirs:
        model['parameters'][f'storage_cost_{reservoir}'] = {
            'type': 'interpolatedvolume',
            'values': [-100,-1],
            'node': f'reservoir_{reservoir}',
            'volumes': [-EPS, volumes[reservoir] + EPS]
        }

    ### current volume stored in each reservoir, plus the aggregated storage node
    for reservoir in nyc_reservoirs + ['agg_nyc']:
        model['parameters'][f'volume_{reservoir}'] = {
            'type': 'interpolatedvolume',
            'values': [-EPS, 1000000],
            'node': f'reservoir_{reservoir}',
            'volumes': [-EPS, 1000000]
        }

    ### aggregated inflows to NYC reservoirs
    model['parameters']['flow_agg_nyc'] = {
            'type': 'aggregated',
            'agg_func': 'sum',
            'parameters': [f'flow_{reservoir}' for reservoir in nyc_reservoirs]
        }

    ### aggregated max volume across to NYC reservoirs
    model['parameters']['max_volume_agg_nyc'] = {
            'type': 'aggregated',
            'agg_func': 'sum',
            'parameters': [f'max_volume_{reservoir}' for reservoir in nyc_reservoirs]
        }

    ### Target release from each NYC reservoir to satisfy NYC demand - part 1.
    ### Uses custom Pywr parameter.
    for reservoir in nyc_reservoirs:
        model['parameters'][f'volbalance_target_max_flow_delivery_nyc_{reservoir}'] = {
            'type': 'VolBalanceNYCDemandTarget',
            'node': f'reservoir_{reservoir}'
        }

    ### Sum of target releases from step above
    model['parameters'][f'volbalance_target_max_flow_delivery_agg_nyc'] = {
            'type': 'aggregated',
            'agg_func': 'sum',
            'parameters': [f'volbalance_target_max_flow_delivery_nyc_{reservoir}' for reservoir in nyc_reservoirs]
        }

    ### Target release from each NYC reservoir to satisfy NYC demand - part 2,
    ### rescaling to make sure total contribution across 3 reservoirs is equal to total demand.
    ### Uses custom Pywr parameter.
    for reservoir in nyc_reservoirs:
        model['parameters'][f'volbalance_max_flow_delivery_nyc_{reservoir}'] = {
            'type': 'VolBalanceNYCDemandFinal',
            'node': f'reservoir_{reservoir}'
        }



    ### Baseline flow target at Montague & Trenton
    mrfs = ['delMontague', 'delTrenton']
    for mrf in mrfs:
        model['parameters'][f'mrf_baseline_{mrf}'] = {
                'type': 'constant',
                'url': 'drb_model_constants.csv',
                'column': 'value',
                'index_col': 'parameter',
                'index': f'mrf_baseline_{mrf}'
            }

    ### Seasonal multiplier factors for Montague & Trenton flow targets based on drought level of NYC aggregated storage
    for mrf in mrfs:
        for level in levels:
            model['parameters'][f'level{level}_factor_mrf_{mrf}'] = {
                'type': 'monthlyprofile',
                'url': 'drb_model_monthlyProfiles.csv',
                'index_col': 'profile',
                'index': f'level{level}_factor_mrf_{mrf}'
            }

    ### Current value of seasonal multiplier factor for Montague & Trenton flow targets based on drought level of NYC aggregated storage
    for mrf in mrfs:
        model['parameters'][f'mrf_drought_factor_{mrf}'] = {
            'type': 'indexedarray',
            'index_parameter': 'drought_level_agg_nyc',
            'params': [f'level{level}_factor_mrf_{mrf}' for level in levels]
        }

    ### Total Montague & Trenton flow targets based on drought level of NYC aggregated storage
    for mrf in mrfs:
        model['parameters'][f'mrf_target_{mrf}'] = {
            'type': 'aggregated',
            'agg_func': 'product',
            'parameters': [
                f'mrf_baseline_{mrf}',
                f'mrf_drought_factor_{mrf}'
            ]
        }

        ### total non-NYC inflows to Montague & Trenton
        upstream_nodes = ['prompton', 'wallenpaupack', 'shoholaMarsh', 'mongaupeCombined',  'delLordville', 'delMontague']
                          # 'delTrenton': ['beltzvilleCombined', 'fewalter', 'merrillCreek', 'hopatcong', 'nockamixon']}
        model['parameters'][f'volbalance_flow_agg_nonnyc_delMontague'] = {
                'type': 'aggregated',
                'agg_func': 'sum',
                'parameters': [f'flow_{node}' for node in upstream_nodes]
            }
        upstream_nodes = ['prompton', 'wallenpaupack', 'shoholaMarsh', 'mongaupeCombined',  'delLordville', 'delMontague', \
                          'beltzvilleCombined', 'fewalter', 'merrillCreek', 'hopatcong', 'nockamixon', 'delTrenton']
        model['parameters'][f'volbalance_flow_agg_nonnyc_delTrenton'] = {
                'type': 'aggregated',
                'agg_func': 'sum',
                'parameters': [f'flow_{node}' for node in upstream_nodes]
            }

        ### Get total release needed from NYC reservoirs to satisfy Montague & Trenton flow targets.
        ### Uses custom Pywr parameter.
        model['parameters']['volbalance_relative_mrf_montagueTrenton'] = {
            'type': 'VolBalanceNYCDownstreamMRFTargetAgg'
        }

        ### Target release from each NYC reservoir to satisfy Montague & Trenton flow targets - part 1.
        ### Uses custom Pywr parameter.
        for reservoir in nyc_reservoirs:
            model['parameters'][f'volbalance_target_max_flow_montagueTrenton_{reservoir}'] = {
                'type': 'VolBalanceNYCDownstreamMRFTarget',
                'node': f'reservoir_{reservoir}'
            }

        ### Sum of target releases from step above
        model['parameters'][f'volbalance_target_max_flow_montagueTrenton_agg_nyc'] = {
            'type': 'aggregated',
            'agg_func': 'sum',
            'parameters': [f'volbalance_target_max_flow_montagueTrenton_{reservoir}' for reservoir in nyc_reservoirs]
        }

        ### Target release from each NYC reservoir to satisfy Montague & Trenton flow targets - part 2,
        ### rescaling to make sure total contribution across 3 reservoirs is equal to total flow requirement.
        ### Uses custom Pywr parameter.
        for reservoir in nyc_reservoirs:
            model['parameters'][f'volbalance_max_flow_montagueTrenton_{reservoir}'] = {
                'type': 'VolBalanceNYCDownstreamMRFFinal',
                'node': f'reservoir_{reservoir}'
            }

    ### finally, get final effective mandated release from each NYC reservoir, which is the max of its
    ###    individually-mandated release from FFMP and its individual contribution to the Montague/Trenton targets
    for reservoir in nyc_reservoirs:
        model['parameters'][f'mrf_target_{reservoir}'] = {
            'type': 'aggregated',
            'agg_func': 'max',
            'parameters': [f'mrf_target_individual_{reservoir}', f'volbalance_max_flow_montagueTrenton_{reservoir}']
        }

    #######################################################################
    ### save full model as json
    #######################################################################

    with open(model_full_file, 'w') as o:
        json.dump(model, o, indent=4)