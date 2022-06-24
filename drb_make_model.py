import json
import ast
import numpy as np
import pandas as pd

### first load baseline model
model_base_file = 'model_data/drb_model_base.json'
model_sheets_file = 'model_data/drb_model_sheets.xlsx'
model_full_file = 'model_data/drb_model_full.json'
model = json.load(open(model_base_file, 'r'))

### load nodes from spreadsheet & add elements as dict items
for sheet in ['nodes']:
    df = pd.read_excel(model_sheets_file, sheet_name=sheet)
    model[sheet] = []
    for i in range(df.shape[0]):
        nodedict = {}
        for j, col in enumerate(df.columns):
            val = df.iloc[i,j]
            if isinstance(val, int):
                nodedict[col] = val
            elif isinstance(val, str):
                ### if it's a list, need to convert from str to actual list
                if '[' in val:
                    val = ast.literal_eval(val)
                ### if it's "true" or "false", convert to bool
                if val in ["\"true\"", "\"false\""]:
                    val = {"\"true\"": True, "\"false\"": False}[val]
                nodedict[col] = val
            elif isinstance(val, float) or isinstance(val, np.float64):
                if not np.isnan(val):
                    nodedict[col] = val
            else:
                print(f'type not supported: {type(val)}, instance: {val}')
        model[sheet].append(nodedict)


### load edges from spreadsheet & add elements as dict items
for sheet in ['edges']:
    df = pd.read_excel(model_sheets_file, sheet_name=sheet)
    model[sheet] = []
    for i in range(df.shape[0]):
        edge = []
        for j, col in enumerate(df.columns):
            val = df.iloc[i,j]
            if isinstance(val, int):
                edge.append(val)
            elif isinstance(val, str):
                ### if it's a list, need to convert from str to actual list
                if '[' in val:
                    val = ast.literal_eval(val)
                ### if it's "true" or "false", convert to bool
                if val in ["\"true\"", "\"false\""]:
                    val = {"\"true\"": True, "\"false\"": False}[val]
                edge.append(val)
            elif isinstance(val, float) or isinstance(val, np.float64):
                if not np.isnan(val):
                    edge.append(val)
            else:
                print(f'type not supported: {type(val)}, instance: {val}')
        model[sheet].append(edge)


### function for writing all relevant parameters to simulate starfit reservoir
def create_starfit_params(d, r):
    ### d = param dictionary, r = reservoir name

    ### first get starfit const params for this reservoir
    for s in ['NORhi_alpha', 'NORhi_beta', 'NORhi_max', 'NORhi_min', 'NORhi_mu',
              'NORlo_alpha', 'NORlo_beta', 'NORlo_max', 'NORlo_min', 'NORlo_mu',
              'Release_alpha1', 'Release_alpha2', 'Release_beta1', 'Release_beta2',
              'Release_c', 'Release_max', 'Release_min', 'Release_p1', 'Release_p2',
              'GRanD_CAP_MG', 'GRanD_MEANFLOW_MGD']:
        name = 'starfit_' + s + '_' + r
        d[name] = {}
        d[name]['type'] = 'constant'
        d[name]['url'] = 'drb_model_sheets.xlsx'
        d[name]['sheet_name'] = 'istarf_conus'
        d[name]['column'] = s
        d[name]['index_col'] = 'reservoir'
        d[name]['index'] = r

    ### aggregated params - each needs agg function and list of params to agg
    for s, f, lp in [('Release_max_unnorm_pt1', 'sum', ['Release_max', 1]),
                     ('Release_max_unnorm_final', 'product', ['Release_max_unnorm_pt1', 'GRanD_MEANFLOW_MGD']),
                     ('Release_min_unnorm_pt1', 'sum', ['Release_min', 1]),
                     ('Release_min_unnorm_final', 'product', ['Release_min_unnorm_pt1', 'GRanD_MEANFLOW_MGD']),
                     ('NORhi_sin', 'product', ['sin_weekly', 'NORhi_alpha']),
                     ('NORhi_cos', 'product', ['cos_weekly', 'NORhi_beta']),
                     ('NORhi_sum', 'sum', ['NORhi_mu', 'NORhi_sin', 'NORhi_cos']),
                     ('NORhi_minbound', 'max', ['NORhi_sum', 'NORhi_min']),
                     ('NORhi_final', 'min', ['NORhi_minbound', 'NORhi_max']),
                     ('NORlo_sin', 'product', ['sin_weekly', 'NORlo_alpha']),
                     ('NORlo_cos', 'product', ['cos_weekly', 'NORlo_beta']),
                     ('NORlo_sum', 'sum', ['NORlo_mu', 'NORlo_sin', 'NORlo_cos']),
                     ('NORlo_minbound', 'max', ['NORlo_sum', 'NORlo_min']),
                     ('NORlo_final', 'min', ['NORlo_minbound', 'NORlo_max']),
                     ('aboveNOR_final', 'sum', ['volume', 'neg_NORhi_final', 'flow']),
                     ('inNOR_sin', 'product', ['sin_weekly', 'Release_alpha1']),
                     ('inNOR_cos', 'product', ['cos_weekly', 'Release_beta1']),
                     ('inNOR_sin2x', 'product', ['sin2x_weekly', 'Release_alpha2']),
                     ('inNOR_cos2x', 'product', ['cos2x_weekly', 'Release_beta2']),
                     ('inNOR_p1a_num', 'sum', ['volume', 'neg_NORlo_final']),
                     ('inNOR_p1a_denom', 'sum', ['NORhi_final', 'neg_NORlo_final']),
                     ('inNOR_p1a_final', 'product', ['inNOR_p1a_div', 'Release_p2']),
                     ('inNOR_inorm_pt1', 'sum', ['flow', 'neg_GRanD_CAP_MG']),
                     ('inNOR_p2i', 'product', ['inNOR_inorm_final', 'Release_p2']),
                     ('inNOR_norm', 'sum', ['inNOR_sin', 'inNOR_cos', 'inNOR_sin2x', 'inNOR_cos2x', 'Release_c', 'inNOR_p1a_final', 'inNOR_p2i', 1]),
                     ('inNOR_final', 'product', ['inNOR_norm', 'GRanD_MEANFLOW_MGD']),
                     ('belowNOR_final', 'sum', ['Release_min_unnorm_final']),
                     ('target_pt2', 'max', ['target_pt1', 'Release_min_unnorm_final']),
                     ('target_final', 'min', ['target_pt2', 'Release_max_unnorm_final']),
                     ('release_pt1', 'sum', ['flow', 'volume']),
                     ('release_pt2', 'min', ['release_pt1', 'target_final']),
                     ('release_pt3', 'sum', ['release_pt1', 'neg_GRanD_CAP_MG']),
                     ('release_final', 'max', ['release_pt2', 'release_pt3'])]:
        name = 'starfit_' + s + '_' + r
        d[name] = {}
        d[name]['type'] = 'aggregated'
        d[name]['agg_func'] = f
        d[name]['parameters'] = []
        for p in lp:
            if type(p) is int:
                param = p
            elif type(p) is str:
                if p.split('_')[0] in ('sin','cos','sin2x','cos2x'):
                    param = p
                elif p.split('_')[0] in ('volume', 'flow'):
                    param = p + '_' + r
                else:
                    param = 'starfit_' + p + '_' + r
            else:
                print('unsupported type in parameter list, ', p)
            d[name]['parameters'].append(param)

    ### negative params
    for s in ['NORhi_final', 'NORlo_final', 'GRanD_MEANFLOW_MGD', 'GRanD_CAP_MG']:
        name = 'starfit_neg_' + s + '_' + r
        d[name] = {}
        d[name]['type'] = 'negative'
        d[name]['parameter'] = 'starfit_' + s + '_' + r

    ### division params
    for s, num, denom in [('inNOR_p1a_div', 'inNOR_p1a_num', 'inNOR_p1a_denom'),
                          ('inNOR_inorm_final', 'inNOR_inorm_pt1', 'GRanD_MEANFLOW_MGD')]:
        name = 'starfit_' + s + '_' + r
        d[name] = {}
        d[name]['type'] = 'division'
        d[name]['numerator'] = 'starfit_' + num + '_' + r
        d[name]['denominator'] = 'starfit_' + denom + '_' + r

    ### other params
    other = {'starfit_level_wallenpaupack': {'type': 'controlcurveindex',
                                            'storage_node': 'reservoir_wallenpaupack',
                                            'control_curves': ['starfit_NORhi_final_wallenpaupack',
                                                               'starfit_NORlo_final_wallenpaupack']},
             'volume_wallenpaupack': {'type': 'interpolatedvolume',
                                      'values': [0, 1000000],
                                      'node': 'reservoir_wallenpaupack',
                                      'volumes': [0, 1000000]},
             'starfit_target_pt1_wallenpaupack': {'type': 'indexedarray',
                                                  'index_parameter': 'starfit_level_wallenpaupack',
                                                  'params': ['starfit_aboveNOR_final_wallenpaupack',
                                                             'starfit_inNOR_final_wallenpaupack',
                                                             'starfit_belowNOR_final_wallenpaupack']}}
    for name, params in other.items():
        d[name] = {}
        for k,v in params.items():
            d[name][k] = v



    return d


### load parameters from spreadsheet & add elements as dict items
for sheet in ['parameters']:
    df = pd.read_excel(model_sheets_file, sheet_name=sheet)
    model[sheet] = {}
    for i in range(df.shape[0]):
        name = df.iloc[i,0]
        ### skip empty line
        if type(name) is not str:
            pass
        ### other than starfit types, print elements from excel directly into json
        elif df.iloc[i,1] != 'starfit':
            model[sheet][name] = {}
            for j, col in enumerate(df.columns[1:], start=1):
                val = df.iloc[i,j]
                if isinstance(val, int):
                    model[sheet][name][col] = val
                elif isinstance(val, str):
                    ### if it's a list, need to convert from str to actual list
                    if '[' in val:
                        val = ast.literal_eval(val)
                    ### if it's "true" or "false", convert to bool
                    if val in ["\"true\"", "\"false\""]:
                        val = {"\"true\"": True, "\"false\"": False}[val]
                    model[sheet][name][col] = val
                elif isinstance(val, float) or isinstance(val, np.float64):
                    if not np.isnan(val):
                        model[sheet][name][col] = val
                else:
                    print(f'type not supported: {type(val)}, instance: {val}')
        ### for starfit types, follow function to create all starfit params for this reservoir
        else:
            reservoir = name.split('_')[1]
            model[sheet] = create_starfit_params(model[sheet], reservoir)


### save full model as json
with open(model_full_file, 'w') as o:
    json.dump(model, o, indent=4)
