import numpy as np
import pandas as pd
import sys

from plotting.plotting_functions import *

### I was having trouble with interactive console plotting in Pycharm for some reason - comment this out if you want to use that and not having issues
#mpl.use('TkAgg')

### directories
output_dir = 'output_data/'
input_dir = 'input_data/'
fig_dir = 'figs/'

# Constants
cms_to_mgd = 22.82
cm_to_mg = 264.17/1e6
cfs_to_mgd = 0.0283 * 22824465.32 / 1e6


### list of reservoirs and major flow points to compare across models
reservoir_list = ['cannonsville', 'pepacton', 'neversink', 'wallenpaupack', 'prompton', 'shoholaMarsh', \
                   'mongaupeCombined', 'beltzvilleCombined', 'fewalter', 'merrillCreek', 'hopatcong', 'nockamixon', \
                   'assunpink', 'ontelaunee', 'stillCreek', 'blueMarsh', 'greenLane', 'marshCreek']

majorflow_list = ['delLordville', 'delMontague', 'delTrenton', 'outletAssunpink', 'outletSchuylkill', 'outletChristina',
                  '01425000', '01417000', '01436000', '01433500', '01449800',
                  '01447800', '01463620', '01470960']

reservoir_link_pairs = {'cannonsville': '01425000',
                           'pepacton': '01417000',
                           'neversink': '01436000',
                           'mongaupeCombined': '01433500',
                           'beltzvilleCombined': '01449800',
                           'fewalter': '01447800',
                           'assunpink': '01463620',
                           'blueMarsh': '01470960'}




## Execution - Generate all figures
if __name__ == "__main__":

    ## System inputs
    rerun_all = True
    use_WEAP = False

    ### User-specified date range, or default to minimum overlapping period across models
    if use_WEAP:
        start_date = sys.argv[1] if len(sys.argv) > 1 else '1999-06-01'
        end_date = sys.argv[2] if len(sys.argv) > 2 else '2010-05-31'
    else:
        start_date = sys.argv[1] if len(sys.argv) > 1 else '1983-10-01'
        end_date = sys.argv[2] if len(sys.argv) > 2 else '2017-01-01'

    # start_date = sys.argv[1] if len(sys.argv) > 1 else '1999-06-01'
    # end_date = sys.argv[2] if len(sys.argv) > 2 else '2010-05-31'

    ## Load data    
    # Load Pywr-DRB simulation models
    print('Retrieving simulation data.')
    if use_WEAP:
        pywr_models = ['obs_pub', 'nhmv10', 'nwmv21', 'WEAP_23Aug2022_gridmet_nhmv10']
    else:
        pywr_models = ['obs_pub', 'nhmv10', 'nwmv21']

    res_releases = {}
    major_flows = {}
    
    for model in pywr_models:
        res_releases[f'pywr_{model}'] = get_pywr_results(output_dir, model, 'res_release').loc[start_date:end_date,:]
        major_flows[f'pywr_{model}'] = get_pywr_results(output_dir, model, 'major_flow').loc[start_date:end_date,:]
    pywr_models = [f'pywr_{m}' for m in pywr_models]

    # Load base (non-pywr) models
    if use_WEAP:
        base_models = ['obs', 'obs_pub', 'nhmv10', 'nwmv21', 'WEAP_23Aug2022_gridmet']
    else:
        base_models = ['obs', 'obs_pub', 'nhmv10', 'nwmv21']

    datetime_index = list(res_releases.values())[0].index
    for model in base_models:
        res_releases[model] = get_base_results(input_dir, model, datetime_index, 'res_release').loc[start_date:end_date,:]
        major_flows[model] = get_base_results(input_dir, model, datetime_index, 'major_flow').loc[start_date:end_date,:]

    # Verify that all datasets have same datetime index
    for r in res_releases.values():
        assert ((r.index == datetime_index).mean() == 1)
    for r in major_flows.values():
        assert ((r.index == datetime_index).mean() == 1)
    print(f'Successfully loaded {len(base_models)} base model results & {len(pywr_models)} pywr model results')

    ## 3-part flow figures with releases
    if rerun_all:
        print('Plotting 3-part flows at reservoirs.')
        ### nhm only - slides 36-39 in 10/24/2022 presentation
        plot_3part_flows(res_releases, ['nhmv10'], 'pepacton')
        ### nhm vs nwm - slides 40-42 in 10/24/2022 presentation
        plot_3part_flows(res_releases, ['nhmv10', 'nwmv21'], 'pepacton')
        if use_WEAP:
            ### nhm vs weap (with nhm backup) - slides 60-62 in 10/24/2022 presentation
            plot_3part_flows(res_releases, ['nhmv10', 'WEAP_23Aug2022_gridmet'], 'pepacton')
        ### nhm vs pywr-nhm - slides 60-62 in 10/24/2022 presentation
        plot_3part_flows(res_releases, ['nhmv10', 'pywr_nhmv10'], 'pepacton')
        ## obs-pub only
        plot_3part_flows(res_releases, ['obs_pub'], 'pepacton')
        plot_3part_flows(res_releases, ['pywr_obs_pub'], 'pepacton')
        plot_3part_flows(res_releases, ['obs_pub', 'nhmv10'], 'pepacton')
        plot_3part_flows(res_releases, ['pywr_obs_pub', 'pywr_nhmv10'], 'pepacton')

        plot_3part_flows(res_releases, ['obs_pub', 'nhmv10'], 'cannonsville')
        plot_3part_flows(res_releases, ['pywr_obs_pub', 'pywr_nhmv10'], 'cannonsville')
        plot_3part_flows(res_releases, ['pywr_obs_pub', 'pywr_nhmv10'], 'neversink')


    if rerun_all:
        print('Plotting weekly flow distributions at reservoirs.')
        ### nhm only - slides 36-39 in 10/24/2022 presentation
        plot_weekly_flow_distributions(res_releases, ['nhmv10'], 'pepacton')
        ### nhm vs nwm - slides 35-37 in 10/24/2022 presentation
        plot_weekly_flow_distributions(res_releases, ['nhmv10', 'nwmv21'], 'pepacton')
        if use_WEAP:
            ### nhm vs weap (with nhm backup) - slides 68 in 10/24/2022 presentation
            plot_weekly_flow_distributions(res_releases, ['nhmv10', 'WEAP_23Aug2022_gridmet'], 'pepacton')
        ### nhm vs pywr-nhm - slides 68 in 10/24/2022 presentation
        plot_weekly_flow_distributions(res_releases, ['nhmv10', 'pywr_nhmv10'], 'pepacton')

        ## obs_pub
        plot_weekly_flow_distributions(res_releases, ['obs_pub'], 'pepacton')
        plot_weekly_flow_distributions(res_releases, ['nhmv10', 'obs_pub'], 'pepacton')
        plot_weekly_flow_distributions(res_releases, ['pywr_obs_pub', 'pywr_nhmv10'], 'pepacton')
        plot_weekly_flow_distributions(res_releases, ['pywr_obs_pub', 'pywr_nhmv10'], 'cannonsville')
        plot_weekly_flow_distributions(res_releases, ['pywr_obs_pub', 'pywr_nhmv10'], 'neversink')

        


    nodes = ['cannonsville', 'pepacton', 'neversink', 'fewalter', 'beltzvilleCombined', 'blueMarsh']
    if use_WEAP:
        radial_models = ['nhmv10', 'nwmv21', 'WEAP_23Aug2022_gridmet', 'pywr_nhmv10', 'pywr_nwmv21', 'pywr_WEAP_23Aug2022_gridmet_nhmv10']
    else:
        radial_models = ['nhmv10', 'nwmv21', 'obs_pub', 'pywr_nhmv10', 'pywr_nwmv21', 'pywr_obs_pub']
    radial_models = radial_models[::-1]

    ### compile error metrics across models/nodes/metrics
    if rerun_all:

        print('Plotting radial figures for reservoir releases')

        res_release_metrics = get_error_metrics(res_releases, radial_models, nodes)
        ### nhm vs nwm only, pepacton only - slides 48-54 in 10/24/2022 presentation
        #plot_radial_error_metrics(res_release_metrics, radial_models, nodes, useNonPep = False, useweap = False, usepywr = False)
        ### nhm vs nwm only, all reservoirs - slides 55-58 in 10/24/2022 presentation
        plot_radial_error_metrics(res_release_metrics, radial_models, nodes, useNonPep = True, useweap = False, usepywr = False)
        ### nhm vs nwm vs weap only, pepaction only - slides 69 in 10/24/2022 presentation
        plot_radial_error_metrics(res_release_metrics, radial_models, nodes, useNonPep = False, useweap = True, usepywr = False)
        ### nhm vs nwm vs weap only, all reservoirs - slides 70 in 10/24/2022 presentation
        plot_radial_error_metrics(res_release_metrics, radial_models, nodes, useNonPep = True, useweap = True, usepywr = False)
        ### all models, pepaction only - slides 72-73 in 10/24/2022 presentation
        plot_radial_error_metrics(res_release_metrics, radial_models, nodes, useNonPep = False, useweap = True, usepywr = True)
        ### all models, all reservoirs - slides 74-75 in 10/24/2022 presentation
        plot_radial_error_metrics(res_release_metrics, radial_models, nodes, useNonPep = True, useweap = True, usepywr = True)



    ### now do figs for major flow locations
    if rerun_all:
        print('Plotting radial error metrics for major flows.')
        nodes = ['delMontague', 'delTrenton', 'outletSchuylkill']  # , 'outletChristina', 'delLordville']
        major_flow_metrics = get_error_metrics(major_flows, radial_models, nodes)
        plot_radial_error_metrics(major_flow_metrics, radial_models, nodes, useNonPep = True, useweap = True, usepywr = True, usemajorflows=True)


    ### flow comparisons for major flow nodes
    if rerun_all:
        print('Plotting 3-part flows at major nodes.')
        plot_3part_flows(major_flows, ['nhmv10', 'nwmv21'], 'delMontague')
        plot_3part_flows(major_flows, ['nhmv10', 'nwmv21'], 'delTrenton')
        plot_3part_flows(major_flows, ['nhmv10', 'nwmv21'], 'outletSchuylkill')
        plot_3part_flows(major_flows, ['nhmv10', 'pywr_nhmv10'], 'delMontague')
        plot_3part_flows(major_flows, ['nhmv10', 'pywr_nhmv10'], 'delTrenton')
        plot_3part_flows(major_flows, ['nhmv10', 'pywr_nhmv10'], 'outletSchuylkill')
        plot_3part_flows(major_flows, ['nwmv21', 'pywr_nwmv21'], 'delMontague')
        plot_3part_flows(major_flows, ['nwmv21', 'pywr_nwmv21'], 'delTrenton')
        plot_3part_flows(major_flows, ['nwmv21', 'pywr_nwmv21'], 'outletSchuylkill')
        if use_WEAP:
            plot_3part_flows(major_flows, ['WEAP_23Aug2022_gridmet', 'pywr_WEAP_23Aug2022_gridmet_nhmv10'], 'delMontague')
            plot_3part_flows(major_flows, ['WEAP_23Aug2022_gridmet', 'pywr_WEAP_23Aug2022_gridmet_nhmv10'], 'delTrenton')
            plot_3part_flows(major_flows, ['WEAP_23Aug2022_gridmet', 'pywr_WEAP_23Aug2022_gridmet_nhmv10'], 'outletSchuylkill')
        plot_3part_flows(major_flows, ['obs_pub', 'pywr_obs_pub'], 'delMontague')
        plot_3part_flows(major_flows, ['obs_pub', 'pywr_obs_pub'], 'delTrenton')
        plot_3part_flows(major_flows, ['obs_pub', 'pywr_obs_pub'], 'outletSchuylkill')
        plot_3part_flows(major_flows, ['pywr_obs_pub', 'pywr_nhmv10'], 'delTrenton')
        plot_3part_flows(major_flows, ['pywr_obs_pub', 'pywr_nhmv10'], 'delMontague')
        plot_3part_flows(major_flows, ['nhmv10', 'pywr_obs_pub'], 'delMontague')

        ### weekly flow comparison for major flow nodes
        print('Plotting weekly flow distributions at major nodes.')
        plot_weekly_flow_distributions(major_flows, ['nhmv10', 'nwmv21'], 'delMontague')
        plot_weekly_flow_distributions(major_flows, ['nhmv10', 'nwmv21'], 'delTrenton')
        plot_weekly_flow_distributions(major_flows, ['nhmv10', 'nwmv21'], 'outletSchuylkill')
        plot_weekly_flow_distributions(major_flows, ['nhmv10', 'pywr_nhmv10'], 'delMontague')
        plot_weekly_flow_distributions(major_flows, ['nhmv10', 'pywr_nhmv10'], 'delTrenton')
        plot_weekly_flow_distributions(major_flows, ['nhmv10', 'pywr_nhmv10'], 'outletSchuylkill')
        plot_weekly_flow_distributions(major_flows, ['nwmv21', 'pywr_nwmv21'], 'delMontague')
        plot_weekly_flow_distributions(major_flows, ['nwmv21', 'pywr_nwmv21'], 'delTrenton')
        plot_weekly_flow_distributions(major_flows, ['nwmv21', 'pywr_nwmv21'], 'outletSchuylkill')
        if use_WEAP:
            plot_weekly_flow_distributions(major_flows, ['WEAP_23Aug2022_gridmet', 'pywr_WEAP_23Aug2022_gridmet_nhmv10'], 'delMontague')
            plot_weekly_flow_distributions(major_flows, ['WEAP_23Aug2022_gridmet', 'pywr_WEAP_23Aug2022_gridmet_nhmv10'], 'delTrenton')
            plot_weekly_flow_distributions(major_flows, ['WEAP_23Aug2022_gridmet', 'pywr_WEAP_23Aug2022_gridmet_nhmv10'], 'outletSchuylkill')
        plot_weekly_flow_distributions(major_flows, ['obs_pub', 'pywr_obs_pub'], 'delMontague')
        plot_weekly_flow_distributions(major_flows, ['obs_pub', 'pywr_obs_pub'], 'delTrenton')
        plot_weekly_flow_distributions(major_flows, ['obs_pub', 'pywr_obs_pub'], 'outletSchuylkill')
        plot_weekly_flow_distributions(major_flows, ['pywr_obs_pub','pywr_nhmv10'], 'delMontague')
        plot_weekly_flow_distributions(major_flows, ['pywr_obs_pub','pywr_nhmv10'], 'delTrenton')

    ## RRV metrics
    if rerun_all:
        print('Plotting RRV metrics.')
        if use_WEAP:
            rrv_models = ['obs', 'obs_pub', 'nhmv10', 'nwmv21', 'WEAP_23Aug2022_gridmet', 'pywr_obs_pub', 'pywr_nhmv10', 'pywr_nwmv21', 'pywr_WEAP_23Aug2022_gridmet_nhmv10']
        else:
            rrv_models = ['obs', 'obs_pub', 'nhmv10', 'nwmv21', 'pywr_obs_pub', 'pywr_nhmv10', 'pywr_nwmv21']

        nodes = ['delMontague','delTrenton']
        rrv_metrics = get_RRV_metrics(major_flows, rrv_models, nodes)
        plot_rrv_metrics(rrv_metrics, rrv_models, nodes)

    ## Plot flow contributions at Trenton
    if rerun_all:
        print('Plotting flow contributions at major nodes.')
        
        node = 'delTrenton'
        models = ['pywr_obs_pub', 'pywr_nhmv10', 'pywr_nwmv21']
        for model in models:  
            plot_flow_contributions(res_releases, major_flows, model, node,
                                    separate_pub_contributions = False,
                                    percentage_flow = True,
                                    plot_target = False)
            plot_flow_contributions(res_releases, major_flows, model, node,
                                    separate_pub_contributions = False,
                                    percentage_flow = False,
                                    plot_target = True)


    ## Plot inflow comparison
    if rerun_all:
        inflows = {}
        inflow_comparison_models = ['obs_pub', 'nhmv10', 'nwmv21']
        for model in inflow_comparison_models:
            inflows[model] = get_pywr_results(output_dir, model, results_set='inflow')
        compare_inflow_data(inflows, nodes = reservoir_list)


    ###

        
    print(f'Done! Check the {fig_dir} folder.')