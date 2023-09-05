import numpy as np
import pandas as pd
import sys

sys.path.append('./')
sys.path.append('../')

from pywrdrb.plotting.plotting_functions import *
from pywrdrb.plotting.styles import *
from pywrdrb.utils.lists import reservoir_list, reservoir_list_nyc, majorflow_list, majorflow_list_figs, reservoir_link_pairs
from pywrdrb.utils.constants import cms_to_mgd, cm_to_mg, cfs_to_mgd
from pywrdrb.utils.directories import input_dir, output_dir, fig_dir
from pywrdrb.post.get_results import get_base_results, get_pywr_results

### I was having trouble with interactive console plotting in Pycharm for some reason - comment this out if you want to use that and not having issues
# mpl.use('TkAgg')




## Execution - Generate all figures
if __name__ == "__main__":

    rerun_all = False

    ## Load data    
    # Load Pywr-DRB simulation models
    print(f'Retrieving simulation data.')
    pywr_models = ['nhmv10', 'nwmv21', 'nhmv10_withObsScaled', 'nwmv21_withObsScaled']

    reservoir_downstream_gages = {}
    major_flows = {}
    storages = {}
    reservoir_releases = {}
    ffmp_levels = {}
    ffmp_level_boundaries = {}
    inflows = {}
    nyc_release_components = {}
    diversions = {}
    consumptions = {}

    datetime_index = None
    for model in pywr_models:
        print(f'pywr_{model}')
        reservoir_downstream_gages[f'pywr_{model}'], datetime_index = get_pywr_results(output_dir, model, results_set='reservoir_downstream_gage', datetime_index=datetime_index)
        reservoir_downstream_gages[f'pywr_{model}']['NYCAgg'] = reservoir_downstream_gages[f'pywr_{model}'][reservoir_list_nyc].sum(axis=1)
        major_flows[f'pywr_{model}'], datetime_index = get_pywr_results(output_dir, model, results_set='major_flow', datetime_index=datetime_index)
        storages[f'pywr_{model}'], datetime_index = get_pywr_results(output_dir, model, results_set='res_storage', datetime_index=datetime_index)
        reservoir_releases[f'pywr_{model}'], datetime_index = get_pywr_results(output_dir, model, results_set='res_release', datetime_index=datetime_index)
        ffmp_levels[f'pywr_{model}'], datetime_index = get_pywr_results(output_dir, model, results_set='res_level', datetime_index=datetime_index)
        inflows[f'pywr_{model}'], datetime_index = get_pywr_results(output_dir, model, 'inflow', datetime_index=datetime_index)
        nyc_release_components[f'pywr_{model}'], datetime_index = get_pywr_results(output_dir, model, 'nyc_release_components', datetime_index=datetime_index)
        diversions[f'pywr_{model}'], datetime_index = get_pywr_results(output_dir, model, 'diversions', datetime_index=datetime_index)
        consumptions[f'pywr_{model}'], datetime_index = get_pywr_results(output_dir, model, 'consumption', datetime_index=datetime_index)

    ffmp_level_boundaries, datetime_index = get_pywr_results(output_dir, model, results_set='ffmp_level_boundaries', datetime_index=datetime_index)

    pywr_models = [f'pywr_{m}' for m in pywr_models]

    ### Load base (non-pywr) models
    base_models = ['obs', 'nhmv10', 'nwmv21', 'nhmv10_withObsScaled', 'nwmv21_withObsScaled']

    datetime_index = list(reservoir_downstream_gages.values())[0].index
    for model in base_models:
        print(model)
        reservoir_downstream_gages[model], datetime_index = get_base_results(input_dir, model, results_set='reservoir_downstream_gage', datetime_index=datetime_index)
        reservoir_downstream_gages[model]['NYCAgg'] = reservoir_downstream_gages[model][reservoir_list_nyc].sum(axis=1)
        major_flows[model], datetime_index = get_base_results(input_dir, model, results_set='major_flow', datetime_index=datetime_index)

    ### different time periods for different type of figures.
    start_date_obs = pd.to_datetime('2008-01-01')   ### comparison to observed record, error metrics
    end_date_obs = pd.to_datetime('2017-01-01')
    start_date_full = pd.to_datetime('1984-01-01')  ### full NHM/NWM time series
    end_date_full = pd.to_datetime('2017-01-01')
    start_date_short = pd.to_datetime('2011-01-01') ### short 2-year period for zoomed dynamics
    end_date_short = pd.to_datetime('2013-01-01')


    ### first set of subplots showing comparison of modeled & observed flows at 3 locations, top to bottom of basin
    if rerun_all:
        print('Plotting 3x3 flow comparison figure')
        nodes = ['cannonsville','NYCAgg','delTrenton']
        for model in pywr_models:
            ### first with full observational record
            plot_3part_flows_hier(reservoir_downstream_gages, major_flows, [model.replace('pywr_',''), model], uselog=True,
                                  colordict=model_colors_diagnostics_paper2, start_date=start_date_obs, end_date=end_date_obs)
            ### now zoomed in 2 year period where it is easier to see time series
            plot_3part_flows_hier(reservoir_downstream_gages, major_flows, [model.replace('pywr_', ''), model], uselog=True,
                                  colordict=model_colors_diagnostics_paper2, start_date=start_date_short, end_date=end_date_short)



    ### gridded error metrics for NYC reservoirs + major flows
    if rerun_all:
        print('Plotting gridded error metrics')
        ### first do smaller subset of models & locations for main text
        error_models = base_models[1:-1] + pywr_models[:-1]
        nodes = ['cannonsville', 'NYCAgg', 'delTrenton']
        node_metrics = get_error_metrics(reservoir_downstream_gages, major_flows, error_models, nodes,
                                         start_date=start_date_obs, end_date=end_date_obs)
        plot_gridded_error_metrics(node_metrics, error_models, nodes, start_date_obs, end_date_obs, figstage=0)
        ### now full set of models & locations, for SI
        error_models = base_models[1:] + pywr_models
        nodes = reservoir_list_nyc + ['NYCAgg','delMontague','delTrenton']
        node_metrics = get_error_metrics(reservoir_downstream_gages, major_flows, error_models, nodes,
                                         start_date=start_date_obs, end_date=end_date_obs)
        plot_gridded_error_metrics(node_metrics, error_models, nodes, start_date_obs, end_date_obs, figstage=1)
        ### repeat full figure but with monthly time step
        plot_gridded_error_metrics(node_metrics, error_models, nodes, start_date_obs, end_date_obs, figstage=2)




    ## RRV metrics
    if rerun_all:
        print('Plotting RRV metrics.')
        rrv_models = base_models + pywr_models

        nodes = ['delMontague','delTrenton']
        rrv_metrics = get_RRV_metrics(major_flows, rrv_models, nodes, start_date=start_date_obs, end_date=end_date_obs)
        plot_rrv_metrics(rrv_metrics, rrv_models, nodes, colordict=model_colors_diagnostics_paper)



    ## Plot NYC storage dynamics
    if rerun_all:
        print('Plotting NYC reservoir storages & releases')
        for reservoir in ['agg']+reservoir_list_nyc:
            plot_combined_nyc_storage(storages, reservoir_releases, ffmp_levels, pywr_models,
                                      start_date=start_date_obs, end_date=end_date_obs, reservoir=reservoir, fig_dir=fig_dir,
                                      colordict=model_colors_diagnostics_paper,
                                      add_ffmp_levels=True, plot_observed=True, plot_sim=True)


    ### xQn grid low flow comparison figure
    if rerun_all:
        print('Plotting low flow grid.')
        plot_xQn_grid(reservoir_downstream_gages, major_flows,  base_models + pywr_models,
                      reservoir_list_nyc + majorflow_list_figs, xlist = [1,7,30,90, 365], nlist = [5, 10, 20, 30],
                      start_date=start_date_obs, end_date=end_date_obs, fig_dir=fig_dir)

    ### plot comparing flow series with overlapping boxplots & FDCs
    if rerun_all:
        print('Plotting monthly boxplot/FDC figures')
        for node in reservoir_list_nyc + majorflow_list_figs:
            plot_monthly_boxplot_fdc_combined(reservoir_downstream_gages, major_flows, base_models, pywr_models, node,
                                              colordict=model_colors_diagnostics_paper, start_date=start_date_obs,
                                              end_date=end_date_obs, fig_dir=fig_dir)


    ### plot breaking down NYC flows & Trenton flows into components
    # if rerun_all:
    print('Plotting NYC releases by components, combined with downstream flow components')
    for model in pywr_models:
        for node in ['delMontague','delTrenton']:
            plot_NYC_release_components_combined(nyc_release_components, reservoir_releases, major_flows, inflows,
                                                 diversions, consumptions, model, node, use_proportional=True, use_log=True,
                                                 start_date=start_date_obs, end_date=end_date_obs, fig_dir=fig_dir)

            plot_NYC_release_components_combined(nyc_release_components, reservoir_releases, major_flows, inflows,
                                                 diversions, consumptions, model, node, use_proportional=True, use_log=True,
                                                 start_date=start_date_short, end_date=end_date_short, fig_dir=fig_dir)


    if rerun_all:
        print('Plotting new NYC storage figure')
        plot_combined_nyc_storage_new(storages, ffmp_level_boundaries, pywr_models,
                                      start_date=start_date_obs, end_date=end_date_obs, fig_dir=fig_dir)
        plot_combined_nyc_storage_new(storages, ffmp_level_boundaries, pywr_models,
                                      start_date=start_date_short, end_date=end_date_short, fig_dir=fig_dir)

    print(f'Done! Check the {fig_dir} folder.')