"""
Trevor Amestoy

Exploring ways to interact with simulated streamflows. 

Stacked streamgraph here. 
"""

import pandas as pd
import numpy as np
import altair as alt
from drb_make_figs import get_pywr_results, get_base_results, reservoir_list, majorflow_list

lags_to_delTrenton = {}


node_labels = {'cannonsville': 'Cannonsville',
               'pepacton': 'Pepacton',
               'neversink': 'Neversink',
               'wallenpaupack': 'Wallenpaupack',
               'prompton': 'Prompton',
               'shoholaMarsh': 'Shohola Marsh',
               'mongaupeCombined': 'Mongaupe System', 
               'beltzvilleCombined': 'Beltzville System',
               'fewalter': 'FE Walter',
               'merrillCreek': 'Merrill Creek',
               'hopatcong': 'Hoptacong',
               'nockamixon': 'Nockamixon',
               'assunpink': 'Assunpink',
               'ontalaunee': 'Ontalaunee', 
               'stillCreek': 'Still Creek',
               'blueMarsh': 'Blue Marsh',
               'greenLane': 'Green Lane',
               'marshCreek': 'Marsh Creek', 
               'delLordville': 'Unmanaged Flow Upstream of Lordville',
               'delMontague': 'Unmanaged Flow Upstream of Montague',
               'delTrenton': 'Unmanaged Flow Upstream of Trenton',
               }

unmanaged = ['delLordville', 'delMontague']


site_matches_link = [['delLordville', ['01427207'], ['cannonsville', 'pepacton', '01425000', '01417000']],
                    # '01432055': '01432055',  ### lackawaxen river gage downstream of wallenpaupack & prompton
                    ['delMontague', ['01438500'], ['cannonsville', 'pepacton', '01425000', '01417000', 'delLordville',
                                                    'prompton', 'wallenpaupack', 'shoholaMarsh', 'mongaupeCombined', 'neversink', '01436000', '01433500']],
                    ['delTrenton', ['01463500'], ['cannonsville', 'pepacton', '01425000', '01417000', 'delLordville',
                                                    'prompton', 'wallenpaupack', 'shoholaMarsh', 'mongaupeCombined', 'neversink', '01436000', '01433500', 'delMontague',
                                                'beltzvilleCombined', 'fewalter', 'merrillCreek', 'hopatcong', 'nockamixon', '01449800', '01447800']],
                    ['outletAssunpink', ['01463620'], ['assunpink', '01463620']], ## note, should get downstream junction, just using reservoir-adjacent gage for now
                    ['outletSchuylkill', ['01474500'], ['ontelaunee', 'stillCreek', 'blueMarsh', 'greenLane', '01470960']],
                    ['outletChristina', ['01480685'], ['marshCreek']]
                    ]

contributing_nodes = site_matches_link[2][2]




def plot_interactive_streamflow_stack(node, model, 
                                      group_flow = True,
                                    output_dir = 'output_data/', 
                                    fig_dir = 'figs/' , 
                                    plot_target = False):
    """
    Generates an HTML based interactive stacked timeseries plot of the contributing streamflows at a node of interest.
    
    WARNING: Currently only known to be accurate for Trenton.
    """
    
    cms_to_mgd = 22.82
    cm_to_mg = 264.17/1e6
    cfs_to_mgd = 0.0283 * 22824465.32 / 1e6

    flow_data = get_pywr_results(output_dir, model, results_set='major_flow')
    release_data = get_pywr_results(output_dir, model, results_set='res_release')
    inflow_data = get_pywr_results(output_dir, model, results_set='inflow')
    base_flow = get_base_results('input_data/','obs', release_data.index, results_set='major_flow')

    ### Find contributions
    contributing = []
    if node == 'delLordville':
        contributing = site_matches_link[0][2].copy()
        title_text = 'Contributing flows at Lordville'
    elif node == 'delMontague':
        contributing = site_matches_link[1][2].copy()
        title_text = 'Contributing flows at Montague'
        target = 1750*cfs_to_mgd
    elif node == 'delTrenton':
        contributing = site_matches_link[2][2].copy()
        title_text = 'Contributing flows at Trenton'
        target = 3000*cfs_to_mgd
    else:
        print('Invalid node specification.')

    # Group flow contributions by type        
    nyc_reservoirs = ['cannonsville', 'pepacton', 'neversink']
    non_nyc_reservoirs = [i for i in contributing if (i not in nyc_reservoirs) and (i in reservoir_list)]
    unmanaged_flows = [i for i in contributing if (i not in reservoir_list)]

    flow_groups = {'NYC Reservoir Releases': nyc_reservoirs,
                        'Other Reservoir Releases': non_nyc_reservoirs,
                        'Unmanaged Flows Upstream': unmanaged_flows}

    # Pull just contributing data
    use_releases = [i for i in contributing if i in release_data.columns]
    use_releases = nyc_reservoirs + non_nyc_reservoirs
    use_inflows = [i for i in contributing if (i not in use_releases) and (i in contributing)]

    release_contributions = release_data[use_releases]
    inflow_contributions = inflow_data[use_inflows]
    node_inflow = inflow_data[node]
    
    contributions = pd.concat([release_contributions, inflow_contributions, node_inflow], axis=1)
    
    # Get total simulated and obs flow at the node
    total_simulated = flow_data[node]
    
    total_observed = base_flow[node]
    
    total_flows = pd.concat([total_observed, total_simulated], axis=1)
    total_flows.columns = ['Observed Flow (MGD)', 'Simulated Flow (MGD)']


    total_flows_stacked = total_flows.stack().reset_index()
    total_flows_stacked.columns = ['Date', 'Data Source', 'Flow']

    # Account for lag
    lag_1 = ['wallenpaupack', 'prompton', 'shoholaMarsh', 'mongaupeCombined','neversink', 'delLordville']
    lag_2 = ['cannonsville', 'pepacton']
    direct = [i for i in contributions.columns if (i not in lag_1) and (i not in lag_2)]
    
    #shifted = pd.concat([contributions[direct].iloc[2:,:], contributions[lag_1].iloc[1:-1,:], contributions[lag_2].iloc[0:-2,:]], ignore_index=True, axis=1)
    #contributions = shifted.dropna(axis=1)
    
    if group_flow:
        for key in flow_groups.keys():
            contributions[key] = contributions.loc[:,flow_groups[key]].sum(axis=1)
        contributions.loc[:, 'Unmanaged Flows Upstream'] = contributions.loc[:, 'Unmanaged Flows Upstream'] + contributions.loc[:, node]  
        contributions = contributions.loc[:, list(flow_groups.keys())]        
        
    else:
        labels = [node_labels[i] for i in contributions.columns]        
        contributions.columns = labels
        
    percent_contributions = contributions.divide(total_observed, axis=0)*100
    sim_percentages = percent_contributions.stack().reset_index()
    sim_percentages.columns = ['Date', 'Source', 'Flow']
    
    
    ## PLOTTING via altair
    filenameadd = 'percent'
    upper_plot_height = 100
    lower_plot_height = 300
    plot_width = 600
    
    # Setup interactive selection via legend and zooom functionality
    selection = alt.selection_multi(fields=['Source'], bind='legend')
    zoom = alt.selection_interval(encodings=['x'])

    # Generate plot
    total_flow_plot = alt.Chart(total_flows_stacked, width = plot_width, height = upper_plot_height).mark_line(strokeWidth = 0.5).encode(
        x = alt.X('Date:T', axis=alt.Axis(labels=False)),
        y = alt.Y('Flow:Q', title = 'Flow (MGD)', scale = alt.Scale(type = 'log', domain = [1000, 100000])),
        color = alt.Color('Data Source:N', scale = alt.Scale(
            domain = ['Observed Flow (MGD)', 'Simulated Flow (MGD)'],
            range = ['teal', 'maroon']), legend = alt.Legend(values = total_flows.columns.to_list()))
    ).add_selection(zoom)
    
    target_line = alt.Chart(pd.DataFrame({'Flow Target': [target]})).mark_rule().encode(y='Flow Target')
    
    contribution_plot = alt.Chart(sim_percentages, width = plot_width, height = lower_plot_height).mark_area().encode(
        alt.X('Date:T', scale = alt.Scale(domain = zoom)),
        alt.Y('sum(Flow):Q', title = 'Percentage of Total Observed Flow', scale = alt.Scale(domain=[0, 120])),
        alt.Color('Source:N', scale=alt.Scale(scheme='dark2'),
                  legend = alt.Legend(values = percent_contributions.columns.to_list())),
        opacity = alt.condition(selection, alt.value(1), alt.value(0.2))
    ).add_selection(selection)
    
    ideal_line = alt.Chart(pd.DataFrame({'Ideal': [100]})).mark_rule().encode(y='Ideal')

    plot = alt.vconcat((total_flow_plot + target_line), (contribution_plot + ideal_line)).configure_axis(
        labelFontSize= 12,
        titleFontSize= 14).resolve_legend(color = 'independent')

    plot.save(f'{fig_dir}{model}_interactive_streamflow_stack.html', scale_factor=3.0)
    
    return_contributions = False
    if return_contributions:
        return release_contributions, inflow_contributions
    else:
        return
    
"""
, legend = alt.Legend(values = percent_contributions.columns.to_list())
"""