# Pywr-DRB: a Pywr model for Delaware River Basin

Pywr-DRB is an integrated water resource model of the Delaware River Basin (DRB) designed to assist in water resource decision making within the basin. User guides and documentation are available [here](https://pywr-drb.github.io/Pywr-DRB).

A graphical representation of the Pywr-DRB model is shown below:

<div style="padding-bottom:75%; position:relative; display:block; width: 100%">
  <iframe src="/DRB_spatial/drb_model_map.html"
  height = "100%" width = "100%"
  title = "Graphical Representation of Pywr-DRB Model"
  frameborder="0" allowfullscreen="" style="position:absolute; top:0; left: 0">
  </iframe>
</div>

## Installation

Pywr-DRB requires an installation of Python 3+. To clone a copy of the Pywr-DRB model repository to your local machine, run the following command:

```
git clone https://github.com/Pywr-DRB/Pywr-DRB.git
```

This project requires several dependencies, listed in [`requirements.txt`](https://github.com/DRB_water_managment/requirements.txt), including:
- pywr
- glob2
- h5py
- hydroeval

You can install all of the necessary dependencies in a virtual environment:
```Bash
py -m virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Quick-Start

To run the Pywr-DRB simulation for all [available inflow datasets](https://trevorja.github.io/Pywr_DRB_documentation/Supplemental/data_summary.html) you can execute the [`drb_run_all.sh`](https://trevorja.github.io/Pywr_DRB_documentation/API_References/drb_run_all.html) script:

```Bash
sh drb_run_all.sh
```

Running drb_run_all.sh from the command line will do three things:

First, it will run `prep_input_data.py`, which preps input data files on observed streamflows, modeled flows from NHMv1.0, NWMv2.1, and WEAP (Aug 23, 2022, version), and saves data in Pywr-ready formats. 

Second, it loops over these four input data types and runs each through the Pywr-DRB model, using `drb_run_sim.py`. 

Lastly, it analyzes the results and creates the figures all figures used for analysis using `drb_make_figs.py`.

## Executables

- [`prep_input_data.py`](https://trevorja.github.io/Pywr_DRB_documentation/API_References/prep_input_data.md)
> This module prepares input-streamflow data files from different potential sources, and saves data in Pywr-ready formats. Sources include observed streamflows, modeled flows from a reconstructed historic period using [prediction in ungauged basins](../Supplemental/pub.md) NHMv1.0, NWMv2.1, and WEAP (Aug 23, 2022, version). For more information on different inflow datasets, see the [Data Summary page.](../Supplemental/data_summary.md)

- [`drb_run_sim.py`](https://trevorja.github.io/Pywr_DRB_documentation/API_References/drb_run_sim.md)
> This module is used to execute a simulation of the Pywr-DRB model using a specified streamflow input source, generated by `prep_input_data.py`. The model features (nodes and edges) contained within the `DRB_water_management/model_data/` directory, and are passed to `drb_make_model.py` which constructs a JSON file defining the Pywr model. Once the model is constructed, the simulation is run.

- [`drb_make_figs.py`](https://trevorja.github.io/Pywr_DRB_documentation/API_References/api_references.md)
> This module contains several plotting functions. Executing this script after performing a simulation will result in figures being generated and stored in `DRB_water_managment/figs/`.


## Geospatial analysis
The ``DRB_spatial/`` directory contains the Jupyter Notebook ``DRB_spatial.ipynb`` that creates some (rudimentary for now) maps of the system, which can be helpful for visualizing the node network used in the pywr simulation. To recreate the maps, you will need to download additional geospatial data, as outlined in the separate README file in that directory.

