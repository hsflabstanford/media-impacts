# media-impacts
Code and data to reproduce the paper "Documentary films can increase nationwide interest in plant-based food".

To reproduce figures from saved results: 
run the notebooks in the **visualization** directory. Specifically:
- create-media-plots.ipynb reproduces Figure 2.
- visualize-time-series.ipynb reproduces Figures 3 and 7.
- visualize-benchmarking-time-series.ipynb reproduces Figure 12.
- create-all-docs-table.ipynb reproduces Tables 20 and 21.
- create-ci-plot-individual-docs-assoc-contemp-lagged.ipynb reproduces the forest plots for association, contemporaneous, and lagged analyses.
- create-ci-plot-individual-docs-bin.ipynb reproduces the forest plots for interrupted time series analyses.
- create-ci-plot-individual-docs-ksu-lags.ipynb reproduces the forest plots for the KSU lagged analyses 
- create-tables.ipynb creates the tables for association, contemporaneous and lagged analyses, other than Tables 20 and 21.

To reproduce figures from pre-processed data: 
- run the notebooks in the **analyses** directory, specifically ardl_and_its.ipynb and descriptive-ipynb.
- run the notebooks in the **visualization** directory as described above. You will need to change the names of the results directories.

To reproduce figures from raw data: 
- run the notebooks in the **data-preparation** directory.
- run the notebooks in the **analyses** directory, specifically ardl_and_its.ipynb and descriptive-ipynb.
- run the notebooks in the **visualization** directory as described above. You will need to change the names of the results directories.

Please contact Anna Thomas (thomasat@stanford.edu) with questions. 

