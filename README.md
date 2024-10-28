# media-impacts
Code and data to reproduce the paper "Documentary films can increase nationwide interest in plant-based food".

Dependencies: matplotlib (3.1.1), statsmodels (0.13.5), scipy (1.7.3), numpy (1.21.5), pandas (1.3.5), seaborn (0.12.2).
Tested on Mac OS Sonoma 14.6.1

Installation: git clone https://github.com/hsflabstanford/media-impacts.git
Install time: < 1 minute. 

To reproduce figures from saved results (< 1 minute): 
run the notebooks in the **visualization** directory. Specifically:
- create-media-plots.ipynb reproduces Figure 2.
- visualize-time-series.ipynb reproduces Figures 3 and 7.
- visualize-benchmarking-time-series.ipynb reproduces Figure 12.
- create-all-docs-table.ipynb reproduces Tables 20 and 21.
- create-ci-plot-individual-docs-assoc-contemp-lagged.ipynb reproduces the forest plots for association, contemporaneous, and lagged analyses.
- create-ci-plot-individual-docs-bin.ipynb reproduces the forest plots for interrupted time series analyses.
- create-ci-plot-individual-docs-ksu-lags.ipynb reproduces the forest plots for the KSU lagged analyses 
- create-tables.ipynb creates the tables for association, contemporaneous and lagged analyses, other than Tables 20 and 21.

To reproduce figures from pre-processed data (< 5 minutes): 
- run the notebooks in the **analyses** directory, specifically ardl-and-its.ipynb and descriptive-ipynb.
- run the notebooks in the **visualization** directory as described above. You will need to change the names of the results directories.

To reproduce figures from raw data (< 10 minutes): 
- run the notebooks in the **data-preparation** directory.
- run the notebooks in the **analyses** directory, specifically ardl-and-its.ipynb and descriptive-ipynb.
- run the notebooks in the **visualization** directory as described above. You will need to change the names of the results directories.

Please contact Anna Thomas (thomasat@stanford.edu) with questions. 

