# SolarSense - EECE 490 Project
### Many of the notebooks in this repository contain PATH variables in the first cell. The user needs to change them to their desired paths.
### Any .py has only been added to be imported into notebooks. They contain the same code as their .ipynb equivalent.
### Only the unsupervised learning notebooks were run on google colab. Everything else in this respository was run locally on Jupyter.

## Data Preparation
Our datasets included:
* Plegma: a greek dataset showing the power usage of different appliances accross 13 households over the span of one year (exact timeline differs from house to house). This dataset was used for the cooling, hot water, and washing machine models.
* REFIT: a british daatset showing the power usage of different appliances accross 20 households over the span of one year. This dataset was used for the heating model.
* ATUS activity: data published by the US bureau of labor statistics describing the time spent on different activities by many individuals. Activities include watching TV, which we used to train our television model.
The files $prepare_plegma.ipynb$ $prepare_refit.ipynb$ are used to filter unwanted appliances and turn the data from continuous power measurements to binary On/Off using predefined thresholds. The user needs to extract $Plegma_Dataset.zip$ and $REFIT_Dataset.zip$ before running them. $prepare_tv.ipynb$ is used to parse the TV usage times from the $atusact.csv$ file.
