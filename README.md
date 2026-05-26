# SolarSense - EECE 490 Project
### Many of the notebooks in this repository contain PATH variables in the first cell. The user needs to change them to their desired paths.
### Any .py has only been added to be imported into notebooks. They contain the same code as their .ipynb equivalent.
### Only the unsupervised learning notebooks were run on google colab. Everything else in this respository was run locally on Jupyter.

## Data Preparation
Our datasets included:
* Plegma: a greek dataset showing the power usage of different appliances accross 13 households over the span of one year (exact timeline differs from house to house). This dataset was used for the cooling, hot water, and washing machine models.
* REFIT: a british daatset showing the power usage of different appliances accross 20 households over the span of one year. This dataset was used for the heating model.
* ATUS activity: data published by the US bureau of labor statistics describing the time spent on different activities by many individuals. Activities include watching TV, which we used to train our television model.


The files `prepare_plegma.ipynb` `prepare_refit.ipynb` are used to filter unwanted appliances and turn the data from continuous power measurements to binary On/Off using predefined thresholds. The user needs to download the corresponding datasets and place them in `Plegma_Dataset` and `REFIT_Dataset` folders before running them. `prepare_tv.ipynb` is used to parse the TV usage times from the `atusact.csv` file which the user needs to download as well.

## Training Classifiers
The files `XGBoost_plegma.ipynb` `XGBoost_Refit.ipynb` `XGBoost_tv.ipynb` train XGBoost models on theri respective datasets. All of them use GridSearchCV with 5 fold cross validation to get optimal models, along with an 80/20 train/test split. We also used `scale_pos_weight` since many of the datasets are imbalanced, with many appliances being off most of the time, resulting in the majority of samples being Off, and only a few being On. Make sure to run the data preparation notebooks before running the training notebooks.

## Tuning the Classifiers
Every household uses their appliances in a different way. No matter the size of dataset we use for training, a model can only learn th genral patterns of usage of certain appliances (e.g. ACs turn on when the weather is warmer, TVs turn on mostly during the evening, etc). Some appliances such as washing machines do not have such patterns. One way to solve this issue is to create personalized models for each household, which can be trained on data provided by the household itself and learn their specific usage patterns.

The files `tuning.ipynb` takes a base model a fine tunes on using new data. It requires a minimum of 168 hours (rows) of data. The tunig is done by adding 150 decision trees based on the new data. The result is a model that lears the patterns of the household while keeping the patterns it learned from its original training.
