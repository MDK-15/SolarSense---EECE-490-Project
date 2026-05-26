# SolarSense - EECE 490 Project
### This readme only provides a brief overview of the different files and methods we used throughout the project. Details can be found inside the notebooks.
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

The file `tuning_test.ipynb` runs a simulation to test the performance of the tuning. It generates a 2 weeks electricity usage schedule, fine tunes the original models on the first week only, and compares the performance on the second week of the original XGBoost model and the fine tuned one.

## Raspberry Pi and usage disaggregation
The file `logger.py` runs on a respberry pi 3 and collects data from solar inverters. It supports the three most popular brands in Lebanon (Voltronic, Growatt, Deye) as well as any inverter based on the architectures of those brands. For Voltronic inverters, the program uses https://github.com/jblance/mpp-solar#, a library used to communicate with those inverters. For Growatt inverters, the program uses https://github.com/johanmeijer/grott. Similarly, the progrram uses https://github.com/UnknownHero99/pydeye for Deye based inverters.

Voltronic inverters can connect the the Pi via USB, while Growatt and Deye inverters need to connect via RS485 adapters (you can use an RS485 to USB adapter). The device polls the inverter at a rate of 1Hz, and writes directl yto a USB storage device that needs to be plugged into it.

The raspberry Pi was only tested on a Voltronic inverter as this is the only one we had access to (Raggie RG-MH3500W Hybrid Solar Inverter, uses voltronic architecture). The device should also work on Growatt and Deye based inverters, but we were not able to test it.

Since the raspberry Pi can only provide aggregate electricity usage, we needed a method to disaggregate the usage into different eletrical appliances. For this we developed an NILM script that uses template matching to detect different appliances. When the user first installs the system, they need to guide it by telling it when they turned on each appliance and when they turned it back off. This only needs to be done once per appliance. With these timestamps, the system can look at the aggregate power provided by the raspberry Pi and identify the pattern of the appliance, which it can memorize and use to detect the appliance later on. This all can be seen in the `NILM.ipynb` notebook. The template is based on peak_power, steady_state_power, settle_time, and variance.

## Schedule OPtimization
After the user enters their appliances and provides one week's data for finr tuning, we need to optimize their electricity usage patterns and give them a better schedule that take sadvantage of the power their panels are able to generate without running out of electricity. For this task, we a Linear Programmer. Linear programming is a popular method for energy optimization, and a lot of academic research has been made about it.

Our LP works around the following constraints:
* Acceptable Hours: User can specify what hours are acceptable to use certain appliance (e.g. the LP cannot recommend turning on the washing machine at 3am)
* Min SoC protection: Battery level should never fall below a specified percentage chosen by the user
* Load Reduction: LP can recommend using certain appliances less if it is necessary to stay above the Min SoC.

The following goals were set to the LP:
* Waste (100): the battery should not stay at 100% while solar panels are still able to produce energy
* Comfort (50): the LP should try, whenever possible, to keep the keep the number of usage hours per appliance instact. It should only cut hours when necessary.
* Deviation (10): The LP has a weak preference for the original schedule provided by the user.
