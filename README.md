# SolarSense - EECE 490 Project
### This readme only provides a brief overview of the different files and methods we used throughout the project. Details can be found inside the notebooks, including evaluation results and numbers.
### Many of the notebooks in this repository contain PATH variables in the first cell. The user needs to change them to their desired paths.
### Any .py file has only been added to be imported into notebooks. They contain the same code as their .ipynb equivalent.
### Only the unsupervised learning notebooks were run on google colab. Everything else in this respository was run locally on Jupyter. You may need to install the required libraries. Imports can be found in the first cell of every notebook.

## Problem definition
Since the crisis that started in 2019 in Lebanon, state provided electricity has become almost absent. A lot of people have turned for solar power as an alternative. Many of these people also use solar power as their only electricity source. For these people, if their batteries run out of charge, they end up without electricity until the sun comes up again to recharge.

The main problem is the following: charging peaks and usage peaks don't align. Charging peaks usually around noon, when the sun shines the most. However, consumption peaks around the evening, when people come back from work/school, and they need to turn on lights, TVs, ACs, etc.

The goal of this project is to give people who may be struggling to properly manage the power genrated by their solar panel systems recommended schedules that optimize electricity usage. This schedule should maximize the amount of electricity consumed while keeping the battery SoC from running below a certain percentage specified by the user. This schedule should tell the user when to use each of their appliances on an hourly basis.

The project also includes fault detection models that can detect problems with the solar panels or the batteries.

## Non-ML baseline
The baseline is a simple heuristic: turn on appliances when the sun's irradiance is the strongest (usually around noon), and turn them off at night. While this heuristic can create improvements, it is not sufficient on its own.

The heuristic does not account for weather changes in the future. The weaher in Lebanon is chaotic; it may be sunny now but rainy and cloudy in an hour or two. Simply looking at the current conditions is not enough, we also need to account for future circumstances.

We also need to consider the fact that certain appliances cannot be turned on at any time. One example is the TV which needs to be turned on whenever the user needs it. We cannot tell the user to watch TV only when the sun is up.

## Using the system
The user first needs to plug the raspberry pie into their inverter (see the section about raspberry Pi below for more details). The device needs to monitor for a minimum of one week.

Once the Pi is working, the user can log into the server. They need to enter the specifications of their system (their location, capacity of the solar panels, capacity of the batteries, etc). They then need to manually enter what appliances they have at their house.

Each appliance needs to be calibrated; for this, the user needs to turn off their appliance, then turn it back on and instantly press a button as indicated on the website. They should then for a few minutes and turn the appliance back off, pressing again on the same button on the website (see usage disaggregation below for more details).

When the first week has passed, the user can extract the files out of the raspberry Pi. `load.csv` should be uploaded on the schedule tab to generate a schedule. `solar.csv` and `battery.csv` can be uploaded on the diagnostics tab for fault detection.

### If the user's inverter can provide usage history natively without the raspberry Pi, the uploaded files need to be formatted correctly. Refer to the `demo` folder for examples.

## Demo
For a quick demo of the project, you may login to the system with the username/password `demo/demo1234`. This account has already entered and calibrated appliances. You may use the files in teh `demo` folder to upload on the website.

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

## Fault Detection
Since our raspberry Pi already collects logs about solar charging rates and battery charging and discharging rates, we decided to implement two unsupervised learning models for fault detection on both the solar panels and the batteries.

The solar fault detection model uses data from NREL PVDAQ, a database monitoring solar panels accross many years. We used a Conv1D autoencoder trained on healthy windows, and HDBSCAN with K-Means fallback for error clustering. We identified four types of error: Intermittent, Shade, Inverter Clipping, and Soiling/Degradation.

The battery fault detection model uses NASA+CALCE battery datasets. We used an LSTM autoencoder trained on healthy windows, and HDBSCAN+K-Means fallback for error clustering. The detected fault types are: Overcharge, over-discharge, capacity fade, thermal fault, short circuit, internal resistance rise.

## Cloud hosting and Interface
For hosting we used Google Cloud. The app was developed in Flask, and uses SQLite for database storage. Every user gets their own account, which contains their system specifications, their appliances with their respective calibrations, and their own fine tuned classifiers. The interface has three tabs: a Home page, a schedule generation page, and a disgnostics page (for the error detection models).

We used the open-meteo API for weather forecasts. We also used the PVLib library, which can take weather features and solar system specifications, and calculate the amount of energy generated by the panels.

## Previous Pipelines and Failed Attempts
#### You can find files related to this section in the `Previous Attempts` folder
* Resnet Dataset: the first dataset we used was provided by resnet. This datasets conatained power consumption accross various appliances over the span of one year, for hundreds of homes. However, this data comes from a physical simulation, not real measured data. We abandoned this dataset when we found out, and it is not used in any way in the final project.
* Full reliance on pre-trained XGBoost classifiers: Our first pipeline did not involve the raspberry Pi logging device. Instead, we planned on training classifiers using public datasets ans shipping those models as is, without any fine tuning. However, we found that none of our moedls were able to generalize at all, even accross different houses in the same dataset. This is what motivated us to collect data from individual household, and to create fine tuned models.
* Reinforcement Learning for Schedule Optimization: The initial pipeline included using RL for generating optimized schedules instead of the LP we used in the end. However, RL needs much more computational power to train. Additionally, the RL environment was very complex and hard to tune properly. We eventually moved to LP instead, since it does not require the same compute resources, and is very often used for poewr applications thorughout academic literature.
* NILM models: before using the template matching method we created, we tried using machine learning methods to disaggregate electricity usage to individual appliances. We tried CNNs (seq2point model) and FHMM model. These methods eventually failed to generalize well. Any of these models can be trained on one household (assumign you have labeled data) and can genuinely generalize well to that household, bt cannot generalize to any other household. Using ML for NILM disaggregation is still an active area of research. More about it here: https://github.com/nilmtk/nilmtk

## Limitations
* Weaher Forcasts: Our system relies on the accuracy of weather forecasts to generate reliable schedules. If the forecast isn't accurate, the system might recommend turning on ACs when they are not needed, and may also wrongly predict the amount of energy generated by the solar panels. This can be mitigated by upgrading to more premium services (such as accuweather).
* Solar panel degradation: Our schedule generation system does not account for solar panel degradation. This can be solved by training a model to predict the amount of power generated by the solar panels, making a personalized model for every household similarly to how we make personalized consumption models.

## Potential Additions to the project
* Real time monitoring and notifications: This mainly applies to the fault detection models. The raspberry Pi could send its data in real time to the server, and a mobile app could notify the user whenever a fault is detected. This would require the creation of a mobile app, and to set up networking between the Pi and the server.
