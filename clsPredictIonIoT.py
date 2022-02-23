################################################
#### Written By: SATYAKI DE                 ####
#### Written On: 19-Feb-2022                ####
#### Modified On 21-Feb-2022                ####
####                                        ####
#### Objective: This python script will     ####
#### perform the neural-prophet forecast    ####
#### based on the historical input received ####
#### from IoT device.                       ####
################################################

# We keep the setup code in a different class as shown below.
from clsConfig import clsConfig as cf

import psutil
import os
import pandas as p
import json
import datetime
from neuralprophet import NeuralProphet, set_log_level
from neuralprophet import set_random_seed
from neuralprophet.benchmark import Dataset, NeuralProphetModel, SimpleExperiment, CrossValidationExperiment

import time
import clsL as cl

import matplotlib.pyplot as plt

###############################################
###           Global Section                ###
###############################################
# Initiating Log class
l = cl.clsL()

set_random_seed(10)
set_log_level("ERROR", "INFO")
###############################################
###    End of Global Section                ###
###############################################

class clsPredictIonIoT:
    def __init__(self):
        self.sleepTime = int(cf.conf['sleepTime'])
        self.event1 = cf.conf['event1']
        self.event2 = cf.conf['event2']

    def forecastSeries(self, inputDf):
        try:
            sleepTime = self.sleepTime
            event1 = self.event1
            event2 = self.event2

            df = inputDf

            print('IoTData: ')
            print(df)

            ## user specified events
            # history events
            SummerEnd = p.DataFrame(event1)
            LongWeekend = p.DataFrame(event2)

            dfEvents = p.concat((SummerEnd, LongWeekend))

            # NeuralProphet Object
            # Adding events
            m = NeuralProphet(loss_func="MSE")

            # set the model to expect these events
            m = m.add_events(["SummerEnd", "LongWeekend"])

            # create the data df with events
            historyDf = m.create_df_with_events(df, dfEvents)

            # fit the model
            metrics = m.fit(historyDf, freq="D")

            # forecast with events known ahead
            futureDf = m.make_future_dataframe(df=historyDf, events_df=dfEvents, periods=365, n_historic_predictions=len(df))
            forecastDf = m.predict(df=futureDf)

            events = forecastDf[(forecastDf['event_SummerEnd'].abs() + forecastDf['event_LongWeekend'].abs()) > 0]
            events.tail()

            ## plotting forecasts
            fig = m.plot(forecastDf)

            ## plotting components
            figComp = m.plot_components(forecastDf)

            ## plotting parameters
            figParam = m.plot_parameters()

            #################################
            #### Train & Test Evaluation ####
            #################################
            m = NeuralProphet(seasonality_mode= "multiplicative", learning_rate = 0.1)

            dfTrain, dfTest = m.split_df(df=df, freq="MS", valid_p=0.2)

            metricsTrain = m.fit(df=dfTrain, freq="MS")
            metricsTest = m.test(df=dfTest)

            print('metricsTest:: ')
            print(metricsTest)

            # Predict Into Future
            metricsTrain2 = m.fit(df=df, freq="MS")
            futureDf = m.make_future_dataframe(df, periods=24, n_historic_predictions=48)
            forecastDf = m.predict(futureDf)
            fig = m.plot(forecastDf)

            # Visualize training
            m = NeuralProphet(seasonality_mode="multiplicative", learning_rate=0.1)
            dfTrain, dfTest = m.split_df(df=df, freq="MS", valid_p=0.2)

            metrics = m.fit(df=dfTrain, freq="MS", validation_df=dfTest, plot_live_loss=True)

            print('Tail of Metrics: ')
            print(metrics.tail(1))

            ######################################
            #### Time-series Cross-Validation ####
            ######################################
            METRICS = ['SmoothL1Loss', 'MAE', 'RMSE']
            params = {"seasonality_mode": "multiplicative", "learning_rate": 0.1}

            folds = NeuralProphet(**params).crossvalidation_split_df(df, freq="MS", k=5, fold_pct=0.20, fold_overlap_pct=0.5)

            metricsTrain = p.DataFrame(columns=METRICS)
            metricsTest = p.DataFrame(columns=METRICS)

            for dfTrain, dfTest in folds:
                m = NeuralProphet(**params)
                train = m.fit(df=dfTrain, freq="MS")
                test = m.test(df=dfTest)
                metricsTrain = metricsTrain.append(train[METRICS].iloc[-1])
                metricsTest = metricsTest.append(test[METRICS].iloc[-1])

            print('Stats: ')
            dfStats = metricsTest.describe().loc[["mean", "std", "min", "max"]]
            print(dfStats)

            ####################################
            #### Using Benchmark Framework  ####
            ####################################
            print('Starting extracting result set for Benchmark:')
            ts = Dataset(df = df, name = "thermoStatsCPUUsage", freq = "MS")
            params = {"seasonality_mode": "multiplicative"}
            exp = SimpleExperiment(
                model_class=NeuralProphetModel,
                params=params,
                data=ts,
                metrics=["MASE", "RMSE"],
                test_percentage=25,
            )
            resultTrain, resultTest = exp.run()

            print('Test result for Benchmark:: ')
            print(resultTest)
            print('Finished extracting result test for Benchmark!')

            ####################################
            #### Cross Validate Experiment  ####
            ####################################
            print('Starting extracting result set for Corss-Validation:')
            ts = Dataset(df = df, name = "thermoStatsCPUUsage", freq = "MS")
            params = {"seasonality_mode": "multiplicative"}
            exp_cv = CrossValidationExperiment(
                model_class=NeuralProphetModel,
                params=params,
                data=ts,
                metrics=["MASE", "RMSE"],
                test_percentage=10,
                num_folds=3,
                fold_overlap_pct=0,
              )
            resultTrain, resultTest = exp_cv.run()

            print('resultTest for Cross Validation:: ')
            print(resultTest)
            print('Finished extracting result test for Corss-Validation!')

            ######################################################
            #### 3-Phase Train, Test & Validation Experiment  ####
            ######################################################
            print('Starting 3-phase Train, Test & Validation Experiment!')

            m = NeuralProphet(seasonality_mode= "multiplicative", learning_rate = 0.1)

            # create a test holdout set:
            dfTrainVal, dfTest = m.split_df(df=df, freq="MS", valid_p=0.2)
            # create a validation holdout set:
            dfTrain, dfVal = m.split_df(df=dfTrainVal, freq="MS", valid_p=0.2)

            # fit a model on training data and evaluate on validation set.
            metricsTrain1 = m.fit(df=dfTrain, freq="MS")
            metrics_val = m.test(df=dfVal)

            # refit model on training and validation data and evaluate on test set.
            metricsTrain2 = m.fit(df=dfTrainVal, freq="MS")
            metricsTest = m.test(df=dfTest)

            metricsTrain1["split"]  = "train1"
            metricsTrain2["split"]  = "train2"
            metrics_val["split"] = "validate"
            metricsTest["split"] = "test"
            metrics_stat = metricsTrain1.tail(1).append([metricsTrain2.tail(1), metrics_val, metricsTest]).drop(columns=['RegLoss'])

            print('Metrics Stat:: ')
            print(metrics_stat)

            # Train, Cross-Validate and Cross-Test evaluation
            METRICS = ['SmoothL1Loss', 'MAE', 'RMSE']
            params = {"seasonality_mode": "multiplicative", "learning_rate": 0.1}

            crossVal, crossTest = NeuralProphet(**params).double_crossvalidation_split_df(df, freq="MS", k=5, valid_pct=0.10, test_pct=0.10)

            metricsTrain1 = p.DataFrame(columns=METRICS)
            metrics_val = p.DataFrame(columns=METRICS)
            for dfTrain1, dfVal in crossVal:
                m = NeuralProphet(**params)
                train1 = m.fit(df=dfTrain, freq="MS")
                val = m.test(df=dfVal)
                metricsTrain1 = metricsTrain1.append(train1[METRICS].iloc[-1])
                metrics_val = metrics_val.append(val[METRICS].iloc[-1])

            metricsTrain2 = p.DataFrame(columns=METRICS)
            metricsTest = p.DataFrame(columns=METRICS)
            for dfTrain2, dfTest in crossTest:
                m = NeuralProphet(**params)
                train2 = m.fit(df=dfTrain2, freq="MS")
                test = m.test(df=dfTest)
                metricsTrain2 = metricsTrain2.append(train2[METRICS].iloc[-1])
                metricsTest = metricsTest.append(test[METRICS].iloc[-1])

            mtrain2 = metricsTrain2.describe().loc[["mean", "std"]]
            print('Train 2 Stats:: ')
            print(mtrain2)

            mval = metrics_val.describe().loc[["mean", "std"]]
            print('Validation Stats:: ')
            print(mval)

            mtest = metricsTest.describe().loc[["mean", "std"]]
            print('Test Stats:: ')
            print(mtest)

            return 0
        except Exception as e:
            x = str(e)
            print('Error: ', x)

            return 1
