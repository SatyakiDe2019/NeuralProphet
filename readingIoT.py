###############################################
#### Written By: SATYAKI DE                ####
#### Written On: 21-Feb-2022               ####
#### Modified On 21-Feb-2022               ####
####                                       ####
#### Objective: This python script will    ####
#### invoke the main class to use the      ####
#### stored historical IoT data stored &   ####
#### then transform, cleanse, predict &    ####
#### analyze the data points into more     ####
#### meaningful decision-making insights.  ####
###############################################

# We keep the setup code in a different class as shown below.
from clsConfig import clsConfig as cf

import datetime
import logging
import pandas as p

import clsPredictIonIoT as cpt
###############################################
###           Global Section                ###
###############################################

sep = str(cf.conf['SEP'])
Curr_Path = str(cf.conf['INIT_PATH'])
fileName = str(cf.conf['FILE_NAME'])

###############################################
###    End of Global Section                ###
###############################################

def main():
    try:
        # Other useful variables
        debugInd = 'Y'
        var = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        var1 = datetime.datetime.now()

        # Initiating Prediction class
        x1 = cpt.clsPredictIonIoT()

        print('Start Time: ', str(var))
        # End of useful variables

        # Initiating Log Class
        general_log_path = str(cf.conf['LOG_PATH'])

        # Enabling Logging Info
        logging.basicConfig(filename=general_log_path + 'IoT_NeuralProphet.log', level=logging.INFO)

        # Reading the source IoT data
        iotData = p.read_csv(fileName)
        df = iotData.rename(columns={'MonthlyDate': 'ds', 'AvgIoTCPUUsage': 'y'})[['ds', 'y']]

        r1 = x1.forecastSeries(df)

        if (r1 == 0):
            print('Successfully IoT forecast predicted!')
        else:
            print('Failed to predict IoT forecast!')

        var2 = datetime.datetime.now()

        c = var2 - var1
        minutes = c.total_seconds() / 60
        print('Total Run Time in minutes: ', str(minutes))

        print('End Time: ', str(var1))

    except Exception as e:
        x = str(e)
        print('Error: ', x)

if __name__ == "__main__":
    main()
