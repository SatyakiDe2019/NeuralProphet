################################################
#### Written By: SATYAKI DE                 ####
#### Written On:  15-May-2020               ####
#### Modified On: 28-Dec-2021               ####
####                                        ####
#### Objective: This script is a config     ####
#### file, contains all the keys for        ####
#### Machine-Learning & streaming dashboard.####
####                                        ####
################################################

import os
import platform as pl
import pandas as p

class clsConfig(object):
    Curr_Path = os.path.dirname(os.path.realpath(__file__))

    os_det = pl.system()
    if os_det == "Windows":
        sep = '\\'
    else:
        sep = '/'

    conf = {
        'APP_ID': 1,
        'ARCH_DIR': Curr_Path + sep + 'arch' + sep,
        'PROFILE_PATH': Curr_Path + sep + 'profile' + sep,
        'LOG_PATH': Curr_Path + sep + 'log' + sep,
        'REPORT_PATH': Curr_Path + sep + 'report',
        'FILE_NAME': Curr_Path + sep + 'Data' + sep + 'thermostatIoT.csv',
        'SRC_PATH': Curr_Path + sep + 'data' + sep,
        'APP_DESC_1': 'Old Video Enhancement!',
        'DEBUG_IND': 'N',
        'INIT_PATH': Curr_Path,
        'SUBDIR': 'data',
        'SEP': sep,
        'testRatio':0.2,
        'valRatio':0.2,
        'epochsVal':8,
        'sleepTime':3,
        'sleepTime1':6,
        'factorVal':0.2,
        'learningRateVal':0.001,
        'event1': {
            'event': 'SummerEnd',
            'ds': p.to_datetime([
                '2010-04-01', '2011-04-01', '2012-04-01',
                '2013-04-01', '2014-04-01', '2015-04-01',
                '2016-04-01', '2017-04-01', '2018-04-01',
                '2019-04-01', '2020-04-01', '2021-04-01',
            ]),},
        'event2': {
            'event': 'LongWeekend',
            'ds': p.to_datetime([
                '2010-12-01', '2011-12-01', '2012-12-01',
                '2013-12-01', '2014-12-01', '2015-12-01',
                '2016-12-01', '2017-12-01', '2018-12-01',
                '2019-12-01', '2020-12-01', '2021-12-01',
            ]),}
    }
