# forecaster.py
# Implements a MLP model to forecast load over a 24-hour period. 

import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
import warnings
from dateutil import parser
from pytz import timezone
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# fills in temperature gaps using spline interpolation over small (2-day)
# segments of data.
def interpolate_temps(df):
    # in order to catch runtimewarnings for spline
    warnings.filterwarnings('error', "The maximal number of iterations \(20\) "
                            "allowed for finding smoothing[\r\n]+spline.*")  
    step = 200
    offset = 5  # avoid boundary effects
    overlap = 3
    start_indices = np.arange(0, df.shape[0], step - offset*overlap)[:-1]
    end_indices = start_indices + step
    end_indices[-1] = df.shape[0]
    temp_range = (min(df['actual_temperature']), max(df['actual_temperature']))
    tolerance = 5
    
    for start, end in zip(start_indices, end_indices):
        segment = df[start:end]
        df_drop = segment.dropna(subset=['actual_temperature'])   
        smooth_factor = 0.2
        spline_found = False
        # if spline fails for any reason, increase smoothing factor
        while(not spline_found):
            try:
                spl = interpolate.splrep(df_drop['timestamp'], 
                                         df_drop['actual_temperature'], 
                                         k=3, s=smooth_factor)
            except(RuntimeWarning):
                smooth_factor += 0.3
                continue
        
            interp_temp = interpolate.splev(segment.loc[start+offset : 
                                                        end-offset-1, 'timestamp'], spl)
            if (np.isnan(interp_temp).all() or 
                (max(interp_temp) > temp_range[1]+tolerance) or 
                (min(interp_temp) < temp_range[0]-tolerance)):
                smooth_factor += 0.3
            else: spline_found = True
       
        df.loc[start+offset : end-offset-1, 'interp_temp'] = interp_temp
    

# returns a matrix of predictors for training & testing data
def gen_predictors(df, holidays=None, short_term=False):
    predictors = pd.DataFrame(df['dow'])
    predictors['hour'] = [dt.hour*60 + dt.minute for dt in df['datetime']]
    predictors['work_day'] = True
    predictors.loc[df['dow'].isin([5,6]), 'work_day'] = False
    if holidays is not None:
        holidays = pd.read_csv(holidays)['date']
        predictors.loc[df['date'].isin(holidays), 'work_day'] = False
    predictors['doy'] = [(date-timezone('US/Pacific').localize(datetime(year=date.year,month=1,day=1))).days 
                         for date in df['datetime']]
    
    if (short_term):
        predictors['interp_temp'] = df['interp_temp']
        predictors['prev_day_load'] = pd.concat([pd.Series([np.nan]*24*4), 
                df.loc[df.index[:-24*4], 'actual_kwh']]).reset_index(drop=True)
        predictors['prev_week_load'] = pd.concat([pd.Series([np.nan]*24*7*4), 
              df.loc[df.index[:-24*7*4], 'actual_kwh']]).reset_index(drop=True)
        #generate prev day average load
        n, total = 24*4, 0
        index = range(len(df))
        prev_day_avg_load = []
        for i in index:
            if not np.isnan(df.loc[i, 'actual_kwh']):
                total += df.loc[i, 'actual_kwh']
            if i % n == 0:
                for b in range(n):
                    if i-b >= 0:
                        prev_day_avg_load.append(total / n)
                total = 0
        predictors['prev_day_avg_load'] = pd.concat([pd.Series([np.nan]*24*4), 
                         pd.Series(prev_day_avg_load)]).reset_index(drop=True)
    return df['timestamp'], predictors


# returns matrix of predictors for the new data; i.e. 24 hr horizon
def gen_new_X(date, holidays=None, short_term=False):
    date = timezone('US/Pacific').localize(date)
    new_X = pd.DataFrame()
    times = [(date+timedelta(minutes=a)) for a in range(0, 60*24,15)]
    new_X['dow'] = [time.weekday() for time in times]
    new_X['hour'] = [time.hour*60 + time.minute for time in times]
    
    new_X['work_day'] = True
    new_X.loc[new_X['dow'].isin([5,6]), 'work_day'] = False
    if holidays is not None:
        holidays = pd.read_csv(holidays)['date']
        dates = pd.Series([date.strftime('%Y-%m-%d') for date in times])
        new_X.loc[dates.isin(holidays), 'work_day'] = False
    new_X['doy'] = [(date - timezone('US/Pacific').localize(datetime(year=date.year, month=1, day=1))).days 
                    for date in times]
    
    if short_term:          # short-term forecasting not yet implemented
        print('Short-term forecasting unavailable; proceeding with long-term')
    
    return new_X, times

# scale data for MLP
def preprocess_data(train_X, test_X):
    scaler = StandardScaler()
    scaler.fit(train_X)
    train_X = scaler.transform(train_X)
    test_X = scaler.transform(test_X)
    return train_X, test_X, scaler

# remove missing data points for preprocessing
def clean_data(predictors, df):
    predictors['load'] = df['actual_kwh']
    predictors.dropna(inplace=True)
    predictors.drop('load', axis=1, inplace=True)
    return predictors, df.loc[predictors.index, 'actual_kwh']


#_____________________________________________________________________________#
#                           Higher level functions                            #
    
# Populates missing temperature data and prepares data for next stages
# of processing.
#
# Input:
#    load_data [str]:  full filepath to load data
# Output:
#    df [DataFrame]:   updated dataframe with temperatures
    
def initialize(load_data):
    df = pd.read_csv(load_data)
    df['datetime'] = [parser.parse(a) for a in df.iloc[:,0]]
    df['timestamp'] = [a.timestamp() for a in df['datetime']]

    interpolate_temps(df)

    # filter dates for which load data is available
    df = df[(df['datetime'] > timezone('US/Pacific').localize(datetime(2012,11,2))) & 
             (df['datetime'] < timezone('US/Pacific').localize(datetime(2013,12,2)))].reset_index(drop=True)
    return df


# Generates predictor matrix, splits data into training and testing sets,
# and trains MLP regressor using training data. Prints the R^2 coefficient
# from in-sample and out-of-sample testing.
#
# Inputs:
#    df [DataFrame]:        initialized dataframe with time, temp, and load
#    holidays [str]:        full filepath to list of holidays. if provided,
#                           will inform the working-day predictor
#    short_term [bool]:     toggle for short-term vs long-term forecasting.
#                           only long-term is available for new data
#    hidden layers [tuple]: configure the size of hidden layers for MLP
# Outputs:
#    mlp [MLPRegressor]:      trained NN model, to be used for forecasting
#    scaler [StandardScaler]: scaler used on training data; for use on new
#                             test data
    
def train_model(df, holidays=None, short_term=False, hidden_layers=(30,30)):
    # generate predictor variables using all data
    times, predictors = gen_predictors(df, holidays, short_term)
    predictors, loads = clean_data(predictors, df)
    
    # split training and testing data; 25% test split
    train_X = predictors[:28000]
    test_X = predictors[28000:]
    train_Y = loads[:28000]
    test_Y = loads[28000:]
    
    # scale predictors for improved MLP regressor performance
    train_X, test_X, scaler = preprocess_data(train_X, test_X)

    # train model
    mlp = MLPRegressor(hidden_layer_sizes=hidden_layers)
    mlp.fit(train_X, train_Y)

    # evaluate model on testing and training data
    print(mlp.score(test_X, test_Y))
    print(mlp.score(train_X, train_Y))
    return mlp, scaler


# Forecasts load over a 24-hr horizon using trained model.
#
# Inputs:
#    forecast_date [str]:     starting date and time of 24-hr interval.
#                             suggested format: "2017-6-30 14:30"
#    mlp [MLPRegressor]:      trained NN model, to be used for forecasting
#    scaler [StandardScaler]: scaler used on training data; for use on new
#                             test data
#    holidays [str]:          full filepath to list of holidays
# Outputs:
#    forecasted_load [list]:  list of (datetime, predicted load) over 24-hr
#                             horizon

def forecast_day(forecast_date, mlp, scaler, holidays=None):
    try:
        date = parser.parse(forecast_date)
    except ValueError:
        print('Try using the following format for the start date/time: '
              '2012-1-1 14:00')
    if(date.minute % 15 != 0): 
        print('Please round the time to the nearest 15-min interval')
        return None
    if (date > datetime(2012,11,2) + timedelta(days=7)) and \
       (date < datetime(2013,12,2)):
        #short_term = True      #short-term forecasting not implemented yet
        short_term = False
    else:
        short_term = False
    
    new_X, times = gen_new_X(date, holidays, short_term)
    new_X_scaled = None
    try:
        new_X_scaled = scaler.transform(new_X)
    except ValueError:
        print('Inconsistent number of predictors used between training set ' 
              'and test set. Try re-training model with short_term=False')
        return None
    new_Y = mlp.predict(new_X_scaled)

    plt.plot_date(times, new_Y)
    return zip(times, new_Y)

#_____________________________________________________________________________#
#                             Wrapper function                                #

# Forecasts load over a 24-hr horizon using trained model.
#
# Inputs:
#    load_data [str]:         full filepath to load data
#    forecast_date [str]:     starting date and time of 24-hr interval.
#                             suggested format: "2017-6-30 14:30"
#    holidays [str]:          full filepath to list of holidays. suggested
#                             file: "US Bank Holidays.csv"
#    short_term [bool]:       toggle for short-term vs long-term forecasting.
#                             only long-term is available for new data
# Outputs:
#    mlp [MLPRegressor]:      trained NN model for forecasting
#    forecasted_load [list]:  list of (datetime, predicted load) over 24-hr
#                             horizon
    
def train_model_and_forecast(load_data, forecast_date, holidays=None, 
                             short_term=False):
    df = initialize(load_data)
    mlp, scaler = train_model(df, holidays, short_term)
    forecasted_load = forecast_day(forecast_date, mlp, scaler, holidays)
    return forecasted_load, mlp
