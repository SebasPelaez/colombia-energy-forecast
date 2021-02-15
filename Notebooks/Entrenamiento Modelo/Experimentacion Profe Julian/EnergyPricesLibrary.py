# -*- coding: utf-8 -*-
"""
Created on Sat Sep  8 09:15:40 2018

@author: julian

Modified on September 2020
@author: Juan Sebastián Peláez
"""
import numpy as np
import tensorflow as tf

from math import sqrt
from numpy import array
from numpy import mean
from pandas import DataFrame
from pandas import concat
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

#Mean Absolute Percentage Error para los problemas de regresión
def MAPE(Y_est,Y):
    N = np.size(Y)
    mape = np.sum(abs((Y_est.reshape(N,1) - Y.reshape(N,1))/Y.reshape(N,1)))/N
    return mape

def create_datasetMultipleTimesBackAhead_inverse(dataset, ds_y=None, n_steps_out=1, n_steps_in = 1, overlap = 1):
    dataX, dataY = [], []
    tem = n_steps_in + n_steps_out - overlap
    range_iter = int((len(dataset) - tem)/overlap)

    for i in range(range_iter,0,-1):
        startx = (i*overlap)-(overlap*int(n_steps_out/24))
        endx = startx + n_steps_in
        starty = endx
        endy = endx + n_steps_out
        dataX.append(dataset[startx:endx, :])
        dataY.append(ds_y[starty:endy, :])

    trainX, trainY = np.array(dataX), np.array(dataY)
    trainX, trainY = trainX[::-1],trainY[::-1]

    return trainX, trainY

def create_datasetMultipleTimesBackAhead_differentTimes_inverse(ds_x, ds_y, n_steps_out=1, n_steps_in = 1, overlap = 1):
    dataX, dataY = [], []
    tem = n_steps_in + (n_steps_out/24) - overlap
    range_iter = int((len(ds_x) - tem)/overlap)

    for i in range(range_iter,0,-1):
        startx = (i*overlap)-(overlap*int(n_steps_out/24))
        endx = startx + n_steps_in
        starty = endx*n_steps_out
        endy =  starty+n_steps_out
        dataX.append(ds_x[startx:endx, :])
        dataY.append(ds_y[starty:endy, -1])

    trainX, trainY = np.array(dataX), np.array(dataY).reshape(-1,n_steps_out,1)
    trainX, trainY = trainX[::-1],trainY[::-1]

    return trainX, trainY

def create_datasetMultipleTimesBackAhead(dataset, ds_y=None, n_steps_out=1, n_steps_in = 1, overlap = 1):
    dataX, dataY = [], []
    tem = n_steps_in + n_steps_out - overlap
    range_iter = int((len(dataset) - tem)/overlap)

    for i in range(range_iter):
        startx = i*overlap
        endx = startx + n_steps_in
        starty = endx
        endy = endx + n_steps_out
        dataX.append(dataset[startx:endx, :])
        dataY.append(ds_y[starty:endy, :])

    return np.array(dataX), np.array(dataY)

def create_datasetMultipleTimesBackAhead_differentTimes(ds_x, ds_y, n_steps_out=1, n_steps_in = 1, overlap = 1):
    dataX, dataY = [], []
    tem = n_steps_in + (n_steps_out/24) - overlap
    range_iter = int((len(ds_x) - tem)/overlap)

    for i in range(range_iter):
        startx = i*overlap
        endx = startx + n_steps_in
        starty = (i*overlap*n_steps_out)+(overlap*n_steps_out)
        endy = (i*overlap*n_steps_out)+(overlap*n_steps_out)+n_steps_out
        dataX.append(ds_x[startx:endx, :])
        dataY.append(ds_y[starty:endy, -1])

    return np.array(dataX), np.array(dataY).reshape(-1,n_steps_out,1)

def SplitTimeseriesMultipleTimesBackAhead(df,day='Monday',start_date_train='2000-02-01',start_date_val='2020-01-01',
                                          start_date_test='2020-04-01',end_date_test='2020-05-01',n_steps_out=1,
                                          n_steps_in=1,overlap=1,input_features=None,output_features=None):
    if day == 'All':

        # split into train and test sets
        df2 = df
        dataset = df2[input_features].values

        train_x   = df.loc[(df.index >=  start_date_train) & (df.index <  start_date_val), input_features].values
        val_x     = df.loc[(df.index >=  start_date_val)   & (df.index < start_date_test), input_features].values
        test_x    = df.loc[(df.index >=  start_date_test)  & (df.index < end_date_test), input_features].values

        train_y   = df.loc[(df.index >=  start_date_train) & (df.index <  start_date_val), output_features].values
        val_y    = df.loc[(df.index >=  start_date_val)   & (df.index < start_date_test), output_features].values
        test_y    = df.loc[(df.index >=  start_date_test)  & (df.index < end_date_test), output_features].values

        # normalize the dataset
        scaler_x = MinMaxScaler(feature_range=(0, 1))
        train_x  = scaler_x.fit_transform(train_x)
        val_x    = scaler_x.transform(val_x)
        test_x   = scaler_x.transform(test_x)

        scaler_y = MinMaxScaler(feature_range=(0, 1))
        train_y  = scaler_y.fit_transform(train_y)
        val_y    = scaler_y.transform(val_y)
        test_y   = scaler_y.transform(test_y)

        trainX, trainY = create_datasetMultipleTimesBackAhead_inverse(train_x,ds_y=train_y,n_steps_out=n_steps_out, n_steps_in = n_steps_in, overlap = overlap)

        val2_x = np.concatenate((train_x[-n_steps_in:].reshape(n_steps_in,-1),val_x),axis=0)
        val2_y = np.concatenate((train_y[-n_steps_in:].reshape(n_steps_in,-1),val_y),axis=0)
        valX, valY = create_datasetMultipleTimesBackAhead_inverse(val2_x,ds_y=val2_y,n_steps_out=n_steps_out, n_steps_in = n_steps_in, overlap = overlap)

        test2_x = np.concatenate((val_x[-n_steps_in:].reshape(n_steps_in,-1),test_x),axis=0)
        test2_y = np.concatenate((val_y[-n_steps_in:].reshape(n_steps_in,-1),test_y),axis=0)
        testX, testY = create_datasetMultipleTimesBackAhead_inverse(test2_x,ds_y=test2_y,n_steps_out=n_steps_out, n_steps_in = n_steps_in, overlap = overlap)

        # reshape input to be [samples, time steps, features]
        trainY = np.reshape(trainY, (-1, n_steps_out, len(output_features)))
        valY   = np.reshape(valY, (-1, n_steps_out, len(output_features)))
        testY  = np.reshape(testY, (-1, n_steps_out, len(output_features)))

        return trainX, trainY, valX, valY, testX, testY, scaler_x, scaler_y, df2, dataset


def SplitTimeseriesMultipleTimesBackAhead_differentTimes(df_x,df_y,day='Monday',start_date_train='2000-02-01',start_date_val='2020-01-01',
                                                         start_date_test='2020-04-01',end_date_test='2020-05-01',n_steps_out=1,
                                                         n_steps_in=1,overlap=1,input_features=None,output_features=None):
    if day == 'All':

        train_x   = df_x.loc[(df_x.index >=  start_date_train) & (df_x.index <  start_date_val), input_features].values
        val_x     = df_x.loc[(df_x.index >=  start_date_val)   & (df_x.index < start_date_test), input_features].values
        test_x    = df_x.loc[(df_x.index >=  start_date_test)  & (df_x.index < end_date_test), input_features].values

        train_y   = df_y.loc[(df_y.index >=  start_date_train) & (df_y.index <  start_date_val), output_features].values
        val_y     = df_y.loc[(df_y.index >=  start_date_val)   & (df_y.index < start_date_test), output_features].values
        test_y    = df_y.loc[(df_y.index >=  start_date_test)  & (df_y.index < end_date_test), output_features].values

        dataset_x = np.concatenate([train_x,val_x,test_x],axis=0)
        dataset_y = np.concatenate([train_y,val_y,test_y],axis=0)

        # normalize the dataset
        scaler_x = MinMaxScaler(feature_range=(0, 1))
        train_x  = scaler_x.fit_transform(train_x)
        val_x    = scaler_x.transform(val_x)
        test_x   = scaler_x.transform(test_x)

        scaler_y = MinMaxScaler(feature_range=(0, 1))
        train_y  = scaler_y.fit_transform(train_y)
        val_y    = scaler_y.transform(val_y)
        test_y   = scaler_y.transform(test_y)

        trainX, trainY = create_datasetMultipleTimesBackAhead_differentTimes_inverse(train_x, train_y, n_steps_out=n_steps_out, n_steps_in=n_steps_in, overlap=overlap)

        val2_x = np.concatenate((train_x[-n_steps_in:].reshape(n_steps_in,-1),val_x),axis=0)
        val2_y = np.concatenate((train_y[-(n_steps_in*24):].reshape((n_steps_in*24),-1),val_y),axis=0)
        valX, valY = create_datasetMultipleTimesBackAhead_differentTimes_inverse(val2_x,val2_y, n_steps_out=n_steps_out, n_steps_in = n_steps_in, overlap = overlap)

        test2_x = np.concatenate((val_x[-n_steps_in:].reshape(n_steps_in,-1),test_x),axis=0)
        test2_y = np.concatenate((val_y[-(n_steps_in*24):].reshape((n_steps_in*24),-1),test_y),axis=0)
        testX, testY = create_datasetMultipleTimesBackAhead_differentTimes_inverse(test2_x,ds_y=test2_y,n_steps_out=n_steps_out, n_steps_in = n_steps_in, overlap = overlap)

        return trainX, trainY, valX, valY, testX, testY, scaler_x,scaler_y, dataset_x, dataset_y
        
    else:

        df_x = df_x.loc[df_x['day_of_week'] == day]
        df_y = df_y.loc[df_y['day_of_week'] == day]

        # split into train and test sets
        train_x = df_x.loc[df_x.index <  TimeSplit, input_features].values
        test_x  = df_x.loc[df_x.index >= TimeSplit, input_features].values

        train_y = df_y.loc[df_y.index <  TimeSplit, output_features].values
        test_y  = df_y.loc[df_y.index >= TimeSplit, output_features].values

        dataset_x = np.concatenate([train_x,test_x],axis=0)
        dataset_y = np.concatenate([train_y,test_y],axis=0)

        # normalize the dataset
        scaler_x = MinMaxScaler(feature_range=(0, 1))
        train_x  = scaler_x.fit_transform(train_x)
        test_x   = scaler_x.transform(test_x)

        scaler_y = MinMaxScaler(feature_range=(0, 1))
        train_y  = scaler_y.fit_transform(train_y)
        test_y   = scaler_y.transform(test_y)

        trainX, trainY = create_datasetMultipleTimesBackAhead_differentTimes_inverse(train_x, train_y, n_steps_out=n_steps_out, n_steps_in=n_steps_in, overlap=overlap)
        
        test2_x = np.concatenate((train_x[-n_steps_in:].reshape(n_steps_in,-1),test_x),axis=0)
        test2_y = np.concatenate((train_y[-n_steps_in:].reshape(n_steps_in,-1),test_y),axis=0)

        testX, testY = create_datasetMultipleTimesBackAhead_differentTimes_inverse(test2_x,test2_y, n_steps_out=n_steps_out, n_steps_in = n_steps_in, overlap = overlap)

        return trainX, trainY, testX, testY, scaler_x,scaler_y, dataset_x, dataset_y

def SplitTimeseriesMultipleTimesBackAhead_DifferentTimes_Images(df_x,df_y,start_date_train='2000-02-01',start_date_val='2020-01-01',
                                                         start_date_test='2020-04-01',end_date_test='2020-05-01',n_steps_out=1,
                                                         n_steps_in = 1, overlap = 1, output_features=None,IMG_HEIGHT=635,IMG_WIDTH=460):

    # split into train and test sets
    train_x   = df_x.loc[(df_x.index >=  start_date_train) & (df_x.index <  start_date_val)].values
    val_x     = df_x.loc[(df_x.index >=  start_date_val)   & (df_x.index < start_date_test)].values
    test_x    = df_x.loc[(df_x.index >=  start_date_test)  & (df_x.index < end_date_test)].values

    train_y   = df_y.loc[(df_y.index >=  start_date_train) & (df_y.index <  start_date_val)].values
    val_y     = df_y.loc[(df_y.index >=  start_date_val)   & (df_y.index < start_date_test)].values
    test_y    = df_y.loc[(df_y.index >=  start_date_test)  & (df_y.index < end_date_test)].values

    dataset_x = np.concatenate([train_x,val_x,test_x],axis=0)
    dataset_y = np.concatenate([train_y,val_y,test_y],axis=0)

    # normalize the dataset
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    train_y  = scaler_y.fit_transform(train_y)
    val_y    = scaler_y.transform(val_y)
    test_y   = scaler_y.transform(test_y)

    img_data_array = _build_images_ds(train_x,IMG_HEIGHT,IMG_WIDTH)
    trainX, trainY = create_datasetMultipleTimesBackAhead_differentTimes(img_data_array, train_y, n_steps_out=n_steps_out, n_steps_in=n_steps_in, overlap=overlap)

    val2_x = np.concatenate((train_x[-n_steps_in:].reshape(n_steps_in,-1),val_x),axis=0)
    val2_y = np.concatenate((train_y[-(n_steps_in*24):].reshape((n_steps_in*24),-1),val_y),axis=0)
    img_data_array = _build_images_ds(val2_x,IMG_HEIGHT,IMG_WIDTH)
    valX, valY = create_datasetMultipleTimesBackAhead_differentTimes(img_data_array,val2_y, n_steps_out=n_steps_out, n_steps_in = n_steps_in, overlap = overlap)

    test2_x = np.concatenate((val_x[-n_steps_in:].reshape(n_steps_in,-1),test_x),axis=0)
    test2_y = np.concatenate((val_y[-(n_steps_in*24):].reshape((n_steps_in*24),-1),test_y),axis=0)
    img_data_array = _build_images_ds(test2_x,IMG_HEIGHT,IMG_WIDTH)
    testX, testY = create_datasetMultipleTimesBackAhead_differentTimes(img_data_array,test2_y, n_steps_out=n_steps_out, n_steps_in = n_steps_in, overlap = overlap)

    return trainX, trainY, valX, valY, testX, testY, scaler_y, dataset_x, dataset_y

def _build_images_ds(ds,IMG_HEIGHT,IMG_WIDTH):
    img_data_list = list()

    for path in ds:
        image_prcp = _load_images(path[0],IMG_HEIGHT,IMG_WIDTH)
        image_tavg = _load_images(path[1],IMG_HEIGHT,IMG_WIDTH)

        image_concat = np.concatenate([image_prcp,image_tavg],axis=-1)
        img_data_list.append(image_concat)
    
    img_data_array = np.array(img_data_list)

    return img_data_array

def _load_images(path,IMG_HEIGHT,IMG_WIDTH):
    image = np.array(Image.open(path))
    image = np.resize(image,(IMG_HEIGHT,IMG_WIDTH,3))
    image = image.astype('float32')
    image /= 255

    return image


def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# root mean squared error or rmse
def measure_rmse(actual, predicted):
    return sqrt(mean_squared_error(actual, predicted))