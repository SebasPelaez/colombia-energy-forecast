import os
import numpy as np
import pandas as pd
import tensorflow as tf
import time
import math
import sys
import matplotlib.pyplot as plt

sys.path.append('..')
import EnergyPricesLibrary as Ep
import CustomMetrics
import CustomHyperModelCompletos

from kerastuner.tuners import BayesianOptimization
from sklearn.metrics import mean_squared_error

def make_predictions(model,scaler_D_x,scaler_D_y,scaler_H_x,scaler_H_y,
                     trainX_D, trainY_D, testX_D, testY_D,
                     trainX_H, trainY_H, testX_H, testY_H,
                     trainX_I, trainY_I, testX_I, testY_I,
                     n_steps_out,len_output_features):
    
    # make predictions
    trainPredict = model.predict([trainX_H,trainX_D,trainX_I])
    trainPredict = trainPredict.reshape(trainPredict.shape[0]*n_steps_out,len_output_features)
    testPredict  = model.predict([testX_H,testX_D,testX_I])
    testPredict  = testPredict.reshape(testPredict.shape[0]*n_steps_out,len_output_features)
    
    # invert predictions
    trainPredict = scaler_D_y.inverse_transform(trainPredict)
    trainY = scaler_D_y.inverse_transform(trainY_D.reshape(trainY_D.shape[0]*n_steps_out,len_output_features))
    
    testPredict = scaler_D_y.inverse_transform(testPredict)
    testY = scaler_D_y.inverse_transform(testY_D.reshape(testY_D.shape[0]*n_steps_out,len_output_features))
        
    return trainPredict,trainY,testPredict,testY

def get_metrics(trainY,trainPredict,testY,testPredict):
    
    trainMAPE  = Ep.MAPE(trainPredict,trainY)
    testMAPE  = Ep.MAPE(testPredict,testY)
    
    train_sMAPE  = Ep.sMAPE(trainY,trainPredict)
    test_sMAPE  = Ep.sMAPE(testY,testPredict)
    
    return trainMAPE,testMAPE,train_sMAPE,test_sMAPE

data_diaria_path = os.path.join('..','..','..','dataset','Series','Sabanas','Original','Sabana_Datos_Diaria.xlsx')
data_diaria = pd.read_excel(data_diaria_path)
data_diaria = data_diaria.set_index('Fecha')

data_horaria_path = os.path.join('..','..','..','dataset','Series','Sabanas','Original','Sabana_Datos_Horaria.xlsx')
data_horaria = pd.read_excel(data_horaria_path)
data_horaria = data_horaria.set_index('Fecha')

climatic_images_prcp_dir = os.path.join('..','..','..','dataset','Climatic Images','PRCP')
climatic_images_tavg_dir = os.path.join('..','..','..','dataset','Climatic Images','TAVG')

precio_bolsa_path = os.path.join('..','..','..','dataset','Series','Sabanas','Original','Sabana_Datos_Precio_Bolsa.xlsx')
precio_bolsa = pd.read_excel(precio_bolsa_path)
precio_bolsa = precio_bolsa.set_index('Fecha')

nombre_series_diaria = data_diaria.columns.values
nombre_series_horaria = data_horaria.columns.values

data_horaria_full = pd.concat([data_horaria,precio_bolsa],axis=1)


lista_fechas = list()
lista_rutas = list()
for prcp_file,tavg_file in zip(os.listdir(climatic_images_prcp_dir),os.listdir(climatic_images_tavg_dir)):
    fecha = prcp_file.split('.')[0]
    ruta_prcp = os.path.join(climatic_images_prcp_dir,prcp_file)
    ruta_tavg = os.path.join(climatic_images_tavg_dir,tavg_file)
    lista_fechas.append(fecha)
    lista_rutas.append([ruta_prcp,ruta_tavg])

d = 'All'
start_date_train = '2000-02-01'
start_date_val = '2020-01-01'
start_date_test = '2020-04-01'
end_date_test = '2020-05-01'
n_steps_out=24
output_columns = ['$kWh']

dataset_df = pd.DataFrame(lista_rutas,index=lista_fechas,columns=['Precipitacion','Temperatura'])
 
n_steps_in  = 5
overlap = 1
len_output_features = len(output_columns)

IMG_HEIGHT,IMG_WIDTH = 128,128

results = Ep.SplitTimeseriesMultipleTimesBackAhead_DifferentTimes_Images(df_x=dataset_df,df_y=precio_bolsa,
                                                                         start_date_train=start_date_train,
                                                                         start_date_val=start_date_val,
                                                                         start_date_test=start_date_test,
                                                                         end_date_test=end_date_test,n_steps_out=n_steps_out,
                                                                         n_steps_in=n_steps_in,overlap=overlap,
                                                                         output_features=output_columns,
                                                                         IMG_HEIGHT=IMG_HEIGHT,IMG_WIDTH=IMG_WIDTH)

trainX_I,trainY_I,valX_I,valY_I,testX_I,testY_I,scaler_y_I,dataset_x_I,dataset_y_I = results

n_steps_in = 5
overlap = 1
inputs_columns = nombre_series_diaria

len_input_features = len(inputs_columns)
len_output_features = len(output_columns)

results = Ep.SplitTimeseriesMultipleTimesBackAhead_differentTimes(df_x=data_diaria,
                                                                  df_y=precio_bolsa,
                                                                  day=d,
                                                                  start_date_train=start_date_train,start_date_val=start_date_val,
                                                                  start_date_test=start_date_test,end_date_test=end_date_test,
                                                                  n_steps_out=n_steps_out,n_steps_in=n_steps_in,
                                                                  overlap=overlap,input_features=inputs_columns,
                                                                  output_features=output_columns)

trainX_D,trainY_D,valX_D,valY_D,testX_D,testY_D,scaler_D_x,scaler_D_y,dataset_x_D, dataset_y_D = results

n_steps_in = 120
overlap = 24
inputs_columns = nombre_series_horaria

len_input_features = len(inputs_columns)
len_output_features = len(output_columns)

results = Ep.SplitTimeseriesMultipleTimesBackAhead(df=data_horaria_full,
                                                   day=d,
                                                   start_date_train=start_date_train,start_date_val=start_date_val,
                                                   start_date_test=start_date_test,end_date_test=end_date_test,
                                                   n_steps_out=n_steps_out,n_steps_in=n_steps_in,overlap=overlap,
                                                   input_features=inputs_columns,output_features=output_columns)

trainX_H,trainY_H,valX_H,valY_H,testX_H,testY_H,scaler_H_x,scaler_H_y,df2_H,dataset_H = results

print('Diaria:',trainX_D.shape,trainY_D.shape,'Horaria:',trainX_H.shape, trainY_H.shape,'Imagenes:',trainX_I.shape, trainY_I.shape)

print('Diaria:',valX_D.shape,valY_D.shape,'Horaria:',valX_H.shape,valY_H.shape,'Imagenes:',valX_I.shape,valY_I.shape)

print('Diaria:',testX_D.shape, testY_D.shape,'Horaria:',testX_H.shape, testY_H.shape,'Imagenes:',testX_I.shape, testY_I.shape)

callback_reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                          factor=0.1,
                                                          min_lr=1e-5,
                                                          patience=5,
                                                          verbose=1)

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                  patience=10,
                                                  mode='min')

callbacks = [callback_reduce_lr,early_stopping]

hourly_input_shape = (trainX_H.shape[1],trainX_H.shape[2])
daily_input_shape = (trainX_D.shape[1],trainX_D.shape[2])
images_input_shape = trainX_I[0].shape

ModeloCompleto_I2_Concat = CustomHyperModelCompletos.ModeloCompleto_I2_Concat(hourly_input_shape=hourly_input_shape,
                                                                              daily_input_shape=daily_input_shape,
                                                                              image_input_shape=images_input_shape,
                                                                              n_steps_out=n_steps_out)

ModeloCompleto_I2_Suma = CustomHyperModelCompletos.ModeloCompleto_I2_Suma(hourly_input_shape=hourly_input_shape,
                                                                          daily_input_shape=daily_input_shape,
                                                                          image_input_shape=images_input_shape,
                                                                          n_steps_out=n_steps_out)

ModeloCompleto_I3_Concat = CustomHyperModelCompletos.ModeloCompleto_I3_Concat(hourly_input_shape=hourly_input_shape,
                                                                              daily_input_shape=daily_input_shape,
                                                                              image_input_shape=images_input_shape,
                                                                              n_steps_out=n_steps_out)

ModeloCompleto_I3_Suma = CustomHyperModelCompletos.ModeloCompleto_I3_Suma(hourly_input_shape=hourly_input_shape,
                                                                          daily_input_shape=daily_input_shape,
                                                                          image_input_shape=images_input_shape,
                                                                          n_steps_out=n_steps_out)

ModeloCompleto_I5_Concat = CustomHyperModelCompletos.ModeloCompleto_I5_Concat(hourly_input_shape=hourly_input_shape,
                                                                              daily_input_shape=daily_input_shape,
                                                                              image_input_shape=images_input_shape,
                                                                              n_steps_out=n_steps_out)

ModeloCompleto_I5_Suma = CustomHyperModelCompletos.ModeloCompleto_I5_Suma(hourly_input_shape=hourly_input_shape,
                                                                          daily_input_shape=daily_input_shape,
                                                                          image_input_shape=images_input_shape,
                                                                          n_steps_out=n_steps_out)

arq_best_models = dict()
    
bayesian_tuner = BayesianOptimization(
    ModeloCompleto_I5_Suma,
    objective='val_loss',
    num_initial_points=1,
    max_trials=10,
    directory=os.path.normpath('C:/my_dir'),
    project_name='tuning'
)

# Overview of the task
bayesian_tuner.search_space_summary()

# Performs the hyperparameter tuning
search_start = time.time()
bayesian_tuner.search(x=[trainX_H,trainX_D,trainX_I], y=trainY_D,
                      epochs=200,
                      validation_data=([valX_H,valX_D,valX_I],valY_D),
                      callbacks=callbacks)
search_end = time.time()
elapsed_time = search_end - search_start

print('Tiempo Total Transcurrido {}'.format(elapsed_time))

dict_key = 'Arquitectura'

arq_best_models[dict_key] = dict()
bs_model = bayesian_tuner.oracle.get_best_trials(1)[0]

model = bayesian_tuner.get_best_models(num_models=1)[0]

trainPredict,trainY,valPredict,valY = make_predictions(model,scaler_D_x,scaler_D_y,scaler_H_x,scaler_H_y,
                                                     trainX_D, trainY_D, valX_D, valY_D,
                                                     trainX_H, trainY_H, valX_H, valY_H,
                                                     trainX_I, trainY_I, valX_I, valY_I,
                                                     n_steps_out,len_output_features)

trainMAPE,testMAPE,train_sMAPE,test_sMAPE = get_metrics(trainY,trainPredict,valY,valPredict)

arq_best_models[dict_key]['Score'] = bs_model.score
arq_best_models[dict_key]['Tiempo Scaneo'] = elapsed_time
arq_best_models[dict_key]['Mape Train'] = trainMAPE
arq_best_models[dict_key]['Mape Test'] = testMAPE
arq_best_models[dict_key]['sMape Train'] = train_sMAPE
arq_best_models[dict_key]['sMape Test'] = test_sMAPE

if bs_model.hyperparameters.values:
    for hp, value in bs_model.hyperparameters.values.items():
        arq_best_models[dict_key][hp] = value

print(arq_best_models)