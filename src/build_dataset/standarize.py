import argparse
import numpy as np
import os
import pandas as pd
import pickle

from ..utils import preprocessing_utils
from ..utils import utils

from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import StandardScaler

def _standarize(data,scaler):

	scaler.fit(data)
	scaled_data = scaler.transform(data)

	return scaled_data,scaler

def _save_scaler(file_path,file_name,scaler):

	if not os.path.exists(file_path):
		os.makedirs(file_path)

	file_path = os.path.join(file_path,'{}.pkl'.format(file_name))

	with open(file_path, 'wb') as handle:
		pickle.dump(scaler,handle,protocol=pickle.HIGHEST_PROTOCOL)

	print('Archivo: {} Guardado con exito'.format(file_path))


def make_standarization(params,path=None,data=None,standarized_data_name=None,scaler_name='Scaler'):

	if path != None:
		data = pd.read_excel(daily_data_sheet_path)
		data = data.set_index('Fecha')

	scaler = StandardScaler()
	scaled_data,scaler_fit = _standarize(data,scaler)

	data = pd.DataFrame(scaled_data,index=data.index,columns=data.columns)

	file_path = os.path.join(params['data_dir'],params['data_dir_series'],'Sabanas','Estandarizada')
	preprocessing_utils.save_data_files(file_path=file_path,file_name=standarized_data_name,data=data)

	file_path = os.path.join(params['data_objects'],params['data_objects_scaler'])
	_save_scaler(file_path=file_path,file_name=scaler_name,scaler=scaler_fit)

	"""
	daily_data_sheet_path = os.path.join(params['data_dir'],params['data_dir_series'],'Sabanas','Original','Sabana_Datos_Diaria.xlsx')
	daily_data_sheet = pd.read_excel(daily_data_sheet_path)
	daily_data_sheet = daily_data_sheet.set_index('Fecha')

	hourly_data_sheet_path = os.path.join(params['data_dir'],params['data_dir_series'],'Sabanas','Original','Sabana_Datos_Horaria.xlsx')
	hourly_data_sheet = pd.read_excel(hourly_data_sheet_path)
	hourly_data_sheet = hourly_data_sheet.set_index('Fecha')

	hourly_scaler = StandardScaler()
	daily_scaler = StandardScaler()

	daily_scaled_data,daily_scaler = _standarize(daily_data_sheet,daily_scaler)
	hourly_scaled_data,hourly_scaler = _standarize(hourly_data_sheet,hourly_scaler)

	daily_data_sheet = pd.DataFrame(daily_scaled_data,index=daily_data_sheet.index,columns=daily_data_sheet.columns)
	hourly_data_sheet = pd.DataFrame(hourly_scaled_data,index=hourly_data_sheet.index,columns=hourly_data_sheet.columns)

	file_path = os.path.join(params['data_dir'],params['data_dir_series'],'Sabanas','Estandarizada')
	preprocessing_utils.save_data_files(file_path=file_path,file_name='Sabana_Datos_Horaria_Estandarizada',data=hourly_data_sheet)
	preprocessing_utils.save_data_files(file_path=file_path,file_name='Sabana_Datos_Diaria_Estandarizada',data=daily_data_sheet)

	file_path = os.path.join(params['data_objects'],params['data_objects_scaler'])
	_save_scaler(file_path=file_path,file_name='Data Scaler Diario',scaler=daily_scaler)
	_save_scaler(file_path=file_path,file_name='Data Scaler Horario',scaler=hourly_scaler)
	"""

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('-c', '--config', help="path to configuration file", default='config.yml')

	args = parser.parse_args()
	params = utils.yaml_to_dict(args.config)

	daily_data_sheet_path = os.path.join(params['data_dir'],params['data_dir_series'],'Sabanas','Original','Sabana_Datos_Diaria.xlsx')
	hourly_data_sheet_path = os.path.join(params['data_dir'],params['data_dir_series'],'Sabanas','Original','Sabana_Datos_Horaria.xlsx')

	make_standarization(params,path=hourly_data_sheet_path,standarized_data_name='Sabana_Datos_Horaria_Estandarizada',scaler_name='hourly_scaler')
	make_standarization(params,path=daily_data_sheet_path,standarized_data_name='Sabana_Datos_Diaria_Estandarizada',scaler_name='daily_scaler')