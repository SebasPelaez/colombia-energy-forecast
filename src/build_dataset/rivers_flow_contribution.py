import argparse
import os
import wget
import numpy as np
import pandas as pd

from ..utils import utils
from ..utils import preprocessing_utils

def _groupby_dates_river(groups):

    hidrologic_region = groups['Region Hidrologica'].tolist()[0]
    
    columns_contribution = ['Aportes Caudal m3/s','Aportes Energía kWh','Aportes %']
    columns_contribution_sum = groups[columns_contribution].sum()
    
    dataframe_result = pd.Series([hidrologic_region],index=['Region Hidrologica'])
    dataframe_result = dataframe_result.append(columns_contribution_sum)
    
    return dataframe_result

def generate(params):

	data_dir = os.path.join(params['data_dir'],params['data_dir_xm'],'Caudal')
	real_columns_names = ['Fecha','Region Hidrologica','Nombre Río','Aportes Caudal m3/s',
						  'Aportes Energía kWh','Aportes %']

	full_data = preprocessing_utils.load_files(data_dir,real_columns_names)
	full_data = preprocessing_utils.remove_index_in_names(full_data,'Nombre Río')

	print('Proceso de Agrupación Iniciado')
	print('Proceso de Guardado Iniciado')
	group_data = full_data.groupby(['Fecha','Nombre Río']).apply(_groupby_dates_river)
	group_data = group_data.reset_index()

	group_data['Aportes Caudal m3/s'].fillna(value=0, inplace=True)
	group_data['Aportes Energía kWh'].fillna(value=0, inplace=True)
	group_data['Aportes %'].fillna(value=0, inplace=True)

	fecha_idx = group_data[group_data['Fecha'] == 'Fecha'].index
	group_data = group_data.drop(index=fecha_idx)

	dataset_path = os.path.join(params['data_dir'],params['data_dir_series'],'Caudal')
	if not os.path.exists(dataset_path):
		os.makedirs(dataset_path)

	preprocessing_utils.split_and_save_dataframes(group_data,dataset_path,dataset_name='Aporte Rios')

	print('Proceso de Transformación y Guardado Finalizado')

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('-c', '--config', help="path to configuration file", default='config.yml')

	args = parser.parse_args()
	params = utils.yaml_to_dict(args.config)

	generate(params)