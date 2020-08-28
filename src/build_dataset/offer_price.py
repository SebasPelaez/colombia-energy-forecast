import argparse
import numpy as np
import os
import pandas as pd

from ..utils import utils
from ..utils import preprocessing_utils

def _groupby_dates_resources(groups):

    agent_code = groups['Código Agente'].tolist()[0]
    
    columns_price = ['Precio de Oferta Ideal','Precio de Oferta de Despacho','Precio de Oferta Declarado']
    columns_price_sum = groups[columns_price].sum()
    
    dataframe_result = pd.Series([agent_code],index=['Código Agente'])
    dataframe_result = dataframe_result.append(columns_price_sum)
    
    return dataframe_result

def generate(params):

	data_dir = os.path.join(params['data_dir'],params['data_dir_xm'],'Precio','Precio Oferta kWh')
	real_columns_names = ['Fecha','Recurso','Código Agente','Precio de Oferta Ideal',
						  'Precio de Oferta de Despacho','Precio de Oferta Declarado']

	full_data = preprocessing_utils.load_files(data_dir,real_columns_names)
	full_data = preprocessing_utils.remove_index_in_names(full_data,'Recurso')

	print('Proceso de Agrupación Iniciado')
	print('Proceso de Guardado Iniciado')
	group_data = full_data.groupby(['Fecha','Recurso']).apply(_groupby_dates_resources)

	group_data = group_data.reset_index()

	group_data['Precio de Oferta Ideal'].fillna(value=0, inplace=True)
	group_data['Precio de Oferta de Despacho'].fillna(value=0, inplace=True)
	group_data['Precio de Oferta Declarado'].fillna(value=0, inplace=True)

	fecha_idx = group_data[group_data['Fecha'] == 'Fecha'].index
	group_data = group_data.drop(index=fecha_idx)

	dataset_path = os.path.join(params['data_dir'],params['data_dir_series'],'Precio','Oferta')
	if not os.path.exists(dataset_path):
		os.makedirs(dataset_path)

	preprocessing_utils.split_and_save_dataframes(group_data,dataset_path,dataset_name='Oferta')

	print('Proceso de Transformación y Guardado Finalizado')

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('-c', '--config', help="path to configuration file", default='config.yml')

	args = parser.parse_args()
	params = utils.yaml_to_dict(args.config)

	generate(params)