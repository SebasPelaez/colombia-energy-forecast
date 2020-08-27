import argparse
import numpy as np
import os
import pandas as pd

import utils
import preprocessing_utils

def _groupby_dates_resources(groups):
    
    generation_type = groups['Tipo Generación'].tolist()[0]
    combustible = groups['Combustible'].tolist()[0]
    agent_code = groups['Código Agente'].tolist()[0]
    dispatch_type = groups['Tipo Despacho'].tolist()[0]
    is_less = groups['Es Menor'].tolist()[0]
    is_autogenerator = groups['Es Autogenerador'].tolist()[0]

    hours_columns = [str(i) for i in np.arange(24)]
    hours_columns_sum = groups[hours_columns].sum()

    version = groups['Version'].tolist()[0]
     
    c = ['Tipo Generación','Combustible','Código Agente','Tipo Despacho','Es Menor','Es Autogenerador']

    dataframe_result = pd.Series([generation_type,combustible,agent_code,dispatch_type,is_less,is_autogenerator],index=c)
    dataframe_result = dataframe_result.append(hours_columns_sum)
    dataframe_result = dataframe_result.append(pd.Series([version],index=['Version']))

    return dataframe_result

def generate(params):

	data_dir = os.path.join(params['data_dir'],params['data_dir_xm'],'Generación','Generación')
	real_columns_names = ['Fecha','Recurso','Tipo Generación','Combustible','Código Agente',
						  'Tipo Despacho','Es Menor','Es Autogenerador'] + [str(i) for i in np.arange(24)] + ['Version']

	full_data = preprocessing_utils.load_files(data_dir,real_columns_names)
	full_data = preprocessing_utils.remove_index_in_names(full_data,'Recurso')

	print('Proceso de Agrupación Iniciado')
	group_data = full_data.groupby(['Fecha','Recurso']).apply(_groupby_dates_resources)

	group_data = group_data.reset_index()
	fecha_idx = group_data[group_data['Fecha'] == 'Fecha'].index
	group_data = group_data.drop(index=fecha_idx)

	unique_generation_types = pd.unique(group_data['Tipo Generación'])
	unique_generation_types = [i for i in unique_generation_types if not isinstance(i, float)]

	print('Proceso de Agrupamiento Finalizado')
	print('Proceso de Transformación y Guardado Iniciado')

	for generation_types in unique_generation_types:

	    filter_dataframe = group_data[group_data['Tipo Generación']==generation_types].copy()

	    id_columns = ['Fecha', 'Recurso', 'Tipo Generación', 'Combustible', 'Código Agente',
	    			  'Tipo Despacho', 'Es Menor', 'Es Autogenerador','Version']
	    transpose_dataframe = preprocessing_utils.transpose_hours_into_rows(filter_dataframe,id_columns)

	    columns_to_sort = ['Recurso','Fecha']
	    generation_dataframe = preprocessing_utils.build_dataset(transpose_dataframe,columns_to_sort)

	    dataset_path = os.path.join(params['data_dir'],params['data_dir_series'],'Generacion')
	    if not os.path.exists(dataset_path):
	        os.makedirs(dataset_path)
	        
	    preprocessing_utils.split_and_save_dataframes(generation_dataframe,dataset_path,dataset_name=generation_types)

	print('Proceso de Transformación y Guardado Finalizado')

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('-c', '--config', help="path to configuration file", default='config.yml')

	args = parser.parse_args()
	params = utils.yaml_to_dict(args.config)

	generate(params)