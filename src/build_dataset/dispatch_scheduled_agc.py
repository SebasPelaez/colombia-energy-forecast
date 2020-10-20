import argparse
import numpy as np
import os
import pandas as pd

from ..utils import utils
from ..utils import preprocessing_utils

def _groupby_dates_resources(groups):
    
    hours_columns = [str(i) for i in np.arange(24)]
    hours_columns_sum = groups[hours_columns].sum()
    
    dataframe_result = pd.Series(hours_columns_sum,index=hours_columns)

    return dataframe_result

def generate(params):

	data_dir = os.path.join(params['data_dir'],params['data_dir_xm'],'Despacho','AGC Programado')
	real_columns_names = ['Fecha','Recurso','Código Agente'] + [str(i) for i in np.arange(24)]

	full_data = preprocessing_utils.load_files(data_dir,real_columns_names)
	full_data = preprocessing_utils.remove_index_in_names(full_data,'Recurso')

	print('Proceso de Agrupación Iniciado')
	group_data = full_data.groupby(['Fecha','Recurso','Código Agente']).apply(_groupby_dates_resources)
	group_data = group_data.reset_index()

	id_columns = ['Fecha', 'Recurso', 'Código Agente']
	transpose_dataframe = preprocessing_utils.transpose_hours_into_rows(group_data,id_columns)

	dataset = preprocessing_utils.build_dataset(transpose_dataframe,columns_to_sort=['Recurso','Fecha'])

	dataset_path = os.path.join(params['data_dir'],params['data_dir_series'],'Despacho')

	if not os.path.exists(dataset_path):
	    os.makedirs(dataset_path)

	preprocessing_utils.split_and_save_dataframes(dataset,dataset_path,dataset_name='Despacho')

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('-c', '--config', help="path to configuration file", default='config.yml')

	args = parser.parse_args()
	params = utils.yaml_to_dict(args.config)

	generate(params)