import argparse
import numpy as np
import os
import pandas as pd

from ..utils import utils
from ..utils import preprocessing_utils

def generate(params):

	data_dir = os.path.join(params['data_dir'],params['data_dir_xm'],'Precio','Precio Bolsa Nacional')
	real_columns_names = ['Fecha'] + [str(i) for i in np.arange(24)]

	full_data = preprocessing_utils.load_files(data_dir,real_columns_names)

	print('Proceso de Transformación y Guardado Iniciado')

	id_columns = ['Fecha']
	transpose_dataframe = preprocessing_utils.transpose_hours_into_rows(full_data,id_columns)

	columns_to_sort = ['Fecha']
	generation_dataframe = preprocessing_utils.build_dataset(transpose_dataframe,columns_to_sort)

	dataset_path = os.path.join(params['data_dir'],params['data_dir_series'],'Precio','Bolsa Nacional')
	if not os.path.exists(dataset_path):
		os.makedirs(dataset_path)

	generation_dataframe.rename(columns={"kWh": "$kWh"}, inplace = True)

	preprocessing_utils.split_and_save_dataframes(generation_dataframe,dataset_path,dataset_name='Bolsa Nacional')
	print('Proceso de Transformación y Guardado Finalizado')

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('-c', '--config', help="path to configuration file", default='config.yml')

	args = parser.parse_args()
	params = utils.yaml_to_dict(args.config)

	generate(params)