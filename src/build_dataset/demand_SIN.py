import argparse
import os

from ..utils import utils
from ..utils import preprocessing_utils

def generate(params):

	data_dir = os.path.join(params['data_dir'],params['data_dir_xm'],'Demanda','Demanda Energia SIN')
	real_columns_names = ['Fecha','Demanda Energia SIN','Generación',
						  'Demanda No Atendida','Exportaciones','Importaciones']

	full_data = preprocessing_utils.load_files(data_dir,real_columns_names)

	dataset_path = os.path.join(params['data_dir'],params['data_dir_series'],'Demanda','Energia SIN')

	if not os.path.exists(dataset_path):
	    os.makedirs(dataset_path)

	preprocessing_utils.split_and_save_dataframes(full_data,dataset_path,dataset_name='Demanda Energia SIN')


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('-c', '--config', help="path to configuration file", default='config.yml')

	args = parser.parse_args()
	params = utils.yaml_to_dict(args.config)

	generate(params)