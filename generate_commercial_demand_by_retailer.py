import argparse
import os
import wget
import numpy as np
import pandas as pd

import utils
import preprocessing_utils

def _groupby_dates_market(groups):
    
    hours_columns = [str(i) for i in np.arange(24)]
    hours_columns_sum = groups[hours_columns].sum()
    
    version = groups['Version'].tolist()[0]
    
    dataframe_result = pd.Series(hours_columns_sum,index=hours_columns)
    dataframe_result = dataframe_result.append(pd.Series([version],index=['Version']))

    return dataframe_result

def generate(params):

	data_dir = os.path.join(params['data_dir'],params['data_dir_xm'],'Demanda','Demanda Comercial Por Comercializador')
	real_columns_names = ['Fecha','Codigo Comercializador','Mercado'] + [str(i) for i in np.arange(24)] + ['Version']

	full_data = preprocessing_utils.load_files(data_dir,real_columns_names,skiprows=2)

	print('Proceso de Agrupación Iniciado')
	group_data = full_data.groupby(['Fecha','Codigo Comercializador','Mercado']).apply(_groupby_dates_market)
	group_data = group_data.reset_index()

	fecha_idx = group_data[group_data['Fecha'] == 'Fecha'].index
	group_data = group_data.drop(index=fecha_idx)

	unique_markets = pd.unique(group_data['Mercado'])

	unique_markets = [i for i in unique_markets if not isinstance(i, float)]

	print('Proceso de Agrupamiento Finalizado')
	print('Proceso de Transformación y Guardado Iniciado')

	for market in unique_markets:
	    
	    filter_dataframe = group_data[group_data['Mercado']==market].copy()

	    id_columns = ['Fecha', 'Codigo Comercializador','Mercado','Version']
	    transpose_dataframe = preprocessing_utils.transpose_hours_into_rows(filter_dataframe,id_columns)

	    columns_to_sort = ['Codigo Comercializador','Fecha']
	    market_dataframe = preprocessing_utils.build_dataset(transpose_dataframe,columns_to_sort)
	    
	    dataset_path = os.path.join(params['data_dir'],params['data_dir_series'],'Demanda','Por Comercializador')
	    if not os.path.exists(dataset_path):
	        os.makedirs(dataset_path)
	        
	    preprocessing_utils.split_and_save_dataframes(market_dataframe,dataset_path,dataset_name=market)

	print('Proceso de Transformación y Guardado Finalizado')

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('-c', '--config', help="path to configuration file", default='config.yml')

	args = parser.parse_args()
	params = utils.yaml_to_dict(args.config)

	generate(params)