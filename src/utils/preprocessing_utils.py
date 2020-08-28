import datetime
import math
import numpy as np
import os
import pandas as pd

def load_files(data_dir,columns_names,skiprows=0):
	
	data_list = list()

	shape_count = 0
	for root, dirs, files in os.walk(data_dir, topdown=False):
	    for file in files:
	        f = pd.read_excel(os.path.join(root,file),skiprows=skiprows)
	        f.columns = columns_names
	        shape_count += f.shape[0]
	        data_list.append(f)
	        print(os.path.join(root,file),f.shape)

	full_data = pd.concat(data_list)
	assert shape_count == full_data.shape[0], 'No coinciden las dimensiones'

	full_data.dropna(how='all',inplace=True)
	full_data.fillna(value=0, inplace=True)

	return full_data

def transpose_hours_into_rows(dataset,id_columns):
	transpose_dataframe = pd.melt(dataset,
		id_vars=id_columns,
		value_vars=[str(i) for i in np.arange(24)])

	return transpose_dataframe

def build_dataset(dataset,columns_to_sort):

    hour_lambda = lambda x: pd.Timedelta(datetime.datetime.strptime(x,'%H').hour, unit='hours')
    
    dataset['Fecha'] =  pd.to_datetime(dataset['Fecha'], format='%Y-%m-%d')
    dataset['variable'] = dataset['variable'].map(hour_lambda)
    
    dataset['Fecha'] = dataset['Fecha'] +  dataset['variable']
    dataset = dataset.sort_values(columns_to_sort)
    dataset.drop('variable', axis=1, inplace = True)
    dataset.rename(columns={"value": "kWh"}, inplace = True)
    dataset['kWh'].fillna(value=0, inplace=True)
    
    return dataset

def split_and_save_dataframes(dataset,dataset_path,dataset_name):
    MAX_EXCEL_ROWS = 1048576
    dataframe_rows = dataset.shape[0]
    sets = 1
    
    if dataframe_rows > MAX_EXCEL_ROWS:
        sets = math.ceil(dataframe_rows/MAX_EXCEL_ROWS)
        
    dataset_list = np.array_split(dataset, sets)
    for idx,df in enumerate(dataset_list):
    	file_path = os.path.join(dataset_path,'{}_{}.xlsx'.format(dataset_name,idx))
    	df.to_excel(file_path,index=False)
    	print('Archivo: {} Guardado con exito'.format(file_path))

def remove_index_in_names(dataset,column_to_preprocess):
	dataset[column_to_preprocess] = dataset[column_to_preprocess].str.replace('\d+', '')
	dataset[column_to_preprocess] = dataset[column_to_preprocess].str.replace(r'(?=\b[MDCLXVI]+\b)M{0,4}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})', '')
	dataset[column_to_preprocess] = dataset[column_to_preprocess].map(lambda x: x.strip(' '))

	return dataset