import calendar
import datetime
import math
import numpy as np
import os
import pandas as pd

"""
Este script contiene un conjunto de funciones génericas especializadas para
la etapa de preprocesamiento de la información.
"""

def load_files(data_dir,columns_names,skiprows=0):
	"""
	Esta función tiene como objetivo cargar en memoria todos aquellos archi-
	vos de datos que pertenecen a un directorio particular para   posterior-
	mente combinarlos y retornarlos como uno solo.
	Input:
		- data_dir: String con la ruta en la cual se encuentra la carpeta a 
		analizar.
		- columns_names: Lista con el nombre de las columnas reales de  los
		archivos.
		- skiprows: Entero que indica cuantas filas se dejaran antes de en-
		contrar información en el archivo.
	Output:
		- full_data: Pandas DataFrame que contiene la información de  todos
		los archivos.
	"""
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
	"""
	Esta función se encarga de tomar las horas represantadas en formato  de
	columna y transponerlas a filas conservando el orden de las fechas.
	Input:
		- dataset: Pandas DataFrame con la información de una variable.
		- id_columns: Lista con los nombres de las columnas que se  manten-
		dran.
	Output:
		- transpose_dataframe: Pandas DataFrame con las columnas de las ho-
		ras transpuestas.
	"""
	transpose_dataframe = pd.melt(dataset,
		id_vars=id_columns,
		value_vars=[str(i) for i in np.arange(24)])

	return transpose_dataframe

def build_dataset(dataset,columns_to_sort):
	"""
	Esta función se encarga de sumar el valor de la fecha de un registro con
	el valor de la hora previamente transpuesto. Además ordena los registros
	en función de su fecha+hora. Para realizar este proceso, incialmente  se
	transforma  en  formato  TimeDelta  el  valor  de  la  hora y en formato 
	date_time la fecha, posteriormente se suman y finalmente se ordenan.
	Input:
		- dataset: Pandas DataFrame con la información de la variable.
		- columns_to_sort: Lista con las columnas que deben ser ordenadas.
	Output:
		- dataset: Pandas DataFrame con la información de la variable orde-
		nada por fecha+hora.
	"""
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
	"""
	Esta función se encarga de subdividir un Pandas DataFrame en objetos más
	pequeños de tal manera que al momento de guardarlos en excel no sobrepa-
	se el límite de filas permitidas. Para hacer esto inicialmente la funci-
	ónverifica que el número total de filas del DataFrame no se mayor que el
	límite permitido de filas en excel. En caso de ser mayor, calcula ccuan-
	tos conjuntos se pueden construir y procede a ejecutar la división.  Fi-
	nalmente, se guarda cada uno de los conjuntos.
	Input:
		- dataset: Pandas DataFrame con la información de la variable.
		- dataset_path: String con la ruta en la cual serán guardados   los
		archivos.
		- dataset_name: String con el nombre de los archivos.
	"""
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
	"""
	Esta  función  se  encarga de eliminar de los nombres de los recursos o
	agentes aquellos que tengan números, bien sea en formato Arabigo o   en
	formato Romano.
	Input:
		- dataset: Pandas DataFrame con la información de la variable.
		- column_to_preprocess: Nombre de las columnas que se realizará  el
		cambio de nombre.
	Output:
		- dataset: Pandas DataFrame con los nombres de los recursos o agen-
		tes correctamente formateados.
	"""
	dataset[column_to_preprocess] = dataset[column_to_preprocess].str.replace('\d+', '')
	dataset[column_to_preprocess] = dataset[column_to_preprocess].str.replace(r'(?=\b[MDCLXVI]+\b)M{0,4}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})', '')
	dataset[column_to_preprocess] = dataset[column_to_preprocess].map(lambda x: x.strip(' '))

	return dataset

def get_features_from_date(dataset):
	"""
	Esta función se encarga de extraer características de las fechas.
	Input:
		- dataset: Pandas DataFrame con la información de la variable.
	Output:
		- dataset: Pandas DataFrame al cual se le agrego la información
		de las características de las fechas agde la variable.
	"""
	weekday_or_weekend_lambda = lambda x: 1 if x>=5 else 0

	dataset = dataset.copy()

	dataset['Ano'] = dataset['Fecha'].dt.year
	dataset['Mes'] = dataset['Fecha'].dt.month
	dataset['Dia'] = dataset['Fecha'].dt.day
	dataset['Hora'] = dataset['Fecha'].dt.hour
	dataset['Dia del ano'] = dataset['Fecha'].dt.dayofyear
	dataset['Semana del ano'] = dataset['Fecha'].dt.weekofyear
	dataset['Dia de la semana'] = dataset['Fecha'].dt.dayofweek
	dataset['Trimestre'] = dataset['Fecha'].dt.quarter
	dataset['Es Fin de Semana'] = dataset['Dia de la semana'].map(weekday_or_weekend_lambda)

	return dataset

def save_data_files(file_path,file_name,data):
	"""
	Esta función se encarga de guardar un objeto Pandas Dataframe como 
	archivo de excel en una ruta determinada.
	Input:
		- file_path: String con la ruta donde se guarará el archivo.
		- file_name: String con el nombre con el cual se guardará  el
		archivo.
		- data: Pandas DataFrame con la información de la variable.
	"""
	if not os.path.exists(file_path):
		os.makedirs(file_path)

	file_path = os.path.join(file_path,'{}.xlsx'.format(file_name))
	data.to_excel(file_path,index=True)
	print('Archivo: {} Guardado con exito'.format(file_path))