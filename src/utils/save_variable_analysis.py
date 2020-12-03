import argparse
import numpy as np
import os
import pandas as pd

from . import preprocessing_utils
from . import utils

"""
Este script tiene como función construir un descriptivo de las variables dis-
ponibles en cada una de las sabanas de datos.
"""

def _find_characteristics(data_sheet):
	"""
	Esta función encuentra las caracteristicas más importantes en una sábana
	de datos. Entre la información que extrae está: Valor mínimo, valor  mí-
	nimo diferente de cero, valor máximo, media, mediana, desviación, %   de
	valores diferentes de cero, % de valores en cero.
	Input:
		- data_sheet: DataFrame de pandas con la información de una macro va-
		riable.
	Output:
		- data_sheet_df: DataFrame de pandas con el descriptivo    organizado
		por columnas.   
	"""
	data_sheet_dict = dict()
	for column in data_sheet.columns:
	    data_sheet_dict[column] = dict()

	    column_data_sheet = data_sheet[column]
	   
	    data_sheet_dict[column]['Valor Minimo'] = column_data_sheet.min()
	    data_sheet_dict[column]['Valor Minimo Diferente de Cero'] = column_data_sheet[column_data_sheet>0].min()
	    data_sheet_dict[column]['Valor Máximo'] = column_data_sheet.max()
	    data_sheet_dict[column]['Media'] = column_data_sheet.mean()
	    data_sheet_dict[column]['Mediana'] = column_data_sheet.median()
	    data_sheet_dict[column]['Desviación'] = column_data_sheet.std()
	    
	    mask_date = data_sheet[column] == 0
	    diferentes_de_cero = np.round(np.sum(~mask_date)*100/len(data_sheet),2)
	    iguales_cero = np.round(np.sum(mask_date)*100/len(data_sheet),2)
	    data_sheet_dict[column]['% Valores Diferentes de Cero'] = diferentes_de_cero
	    data_sheet_dict[column]['% Valores en Cero'] = iguales_cero

	columns=['Valor Minimo','Valor Minimo Diferente de Cero','Valor Máximo',
			 'Media','Mediana','Desviación','% Valores Diferentes de Cero',
			 '% Valores en Cero']
	data_sheet_df = pd.DataFrame(data_sheet_dict).T
	data_sheet_df = data_sheet_df.reindex(columns=columns)

	return data_sheet_df


def generate(params):
	"""
	Esta función se encarga de tomar cada una de las sábanas de datos y realiza los lla-
	mados pertinentes a la función que construye el dataframe con el descriptivo.
	Input:
		- params: Diccionario con los parámetros de configuración del proyecto.
	"""
	print('.....Load Files')

	predicted_variable_data_sheet_path = os.path.join(params['data_dir'],params['data_dir_series'],'Sabanas','Original','Sabana_Datos_Precio_Bolsa.xlsx')
	predicted_variable_data_sheet = pd.read_excel(predicted_variable_data_sheet_path)
	predicted_variable_data_sheet = predicted_variable_data_sheet.set_index('Fecha')
	
	daily_data_sheet_path = os.path.join(params['data_dir'],params['data_dir_series'],'Sabanas','Original','Sabana_Datos_Diaria.xlsx')
	daily_data_sheet = pd.read_excel(daily_data_sheet_path)
	daily_data_sheet = daily_data_sheet.set_index('Fecha')
	
	hourly_data_sheet_path = os.path.join(params['data_dir'],params['data_dir_series'],'Sabanas','Original','Sabana_Datos_Horaria.xlsx')
	hourly_data_sheet = pd.read_excel(hourly_data_sheet_path)
	hourly_data_sheet = hourly_data_sheet.set_index('Fecha')
	
	print('.....Make Process')

	predicted_variable_summary = _find_characteristics(predicted_variable_data_sheet)
	daily_data_sheet_summary = _find_characteristics(daily_data_sheet)
	hourly_data_sheet_summary = _find_characteristics(hourly_data_sheet)

	file_path = os.path.join(params['data_dir'],params['data_dir_series'],'Resumen')
	preprocessing_utils.save_data_files(file_path=file_path,file_name='Resumen_Datos_Precio_Bolsa',data=predicted_variable_summary)
	preprocessing_utils.save_data_files(file_path=file_path,file_name='Resumen_Datos_Diario',data=daily_data_sheet_summary)
	preprocessing_utils.save_data_files(file_path=file_path,file_name='Resumen_Datos_Horaria',data=hourly_data_sheet_summary)

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('-c', '--config', help="path to configuration file", default='config.yml')

	args = parser.parse_args()
	params = utils.yaml_to_dict(args.config)

	generate(params)