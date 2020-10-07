import numpy as np
import os
import pandas as pd

def _load_files(data_dir_path,file_name_condition):

	data_list = list()
	for file_name in os.listdir(data_dir_path):
	    if file_name_condition in file_name:
	        file = pd.read_excel(os.path.join(data_dir_path,file_name))
	        data_list.append(file)
	        print(os.path.join(data_dir_path,file_name),file.shape)

	full_data = pd.concat(data_list)
	full_data = full_data[full_data['Fecha'].between('1999-12-31 23:59:59', '2020-04-01', inclusive=False)]

	return full_data


def _build_data_sheet(data_dir_path,file_name_condition,list_filter,pivot_index_column,pivot_stack_column,pivot_value_columns):

	full_data = _load_files(data_dir_path,file_name_condition)
	full_data_filter = full_data[full_data[pivot_stack_column].isin(list_filter)]
	data_sheet = full_data_filter.pivot(index=pivot_index_column, columns=pivot_stack_column, values=pivot_value_columns)
	
	return data_sheet


def _generate_hourly_data_sheet(params,resources_list,active_agents_list):

	generation_data_dir = os.path.join(params['data_dir'],params['data_dir_series'],'Generacion')
	demand_by_retailer_data_dir = os.path.join(params['data_dir'],params['data_dir_series'],'Demanda','Por Comercializador')

	generation_data_sheet = _build_data_sheet(data_dir_path = generation_data_dir,
											  file_name_condition='HIDRAULICA',
											  list_filter=resources_list,
											  pivot_index_column='Fecha',
											  pivot_stack_column='Recurso',
											  pivot_value_columns='kWh')
	generation_data_sheet.columns = generation_data_sheet.columns.name + ' Generacion ' + generation_data_sheet.columns

	demand_by_retailer_data_sheet = _build_data_sheet(data_dir_path=demand_by_retailer_data_dir,
											  file_name_condition='NO REGULADO',
											  list_filter=active_agents_list,
											  pivot_index_column='Fecha',
											  pivot_stack_column='Codigo Comercializador',
											  pivot_value_columns='kWh')
	demand_by_retailer_data_sheet.columns = demand_by_retailer_data_sheet.columns.name + ' ' + demand_by_retailer_data_sheet.columns

	hourly_data_sheet = pd.concat([generation_data_sheet,demand_by_retailer_data_sheet],axis=1)
	hourly_data_sheet = hourly_data_sheet.sort_index(ascending=True)
	hourly_data_sheet.fillna(0, inplace=True)

	return hourly_data_sheet

def _generate_daily_data_sheet(params,resources_list,active_agents_list,rivers_resources_merge_filtered):

	river_flow_data_dir = os.path.join(params['data_dir'],params['data_dir_series'],'Caudal')
	river_flow_data = _load_files(river_flow_data_dir,'Rios')
	river_flow_data_filter = river_flow_data[river_flow_data['Nombre Río'].isin(rivers_resources_merge_filtered['Nombre Río'])]
	river_flow_data_sheet = river_flow_data_filter.pivot(index='Fecha',
														 columns='Nombre Río',
														 values=['Aportes Caudal m3/s','Aportes Energía kWh','Aportes %'])

	demand_sin_dir = os.path.join(params['data_dir'],params['data_dir_series'],'Demanda','Energia SIN')
	demand_data_sheet = _load_files(demand_sin_dir,'Energia SIN')
	demand_data_sheet = demand_data_sheet.set_index('Fecha')
	demand_data_sheet.columns = 'Demanda ' + demand_data_sheet.columns

	offer_price_data_dir = os.path.join(params['data_dir'],params['data_dir_series'],'Precio','Oferta')
	offer_price_data = _load_files(offer_price_data_dir,'Oferta')
	offer_price_data_filtered = offer_price_data[offer_price_data['Recurso'].isin(resources_list)]
	offer_price_data_filtered = offer_price_data_filtered[offer_price_data_filtered['Código Agente'].isin(active_agents_list)]
	offer_price_data_sheet = offer_price_data_filtered.pivot(index='Fecha',
															 columns='Recurso',
															 values=['Precio de Oferta Ideal','Precio de Oferta de Despacho','Precio de Oferta Declarado'])

	daily_data_sheet = pd.concat([demand_data_sheet,river_flow_data_sheet,offer_price_data_sheet],axis=1)
	daily_data_sheet = daily_data_sheet.sort_index(ascending=True)
	daily_data_sheet.fillna(0, inplace=True)

	return daily_data_sheet

def _generate_predicted_variable_data_sheet(params):

	national_stock_exchange_dir = os.path.join(params['data_dir'],params['data_dir_series'],'Precio','Bolsa Nacional')
	national_stock_exchange_data_sheet = _load_files(national_stock_exchange_dir,'Bolsa Nacional')
	national_stock_exchange_data_sheet = national_stock_exchange_data_sheet.set_index('Fecha')
	
	national_stock_exchange_data_sheet = national_stock_exchange_data_sheet.sort_index(ascending=True)
	national_stock_exchange_data_sheet.fillna(0, inplace=True)

	return national_stock_exchange_data_sheet

def _save_data_sheet(params,file_name,data_sheet):

	file_path = os.path.join(params['data_dir'],params['data_dir_series'])
	if not os.path.exists(file_path):
		os.makedirs(file_path)

	file_path = os.path.join(file_path,'{}.xlsx'.format(file_name))
	data_sheet.to_excel(file_path,index=True)
	print('Archivo: {} Guardado con exito'.format(file_path))

def generate(params):
	resources_list = ['ALBAN','BETANIA','CHIVOR','EL QUIMBO','GUATAPE','GUATRON',
	'GUAVIO','LA TASAJERA','MIEL','PAGUA','PLAYAS','PORCE','SAN CARLOS',
	'SOGAMOSO','URRA']

	rivers_resources_merge_path = os.path.join(params['data_dir'],params['data_dir_xm'],'Cruce Rios-Recursos.xlsx')
	rivers_resources_merge = pd.read_excel(rivers_resources_merge_path)
	rivers_resources_merge_filtered = rivers_resources_merge[rivers_resources_merge['Recurso'].isin(resources_list)]

	agents_list_path = os.path.join(params['data_dir'],params['data_dir_xm'],'Listado_Agentes.xlsx')
	agents_list = pd.read_excel(agents_list_path,skiprows=3)
	active_agents_list = agents_list[agents_list['Estado'] == 'OPERACION']

	hourly_data_sheet = _generate_hourly_data_sheet(params,resources_list,active_agents_list['Código'])
	daily_data_sheet = _generate_daily_data_sheet(params,resources_list,active_agents_list,rivers_resources_merge_filtered)
	predicted_variable_data_sheet = _generate_predicted_variable_data_sheet(params)

	_save_data_sheet(params=params,file_name='Sabana_Datos_Horaria',data_sheet=hourly_data_sheet)
	_save_data_sheet(params=params,file_name='Sabana_Datos_Diaria',data_sheet=daily_data_sheet)
	_save_data_sheet(params=params,file_name='Sabana_Datos_Precio_Bolsa',data_sheet=predicted_variable_data_sheet)

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('-c', '--config', help="path to configuration file", default='config.yml')

	args = parser.parse_args()
	params = utils.yaml_to_dict(args.config)

	generate(params)