import yaml
import os

"""
Script con funciones genéricas que pueden ser usadas en cualquier etapa
del proyecto.
"""

def yaml_to_dict(yml_path):
	"""
	Esta función se encarga de cargar un archivo en formato .yml y retor-
	nar un diccionario con cada uno de los elementos allí incluidos.
	Input:
		- yml_path: String con la ruta del archivo .yml
	Output:
		- params: Diccionario con las configuraciones del archivo.
	"""
	with open(yml_path, 'r') as stream:
		try:
			params = yaml.load(stream,Loader=yaml.FullLoader)
		except yaml.YAMLError as exc:
			print(exc)
	return params
