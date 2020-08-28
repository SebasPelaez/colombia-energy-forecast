import argparse
import os
import wget
import zipfile

import src.utils.utils as utilities
import src.build_dataset.climate_data as climate_data

from src.build_dataset import commercial_demand_by_retailer
from src.build_dataset import demand_SIN
from src.build_dataset import dispatch_scheduled_agc
from src.build_dataset import generation
from src.build_dataset import national_stock_exchange_price
from src.build_dataset import offer_price

def download_data(url_tar_file,downloads_dir):
	if not os.path.exists(downloads_dir):
	  	os.makedirs(downloads_dir)

	wget.download(url_tar_file, downloads_dir)

def extract_data(params,compressed_file_name):

	zip_file = os.path.join(params['downloads_dir'],compressed_file_name)

	zip_ref = zipfile.ZipFile(zip_file, 'r')
	zip_ref.extractall(params['data_dir'])
	zip_ref.close()

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('-c', '--config', help="path to configuration file", default='config.yml')
	parser.add_argument('-full_download', '--full_download',
	 help="Set Y if you want to download, extract and preprocess data from raw files, otherwise set N if you only want extract preprocess data",
	 choices=['Y','N'],
	 default='N')

	args = parser.parse_args()
	params = utilities.yaml_to_dict(args.config)

	download_data(params['full_raw_dataset_url'],params['downloads_dir'])
	extract_data(params,params['compressed_full_raw_dataset'])

	if args.full_download == 'Y':
		climate_data.generate(params)
		commercial_demand_by_retailer.generate(params)
		demand_SIN.generate(params)
		dispatch_scheduled_agc.generate(params)
		generation.generate(params)
		national_stock_exchange_price.generate(params)
		offer_price.generate(params)
	else:
		download_data(params['preprocess_dataset_url'],params['downloads_dir'])
		extract_data(params,params['compressed_preprocess_dataset'])