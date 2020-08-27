import argparse
import os
import wget
import zipfile

import utils
import generate_climate_data

def download_data(params):
	url_tar_file = params['url_dataset']

	if not os.path.exists(params['downloads_dir']):
	  	os.makedirs(params['downloads_dir'])

	wget.download(url_tar_file, params['downloads_dir'])

def extract_data(params):

	zip_file = os.path.join(params['downloads_dir'],params['compressed_data_name'])

	zip_ref = zipfile.ZipFile(zip_file, 'r')
	zip_ref.extractall(params['data_dir'])
	zip_ref.close()

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', help="path to configuration file", default='config.yml')

    args = parser.parse_args()

    params = utils.yaml_to_dict(args.config)

	download_data(params)
	extract_data(params)
	generate_climate_data.generate(params)