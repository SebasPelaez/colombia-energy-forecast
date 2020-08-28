import argparse
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from scipy import ndimage
from ..utils import utils

def _build_complete_df(geodf,variable):
    N = len(geodf)
    df_complete = geodf.copy().reset_index(drop=True) 
    for idx in range(N):
        fila_tmp = geodf.iloc[idx]
        value = fila_tmp[variable]
        df_complete = df_complete.append([fila_tmp]*int(value),ignore_index=True)
    return df_complete

def _build_heatmap_image_dataset(df_complete,variable=None,image_name=None,params=None):    
    def getx(pt):
        return pt.coords[0][0]

    def gety(pt):
        return pt.coords[0][1]

    x = list(df_complete['geometry'].apply(getx))
    y = list(df_complete['geometry'].apply(gety))

    heatmap, xedges, yedges = np.histogram2d(y, x, bins=(50,50))
    
    heatmap[heatmap == 0] = np.finfo(np.float32).eps
    
    extent = [yedges[0], yedges[-1], xedges[-1], xedges[0]]

    logheatmap = np.log(heatmap)
    logheatmap[np.isneginf(logheatmap)] = 0
    logheatmap = ndimage.gaussian_filter(logheatmap, 1.5, mode='nearest')
    
    climatic_dataset_folder = os.path.join(params['data_dir'],params['data_dir_climatic_images'],variable)
    if not os.path.exists(climatic_dataset_folder):
        os.makedirs(climatic_dataset_folder)
    
    plt.subplots(figsize=(8, 8))
    plt.imshow(logheatmap, cmap='jet', extent=extent)
    plt.gca().invert_yaxis()
    plt.axis('off')
    plt.savefig(os.path.join(climatic_dataset_folder,image_name), bbox_inches='tight')
    plt.close('all')

def generate(params):

    data_path = os.path.join(params['data_dir'],params['data_dir_climatic'],params['meteorologic_stations_data_file'])
    data = pd.read_csv(data_path)

    data['DATE'] =  pd.to_datetime(data['DATE'], format='%Y-%m-%d')
    data.sort_values(by=['DATE'], inplace=True)

    unique_dates = pd.unique(data['DATE'])

    iterator_idx = 0
    for variable in ['TAVG','PRCP']:
        for date in unique_dates:

            if (iterator_idx%5000) == 0:
                print('Se han construido {} imagenes'.format(iterator_idx))

            df_by_date = data[data['DATE'] == date]

            df_points = gpd.points_from_xy(df_by_date['LONGITUDE'], df_by_date['LATITUDE']) 
            geodf_colombia = gpd.GeoDataFrame(df_by_date,geometry=df_points)

            geodf_colombia[variable] = geodf_colombia[variable].fillna(0)
            if variable == 'PRCP':
                geodf_colombia[variable] = geodf_colombia[variable]*100

            df_complete = _build_complete_df(geodf_colombia,variable)

            img_name = str(date).split('T')[0]
            _build_heatmap_image_dataset(df_complete,variable,'{}.jpg'.format(img_name),params=params)

            iterator_idx += 1

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', help="path to configuration file", default='config.yml')

    args = parser.parse_args()

    params = utils.yaml_to_dict(args.config)

    generate(params)