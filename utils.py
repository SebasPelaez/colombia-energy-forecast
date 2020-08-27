import yaml
import os

def yaml_to_dict(yml_path):
  with open(yml_path, 'r') as stream:

    try:
      params = yaml.load(stream,Loader=yaml.FullLoader)
    except yaml.YAMLError as exc:
      print(exc)

  return params