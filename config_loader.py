# config_loader.py

import yaml

def load_config(config_file='config.yml'):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config
