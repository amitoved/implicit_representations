import os
_constants_path = os.path.realpath(__file__)
_constants_dirname = os.path.realpath(os.path.dirname(_constants_path))
PROJECT_DIR = os.path.realpath(os.path.join(_constants_dirname, os.path.pardir))
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
MODELS_DIR = os.path.join(PROJECT_DIR, 'models')


