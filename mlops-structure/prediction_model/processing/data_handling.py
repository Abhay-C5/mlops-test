import os
import pathlib
import pandas as pd 
from pathlib import Path
import os
import sys

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent.parent
sys.path.append(str(PACKAGE_ROOT))

from prediction_model.config import config

#function to load the data
def load_dataset(file_name):
    filepath = os.path.join(config.DATAPATH, file_name)
    _data = pd.read_csv(filepath)
    return _data

#function to save pickle file from model i.e. serialization of the model
def save_pipeline(pipeline_to_save):
    save_path = os.path.join(config.SAVE_MODEL_PATH, config.MODEL_NAME)
    joblib.dump(pipeline_to_save, save_path)
    print(f"Your Model has been saved successfully under the name {config.MODEL_NAME}")

#function to load the pickle i.e. desearialization of the model
def load_pipeline(pipeline_to_load):
    load_path = os.path.join(config.SAVE_MODEL_PATH, config.MODEL_NAME)
    model_loaded = joblib.load(load_path)
    print(f"Model has been loaded")
    return model_loaded