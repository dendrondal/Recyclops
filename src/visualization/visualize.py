from sklearn.metrics import confusion_matrix
from pathlib import Path
from tensorflow.keras.models import load_model
from src.models.predict_model import validation_generator
from typing import Union
import sqlite3

PARENT_DIR = Path(__file__).resolve().parents[2]
KAGGLE_DATA_PATH = PARENT_DIR / 'data/raw/DATASET/TEST'
MODEL_DIR = PARENT_DIR / 'models'
def get_data(prediction_variable: str):
    """Helper function to find all images of a certain class within data directory using metadata file"""
    pass

def plot_confusion_matrix(
    model_path:Path = MODEL_DIR / '2019-08-28 08:03:49.h5',
    prediction_variable:Union[str, Path]
    ):
    """For kaggle data, prediction class is organized by folder structure, but for scraped data, sql metadata is used."""
    clf = load_model(model_path)
    y_hat = clf.predict(validation_generator)
