import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pathlib import Path
import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sqlite3
from cif3r.models.train_model import datagen, process_path

PARENT_DIR = Path(__file__).resolve().parents[2]
KAGGLE_DATA_PATH = PARENT_DIR / 'data/raw/DATASET/TEST'
MODEL_DIR = PARENT_DIR / 'models'
VIZ_DIR= PARENT_DIR / 'reports/figures'


def plot_confusion_matrix(university:str):
    """For kaggle data, prediction class is organized by folder structure, but for scraped data, sql metadata is used."""
    df = datagen(university)
    X_val = df.sample(n=int(len(df)/5), random_state=42)
    clf = tf.keras.models.load_model(MODEL_DIR / f'{university}.h5')
    df['y_hat'] = clf.predict(process_path(df['filename']))
    con_mat = tf.math.confusion_matrix(
        labels=df['class'],
        predictions=df['y_hat']
        ).numpy()
    con_mat_norm = np.around(
        con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], 
        decimals=2
        )
    con_mat_df = pd.DataFrame(con_mat_norm, index=classes, columns=classes)
    figure = plt.figure(figsize=(8,8))
    sns.heatmap(con_mat_df, annot=True, cmap=plt.cm.Blues)
    plt.tight_layout()
    plt.xlabel('True label')
    plt.ylabel('Predicted label')
    plt.savefig(f'{university}_confusion_matrix.png')



