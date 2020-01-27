import tensorflow as tf
from pathlib import Path
import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sqlite3
from cif3r.models.train_model import train_val_split

PARENT_DIR = Path(__file__).resolve().parents[2]
KAGGLE_DATA_PATH = PARENT_DIR / 'data/raw/DATASET/TEST'
MODEL_DIR = PARENT_DIR / 'models'
VIZ_DIR= PARENT_DIR / 'reports/figures'


def plot_confusion_matrix(university:str):
    """For kaggle data, prediction class is organized by folder structure, but for scraped data, sql metadata is used."""
    X_train, y_train, X_val, y_val = train_val_split(university)
    X_val = X_val.reshape((64, 244, 244, 3))
    clf = tf.keras.load_model(MODEL_DIR / university)
    y_hat = clf.predict_classes(X_val)
    con_mat = tf.math.confusion_matrix(labels=y_val, predictions=y_hat).numpy()
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



