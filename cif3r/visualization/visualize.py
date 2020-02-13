import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import auc, roc_curve
from pathlib import Path
import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sqlite3
from PIL import Image
from scipy import interpolate

from cif3r.data import recycling_guidelines
from cif3r.models.train_model import macro_f1, macro_f1_loss
from cif3r.features import preprocessing


PARENT_DIR = Path(__file__).resolve().parents[2]
KAGGLE_DATA_PATH = PARENT_DIR / 'data/raw/DATASET/TEST'
MODEL_DIR = PARENT_DIR / 'models'
VIZ_DIR= PARENT_DIR / 'reports/figures'
DEPS = {'macro_f1_loss': macro_f1_loss, 'macro_f1': macro_f1}


def prediction_mapping(university: str):
    try:
        clf = tf.keras.models.load_model(
            MODEL_DIR / f"{university}.h5", custom_objects=DEPS
        )
    except OSError:
        raise Exception(
            f"Unable to find model. Valid  models include {MODEL_DIR.glob('*.h5')}"
        )
    df = preprocessing.datagen(university, balance_classes=False, verify_paths=True)
    df = df.sample(n=int(len(df) / 10), random_state=42)
    images = ImageDataGenerator().flow_from_dataframe(df, batch_size=64)
    y_hat = list(clf.predict(images))
    df["y_hat"] = y_hat
    return {"df": df, "labels": images.class_indices}


def plot_confusion_matrix(university: str):
    """For kaggle data, prediction class is organized by folder structure, but for scraped data, 
    sql metadata is used."""
    preds = prediction_mapping(university)
    df = preds["df"]
    df["y_hat"] = df["y_hat"].map(lambda x: np.where(x == np.amax(x))[0][0])
    df["y_hat"] = df["y_hat"].map(lambda x: list(preds["labels"].keys())[x])
    print(df.head())
    labels = list(preds['labels'].keys())
    con_mat = confusion_matrix(df['class'], df['y_hat'], labels=labels)
    figure = plt.figure(figsize=(10,8))
    con_mat_df = pd.DataFrame(con_mat, index=labels, columns=labels)
    sb.heatmap(con_mat_df, annot=True, cmap=plt.cm.Blues)
    plt.tight_layout()
    plt.xlabel("True label")
    plt.ylabel("Predicted label")
    plt.savefig(VIZ_DIR / f"{university}_confusion_matrix.png")


def plot_roc(university:str):
    """Creates ROC curve from average ROC value for each class"""
    preds = prediction_mapping(university)
    df = preds["df"]
    df["y_hat"] = df["y_hat"].map(lambda x: np.where(x == np.amax(x))[0][0])
    df["y_test"] = df["class"].map(lambda x: preds['labels'][x])
    labels = list(preds['labels'].values())
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for label in labels:
        y_hat = np.zeros(len(preds['df']))
        y_test = np.zeros(len(preds['df']))
        for i, (true, pred) in enumerate(df[['y_test', 'y_hat']].itertuples(index=False)):
            if true == label:
                y_test[i] = 1
            if pred == label:
                y_test[i] = 1
        print(y_hat, y_test)
        fpr[label], tpr[label], _ = roc_curve(y_test, y_hat)
        roc_auc[label] = auc(fpr[label], tpr[label])
    
    print(roc_auc)

    plt.figure()
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(
        fpr[label], 
        tpr[label], 
        label=f'ROC score for {i}: {roc_auc[label]}'
            )
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.savefig(VIZ_DIR / f"{university}_roc.png")


def make_visualizations():
    for university in recycling_guidelines.UNIVERSITIES.keys():
        plot_roc(university)


if __name__ == "__main__":
    make_visualizations()
