import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
from pathlib import Path
import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sqlite3
from PIL import Image
import networkx as nx
from cif3r.data import recycling_guidelines
from cif3r.features import preprocessing
from cif3r.models import custom_metrics
from cif3r.data.recycling_guidelines import UNIVERSITIES


PARENT_DIR = Path(__file__).resolve().parents[2]
KAGGLE_DATA_PATH = PARENT_DIR / "data/raw/DATASET/TEST"
MODEL_DIR = PARENT_DIR / "models"
VIZ_DIR = PARENT_DIR / "reports/figures"
DEPS = {
    "macro_f1_loss": custom_metrics.macro_f1_loss,
    "macro_f1": custom_metrics.macro_f1,
}
COLORS = {"UTK": "orange", "penn_state": "b"}

def prediction_mapping(university: str):
    try:
        clf = tf.keras.models.load_model(
            MODEL_DIR / f"{university}.h5", custom_objects=DEPS
        )
    except OSError:
        raise Exception(
            f"Unable to find model. Valid  models include {MODEL_DIR.glob('*.h5')}"
        )
    df = preprocessing.sample_all(university, verify_paths=True)
    #df = df.sample(n=int(len(df) / 5), random_state=42)
    images = ImageDataGenerator().flow_from_dataframe(df, batch_size=64)
    y_hat = list(clf.predict(images))
    df["y_hat"] = y_hat
    return {"df": df, "labels": images.class_indices}


def plot_guideline_network(university: str):
    G = nx.Graph()
    G.add_node(university, color='b')
    for key in UNIVERSITIES[university]['R']:
        G.add_node(key, color='b')
        G.add_edge(university, key)
        for sub in UNIVERSITIES[university]['R'][key]:
            G.add_node(sub, color='g')
            G.add_edge(key, sub)
    for key in UNIVERSITIES[university]['O']:
        G.add_node(key, color='b')
        G.add_edge(university, key)
        for sub in UNIVERSITIES[university]['O'][key]:
            G.add_node(sub, color='r')
            G.add_edge(key, sub)
    colors = [color for color in nx.get_node_attributes(G, 'color').values()]
    plt.figure(figsize=(13,13))
    plt.tight_layout()
    nx.draw_networkx(G, node_color=colors)
    plt.savefig(VIZ_DIR / f'{university}_network')


def plot_class_dist(university:str):
    conn = sqlite3.connect(str(PARENT_DIR / 'data/interim/metadata.sqlite3'))
    df = pd.read_sql("SELECT * FROM {}".format(university), conn)
    plt.figure()
    plt.tight_layout()
    df.groupby(['stream']).size().plot(kind='bar', color=COLORS[university])
    plt.savefig(VIZ_DIR / f'{university}_stream_histogram.png', bbox_inches='tight')


def plot_confusion_matrix(university: str):
    """For kaggle data, prediction class is organized by folder structure, but for scraped data, 
    sql metadata is used."""
    preds = prediction_mapping(university)
    df = preds["df"]
    df["y_hat"] = df["y_hat"].map(lambda x: np.where(x == np.amax(x))[0][0])
    df["y_hat"] = df["y_hat"].map(lambda x: list(preds["labels"].keys())[x])
    labels = list(preds["labels"].keys())
    con_mat = confusion_matrix(df["class"], df["y_hat"], labels=labels)
    figure = plt.figure(figsize=(10, 8))
    print(f"Plotting confusion matrix for {university}...")
    con_mat_df = pd.DataFrame(con_mat, index=labels, columns=labels)
    sb.heatmap(con_mat_df, annot=True, cmap=plt.cm.Blues)
    plt.tight_layout()
    plt.xlabel("True label")
    plt.ylabel("Predicted label")
    plt.savefig(VIZ_DIR / f"{university}_confusion_matrix.png")


def make_visualizations():
    for university in recycling_guidelines.UNIVERSITIES.keys():
        plot_class_dist(university)


if __name__ == "__main__":
    make_visualizations()
