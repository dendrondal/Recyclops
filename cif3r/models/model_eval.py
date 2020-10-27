import sqlite3
from dataclasses import dataclass
from pathlib import Path

import boto3
import click
import numpy as np
import pandas as pd
import seaborn as sb
import torch
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

from clustering import feedforward, load_image_paths, load_model
from dataset import transform
from learned_features import embeddings_to_numpy, load_embeddings


def plot_confusion_matrix(df, university, VIZ_DIR, conn):
    """
    Determines classification accuracy of distance-based 
    metric
    """
    labels = df["stream"].unique()
    le = LabelEncoder()
    le.fit(labels)
    print(df["y_hat"].apply(lambda x: le.inverse_transform(x)[0]).values[:5])
    con_mat = confusion_matrix(
        df["stream"].values,
        df["y_hat"].apply(lambda x: le.inverse_transform(x)[0]).values,
        labels=labels,
    )
    con_mat = con_mat / con_mat.max()
    figure = plt.figure(figsize=(10, 8))
    con_mat_df = pd.DataFrame(con_mat, index=labels, columns=labels)
    sb.heatmap(con_mat_df, annot=True)
    plt.tight_layout()
    plt.xlabel("True label")
    plt.ylabel("Predicted label")
    plot_name = VIZ_DIR / f"{university}_confustion_matrix.png"
    plt.savefig(plot_name, bbox_inches="tight")
    conn.upload_file(str(plot_name), "cif3r", f"{university}_con_mat")


def ETL(model, pca, conn, university):
    """
    Grabs ProtoNet features, performs PCA on them, and 
    outputs train-test split for model consumption.
    """
    le = LabelEncoder()
    df = load_image_paths(conn, str(university))
    print("Getting image embeddings...")
    df["embedding"] = df["hash"].apply(
        lambda x: feedforward(model, transform(x).unsqueeze_(0))
        .numpy()[0]
        .reshape(-1, 1)
    )
    df["X"] = df["embedding"].apply(lambda x: pca.transform(x)[1])
    y = le.fit_transform(df["stream"].values)
    df["y"] = list(y)
    print("ETL finished!")
    return df


def dim_reduction():
    X, y = embeddings_to_numpy(load_embeddings())
    pca = PCA(n_components=27)
    X_supp = pca.fit_transform(X)
    return pca, X_supp


def inference(df, clf):
    """
    Creates prediction column based on trained model
    """
    print("Performing KNN predictions...")
    df["y_hat"] = df["X"].apply(lambda x: clf.predict(x.reshape(1, -1)))
    print(df[["stream", "y_hat"]].head())
    return df


def breakpoint(conn, university):
    le = LabelEncoder()
    df = load_image_paths(conn, str(university))
    y = le.fit_transform(df["stream"].values)
    df["y"] = list(y)


@click.command()
@click.argument("university", default="UTK")
def main(university):
    """
    Glue function for visualiztion operations:
     - Produces confusion matrix of KNN predictions on representations
        learned by protonet
    """
    project_dir = Path(__file__).resolve().parents[2]
    viz_dir = project_dir / "reports" / "figures"
    dfdump = project_dir / "data" / "interim" / "df_dump.pkl"
    conn = sqlite3.connect(project_dir / "data" / "interim" / "metadata.sqlite3")
    breakpoint(conn, university)
    model = load_model(university)
    clf = KNeighborsClassifier(n_neighbors=5, weights="distance", n_jobs=-1)
    pca, X_supp = dim_reduction()
    if not dfdump.exists():
        df = ETL(model, pca, conn, university)
        # since this step takes so long, intermediate dataframe is saved to disk
        df.to_pickle(dfdump)
        X, y = df["X"].array, df["y"].array
    else:
        df = pd.read_pickle(dfdump)
    X, y = df["X"].to_numpy(), df["y"].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=42, train_size=0.5
    )
    clf.fit(
        np.array([arr.reshape(-1, 1) for arr in X_train])[:, :, 0], y_train.flatten()
    )
    pred_df = inference(df, clf)
    conn = boto3.client("s3")
    plot_confusion_matrix(pred_df, university, viz_dir, conn)


if __name__ == "__main__":
    main()
