from pathlib import Path
import sqlite3

import click
import pandas as pd
import torch
import numpy as np

from dataset import transform
from protonet import ProtoNet


def load_model(university):
    """Loads the pre-trained protonet for the university"""
    model_dir = Path(__file__).resolve().parents[2] / "models"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    weights = torch.load(
        f"{model_dir}/{university}_best_model.pth", map_location=torch.device(device,)
    )
    clf = ProtoNet()
    clf.load_state_dict(weights)
    clf.eval()
    return clf


def load_image_paths(connection, university):
    """
    Makes dataframe of all image paths.
    """
    df = pd.read_sql(f"SELECT * FROM {university}", con=connection)
    return df


def filter_by_class(df, subcls: str):
    """Filters dataframe into one specific class"""
    return df.loc[df["subclass"] == subcls]


def feedforward(model, image_vector):
    """Creates 1x1600 embedding of image using saved model."""
    with torch.no_grad():
        return model.forward(image_vector)


def stacking(series):
    """Takes a dataframe with calculated embeddings, and
    creates a stacked tensor of the predictions so the
    centroid can be calculated."""
    arr = torch.stack([var.detach() for var in series.values])
    return torch.from_numpy(arr)


def calculate_centroid(support_vecs: torch.Tensor):
    """
    Calculates centroid for a given class.
    """
    centroid = torch.mean(torch.stack([vec.detach() for vec in support_vecs]), dim=0)
    return centroid


@click.command()
@click.argument("university", required=True)
def main(university):
    model = load_model(university)

    project_dir = Path(__file__).resolve().parents[2]
    data_path = project_dir / "data" / "interim"
    db_path = data_path / "metadata.sqlite3"
    conn = sqlite3.connect(db_path)

    output = dict()
    df = load_image_paths(conn, str(university))
    for stream in df["subclass"].unique():
        print(f"Getting embeddings for {stream}...")
        class_paths = filter_by_class(df, stream)
        class_paths["embedding"] = class_paths["hash"].apply(
            lambda x: feedforward(model, transform(x).unsqueeze_(0))
        )
        output = calculate_centroid(class_paths["embedding"].values)
        torch.save(
            {stream: output}, project_dir / "data" / "final" / f"{stream}_centroids.pt",
        )


if __name__ == "__main__":
    main()
