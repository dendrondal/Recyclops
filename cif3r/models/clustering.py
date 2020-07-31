import click
import torch
import sqlite3
import pandas as pd
from protonet import ProtoNet
from pathlib import Path
from dataset import transform


def load_model(university):
    """Loads the pre-trained protonet for the university"""
    model_dir = Path(__file__).resolve().parents[2]/'models'
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    weights = torch.load(
        f"{model_dir}/{university}_best_model.pth",
        map_location=torch.device(device)
    )
    clf = ProtoNet()
    clf.load_state_dict(weights)
    clf.eval()
    return clf

def load_image_paths(connection, university):
    """
    Makes dataframe of all image paths.
    """
    df = pd.read_sql(university, con=connection)
    return df

def filter_by_class(df, cls: str):
    """Filters dataframe into one specific class"""
    return df.loc[df['stream'] == cls]


def feedforward(model, image_vector):
    """Creates 1x1600 embedding of image using saved model."""
    return model.forward(image_vector)


def stacking(series):
    """Takes a dataframe with calculated embeddings, and
    creates a stacked tensor of the predictions so the
    centroid can be calculated."""
    return torch.tensor(series.values)


def calculate_centroid(support_vecs: torch.Tensor):
    """
    Calculates centroid for a given class.
    """
    centroid = torch.stack([vec.nonzero.squeeze(1).mean(0) for vec in support_vecs])
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
    df = load_image_paths(conn, university)
    for stream in df['subclass'].unique():
        class_paths = filter_by_class(df, stream)
        class_paths['embedding'] =\
            class_paths['subclass'].apply(
                lambda x: feedforward(
                    model, transform(x)
                )
            )
        tensor = stacking(class_paths['embedding'])
        output[stream] = calculate_centroid(model, tensor)

    torch.save(output, project_dir / "data" / "final" / f"{university}_centroids.pt")

if __name__ == '__main__':
    main()
