import torch
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
import pandas as pd
from protonet import ProtoNet
from pathlib import Path
from dataset import transform 


def load_model(university):
    model_dir = Path(__file__).resolve().parents[2]/'models'
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    weights = torch.load(f"{model_dir}/{university}_best_model.pth", map_location=torch.device(device))
    clf = ProtoNet()
    clf.load_state_dict(weights)
    clf.eval()
    return clf


def make_embeddings(model, connection): 
    """
    Creates 1600-length embeddings using saved PyTorch model. Saves them to
    orthogonal table.
    """
    df = pd.read_sql(f"FROM {university} import *", con=connection)
    model_dir = Path(__file__).resolve().parents[2]/'models'
    df['embedding'] = df['hash'].apply(lambda x:
                                       model.forward(transform(x).unsqueeze_(0)))
   df['embedding'] 
