import streamlit as st
import torch
from PIL import Image
from pathlib import Path
from protonet import ProtoNet
from dataset import transform
from torchvision import transforms
import numpy as np
import pickle
from itertools import chain
from torchsummary import summary
import pandas as pd


@st.cache
def load_model(university):
    model_dir = Path(__file__).resolve().parents[2]/'models'
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    weights = torch.load(f"{model_dir}/{university}_best_model.pth", map_location=torch.device(device))
    clf = ProtoNet()
    clf.load_state_dict(weights)
    clf.eval()
    return clf


@st.cache
def get_classes(university):
    data_dir = Path(__file__).resolve().parents[2] / "data/interim"
    dict_path = data_dir.resolve().parents[0] / "external/{}.pickle".format(
            university
        )
    with open(dict_path, "rb") as f:
        uni = pickle.load(f)
    streams = list(chain.from_iterable([key for key in uni["R"].values()]))
    return streams


def main():
    st.write(
    """
    # Recyclops
    ## Keeping an eye on what you recycle
    """
    )

    unis = {'University of Tennessee': 'UTK'}
    option = st.sidebar.selectbox("Select your university:", ('University of Tennessee', 'Penn State'))
    st.write("You selected:", option)
    model = load_model(unis[option])
    classes = get_classes(unis[option])
    st.write(len(classes))
    uploaded_file = st.file_uploader("Choose an image...", type='jpg')
    if uploaded_file is not None:
        image = transform(uploaded_file)
        image.unsqueeze_(0)
        st.write(image.shape)
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
        with torch.no_grad():
            st.write('Classifying...')
            outputs = model.forward(image)
            df = pd.DataFrame(outputs)
            _, index = outputs.max(1)
            y_hat = classes[index.item() // 64]
            st.write(y_hat)


if __name__ == '__main__':
    main()
