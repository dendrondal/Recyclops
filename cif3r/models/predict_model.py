import streamlit as st
import torch
from PIL import Image
from pathlib import Path
<<<<<<< HEAD
from protonet import ProtoNet
from dataset import transform
from torchvision import transforms
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
import numpy as np
import pickle
from itertools import chain
from torchsummary import summary
import pandas as pd
from learned_features import load_embeddings, embeddings_to_numpy


@st.cache
def load_model(university):
    model_dir = Path(__file__).resolve().parents[2] / "models"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    weights = torch.load(
        f"{model_dir}/{university}_best_model.pth", map_location=torch.device(device)
    )
    clf = ProtoNet()
    clf.load_state_dict(weights)
    clf.eval()
    return clf


@st.cache
def get_classes(university):
    data_dir = Path(__file__).resolve().parents[2] / "data/interim"
    dict_path = data_dir.resolve().parents[0] / "external/{}.pickle".format(university)
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

    unis = {"University of Tennessee": "UTK"}
    option = st.sidebar.selectbox(
        "Select your university:", ("University of Tennessee", "Penn State")
    )
    st.write("You selected:", option)
    model = load_model(unis[option])
    classes = get_classes(unis[option])
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")
    if uploaded_file is not None:
        image = transform(uploaded_file)
        image.unsqueeze_(0)
        st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
        with torch.no_grad():
            st.write("Classifying...")
            pca = PCA(n_components=27)
            X_query = model.forward(image)
            X_supp, y = embeddings_to_numpy(load_embeddings())
            X_supp = pca.fit_transform(X_supp)
            X_query = pca.transform(X_query)
            knn = KNeighborsClassifier(n_neighbors=5, weights="distance", n_jobs=-1)
            y_encoded = [i for i in range(len(y))]
            knn.fit(X_supp, y_encoded)
            st.write(y[knn.predict(X_query)[0]])
=======
from app import models
from cif3r.models.train_model import macro_f1, macro_f1_loss
>>>>>>> 431e0eed28d8db687d109a29a0fa434483ccaaa2

custom_deps = {'macro_f1': macro_f1, 'macro_f1_loss':macro_f1_loss}

<<<<<<< HEAD
if __name__ == "__main__":
    main()
=======
def clf_factory(university, img_array, labels):
    clf = tf.keras.models.load_model("/home/dal/CIf3R/models/{}.h5".format(university), custom_objects=custom_deps)
    x = np.expand_dims(img_array, axis=0)
    result = clf.predict(x)
    print(result)
    index = np.where(result[0] == np.amax(result[0]))[0][0]
    print(index)
    return labels[index]

imgs = {
    'paper':
    '/home/dal/CIf3R/data/interim/R/pieces_of_paper/5341292860.jpg',
    'metal':
    '/home/dal/CIf3R/data/interim/R/tin_can/567628875.jpg',
    'plastic':
    '/home/dal/CIf3R/data/interim/R/clean_plastic_bottle/363392046.jpg',
    'trash':
    '/home/dal/CIf3R/data/interim/O/plastic_bag/624999118.jpg'
    
}

>>>>>>> 431e0eed28d8db687d109a29a0fa434483ccaaa2
