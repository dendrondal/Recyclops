from sklearn.metrics import confusion_matrix
from pathlib import Path
from tensorflow.keras.models import load_model


def plot_confusion_matrix(model_path:Path, prediction_variable:Union[str, Path]):
    pass