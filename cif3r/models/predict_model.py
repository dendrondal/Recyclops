import tensorflow as tf
from PIL import Image
import numpy as np
from pathlib import Path
from app import models
from cif3r.models.custom_metrics import macro_f1_loss, macro_f1

custom_metrics = {"macro_f1_loss": macro_f1_loss, "macro_f1": macro_f1}


def clf_factory(university, img_array, labels):
    model_dir = Path(__file__).resolve().parents[1]
    clf = tf.keras.models.load_model(
        f"{model_dir}/{university}.h5", custom_objects=custom_metrics
    )
    x = np.expand_dims(img_array, axis=0)
    result = clf.predict(x)
    index = np.where(result == np.amax(result))[0][0]
    return labels[index]
