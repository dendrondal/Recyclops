import tensorflow as tf
from PIL import Image
import numpy as np
from pathlib import Path
from app import models


def clf_factory(university, img_array)
    clf = tf.keras.models.load_model(Path(__file__).resolve().parents[1] / f'models/{university}.h5')
    x = np.expand_dims(img_array, axis=0)
    clf.predict(x)






