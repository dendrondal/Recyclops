import tensorflow as tf
from PIL import Image
import numpy as np
from pathlib import Path
from app import models
from cif3r.models.train_model import macro_f1_loss, macro_f1

custom_metrics = {'macro_f1_loss': macro_f1_loss, 'macro_f1': macro_f1}

def clf_factory(university, img_array, labels):
    print(university)
    clf = tf.keras.models.load_model("/home/dal/CIf3R/models/{}.h5".format(university), custom_objects=custom_metrics)
    x = np.expand_dims(img_array, axis=0)
    result = clf.predict(x)
    index = np.where(result == np.amax(result))[0][0]
    return labels[index]
