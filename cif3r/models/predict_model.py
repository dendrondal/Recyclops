import tensorflow as tf
from PIL import Image
import numpy as np
from pathlib import Path
from app import models
from cif3r.models.train_model import macro_f1, macro_f1_loss

custom_deps = {'macro_f1': macro_f1, 'macro_f1_loss':macro_f1_loss}

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

