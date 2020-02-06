import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Dense, MaxPooling2D, Flatten, Lambda
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from cif3r.features import preprocessing
from cif3r.data.recycling_guidelines import UNIVERSITIES
import numpy as np
from pathlib import Path
import sqlite3
import random


def siamese_model():
    W_init = RandomNormal(stddev=1e-2, seed=42)
    b_init = RandomNormal(mean=0.5, stddev=1e-2, seed=42)
    input_shape = (105, 105, 3)
    left_input = Input(shape=input_shape)
    right_input = Input(shape=input_shape)
    
    convnet = Sequential()
    convnet.add(Conv2D(64,(10,10),activation='relu',input_shape=input_shape,
                    kernel_initializer=W_init,kernel_regularizer=l2(2e-4)))
    convnet.add(MaxPooling2D())
    convnet.add(Conv2D(128,(7,7),activation='relu',
                    kernel_regularizer=l2(2e-4),kernel_initializer=W_init,bias_initializer=b_init))
    convnet.add(MaxPooling2D())
    convnet.add(Conv2D(128,(4,4),activation='relu',kernel_initializer=W_init,kernel_regularizer=l2(2e-4),bias_initializer=b_init))
    convnet.add(MaxPooling2D())
    convnet.add(Conv2D(256,(4,4),activation='relu',kernel_initializer=W_init,kernel_regularizer=l2(2e-4),bias_initializer=b_init))
    convnet.add(Flatten())
    convnet.add(Dense(4096,activation="sigmoid",kernel_regularizer=l2(1e-3),kernel_initializer=W_init,bias_initializer=b_init))
    
    #encode each of the two inputs into a vector with the convnet
    encoded_l = convnet(left_input)
    encoded_r = convnet(right_input)

    #merge two encoded inputs with the l1 distance between them
    L1_layer = Lambda(lambda x: tf.math.abs(x[0] - x[1]))
    L1_distance = L1_layer([encoded_l, encoded_r])

    prediction = Dense(1, activation='sigmoid', bias_initializer=b_init)(L1_distance)

    siamese_net = Model(inputs=[left_input, right_input], outputs=prediction)

    return siamese_net

    
def get_batch(stream, university:str='UTK', batch_size:int=32):
    data_dir = Path(__file__).resolve().parents[2] / "data/interim"
    conn = sqlite3.connect(str(data_dir / "metadata.sqlite3"))
    cur = conn.cursor()
    query = f"""
    SELECT * FROM 
    (SELECT hash AS hash1, 
    (SELECT hash FROM {university} AS INNER 
    WHERE OUTER.stream = inner.stream
    AND OUTER.recyclable = inner.recyclable) 
    AS hash2, 1 FROM {university} AS OUTER
    WHERE stream='{stream}' 
    ORDER BY Random() 
    LIMIT {batch_size/2}) 
    UNION ALL 
    SELECT * FROM 
    (SELECT hash AS hash1, 
    (SELECT hash FROM {university} AS INNER WHERE OUTER.stream != inner.stream) 
    AS hash2, 0 FROM {university} AS OUTER
    WHERE stream='{stream}'
    ORDER BY Random() LIMIT {batch_size/2}) 
    """

    pairs = []
    labels = []

    for name1, name2, label in cur.execute(query):
        pairs.append([name1, name2])
        labels.append(label)
    
    return pairs, labels



def preprocess_images(image_names, seed, datagen, image_cache):
    np.random.seed(seed)
    X = np.zeros((len(image_names), 105, 105, 3))
    for i, image_name in enumerate(image_names):
        image = cached_imread(os.path.join(IMAGE_DIR, image_name), image_cache)
        X[i] = datagen.random_transform(image)
    return X

def image_triple_generator(image_triples, batch_size):
    datagen_args = dict(
        validation_split=0.2,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        rescale=1./255,
        fill_mode="nearest",
    )
    datagen_left = ImageDataGenerator(**datagen_args)
    datagen_right = ImageDataGenerator(**datagen_args)
    image_cache = {}
    
    while True:
        # loop once per epoch
        num_recs = len(image_triples)
        indices = np.random.permutation(np.arange(num_recs))
        num_batches = num_recs // batch_size
        for bid in range(num_batches):
            # loop once per batch
            batch_indices = indices[bid * batch_size : (bid + 1) * batch_size]
            batch = [image_triples[i] for i in batch_indices]
            # make sure image data generators generate same transformations
            seed = np.random.randint(low=0, high=1000, size=1)[0]
            Xleft = preprocess_images([b[0] for b in batch], seed, 
                                      datagen_left, image_cache)
            Xright = preprocess_images([b[1] for b in batch], seed,
                                       datagen_right, image_cache)
            Y = np_utils.to_categorical(np.array([b[2] for b in batch]))
            yield Xleft, Xright, Y


if __name__ == '__main__':
    model = siamese_model()
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy'
    )
    classes = [key for key in UNIVERSITIES['UTK']['R'].keys()]
    for i in range(1, 9000):
        (inputs, targets) = get_batch(random.choice(classes))
        loss = model.train_on_batch(inputs, targets)
        if i % 200 == 0:
            print(loss)