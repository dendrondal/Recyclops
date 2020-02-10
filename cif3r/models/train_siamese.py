import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Dense, Dropout, MaxPooling2D, Flatten, Lambda
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, SGD
from cif3r.features import preprocessing
from cif3r.data.recycling_guidelines import UNIVERSITIES
import numpy as np
from pathlib import Path
from PIL import Image
import sqlite3
import random
from itertools import chain
import time
from tqdm import tqdm


AUTOTUNE = tf.data.experimental.AUTOTUNE
PROJECT_DIR = Path(__file__).resolve().parents[2]


def siamese_model():
    W_init = RandomNormal(stddev=1e-2, seed=42)
    b_init = RandomNormal(mean=0.5, stddev=1e-2, seed=42)
    input_shape = (105, 105, 3)
    
    img_input = Input(shape=input_shape)
    left_input = Input(shape=input_shape)
    right_input = Input(shape=input_shape)

    x = Conv2D(64,(10,10),activation='relu',input_shape=input_shape,
                    kernel_initializer=W_init,kernel_regularizer=l2(2e-4))(img_input)
    x = MaxPooling2D()(x)
    x = Conv2D(128,(7,7),activation='relu',
                    kernel_regularizer=l2(2e-4),kernel_initializer=W_init,bias_initializer=b_init)(x)
    x = MaxPooling2D()(x)
    x = Conv2D(128,(4,4),activation='relu',kernel_initializer=W_init,kernel_regularizer=l2(2e-4),bias_initializer=b_init)(x)
    x = MaxPooling2D()(x)
    x = Conv2D(
        256,(4,4),activation='relu',kernel_initializer=W_init, 
        kernel_regularizer=l2(2e-4),bias_initializer=b_init)(x)
    x = Flatten()(x)
    x = Dense(2048,  kernel_regularizer=l2(1e-3), 
        kernel_initializer=W_init,bias_initializer=b_init, activation='tanh')(x)
    x = Dropout(0.5)(x)
    out = Dense(
        4096, activation="sigmoid", kernel_regularizer=l2(1e-3), 
        kernel_initializer=W_init,bias_initializer=b_init)(x)

    twin = Model(img_input, out)
    
    #encode each of the two inputs into a vector with the convnet
    encoded_l = twin(left_input)
    encoded_r = twin(right_input)

    #merge two encoded inputs with the l1 distance between them
    L1_layer = Lambda(lambda x: tf.math.abs(x[0] - x[1]))
    L1_distance = L1_layer([encoded_l, encoded_r])

    prediction = Dense(1, activation='sigmoid', bias_initializer=b_init)(L1_distance)

    siamese_net = Model(inputs=[left_input, right_input], outputs=prediction)

    return siamese_net

    
def get_pairs(university:str='UTK', minority_cls_count:int=24, total_pairs:int=16000):
    data_dir = Path(__file__).resolve().parents[2] / "data/interim"
    conn = sqlite3.connect(str(data_dir / "metadata.sqlite3"))
    cur = conn.cursor()

    pairs, labels = [], []
    def _query():

        streams = list(chain.from_iterable([key for key in UNIVERSITIES[university]['R'].values()]))
        for subclass in streams:
            print(f"Starting query for {subclass}")
            query = f"""
            SELECT * FROM 
            (SELECT hash AS hash1, 
            (SELECT hash FROM {university} AS INNER 
            WHERE OUTER.subclass = inner.subclass
            AND OUTER.recyclable = inner.recyclable) 
            AS hash2, 1 FROM {university} AS OUTER
            WHERE subclass='{subclass}' 
            ORDER BY Random() 
            LIMIT {minority_cls_count // 2}) 
            UNION ALL 
            SELECT * FROM 
            (SELECT hash AS hash1, 
            (SELECT hash FROM {university} AS INNER WHERE OUTER.subclass != inner.subclass) 
            AS hash2, 0 FROM {university} AS OUTER
            WHERE subclass='{subclass}'
            ORDER BY Random() LIMIT {minority_cls_count // 2}) 
            """

            for name1, name2, label in cur.execute(query):
                pairs.append([name1, name2])
                labels.append(label)

    while len(pairs) < total_pairs:
        _query()
    
    print("Making dataset...")
    img1_tensor = tf.constant(pairs, shape=(len(pairs), 2))
    label_tensor = tf.constant(labels, shape=(len(labels), 1))
    return tf.data.Dataset.from_tensor_slices((img1_tensor, label_tensor))


def decode_img(img):
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return tf.image.resize(img, [105, 105])


def process_path(imgs, label):
    img1 = tf.io.read_file(imgs[0])
    img2 = tf.io.read_file(imgs[1])
    return decode_img(img1), decode_img(img2), label


def prepare_for_training(ds, batch_size=32, shuffle_buffer_size=1000):

  ds = ds.shuffle(buffer_size=shuffle_buffer_size)

  # Repeat forever
  ds = ds.repeat()

  ds = ds.batch(batch_size)

  # `prefetch` lets the dataset fetch batches in the background while the model
  # is training.
  ds = ds.prefetch(buffer_size=AUTOTUNE)

  return ds


if __name__ == '__main__':
    model = siamese_model()
    model.compile(
        optimizer=SGD(lr=0.01, momentum=0.9),
        loss='binary_crossentropy'
    )
    
    classes = list(chain.from_iterable([key for key in UNIVERSITIES['UTK']['R'].values()]))
    baseline = 100/23
    N = 16000
    train_labeled_ds = get_pairs(total_pairs=N).map(
        process_path,
        num_parallel_calls=AUTOTUNE
        )

    val_labeled_ds = get_pairs(total_pairs=N*0.6).map(
        process_path,
        num_parallel_calls=AUTOTUNE
        )
    for img1, img2, label in train_labeled_ds.take(1):
        print(img1.numpy().shape)
        print(img2.numpy().shape)
        print(label.numpy())

    train_ds = prepare_for_training(train_labeled_ds)
    val_ds = prepare_for_training(val_labeled_ds)

    def scoring(model, classes, N):
        n_correct = 0
        for i in tqdm(range(N)):
            *inputs, targets = next(iter(val_ds)) 
            probs = model.predict(inputs)
            if np.argmax(probs) == np.argmax(targets.numpy()):
                n_correct += 1
        percent_correct = (100*n_correct/N)
        return percent_correct
        
    time_start = time.time()
    for i in range(1, 13000):
        *inputs, targets = next(iter(train_ds))
        loss = model.train_on_batch(inputs, targets)
        print(f'Training Loss (iteration {i}): {loss}')
        if i % 400 == 0:
            print(f'Training Loss: {loss}')
            val_acc = scoring(model, classes, 320)
            print(f'-----Validation Accuracy after {(time.time() - time_start)/60} min: {val_acc}')
            if val_acc > baseline:
                model.save(str(PROJECT_DIR / 'models/UTK_siamese.h5'))
                model.save_weights(str(PROJECT_DIR / 'models/UTK_siamese_weights.h5'))
                baseline = val_acc
