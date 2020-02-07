import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Dense, MaxPooling2D, Flatten, Lambda
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from cif3r.features import preprocessing
from cif3r.data.recycling_guidelines import UNIVERSITIES
import numpy as np
from pathlib import Path
from PIL import Image
import sqlite3
import random
import time


AUTOTUNE = tf.data.experimental.AUTOTUNE
PROJECT_DIR = Path(__file__).resolve().parents[2]


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

    
def get_pairs(university:str='UTK', total_pairs:int=3200):
    data_dir = Path(__file__).resolve().parents[2] / "data/interim"
    conn = sqlite3.connect(str(data_dir / "metadata.sqlite3"))
    cur = conn.cursor()

    pairs, labels = [], []
    streams = [key for key in UNIVERSITIES[university]['R'].keys()]
    for stream in streams:
        print(f"Starting query for {stream}")
        query = f"""
        SELECT * FROM 
        (SELECT hash AS hash1, 
        (SELECT hash FROM {university} AS INNER 
        WHERE OUTER.stream = inner.stream
        AND OUTER.recyclable = inner.recyclable) 
        AS hash2, 1 FROM {university} AS OUTER
        WHERE stream='{stream}' 
        ORDER BY Random() 
        LIMIT {total_pairs // (len(streams)/2)}) 
        UNION ALL 
        SELECT * FROM 
        (SELECT hash AS hash1, 
        (SELECT hash FROM {university} AS INNER WHERE OUTER.stream != inner.stream) 
        AS hash2, 0 FROM {university} AS OUTER
        WHERE stream='{stream}'
        ORDER BY Random() LIMIT {total_pairs // (len(streams)/2)}) 
        """

        for name1, name2, label in cur.execute(query):
            pairs.append([name1, name2])
            labels.append(label)
    
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
  # use `.cache(filename)` to cache preprocessing work for datasets that don't
  # fit in memory

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
        optimizer=Adam(6e-5),
        loss='binary_crossentropy'
    )
    classes = [key for key in UNIVERSITIES['UTK']['R'].keys()]
    baseline = 26.0

    labeled_ds = get_pairs().map(
        process_path,
        num_parallel_calls=AUTOTUNE
        )

    for img1, img2, label in labeled_ds.take(1):
        print(img1.numpy().shape)
        print(img2.numpy().shape)
        print(label.numpy())

    train_ds = prepare_for_training(labeled_ds)

    def scoring(model, classes, N):
        n_correct = 0
        for i in range(N):
            *inputs, targets = next(iter(train_ds)) 
            probs = model.predict(inputs)
            if np.argmax(probs) == np.argmax(targets.numpy()):
                n_correct += 1
        percent_correct = (100*n_correct/N)
        return percent_correct
    time_start = time.time()        
    for i in range(1, 42000):
        *inputs, targets = next(iter(train_ds))
        loss = model.train_on_batch(inputs, targets)
        print(f'Training Loss (iteration {i}): {loss}')
        if i % 200 == 0:
            print(f'Training Loss: {loss}')
            val_acc = scoring(model, classes, 250)
            print(f'-----Validation Accuracy after {(time.time() - time_start)/60} min: {val_acc}')
            if val_acc > baseline:
                model.save()
                baseline = val_acc
