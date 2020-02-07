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
from PIL import Image
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

    pairs = dict()
    labels, input1, input2 = [], [], []

    for name1, name2, label in cur.execute(query):
        input1.append(name1)
        input2.append(name2)
        labels.append(label)
    
    if len(input1) != len(input2):
        get_batch(stream, university, batch_size)

    pairs['input_1'] = load_images(input1)
    pairs['input_2'] = load_images(input2)

    return pairs, np.array(labels)


def load_images(filenames):
    image_queue = tf.data.Dataset.list_files(filenames)
    
    def _decode_img(file_path):
        img = tf.io.read_file(file_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        return tf.image.resize(img, [105, 105])

    imgs = [_decode_img(file) for file in image_queue]
    batch = tf.stack(imgs, 0)
    print(batch.shape)
    return batch


def scoring(model, classes, N):
    n_correct = 0
    for i in range(N):
        for stream in classes:
            inputs, targets = get_batch(stream, batch_size=8) 
        probs = model.predict(inputs)
        if np.argmax(probs) == np.argmax(targets):
            n_correct += 1
    percent_correct = (100*n_correct/N)
    return percent_correct

            
if __name__ == '__main__':
    model = siamese_model()
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy'
    )
    classes = [key for key in UNIVERSITIES['UTK']['R'].keys()]
    baseline = 26.0
    for i in range(1, 42000):
        (inputs, targets) = get_batch(random.choice(classes))
        loss = model.train_on_batch(inputs, targets)
        if i % 2 == 0:
            print(f'Training Loss: {loss}')
            val_acc = scoring(model, classes, 2)
            print(f'Validation Accuracy: {val_acc}')