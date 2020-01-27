from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from pathlib import Path
import sqlite3
import numpy as np
import random
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

 
def train_val_split(university:str, test_size:float=0.2):
    data_dir = Path(__file__).resolve().parents[2] / 'data/interim'
    conn = sqlite3.connect(str(data_dir/'metadata.sqlite3'))
    cur = conn.cursor()
    query = """
    SELECT
        hash,
        CASE
            WHEN(recyclable = 'O') THEN 'trash'
            WHEN(recyclable = 'R') THEN stream
        END
    FROM {}
    """.format(university)
    imgs, labels = [], []
    for img, label in cur.execute(query):
        imgs.append(img), labels.append(label)
    return train_test_split(imgs, labels, test_size=test_size, random_state=42)

def label_encoding(y_train, y_val):
    mlb = MultiLabelBinarizer()
    mlb.fit([y_train])
    return mlb.transform(y_train), mlb.transform(y_val)


def datagen(university:str):
    data_dir = Path(__file__).resolve().parents[2] / 'data/interim'
    conn = sqlite3.connect(str(data_dir/'metadata.sqlite3'))
    query = """
    SELECT hash, 
    CASE WHEN(recyclable = 'O') THEN 'trash' 
    WHEN(recyclable = 'R') THEN stream 
    END FROM {}
    """.format(university)

    df = pd.read_sql(sql=query, con=conn)
    df.columns = ['filename', 'class']
    return df


def process_path(file_paths):
    for file_path in file_paths:
        img = tf.io.read_file(file_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = img / 255.0
        yield tf.image.resize(img, [224, 224])


def labeled_ds():
    list_ds = [i for i in datagen()]
    random.shuffle(list_ds)
    for datum in list_ds:
        yield process_path(datum)


def create_dataset(Xs, ys):
  # This is a small dataset, only load it once, and keep it in memory.
  # use `.cache(filename)` to cache preprocessing work for datasets that don't
  # fit in memory.
    return [X for X in Xs], [y for y in ys]


def load_base_model(depth: int, n_labels:int):
    """Loads in MobileNetV2 pre-trained on image net. Prevents layers until
    desired depth from being trained."""
    base_model = MobileNetV2(include_top=False)
    for layer in base_model.layers[:depth]:
        layer.trainable = False
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(n_labels, activation="sigmoid", name="output")(x)
    model = Model(inputs=base_model.inputs, outputs=predictions)
    return model


def checkpoint(filename):
    return ModelCheckpoint(
        str(filename),
        monitor="val_acc",
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        mode="auto",
        period=1,
    )   
    


def early():
    return EarlyStopping(
        monitor="val_accuracy", min_delta=0, patience=10, verbose=1, mode="auto"
    )


def tensorboard():
    return TensorBoard(
        log_dir="../../reports", histogram_freq=0, write_graph=True, write_images=False
    )


@tf.function
def macro_soft_f1(y, y_hat):
    """Compute the macro soft F1-score as a cost (average 1 - soft-F1 across all labels).
    Use probability values instead of binary predictions.
    
    Args:
        y (int32 Tensor): targets array of shape (BATCH_SIZE, N_LABELS)
        y_hat (float32 Tensor): probability matrix from forward propagation of shape (BATCH_SIZE, N_LABELS)
        
    Returns:
        cost (scalar Tensor): value of the cost function for the batch
    """
    y = tf.cast(y, tf.float32)
    y_hat = tf.cast(y_hat, tf.float32)
    tp = tf.reduce_sum(y_hat * y, axis=0)
    fp = tf.reduce_sum(y_hat * (1 - y), axis=0)
    fn = tf.reduce_sum((1 - y_hat) * y, axis=0)
    soft_f1 = 2*tp / (2*tp + fn + fp + 1e-16)
    cost = 1 - soft_f1 # reduce 1 - soft-f1 in order to increase soft-f1
    macro_cost = tf.reduce_mean(cost) # average on all labels
    return macro_cost


@tf.function
def macro_f1(y, y_hat, thresh=0.5):
    """Compute the macro F1-score on a batch of observations (average F1 across labels)
    key)
    Args:key)
        y (int32 Tensor): labels array of shape (BATCH_SIZE, N_LABELS)
        y_hat (float32 Tensor): probability matrix from forward propagation of shape (BATCH_SIZE, N_LABELS)
        thresh: probability value above which we predict positive
        
    Returns:
        macro_f1 (scalar Tensor): value of macro F1 for the batch
    """
    y_pred = tf.cast(tf.greater(y_hat, thresh), tf.float32)
    tp = tf.cast(tf.math.count_nonzero(y_pred * y, axis=0), tf.float32)
    fp = tf.cast(tf.math.count_nonzero(y_pred * (1 - y), axis=0), tf.float32)
    fn = tf.cast(tf.math.count_nonzero((1 - y_pred) * y, axis=0), tf.float32)
    f1 = 2*tp / (2*tp + fn + fp + 1e-16)
    macro_f1 = tf.reduce_mean(f1)
    return macro_f1


if __name__ == "__main__":
    project_dir = Path(__file__).resolve().parents[2]
    UNI = 'UTK'
    X_train, X_val, y_train, y_val = train_val_split('UTK')
    y_train_bin, y_val_bin = label_encoding(y_train, y_val)
    val_ds = create_dataset(X_val, y_val_bin)

    model = load_base_model(-10, len(y_train_bin[0]))
    optimizer = Adam(1e-5)
    model.compile(
        optimizer="adam", 
        loss="binary_crossentropy",
        metrics=[tf.metrics.AUC()]
         )

    df = datagen(UNI)
    print(df.columns)

    model.fit(
        ImageDataGenerator(validation_split=0.2).flow_from_dataframe(df, batch_size=64),
        steps_per_epoch=256,
        epochs=300,
        callbacks=[
            checkpoint(
                (project_dir / "models" / "UTK.h5")
            ),
            early(),
            tensorboard(),
        ],
    )