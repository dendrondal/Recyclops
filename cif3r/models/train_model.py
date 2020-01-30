from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from pathlib import Path
import sqlite3
import numpy as np
import random
import pandas as pd
from cif3r.features.preprocessing import datagen, calc_class_weights

 
def load_base_model(depth: int, n_labels:int):
    """Loads in MobileNetV2 pre-trained on image net. Prevents layers until
    desired depth from being trained."""
    base_model = MobileNetV2(include_top=False)
    for layer in base_model.layers[:depth]:
        layer.trainable = False
    x = base_model.output
    x = Dropout(0.5)(x)
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(n_labels, activation="sigmoid", name="output")(x)
    model = Model(inputs=base_model.inputs, outputs=predictions)
    return model


def checkpoint(filename):
    return ModelCheckpoint(
        str(filename),
        monitor="macro_f1",
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        mode="auto",
        period=1,
    )   


def write_model_data(university, model_name, class_mapping_dict):
    """Creates model metadata to be consumed by the frontend in choosing a prediction
    model and mapping output to classes"""

    data_dir = Path(__file__).resolve().parents[2] / 'data/interim'
    conn = sqlite3.connect(str(data_dir/'metadata.sqlite3'))
    cur = conn.cursor()
    init = """CREATE TABLE IF NOT EXISTS models (
        university text PRIMARY KEY,
        model_name text NOT NULL
    )
    """
    cur.execute(init)
    conn.commit()

    subtbl = """ CREATE TABLE IF NOT EXISTS class_mapping (
        university text PRIMARY KEY,
        label text NOT NULL,
        index int NOT NULL
    )"""
    cur.execute(subtbl)
    conn.commit()

    insert ="""INSERT INTO models 
    (university, model_name) 
    VALUES (?,?)
    """
    cur.execute(insert, (university, model_name))
    conn.commit()

    insert = """INSERT INTO class_mapping 
    (university, label, index)
    VALUES (?,?,?)
    """

    for key, val in class_mapping_dict.items():
        cur.execute(insert, (university, key, val))
        conn.commit()


def early():
    return EarlyStopping(
        monitor="macro_f1", min_delta=1e-3, patience=10, verbose=1, mode="max"
    )


def tensorboard():
    return TensorBoard(
        log_dir=Path(__file__).resolve().parents[2] / 'reports', histogram_freq=0, write_graph=True, write_images=False
    )


@tf.function
def macro_f1_loss(y, y_hat):
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

    Args:
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

    model = load_base_model(-4, 4)
    model.compile(
        optimizer=optimizers.RMSprop(), 
        loss=macro_f1_loss,
        metrics=[tf.metrics.AUC(), macro_f1, 'accuracy']
         )

    df = datagen(UNI)
    class_weights = calc_class_weights(df)

    imagegen = ImageDataGenerator(
        validation_split=0.2,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
        )
    data = imagegen.flow_from_dataframe(df)
    encoded_weights = {data.class_indices[key]: val for key, val in class_weights.items()}
    model.fit(
        data,
        steps_per_epoch=128,
        epochs=300,
        class_weight = encoded_weights,
        callbacks=[
            checkpoint(
                (project_dir / "models" / f"{UNI}.h5")
            ),
            early(),
            tensorboard(),
        ],
    )
    model_path = str(project_dir / "models" / f"{UNI}.h5")
    write_model_data(UNI, model_path, data.class_indices)
