from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, GlobalAveragePooling2D, Conv2D, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from pathlib import Path
import sqlite3
import click
import numpy as np
import random
from pathlib import Path
import pandas as pd
from cif3r.features.preprocessing import datagen, binary_datagen
from cif3r.models.custom_metrics import macro_f1, macro_f1_loss
from cif3r.data.recycling_guidelines import UNIVERSITIES
from cif3r.visualization.visualize import plot_confusion_matrix
from app.models import Models, ClassMapping


def ad_hoc_cnn(n_labels:int):
    """Custom CNN for non-transfer learning"""
    inputs = Input(shape=(400,400,1))
    x = Conv2D(32, (5,5), activation='relu')(inputs)
    x = MaxPooling2D((5,5))(x)
    x = Conv2D(64, (5,5), activation='relu')(x)
    x = Conv2D(128, (5,5), activation='relu')(x)
    x = MaxPooling2D((5,5))(x)
    x = Conv2D(256, (5,5), activation='relu')(x)
    x = MaxPooling2D((2,2))(x)
    x = Conv2D(64, (2,2), activation='relu')(x)
    x = Conv2D(32, (2,2), activation='relu')(x)
    x = MaxPooling2D((2,2))(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(2048, activation='tanh')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(n_labels, activation="softmax", name="output")(x)
    model = Model(inputs=inputs, outputs=predictions)
    return model
    

def load_base_model(depth: int, n_labels: int):
    """Loads in MobileNetV2 pre-trained on image net. Prevents layers until
    desired depth from being trained."""
    base_model = MobileNetV2(include_top=False)
    for layer in base_model.layers[:depth]:
        layer.trainable = False
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Conv2D(64, (5,5), activation='relu')(x)
    x = Conv2D(32, (5,5), activation='relu')(x)
    x = MaxPooling2D((5,5))(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.6)(x)
    x = Dense(2048, activation='tanh')(x)
    x = Dropout(0.7)(x)
    predictions = Dense(n_labels, activation="sigmoid", name="output")(x)
    model = Model(inputs=base_model.inputs, outputs=predictions)
    return model


def checkpoint(filename):
    return ModelCheckpoint(
        str(filename),
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        mode="auto",
        period=1,
    )


def write_model_data(university, class_mapping_dict):
    """Creates model metadata to be consumed by the frontend in choosing a prediction
    model and mapping output to classes"""

    db_path = Path(__file__).resolve().parents[2] / "data/interim/metadata.sqlite3"
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    insert = """INSERT INTO class_mapping 
    (university, label, key_index)
    VALUES (?,?,?)
    """
    for key, val in class_mapping_dict.items():
        cur.execute(insert, (university, key, val))
    conn.commit()


def early():
    return EarlyStopping(
        monitor="val_loss", min_delta=1e-3, patience=5, verbose=1, mode="auto"
    )


def tensorboard():
    return TensorBoard(
        log_dir=Path(__file__).resolve().parents[2] / "reports",
        histogram_freq=0,
        write_graph=True,
        write_images=False,
    )


def get_optimizer():
    """Helper function to map CLI argument to Keras method"""
    options = {"adam": optimizers.Adam, "rmsprop": optimizers.RMSprop, "sgd": optimizers.SGD, "nadam":optimizers.Nadam}
    return options


@click.command()
@click.argument(
    "university",
    required=True,
    type=click.Choice([key for key in UNIVERSITIES.keys()]),
)
@click.option(
    "--optimizer",
    default="rmsprop",
    type=click.Choice([key for key in get_optimizer()]),
)
@click.option("--lr", help="Learning rate passed to optimizer")
@click.option("--sampling", help="Whether to over- or under-sample training set")
@click.option("--batch_size", default=32)
@click.option(
    "--trainable_layers",
    default=1,
    help="How many layers at the end of MobileNet are trainable",
)
@click.option(
    "--loss",
    default="categorical_crossentropy",
    help="Loss metric used for model training. Valid options are the standard keras.optimizers, or macro_f1",
)
@click.option(
    "--plot_confusion",
    default=True,
    help="Whether to plot confusion matrix after training",
)
def train_model(
    university, optimizer, lr, sampling, batch_size, trainable_layers, loss, plot_confusion
):
    """Command line tool for model training. Loads image URIs from SQL metadata, 
    creates an augmented image generator, and loads in MobileNetV2. Trains over 300 epochs
    with early stopping condition based on validation loss (80-20 train-val split)"""
    #model = load_base_model( -int(trainable_layers), 1)
    model = ad_hoc_cnn(len([key for key in UNIVERSITIES['R'].keys])+1)
    if lr:
        optimizer = get_optimizer()[optimizer](lr=float(lr))
    if loss == "macro_f1" or "marco_f1_loss":
        loss = macro_f1_loss
    else:
        optimizer = get_optimizer()[optimizer]()
    project_dir = Path(__file__).resolve().parents[2]
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[tf.metrics.AUC(), macro_f1, "accuracy"],
    )
    print(model.summary())

    imagegen = ImageDataGenerator(
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
    df = binary_datagen(university)
    train = imagegen.flow_from_dataframe(df, batch_size=batch_size, color_mode='grayscale', target_size=(400,400), subset="training")
    validation = imagegen.flow_from_dataframe(
        df, batch_size=batch_size, color_mode='grayscale', target_size=(400,400), subset="validation"
    )

    model.fit(
        train,
        steps_per_epoch=train.samples // batch_size,
        epochs=300,
        validation_data=validation,
        validation_steps=validation.samples // batch_size,
        callbacks=[
            checkpoint((project_dir / "models" / f"{university}_binary.h5")),
            early(),
            tensorboard(),
        ],
    )
    write_model_data(university, train.class_indices)
    
    if plot_confusion:
        plot_confusion_matrix(university)


if __name__ == "__main__":
    train_model()
