import tensorflow.keras as keras
from keras.applications.mobilenet_v2 import MobileNetV2
from datetime import datetime
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import RMSprop, Adam
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from pathlib import Path
import click


def load_base_model(depth: int):
    """Loads in MobileNetV2 pre-trained on image net. Prevents layers until
    desired depth from being trained."""
    base_model = MobileNetV2(include_top=False)
    for layer in base_model.layers[:depth]:
        layer.trainable = False
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.inputs, outputs=predictions)
    return model


def rmsprop(lr: float, decay: float):
    return RMSprop(lr=lr, decay=decay)

def adam(lr: float):
    return Adam(lr=lr)

def train_generator(training_data_directory):
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        horizontal_flip=True,
        fill_mode="nearest",
        zoom_range=0.3,
        width_shift_range=0.3,
        height_shift_range=0.3,
        rotation_range=30
    )
    return train_datagen.flow_from_directory(
        training_data_directory,
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary'
    )


def validation_generator(validation_data_directory):
    test_datagen = ImageDataGenerator(
        rescale=1. / 255,
        horizontal_flip=True,
        fill_mode="nearest",
        zoom_range=0.3,
        width_shift_range=0.3,
        height_shift_range=0.3,
        rotation_range=30
    )
    return test_datagen.flow_from_directory(
        validation_data_directory,
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary'
    )


def checkpoint(filename):
    return ModelCheckpoint(
        str(filename),
        monitor='val_acc',
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        mode='auto',
        period=1
    )


def early():
    return EarlyStopping(monitor='val_acc', min_delta=0, patience=10,
                         verbose=1, mode='auto')


def tensorboard():
    return TensorBoard(log_dir='../../reports', histogram_freq=0,
                       write_graph=True, write_images=False)


if __name__ == '__main__':
    project_dir = Path(__file__).resolve().parents[2]
    model = load_base_model(-10)
    optimizer = adam(0.01)
    model.compile(optimizer='rmsprop', loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.fit_generator(
        train_generator(project_dir/'data'/'raw'/'DATASET'/'TRAIN'),
        steps_per_epoch=256,
        epochs=300,
        validation_data=validation_generator(project_dir/'data'/'raw'/'DATASET'/'TEST'),
        validation_steps=64,
        callbacks=[
            checkpoint((project_dir/"models"/str(datetime.now()
                                                 )).with_suffix('.h5')),
             early(), tensorboard()]
    )
