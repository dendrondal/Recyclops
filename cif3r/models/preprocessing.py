import sqlite3
from pathlib import Path
import tensorflow as tf
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split


def train_val_split(university:str, test_size:float=0.2):
    data_dir = Path(__file__).resolve().parents[2] / 'data/interim'
    conn = sqlite3.connect(data_dir/'metadata.sqlite3')
    cur = conn.cursor()
    query = """
    SELECT
        hash,
        CASE
            WHEN(recyclable = 'O') THEN 'trash'
            WHEN(recyclable = 'R') THEN 'stream'
        END
    FROM {}
    """.format(university)
    imgs, labels = [], []
    for img, label in cur.execute(query):
        imgs.append(img), labels.append(label)
    return train_test_split(imgs, labels, test_size=test_size, random_state=42)


def label_encoding(y_train, y_val):
    mlb = MultiLabelBinarizer()
    mlb.fit(y_train)
    return mlb.transform(y_train), mlb.transform(y_val)


def datagen():
    data_dir = Path(__file__).resolve().parents[2] / 'data/interim'
    conn = sqlite3.connect(data_dir/'metadata.sqlite3')
    cur = conn.cursor()
    query = """
    SELECT
        hash,
        CASE
            WHEN(recyclable = 'O') THEN 'trash'
            WHEN(recyclable = 'R') THEN 'stream'
        END
    FROM UTK
    """
    for row in cur.execute(query):
        yield row


def decode_img(img):
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = img / 255.0
    return tf.image.resize(img, [224, 224])


def process_path(file_path, label):
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label


def labeled_ds():
    list_ds = tf.data.Dataset.from_generator(
        datagen, (str, str)
    )
    return list_ds.map(process_path, num_parallel_calls=AUTOTUNE)


def create_dataset(ds, cache=True, shuffle_buffer_size=1024):
  # This is a small dataset, only load it once, and keep it in memory.
  # use `.cache(filename)` to cache preprocessing work for datasets that don't
  # fit in memory.

  ds = labeled_ds()

  if cache:
    if isinstance(cache, str):
      ds = ds.cache(cache)
    else:
      ds = ds.cache()

  ds = ds.shuffle(buffer_size=shuffle_buffer_size)

  # Repeat forever
  ds = ds.repeat()

  ds = ds.batch(BATCH_SIZE)

  # `prefetch` lets the dataset fetch batches in the background while the model
  # is training.
  ds = ds.prefetch(buffer_size=AUTOTUNE)

  return ds

