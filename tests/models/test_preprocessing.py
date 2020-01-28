import pytest
import os
from cif3r.models.train_model import *


def test_train_val_split():
    X_train, X_val, y_train, y_val = train_val_split('UTK')
    assert type(X_train[1]) == str
    assert type(y_train[1]) == str
    assert len(X_train) == len(y_train)
    assert os.path.exists(X_train[1]) == True


def test_datagen():
    gen = datagen('UTK')
    assert gen.columns[1] == 'class'
    assert len(gen) > 0
    assert len(gen['class'].unique()) == 4
    assert os.path.exists(gen['filename'][0]) == True


def test_labeled_ds():
    result = labeled_ds()
    result_lst = list(next(result))
    assert len(result_lst[0]) == 224
    assert type(result_lst[1]) == str


def test_label_encoding():
    _, _, y_train, y_val = train_val_split('UTK')
    y_train_bin, y_val_bin = label_encoding(y_train, y_val)
    assert len(y_train_bin[0]) == 4


def test_create_dataset():
    X_train, _, y_train, _ = train_val_split('UTK')
    X_result, y_result = create_dataset(X_train, y_train)
    assert os.path.exists(X_result[0]) == True