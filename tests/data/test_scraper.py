import os
from PIL import Image
from selenium import webdriver
from pathlib import Path
from tensorflow.keras.utils import to_categorical
import numpy as np
from src.data.scraper import *

try:
    wd = webdriver.Chrome("/home/dal/chromedriver/chromedriver")
except NotADirectoryError:
    wd = webdriver.Chrome("/home/dal/chromedriver")

TEST_DATA_DIR = Path(__file__).parents[1] / "mock_data"

def test_fetch_image_urls():
    result = fetch_image_urls("jaberwocky", 3, wd=wd)
    assert len(result) == 3
    assert type(result[0]) == str


def test_hash_urls():
    url_list = ["http://google.com", "http://facebook.com"]
    result = hash_urls(url_list)
    assert len(result) == len(url_list)
    assert list(result.values())[0] in url_list
    assert type(list(result.values())[0]) == str


def test_resize_small_img():
    test_img = Image.open(TEST_DATA_DIR / 'small_img.jpg')
    result = resize_img(test_img)
    assert result.size[0] < 300

def test_resize_lg_img():
    test_img = Image.open(TEST_DATA_DIR / 'big_img.jpg')
    result = resize_img(test_img)
    assert result.size[0] == 300
    assert round(result.size[1] / result.size[0]) == round(test_img.size[1] / test_img.size[0])


def test_dict_chunker():
    test_dict = {'foo': 0, 'bar': 2, 'whozits': 3, 'whatzits': 4, 'stuff': 5, 'things': 6}
    result = dict_chunker(test_dict, 3)
    assert len(result) == 2
    assert len(result[1]) == 2
    assert result[0][0] == ['foo', 'bar', 'whozits']
    assert result[1][1] == [4, 5, 6]


def test_pre_prediction_positive():
    prediction_img = Image.open(
        Path(__file__).parents[1] / "mock_data/R_example.jpg",
        target_size=(224, 224)
    )
    true_result = pre_prediction(
        prediction_img, Path(__file__).parents[2] / "models/2019-08-28 08:03:49.h5"
    )
    assert to_categorical(true_result) == 1


def test_pre_prediction_negative():
    prediction_img = Image.open(
        TEST_DATA_DIR/"O_example.jpg",
        target_size=(224, 224)
    )    
    true_result = pre_prediction(
        prediction_img, Path(__file__).parents[2] / "models/2019-08-28 08:03:49.h5"
    )
    assert to_categorical(true_result) == 0
