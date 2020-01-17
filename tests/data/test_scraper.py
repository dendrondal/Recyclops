import pytest
import os
from selenium import webdriver
from pathlib import Path
from src.data.scraper import *

wd = webdriver.Chrome('/home/dal/chromedriver/chromedriver')


def test_fetch_image_urls():
    result = fetch_image_urls('jaberwocky', 5, wd=wd)
    assert len(result) == 5
    assert type(result[0]) == str


def test_hash_urls():
    url_list = ['http://google.com', 'http://facebook.com']
    result = hash_urls(url_list)
    assert len(result) == len(url_list)
    assert list(result.values())[0] in url_list
    assert type(result.values())[0] == str