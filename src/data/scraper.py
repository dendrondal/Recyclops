import click
from selenium import webdriver
from PIL import Image
import sqlite3
import pandas as pd
import hashlib
import time
import requests
import io
from typing import List
import os
from pathlib import Path
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np


def fetch_image_urls(
    query: str,
    max_links_to_fetch: int,
    wd: webdriver,
    sleep_between_interactions: int = 1,
) -> List[str]:
    def scroll_to_end(wd):
        wd.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(sleep_between_interactions)

    # build the google query
    search_url = "https://www.google.com/search?safe=off&site=&tbm=isch&source=hp&q={q}&oq={q}&gs_l=img"

    # load the page
    wd.get(search_url.format(q=query))

    image_urls = set()
    image_count = 0
    results_start = 0
    while image_count < max_links_to_fetch:
        scroll_to_end(wd)

        # get all image thumbnail results
        thumbnail_results = wd.find_elements_by_css_selector("img.rg_ic")
        number_results = len(thumbnail_results)

        for img in thumbnail_results[results_start:number_results]:
            # try to click every thumbnail such that we can get the real image behind it
            try:
                img.click()
                time.sleep(sleep_between_interactions)
            except Exception:
                continue

            # extract image urls
            actual_images = wd.find_elements_by_css_selector("img.irc_mi")
            for actual_image in actual_images:
                if actual_image.get_attribute("src"):
                    image_urls.add(actual_image.get_attribute("src"))

            image_count = len(image_urls)

            if len(image_urls) >= max_links_to_fetch:
                print(f"Found: {len(image_urls)} image links, done!")
                break

        else:
            time.sleep(1)
            load_more_button = wd.find_element_by_css_selector(".ksb")
            if load_more_button:
                wd.execute_script("document.querySelector('.ksb').click();")

        # move the result startpoint further down
        results_start = len(thumbnail_results)

    return list(image_urls)


def hash_urls(img_urls: List[str]):
    hashed_urls = dict()
    for url in img_urls:
        key = str(abs(hash(url)) % (10 ** 10))
        hashed_urls[key] = url

    return hashed_urls


def resize_img(img:Image):
    pass


def pre_prediction(img:Image, model_name:Path):
    """Function to help validation during curation process.
    Theoretically, first CNN should label all images as 
    recycleable"""
    clf = load_model(model_name)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    valid = clf.predict_classes(x)
    return valid


def download_image(folder_path: str, url: str, name: str):
    try:
        image_content = requests.get(url).content

    except Exception as e:
        print(f"ERROR - Could not download {url} - {e}")

    try:
        image_file = io.BytesIO(image_content)
        image = Image.open(image_file).convert("RGB")
        file_path = os.path.join(folder_path, name + ".jpg")
        with open(file_path, "wb") as f:
            image.save(f, "JPEG", quality=85)

    except Exception as e:
        print(f"ERROR - Could not save {url} - {e}")


def get_cursor(db_path: str):
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    return cur


def image_metadata_init(cursor):
    make_images = """
    CREATE TABLE IF NOT EXISTS images (
        hash text PRIMARY KEY,
        recyclable text NOT NULL,
        stream text NOT NULL,
        clean text NOT NULL
    )
    """
    cursor.execute(make_images)


def write_metadata(cur, hash: str, recyclable: bool, stream: str, clean: bool):
    img_addition = (
        "INSERT INTO images (hash, recyclable, stream, clean) VALUES (?, ?, ?, ?)"
    )
    cur.execute(img_addition, (hash, recyclable, stream, clean))


def cleanup(cur, data_path):
    pass


@click.command()
@click.option("--data_path", type=click.Path(), default="/home/dal/CIf3R/data/interim")
@click.option("--query")
@click.option("--result_count")
@click.option("--recycleable", is_flag=True)
@click.option("--model", default='2019-08-28 08:03:49.h5')
@click.option(
    "--stream", type=click.Choice(["paper", "container"], case_sensitive=True)
)
@click.option("--clean", is_flag=True)
def main(data_path, query, result_count, recycleable, stream, clean):
    db_path = Path(data_path) / "metadata.sqlite3"
    cursor = get_cursor(str(db_path))
    image_metadata_init(cursor)
    wd = webdriver.Chrome("/home/dal/chromedriver/chromedriver")
    google_img_result = fetch_image_urls(query, int(result_count), wd)
    hashed_results = hash_urls(google_img_result)
    for key, val in hashed_results.items():
        download_image(data_path, val, key)
        write_metadata(cursor, key, recycleable, stream, clean)


if __name__ == "__main__":
    main()
