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
#from tensorflow.keras.models import load_model
import numpy as np
import pickle
import logging

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


def hash_urls(img_urls:List[str]):
    hashed_urls = dict()
    for url in img_urls:
        key = str(abs(hash(url)) % (10 ** 10))
        hashed_urls[key] = url

    return hashed_urls


def resize_img(img:Image):
    basewidth = 300
    if img.size[0] <= basewidth:
        return img
    else:
        wpercent = (basewidth/float(img.size[0]))
        hsize = int((float(img.size[1])*float(wpercent)))
        img = img.resize((basewidth, hsize), Image.ANTIALIAS)
        return img

from tensorflow.keras.preprocessing import image

def pre_prediction(img: Image, model_name: Path):
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
        raw_image = Image.open(image_file).convert("RGB")
        image = resize_img(raw_image)
        file_path = os.path.join(folder_path, name + ".jpg")
        with open(file_path, "wb") as f:
            image.save(f, "JPEG", quality=85)

    except Exception as e:
        print(f"ERROR - Could not save {url} - {e}")


def get_cursor(db_path: str):
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    return cur


def create_master_table(cursor):
    query = """
    CREATE TABLE IF NOT EXISTS img_master (
        hash text PRIMARY KEY,
        primary_type text NOT NULL
    )
    """
    cursor.execute(query)


def create_guideline_table(cursor, name:str):
    query = """
    CREATE TABLE IF NOT EXISTS {} (
        hash text PRIMARY KEY,
        recyclable text NOT NULL,
        stream text NOT NULL
    )
    """.format(name)
    cursor.execute(query)


def write_metadata(cursor, tbl_name:str, hash:str, recyclable:str, stream:str):
    img_addition = "INSERT INTO {} (hash, recyclable, stream) VALUES (?, ?, ?)".format(tbl_name)
    cursor.execute(img_addition, (hash, recyclable, stream))
    write_master = "INSERT INTO img_master (hash, primary_type) VALUES (?, ?)"
    cursor.execute(write_master, (hash, stream))

        
def db_init(cur, db_name):
    create_master_table(cur)
    create_guideline_table(cur, db_name)


@click.command()
@click.option("--data_path", type=click.Path(), default="/home/dal/CIf3R/data/interim")
@click.option("--result_count", default=500)
@click.option("--model", default="2019-08-28 08:03:49.h5")
@click.option('--first_run', is_flag=True)
@click.option('--dict_name')
def main(data_path, result_count, model, first_run, dict_name):
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logdir = Path(data_path).parents[1] / 'reports'
    logging.basicConfig(
        filename=logdir / 'data.log',
        level=logging.INFO, 
        format=log_fmt
        )
    logger = logging.getLogger(__name__)
    #instantiation and path definitions go here
    db_path = Path(data_path) / "metadata.sqlite3"
    guideline_path = Path(data_path).parents[0] / 'external'
    cursor = get_cursor(str(db_path))

    if first_run:
        logger.info("Creating new tables...")
        db_init(cursor, dict_name)
    #getting the recycling guidelines
    with open(guideline_path / f'{dict_name}.pickle', 'rb') as f:
        guideline_dict = pickle.load(f) 
    #main scraping iteration
    for broad_category, _dict in guideline_dict.items():
        for primary_category, queries in _dict.items():
            #accounting for discrepancy between Ubuntu 16.04 and 18.04
            try:
                wd = webdriver.Chrome("/home/dal/chromedriver/chromedriver")
            except NotADirectoryError:
                wd = webdriver.Chrome("/home/dal/chromedriver")
            for query in queries:
                logger.info(f'Starting scraping for {query}')
                google_img_result = fetch_image_urls(query, int(result_count), wd)
                logger.info('Image URLs obtained! Hashing URLs...')
                hashed_results = hash_urls(google_img_result)
                logger.info('Saving images and metadata...')
                for key, val in hashed_results.items():
                    download_image(Path(data_path) / broad_category, val, key)
                    write_metadata(cursor, dict_name, key, broad_category, primary_category)
                logger.info(f'Finished saving images for {query}')

if __name__ == "__main__":
    main()
