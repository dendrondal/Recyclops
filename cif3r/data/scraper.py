import click
from selenium import webdriver
from PIL import Image
import hashlib
import time
import requests
import io
from typing import List, Dict, Tuple
import os
from pathlib import Path
import numpy as np
import pickle
from threading import Thread
import logging


def fetch_image_urls(
    query: str,
    max_links_to_fetch: int,
    wd: webdriver,
    sleep_between_interactions: float = 0.3,
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
            time.sleep(0.5)
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


def resize_img(img: Image):
    basewidth = 300
    if img.size[0] <= basewidth:
        return img
    else:
        wpercent = basewidth / float(img.size[0])
        hsize = int((float(img.size[1]) * float(wpercent)))
        img = img.resize((basewidth, hsize), Image.ANTIALIAS)
        return img


def pre_prediction(img: Image, model_name: Path):
    """
    NOT IMPLEMENTED YET

    Function to help validation during curation process.
    Theoretically, first CNN should label all images as 
    recycleable
    """
    clf = load_model(model_name)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    valid = clf.predict_classes(x)
    return valid


def dict_chunker(result_dict:Dict[str, List[str]], n: int) -> List[List[str]]:
    flat_keys, flat_vals = list(result_dict.keys()), list(result_dict.values())
    keys = [flat_keys[i : i + n] for i in range(0, len(flat_keys), n)]
    vals = [flat_vals[i : i + n] for i in range(0, len(flat_vals), n)]
    return (keys, vals)


def download_image(folder_path: str, names: str, urls: str):
    for name, url in zip(names, urls):
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


def multithreaded_save(chunks:Tuple[List[List[str]], List[List[str]]], target_path:Path):
    """Takes a chunked dictionary and saves it in a mulithreaded manner."""
    for names, urls in zip(chunks[0], chunks[1]):
        Thread(
            target=download_image, args=(target_path, names, urls)
        ).start()


@click.group()
@click.option("--data_path", type=click.Path(), default="/home/dal/CIf3R/data/interim")
@click.option("--result_count", default=500)
@click.option("--model", default="2019-08-28 08:03:49.h5")
@click.pass_context
def cli(ctx, data_path, result_count, model):
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logdir = Path(data_path).parents[1] / "reports"
    logging.basicConfig(
        filename=logdir / "data.log", level=logging.INFO, format=log_fmt
    )
    logger = logging.getLogger(__name__)
    # instantiation and path definitions go here
    data_path = Path(data_path)

    ctx.obj = {
        'data_path': data_path,
        'result_count': result_count,
        'model': model
        }


@cli.command()
@click.option("--dict_name", prompt=True)
@click.option(
    "--interrupted_on",
    help="If scrape_multiple is interrupted, this is the last item it scraped",
    prompt=True
)
@click.pass_context
def scrape_multiple(
    ctx,
    dict_name,
    data_path: Path,
    result_count: int,
    logfile,
    interrupted_on=None
):
    """
    Main scraping function that iterates through dict of university guidelines, scraping each google
    images for each individual item, hasing them, and then saving them.
    """
    # getting the recycling guidelines
    guideline_path = data_path.parents[0] / "external"
    with open(guideline_path / f"{dict_name}.pickle", "rb") as f:
        guideline_dict = pickle.load(f)

    for broad_category, _dict in guideline_dict.items():
        for primary_category, queries in _dict.items():
            # accounting for discrepancy between Ubuntu 16.04 and 18.04
            try:
                wd = webdriver.Chrome("/home/dal/chromedriver/chromedriver")
            except NotADirectoryError:
                wd = webdriver.Chrome("/home/dal/chromedriver")

            for query in queries:
               #search for value that was stopped at              
                if interrupted_on and query != interrupted_on:
                    pass
                elif interrupted_on and query == interrupted_on:
                    interrupted_on = False
                    pass
                else:    
                    clean_query = query.replace(" ", "_")
                    target_path = data_path / f"{broad_category}/{clean_query}"
                    target_path.mkdir(parents=False, exist_ok=True)
                    logger.info(f"Starting scraping for {query}")
                    google_img_result = fetch_image_urls(query, int(result_count), wd)
                    logger.info("Image URLs obtained! Hashing URLs...")
                    hashed_results = hash_urls(google_img_result)
                    logger.info("Saving images and metadata...")
                    chunks = dict_chunker(hashed_results, 5)
                    multithreaded_save(chunks, target_path)
                    logger.info(f"Finished saving images for {query}")



@cli.command()
@click.option("--query", prompt=True)
@click.option("--metadata", type=(str, str, str), prompt=True)
@click.pass_context
def scrape_single(ctx, query, metadata):
    """
    Scrapes single class of item. broad_category can be 'R' for recyclable or 'O'
    for not. Example is a piece of paper:
    
    query = 'construction paper'
    metadata = ('R', 'paper', 'piece_of_paper')*

    *Note that this will vary by university. This will be addressed post-scraping by the creation
    of a database with university recycling guidelines
    """
    broad_category, primary_category, folder_name = metadata
    valid_categories = ['O', 'R']
    if broad_category not in valid_categories:
        raise ValueError(f"broad_category must be one of {valid_categories} with R for recyclable")

    # accounting for discrepancy between Ubuntu 16.04 and 18.04
    try:
        wd = webdriver.Chrome("/home/dal/chromedriver/chromedriver")
    except NotADirectoryError:
        wd = webdriver.Chrome("/home/dal/chromedriver")

    clean_query = query.replace(" ", "_")
    target_path = ctx.obj['data_path'] / f"{broad_category}/{folder_name}"
    target_path.mkdir(parents=False, exist_ok=True)
    google_img_result = fetch_image_urls(query, int(ctx.obj['result_count']), wd)
    hashed_results = hash_urls(google_img_result)
    chunks = dict_chunker(hashed_results, 5)
    multithreaded_save(chunks, target_path)


if __name__ == "__main__":
    cli()