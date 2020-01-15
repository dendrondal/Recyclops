import click
import selenium
from PIL import io
import os
import sqlite3
import hashlib
from typing import List


def fetch_image_urls(
    query:str, 
    max_links_to_fetch:int, 
    wd:webdriver, 
    sleep_between_interactions:int=1
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
            actual_images = wd.find_elements_by_css_selector('img.irc_mi')
            for actual_image in actual_images:
                if actual_image.get_attribute('src'):
                    image_urls.add(actual_image.get_attribute('src'))

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

    return image_urls


def hash_urls(img_urls:List[str]):
    hashed_urls = dict()
    for url in img_urls:
        key = hashlib.sha1(image_content).hexdigest()[:10]
        hashed_urls[key] = url

    return hashed_urls

def download_images(folder_path:str, url:str, name:str):
    try:
        image_content = requests.get(url).content

    except Exception as e:
        print(f"ERROR - Could not download {url} - {e}")

    try:
        image_file = io.BytesIO(image_content)
        image = Image.open(image_file).convert('RGB')
        file_path = os.path.join(folder_path, + '.jpg')
        with open(file_path, 'wb') as f:
            image.save(f, "JPEG", quality=85)

    except Exception as e:
        print(f"ERROR - Could not save {url} - {e}")


def resize_image(img_file:str):
    pass


def write_metadata(hash:str, recyclable:bool, stream:str, )