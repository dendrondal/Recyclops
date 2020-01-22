import sqlite3
from pathlib import Path
import subprocess
import click


def get_cursor(db_path:str):
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    return cur

RECYCLEABLE_UTK_GUIDELINES = {
    'paper': ['printer paper', 'cardboard', 'cereal boxes', 'envelopes', 'sticky notes', 
    'newspapers', 'hardback books', 'journals', 'magazines', 'spiral bound notebooks', 
    'phone books', 'catalogs', 'file folders'],
    'cans': ['aluminum beverage can', 'tin can', 'soup can', 'vegetable can', 'tuna can', 
    'pet food can'],
    'plastic': ['clean plastic bottle', 'clean plastic cup', 'clean milk jug', 
    'clean plastic detergent container'],
    'cardboard': ['corrugated cardboard', 'shipping boxes', 'paper board', 'cereal boxes'],
}

TRASH_UTK_GUIDELINES = {
    'paper': ['paper cups', 'paper plates', 'milk carton', 'juice carton', 'paper towel', 
    'muffin liner', 'pastry wrap', 'envelope with bubble wrap', 'three ring binders'],
    'cans': ['aerosol spray cans', 'aluminum foil', 'aluminum tray', 'pie tin'],
    'plastic': ['styrofoam containers', 'dirty plasticware', 'plastic bottle with liquid',
     'plastic bag', 'film', 'plastic straws', 'glass jars', 'glass bottles', 'CDs'],
    'cardboard': ['cardboard food container', 'packing peanus', 'styrofoam'],
}

UTK = {'R': RECYCLEABLE_UTK_GUIDELINES, 'O': TRASH_UTK_GUIDELINES}

UNIVERSITIES = [UTK]

def create_master_table(cursor):
    query = """
    CREATE TABLE IF NOT EXISTS img_master (
        hash text PRIMARY KEY,
        primary_type text NOT NULL
    )
    """
    cursor.execute(query)


def create_guideline_table(cursor, name:str):
    cur = get_cursor()
    query = """
    CREATE TABLE IF NOT EXISTS ? (
        hash text PRIMARY KEY,
        recyclable text NOT NULL,
        stream text NOT NULL
    )
    """
    cur.execute(query, [name])


def write_metadata(cursor, tbl_name:str, hash:str, reyclable:str, stream:str):
    img_addition = "INSERT INTO ? (hash, recyclable, stream) VALUES (?, ?, ?)"
    cursor.execute(img_addition, (tbl_name, hash, recyclable, stream))
    write_master = "INSERT INTO img_master (hash, primary_type) VALUES (?, ?)"
    cursor.execute(write_master, (hash, stream))

        
@click.command()        
@click.option('--db_name')
def db_init(cur, db_name):
    create_master_table(cur)
    create_guideline_table(cur, db_name)


@click.command()
@click.option(
    '--db_path', 
    default=Path(__file__).parents[2] / 'data/interim/metadata.sqlite3'
    )
@click.option('--first_run', is_flag=True)
@click.option('--dict_name')
def create_schema(db_path, first_run, dict_name):
    cur = get_cursor(db_path)
    for key, val in guideline_dict.items():

