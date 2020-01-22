import sqlite3


def get_cursor(db_path: str):
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    return cur

RECYCLEABLE_UTK_GUIDELINES = {
    'paper': ['printer paper', 'cardboard', 'cereal boxes', 'envelopes', 'sticky notes', 'newspapers', 'hardback books',
    'journals', 'magazines', 'spiral bound notebooks', 'phone books', 'catalogs', 'file folders'],
    'cans': ['aluminum beverage can', 'tin can', 'soup can', 'vegetable can', 'tuna can', 'pet food can'],
    'plastic': ['clean plastic bottle', 'clean plastic cup', 'clean milk jug', 'clean plastic detergent container'],
    'cardboard': ['corrugated cardboard', 'shipping boxes', 'paper board', 'cereal boxes'],
}

TRASH_UTK_GUIDELINES = {
    'paper': ['paper cups', 'paper plates', 'milk carton', 'juice carton', 'paper towel', 'muffin liner', 'pastry wrap',
    'envelope with bubble wrap', 'three ring binders'],
    'cans': ['aerosol spray cans', 'aluminum foil', 'aluminum tray', 'pie tin'],
    'plastic': ['styrofoam containers', 'dirty plasticware', 'plastic bottle with liquid', 'plastic bag', 'film', 
    'plastic straws', 'glass jars', 'glass bottles', 'CDs'],
    'cardboard': ['cardboard food container', 'packing peanus', 'styrofoam'],
}

UTK = {'R': RECYCLEABLE_UTK_GUIDELINES, 'O': TRASH_UTK_GUIDELINES}


def create_schema(cursor, guideline_dict):
    for key, val in guideline_dict.items():


