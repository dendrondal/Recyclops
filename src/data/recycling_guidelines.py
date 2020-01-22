import pickle
from pathlib import Path


RECYCLEABLE_UTK_GUIDELINES = {
    'paper': ['pieces of paper', 'cardboard', 'cereal boxes', 'envelopes', 'sticky notes', 
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
    'cardboard': ['cardboard food container', 'packing peanuts', 'styrofoam'],
}


UNIVERSITIES = {'UTK': {'R': RECYCLEABLE_UTK_GUIDELINES, 'O': TRASH_UTK_GUIDELINES}}


def dump_guidelines():
    for name, guidelines in UNIVERSITIES.items():
        with open(
            Path(__file__).parents[2] / f'data/external/{name}.pickle',
             'wb') as f:
            pickle.dump(guidelines, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    dump_guidelines()