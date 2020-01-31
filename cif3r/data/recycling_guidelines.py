import pickle
from pathlib import Path


RECYCLEABLE_UTK_GUIDELINES = {
    "paper": [
        "pieces of paper",
        "cardboard",
        "ledger paper",
        "computer printout paper",
        "cereal boxes",
        "envelopes",
        "sticky notes",
        "newspapers",
        "hardback books",
        "journals",
        "magazines",
        "paperback books",
        "spiral bound notebooks",
        "phone books",
        "catalogs",
        "poster board" "file folders",
    ],
    "cans": ["aluminum beverage can", "tin can", "soup can", "pet food can"],
    "plastic": [
        "clean plastic bottle",
        "clean plastic cup",
        "clean milk jug",
        "clean plastic detergent container",
    ],
}

TRASH_UTK_GUIDELINES = {
    "paper": [
        "paper cups",
        "paper plates",
        "milk carton",
        "juice carton",
        "paper towel",
        "muffin liner",
        "pastry wrap",
        "envelope with bubble wrap",
        "three ring binders",
    ],
    "cans": ["aerosol spray cans", "aluminum foil", "aluminum tray", "pie tin"],
    "plastic": [
        "styrofoam containers",
        "dirty plasticware",
        "plastic bottle with liquid",
        "plastic bag",
        "camera film",
        "plastic straws",
        "glass jars",
        "glass bottles",
        "CDs",
    ],
    "cardboard": ["cardboard food container", "packing peanuts", "styrofoam"],
}

RECYCLABLES_PENN_STATE  = {
    "paper": [
        "pieces of paper",
        "ledger paper",
        "computer printout paper",
        "brochures",
        "magazines",
        "envelopes", 
        "sticky notes",
        "poster board",
        "cereal boxes",
        "wrapping paper",
        "paperback books",
        "journals",
        "phone books",
        "catalogs"
    ],
    "glass": [
        "glass bottles",
        "clear glass",
        "blue glass",
        "brown glass",
        "green glass"
    ],
    "metal": [
        "aluminum beverage can",
        "tin can",
        "soup can",
        "pet food can",
        "aluminum foil",
        "aerosol spray cans",
        "empty paint cans"
    ],
    "bottles and film": [
        "clean plastic bottle",
        "clingwrap",
        "bubble wrap",
        "packing pillows",
        "plastic bags"
    ],
    "misc. plastic": [
        "clean yogurt container",
        "clean plastic take-out food container",
        "tupperware"
    ]
}

TRASH_PENN_STATE = {
    "paper": [
        
    ]
}

UNIVERSITIES = {"UTK": {"R": RECYCLEABLE_UTK_GUIDELINES, "O": TRASH_UTK_GUIDELINES}}


def dump_guidelines():
    for name, guidelines in UNIVERSITIES.items():
        with open(
            Path(__file__).parents[2] / f"data/external/{name}.pickle", "wb"
        ) as f:
            pickle.dump(guidelines, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    dump_guidelines()
