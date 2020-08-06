import pickle
from pathlib import Path


RECYCLEABLE_UTK_GUIDELINES = {
    "paper": [
        "pieces of paper",
        "cardboard",
        "cereal boxes",
        "envelopes",
        "sticky notes",
        "newspapers",
        "hardback books",
        "magazines",
        "paperback books",
        "spiral bound notebooks",
        "phone books",
        "catalogs",
        "poster board",
        "file folders",
    ],
    "cans": ["aluminum beverage can", "tin can", "soup can", "pet food can"],
    "plastic": [
        "clean plastic bottle",
        "clean plastic cup",
        "clean milk jug",
        "clean plastic detergent container",
        "tupperware",
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
        "cardboard food container",
        "packing peanuts",
        "styrofoam",
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
}

RECYCLABLES_PENN_STATE = {
    "compost": ["food", "tea bags", "coffee grounds"],
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
        "catalogs",
    ],
    "glass": [
        "clean empty glass bottle",
        "clear glass",
        "blue glass",
        "brown glass",
        "green glass",
    ],
    "metal": [
        "aluminum beverage can",
        "tin can",
        "soup can",
        "pet food can",
        "aluminum foil",
        "aerosol spray cans",
        "empty paint can",
    ],
    "bottles and films": [
        "clean plastic bottle",
        "clingwrap",
        "plastic bag",
        "bubble wrap",
        "packing pillow",
        "clean ziploc bag",
    ],
    "misc. plastic": [
        "clean yogurt container",
        "clean plastic take-out food container",
        "tupperware",
    ],
}

TRASH_PENN_STATE = {
    "paper": ["shredded paper", "paper towel", "wax paper", "food wrappers",],
    "metal": [
        "empty chip bag",
        "granola bar wrapper",
        "clean pringles can",
        "alkaline battery",
    ],
    "bottles and films": ["plastic bottle with liquid"],
    "misc. plastic": [
        "styrofoam",
        "single-use coffee cup",
        "k-cups",
        "plastic straws",
        "condiment packet",
        "pen",
        "marker",
    ],
}

UNIVERSITIES = {
    "UTK": {"R": RECYCLEABLE_UTK_GUIDELINES, "O": TRASH_UTK_GUIDELINES},
    "penn_state": {"R": RECYCLABLES_PENN_STATE, "O": TRASH_PENN_STATE},
}


def dump_guidelines():
    """This file is used for hand curation of university of recycling guidelines (see schema below)
    This function dumps pickled dictionaries into the external data directories.

    Dict Schema:
    University name
        R (Recyclable)
            stream name (paper, plastic, etc.)
                subcategories (clean plastic bottles, cardstock, etc.)
        O (Trash)
            stream name (stream names don't have to match R exactly)
                subcategories (true negatives that are explicitly defined by the university, or dirty items)
    """

    for name, guidelines in UNIVERSITIES.items():
        with open(
            Path(__file__).parents[2] / f"data/external/{name}.pickle", "wb"
        ) as f:
            pickle.dump(guidelines, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    dump_guidelines()
