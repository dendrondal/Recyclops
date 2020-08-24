import sqlite3
from pathlib import Path
import pickle
import click


@click.command()
@click.argument("table_name", required=True)
def create_metadata(table_name):
    """Takes a pickled dictionary of recycling guidelines (see cif3r/data/recycling_guidelines.py)
    and recursively searches for all sub-directories in the interim data directory to see if they
    match sub-categories of major recycling streams (i.e. 'tin can' -> 'metal' -> Recyclable ('R')).
    
    Saves it to the metadata sqlite db, deleting the existing table if it already exists, to take into
    account changes in the guideline dict or filesystem paths. 
    
    Also creates the class_mapping table to store model result mapping if it doesn't already exist"""

    project_dir = Path(__file__).resolve().parents[2]
    data_path = project_dir / "data" / "interim"
    db_path = data_path / "metadata.sqlite3"
    guideline_path = project_dir / "data/external" / f"{table_name}.pickle"
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    cleanup = "DROP TABLE {}".format(table_name)
    try:
        cur.execute(cleanup)
    except Exception as e:  # keeps function from halting here if this table name doesn't exist.
        print(e)
        pass

    init = """CREATE TABLE {} (
        hash text PRIMARY KEY,
        recyclable text NOT NULL,
        stream text NOT NULL,
        subclass text NOT NULL
    )
    """.format(
        table_name
    )
    cur.execute(init)

    with open(guideline_path, "rb") as f:
        guideline_dict = pickle.load(f)
    for key in ["R", "O"]:
        for stream in guideline_dict[key]:
            for subclass in guideline_dict[key][stream]:
                folder = data_path / key / subclass.replace(" ", "_")
                for img in folder.glob("*.jp*g"):
                    query = """
                    INSERT INTO {} (hash, recyclable, stream, subclass) VALUES (?, ?, ?, ?)
                    """.format(
                        table_name
                    )
                    try:
                        cur.execute(
                            query, (str(data_path / img), key, stream, subclass)
                        )
                    except sqlite3.IntegrityError:
                        pass

    subtbl = """ CREATE TABLE IF NOT EXISTS class_mapping (
        university text NOT NULL,
        label text NOT NULL,
        key_index integer NOT NULL
    )"""
    cur.execute(subtbl)

    conn.commit()
    conn.close()


if __name__ == "__main__":
    create_metadata()
