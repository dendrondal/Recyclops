import sqlite3
from pathlib import Path
import pickle
import click


@click.command()
@click.option('--table_name')
def create_metadata(table_name):
    project_dir = Path(__file__).resolve().parents[2]
    print(project_dir)
    data_path = project_dir / 'data' / 'interim'
    db_path = data_path / 'metadata.sqlite3'
    guideline_path = project_dir / 'data/external' / f'{table_name}.pickle'
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    init = """CREATE TABLE IF NOT EXISTS {} (
        hash text PRIMARY KEY,
        recyclable text NOT NULL,
        stream text NOT NULL,
        subclass text NOT NULL
    )
    """.format(table_name)
    cur.execute(init)

    with open(guideline_path, 'rb') as f:
        guideline_dict = pickle.load(f)
    for key in ['R', 'O']:
        for stream in guideline_dict[key]:
            for subclass in guideline_dict[key][stream]:
                folder = data_path / key / subclass.replace(' ', '_')
                for img in folder.glob('*.jpg'):
                    query = """
                    INSERT INTO {} (hash, recyclable, stream, subclass) VALUES (?, ?, ?, ?)
                    """.format(table_name) 
                    try:
                        cur.execute(query, (str(img.name), key, stream, subclass))
                    except sqlite3.IntegrityError:
                        pass
    conn.commit()
    conn.close()

if __name__ == '__main__':
    create_metadata()