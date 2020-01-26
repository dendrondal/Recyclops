import sqlite3
from pathlib import Path
import random
from tempfile import TemporaryDirectory
import shutil
import pickle


def train_test_split(tbl_name:str, split:float=0.7):


    get_recyclables = """
    SELECT 
    """
class TrainTestSplit:
    def __init__(self, tbl_name:str, split:float=0.7):
        data_dir = Path(__file__).resolve().parents[2] / 'data/interim'
        conn = sqlite3.connect(data_dir/'metadata.sqlite3')
        cur = conn.cursor()
        self.tbl_name = tbl_name
        self.split = split
        self.train = TemporaryDirectory(dir=dir)
        self.test = TemporaryDirectory(dir=dir)
        get_streams = "SELECT DISTINCT streams FROM {}".format(tbl_name)
        self.streams = cur.execute(get_streams)
        for stream in self.streams:
            filter = """
            SELECT 
                hash 
            FROM 
                {}
            WHERE 
                recyclable = 'R' AND
                stream = {}
            """.format(self.tbl_name, stream)
            setattr(self, stream, cur.execute(filter))

    @property
    def trash(self):
        get_trash = """
        SELECT 
            hash 
        FROM 
            {} 
        WHERE 
            recyclable = 'O'
        """.format(self.tbl_name)
        return [img for img in cur.execute(get_trash)]
    
    def shuffle_and_split(self, lst):
        random.shuffle(lst)
        train_size = int(len(lst) * 0.7)bes
        train = lst[:train_size]
        test = lst[train_size:]
        return train, test

    def copy_to_tmp_dir(self):
        for stream in self.streams:
            TemporaryDirectory()
    true_neg = list((dir/'O').rglob(".jpg"))
    
    random.shuffle(true_neg)

        dir/'O/TRAIN'.mkdir(parents=False,exist_ok=True)
        dir/'O/TEST'.mkdir(parents=False,exist_ok=True)
        dir/'R/TRAIN'.mkdir(parents=False,exist_ok=True)
        dir/'R/TEST'.mkdir(parents=False,exist_ok=True)

    #Copying into R directories
    train_size = int(len(true_pos) * split)
    print(f"Splitting recyclable data into {train_size} training points\
         and {len(true_pos) -train_size} test points")
    training_data = true_pos[:train_size]
    test_data = true_pos[train_size:]
    for pic in training_data:
        shutil.copyfile(pic, dir/f'R/TRAIN/{pic.name}')
    for pic in test_data:
        shutil.copyfile(pic, dir/f'R/TEST/{pic.name}')

    #Copying into O Directories
    train_size = int(len(true_neg) * split)
    print(f"Splitting trash data into {train_size} training points\
         and {len(true_neg) -train_size} test points")
    training_data = true_neg[:train_size]
    test_data = true_neg[train_size:]
    for pic in training_data:
        shutil.copyfile(pic, dir/f'O/TRAIN/{pic.name}')
    for pic in test_data:
        shutil.copyfile(pic, dir/f'O/TEST/{pic.name}')
