from pathlib import Path
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd


def train_val_split(university: str, test_size: float = 0.2):
    data_dir = Path(__file__).resolve().parents[2] / "data/interim"
    conn = sqlite3.connect(str(data_dir / "metadata.sqlite3"))
    cur = conn.cursor()
    query = """
    SELECT
        hash,
        CASE
            WHEN(recyclable = 'O') THEN 'trash'
            WHEN(recyclable = 'R') THEN stream
        END
    FROM {}
    """.format(
        university
    )
    imgs, labels = [], []
    for img, label in cur.execute(query):
        imgs.append(img), labels.append(label)
    return train_test_split(imgs, labels, test_size=test_size, random_state=42)


def label_encoding(y_train, y_val):
    mlb = MultiLabelBinarizer()
    mlb.fit([y_train])
    return mlb.transform(y_train), mlb.transform(y_val)


def datagen(university: str, balance_classes=True):
    """
    Creates dataframe to be consumed by the Keras stream_from_dataframe method
    with columns 'filename' and 'class'. Joins together both trash and recycling data, 
    downsampling trash to prevent class imbalances. 
    
    balance_classes ensures all classes are perfectly balanced.
    """

    data_dir = Path(__file__).resolve().parents[2] / "data/interim"
    conn = sqlite3.connect(str(data_dir / "metadata.sqlite3"))
    q1 = """
    SELECT hash,
       stream
    FROM   {}
    WHERE  recyclable='R'
    """.format(
        university
    )
    q2 = """
    SELECT hash, (CASE WHEN(recyclable = 'O') THEN 'trash' END)
    FROM   {}
    WHERE  recyclable = 'O'
    ORDER BY RANDOM() 
    LIMIT (
              SELECT count(*)
              FROM   {}
              WHERE  recyclable = 'R')
    """.format(
        university, university
    )
    df1 = pd.read_sql(sql=q1, con=conn)
    df2 = pd.read_sql(sql=q2, con=conn)
    all_dfs = [df1, df2]
    for df in all_dfs:
        df.columns = ["filename", "class"]
    master_df = pd.concat(all_dfs).reset_index(drop=True)

    if balance_classes:
        grouped = master_df.groupby("class")
        grouped.apply(lambda x: x.sample(grouped.size().min()).reset_index(drop=True))
        print(f"Sampling {grouped.size().min()} samples from each class...")

    class_balances = master_df.groupby(["class"]).nunique()["filename"]
    print(f"Full data:/n {class_balances}")
    return master_df
