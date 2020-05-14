import torch.utils.data as data
from pathlib import Path
from PIL import Image
from itertools import chain
from sklearn import preprocessing
import torch
from torchvision import transforms
import pickle
import sqlite3

class Recyclables(data.Dataset):
    def __init__(self, university, minority_cls_count):

        super(Recyclables, self).__init__()
        self.university = university
        self.minority_cls_count = minority_cls_count
        # Datbase connection
        data_dir = Path(__file__).resolve().parents[2] / "data/interim"
        conn = sqlite3.connect(str(data_dir / "metadata.sqlite3"))
        self.cur = conn.cursor()
        # Get all recycling streams
        dict_path = data_dir.resolve().parents[0] / 'external/{}.pickle'.format(university)
        with open(dict_path, 'rb') as f:
            uni = pickle.load(f)
        self.streams = list(
            chain.from_iterable(
                [key for key in uni['R'].values()]
                )
            )
        self.le = preprocessing.LabelEncoder().fit(self.streams)
        self.images, labels = self._query()
        self.labels = self.le.transform(labels)
    

    @staticmethod
    def transform(path):
        img = Image.open(path).convert('RGB')
        operations = transforms.Compose([
            transforms.Resize(84),
            transforms.CenterCrop(84),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.485, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])
        return operations(img)


    def _query(self):
        images, labels = [], []
        for subclass in self.streams:
            print(f"Starting query for {subclass}")
            query = f"""
            SELECT 
                hash, subclass 
            FROM 
                {self.university} 
            WHERE 
                subclass='{subclass}' 
            ORDER BY 
                Random() 
            LIMIT 
                {self.minority_cls_count}
            """

            for name, label in self.cur.execute(query):
                images.append(name)
                labels.append(label)

        return images, labels

    def __len__(self):
        query = f"SELECT COUNT(*) FROM {self.university}"
        return self.cur.execute(query)

    def __getitem__(self, i):
        imgs, labels = self.images[i], self.labels[i]
        imgs = self.transform(imgs)
        return imgs, labels