import torch.utils.data as data
from pathlib import Path
from PIL import Image
from cif3r.data.recycling_guidelines import UNIVERSITIES
from itertools import chain
import torch
from torchvision import transforms

class Recyclables(data.Dataset):
    def __init__(
        self,
        university:str='UTK', 
        minority_cls_count:int=10, 
        total_imgs:int=16000
        ):

        super(BatchSampler, self).__init__()
        self.university = university
        self.minority_cls_count = minority_cls_count
        self.total_imgs = total_imgs
        # Datbase connection
        data_dir = Path(__file__).resolve().parents[2] / "data/interim"
        conn = sqlite3.connect(str(data_dir / "metadata.sqlite3"))
        self.cur = conn.cursor()
        # Get all recycling streams
        self.streams = list(
            chain.from_iterable(
                [key for key in UNIVERSITIES[university]['R'].values()]
                )
            )

        self.images, self.labels = _query() 
        self.transform = transforms.Compose([
            transforms.Resize(84),
            transforms.CenterCrop(84),
            transforms.ToTensor(),
            trasforms.Normalize(mean=[0.485, 0.485, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])
  
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
        img, label = self.images[i], self.labels[i]
        image = self.transform(Image.open(img).convert('RGB'))
        return img, label