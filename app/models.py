from .config import db


class Models(db.Model):
    __tablename__ = "models"
    university = db.Column(db.String, primary_key=True)
    model_name = db.Column(db.String)


class ClassMapping(db.Model):
    __tablename__ = "class_mapping"
    university = db.Column(db.String, primary_key=True)
    label = db.Column(db.String)
    index = db.Column(db.Integer)