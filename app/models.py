from .config import db


class Models(db.Model):
    __tablename__ = 'models'
    university = db.Column(db.String, primary_key=True)
    model_name = db.Column(db.String)
    prediction_index = db.Column(db.Integer)
    prediction_class = db.Column(db.String)