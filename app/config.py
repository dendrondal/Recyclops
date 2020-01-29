from pathlib import Path
from flask_sqlalchemy import SQLAlchemy
from flask import Flask


app = Flask(__name__)
db_uri = f"sqlite:////{Path(__file__).parents[1]}/data/interim/metadata.sqlite3"
app.config["SQLALCHEMY_DATABASE_URI"] = db_uri
app.config["SQLALCHEMY_ECHO"] = True
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.debug = True

db = SQLAlchemy(app)
