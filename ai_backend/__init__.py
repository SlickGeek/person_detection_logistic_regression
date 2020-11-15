from flask import Flask
from flask_marshmallow import Marshmallow

app = Flask(__name__)

ma = Marshmallow(app)

from melp_backend.restaurants.routes import restaurants

app.register_blueprint(restaurants)
