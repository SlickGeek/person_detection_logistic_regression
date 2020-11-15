from flask import Flask
from flask_marshmallow import Marshmallow

app = Flask(__name__)

ma = Marshmallow(app)

from ai_backend.video_stream.routes import video_stream

app.register_blueprint(video_stream)
