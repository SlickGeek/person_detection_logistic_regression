from flask import Flask
from flask_marshmallow import Marshmallow
from ai_backend.neural_network.predict_image import PredictImage

app = Flask(__name__)
ma = Marshmallow(app)
predict = PredictImage()

# Register Blueprint video_stream
from ai_backend.video_stream.routes import video_stream

app.register_blueprint(video_stream)
