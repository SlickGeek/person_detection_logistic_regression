from ai_backend.neural_network.predict import predict
from ai_backend.neural_network.logistic_regression import logistic_regression


class PredictImage:

    def __init__(self):
        self.w, self.b = logistic_regression()

    def predict_image(self, img):
        pred = predict(self.w, self.b, img)
        return pred[0, 0]
