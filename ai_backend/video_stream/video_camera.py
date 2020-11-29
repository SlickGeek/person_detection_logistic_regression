import cv2
import numpy as np


class VideoCamera:
    def __init__(self):
        self.video = cv2.VideoCapture('/Users/davidbanda/Coding/person_detection_logistic_regression/ai_backend'
                                      '/static/video/person.mp4')
        # self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        ret, image = self.video.read()
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()

    def detect_person(self):
        ret, image = self.video.read()
        ret, image = cv2.imencode('.jpg', image)
        image = cv2.resize(image, (64, 64), interpolation=cv2.INTER_CUBIC)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.array(image[None])
        image = image.reshape(image.shape[0], -1).T
        image = image / 255.
        return image
