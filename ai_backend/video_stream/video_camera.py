import cv2
import numpy as np


class VideoCamera:
    def __init__(self, path):
        self.video = cv2.VideoCapture('/Users/davidbanda/Coding/person_detection_logistic_regression/ai_backend'
                                      '/static/video/person.mp4')
        # self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    # def get_frame(self):
    #     ret, image = self.video.read()
    #     ret, jpeg = cv2.imencode('.jpg', image)
    #     return jpeg.tobytes()

    def get_frame(self):
        ret, image = self.video.read()
        # ret, image = cv2.imencode('.jpg', image)
        frame = cv2.resize(image, (128, 128), interpolation=cv2.INTER_CUBIC)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.array(frame[None])
        frame = frame.reshape(frame.shape[0], -1).T
        frame = frame / 255.
        return image, frame
