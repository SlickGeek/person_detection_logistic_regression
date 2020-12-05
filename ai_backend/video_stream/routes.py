from flask import Blueprint, Response, request, jsonify
from ai_backend.video_stream.video_camera import VideoCamera
import cv2
from ai_backend import predict

video_stream = Blueprint('video_stream', __name__)


def gen_frame(camera):
    while True:
        jpeg, frame = camera.get_frame()

        if int(predict.predict_image(frame)) == 1:
            label_image = 'Persona detectada'
            color = (31, 31, 234)
        else:
            color = (23, 211, 23)
            label_image = 'Persona no detectada'
        # # print(predict.predict_image(frame))
        
        # # process and add text to the image
        height, width, channels = jpeg.shape
        # # print(height, width)
        cv2.putText(jpeg, label_image, (int(width * .03), int(height * .9)), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3)
        ret, jpeg = cv2.imencode('.jpg', jpeg)

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')


@video_stream.route('/video_feed', methods=['GET'])
def video_feed():
    path = request.args.get('path')
    return Response(gen_frame(VideoCamera(path)), mimetype='multipart/x-mixed-replace; boundary=frame')
