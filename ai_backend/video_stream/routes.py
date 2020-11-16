from flask import Blueprint, Response
from camera import Camera
from ai_backend.video_stream.video_camera import VideoCamera
from flask_gzip import Gzip

video_stream = Blueprint('video_stream', __name__)


def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@video_stream.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()), mimetype='multipart/x-mixed-replace; boundary=frame')
