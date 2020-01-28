from flask import render_template, Response
from .config import app
from .models import Models
from cif3r.models import predict_model
import cv2

application = app


@app.route("/video_feed")
def video_feed():
    img = next(capture())
    frame = cv2.imencode('.jpg', img)[1].tobytes()
    return Response(
        (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'),
        mimetype='multipart/x-mixed-replace; boundary=frame'
        )


def capture():
    camera = cv2.VideoCapture(0)

    while True:
        ret, img = camera.read()

        if ret:
            yield img
        else:
            break


@app.route("/")
def main():
    return render_template("index.html")


@app.route("/<university>")
def get_university_guidelines(university):
    img_array = next(capture())
    flash(predict_model.clf_factory(university, img_array, ['paper', 'cans', 'plastic', 'trash']))
    return render_template('uni.html')


if __name__ == "__main__":
    main()
