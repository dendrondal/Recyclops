from flask import render_template, Response
from .config import app
from .models import Models
import cv2

application = app


@app.route("/video_feed")
def video_feed():
    return Response(capture(), mimetype='multipart/x-mixed-replace; boundary=frame')


def capture():
    camera = cv2.VideoCapture(0)

    while True:
        ret, img = camera.read()

        if ret:
            frame = cv2.imencode('.jpg', img)[1].tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            break


@app.route("/")
def main():
    return render_template("index.html")


@app.route("/<university>")
def get_university_guidelines(university):
    view = Models.query.filter(Models.university == university).first()



if __name__ == "__main__":
    main()
