from flask import render_template, Response, url_for, request
from .config import app
from .models import Models
#from cif3r.models import predict_model
import cv2

application = app


@app.route("/video_feed")
def video_feed():
    camera = capture()
    for img in camera:
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


@app.route("/guidelines", methods=['GET', 'POST'])
def get_university_guidelines():
    if request.method == 'GET':
        university = request.args.get('location', 'university')
    if request.method == 'POST':
        return render_template('uni.html', result="Worked!")
    return render_template('uni.html')

#predict_model.clf_factory(university, next(capture()), ['paper', 'cans', 'plastic', 'trash'])
if __name__ == "__main__":
    app.run()
