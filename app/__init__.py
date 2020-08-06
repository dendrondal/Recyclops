from flask import render_template, Response, url_for, request
from .config import app
from .models import ClassMapping
from cif3r.models import predict_model
from cif3r.data.recycling_guidelines import UNIVERSITIES
import cv2

application = app


@app.route("/video_feed", methods=["GET", "POST"])
def video_feed():
    return Response(capture(), mimetype="multipart/x-mixed-replace; boundary=frame")


def capture():
    camera = cv2.VideoCapture(0)

    while True:
        global img
        ret, img = camera.read()

        if ret:
            frame = cv2.imencode(".jpg", img)[1].tobytes()
            yield b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        else:
            break


@app.route("/")
def main():
    return render_template("index.html")


@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500


@app.route("/guidelines", methods=["GET", "POST"])
def get_university_guidelines():
    if request.method == "GET":
        university = request.args.get("location", "university") 
        
    if request.method == "POST":
        university = request.args.get("location")
<<<<<<< HEAD
        class_mappings = ClassMapping.query\
            .filter(ClassMapping.university == university)\
            .order_by(ClassMapping.key_index)
        classes = [row for row in class_mappings.label]

=======
        class_mappings = recycling_guidelines.UNIVERSITIES[university]['R']
        classes = [row for row in class_mappings.keys()]
        print(classes)
        classes.append('trash')
>>>>>>> 431e0eed28d8db687d109a29a0fa434483ccaaa2
        return render_template(
            "uni.html",
            result=predict_model.clf_factory(
                university, img, classes
            ),
        )

    return render_template("uni.html")


if __name__ == "__main__":
    app.run()
