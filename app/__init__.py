from flask import Flask, render_template


app = Flask(__name__)


@app.route("/")
def main():
    return render_template("index.html")


@app.route("/<university>")
def get_university_guidelines(university):
    pass


if __name__ == "__main__":
    main()
