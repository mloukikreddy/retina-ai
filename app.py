import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from predict import predict_dr

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join("static", "uploads")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return render_template("predict.html")

    if "fundus" not in request.files or "oct" not in request.files:
        return jsonify({"error": "Both fundus and OCT images are required."}), 400

    fundus_file = request.files["fundus"]
    oct_file    = request.files["oct"]

    if fundus_file.filename == "" or oct_file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    if not allowed_file(fundus_file.filename) or not allowed_file(oct_file.filename):
        return jsonify({"error": "Only PNG, JPG, JPEG files are allowed."}), 400

    fundus_filename = secure_filename(fundus_file.filename)
    oct_filename    = secure_filename(oct_file.filename)

    fundus_path = os.path.join(app.config["UPLOAD_FOLDER"], fundus_filename)
    oct_path    = os.path.join(app.config["UPLOAD_FOLDER"], oct_filename)

    fundus_file.save(fundus_path)
    oct_file.save(oct_path)

    try:
        prediction, confidence = predict_dr(fundus_path, oct_path)
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    return jsonify({
        "prediction": prediction,
        "confidence": confidence,
        "fundus_img": fundus_filename,
        "oct_img":    oct_filename
    })


@app.route("/result")
def result():
    return render_template("result.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/signin")
def signin():
    return render_template("signin.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
