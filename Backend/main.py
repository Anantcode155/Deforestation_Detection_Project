from flask import Flask, render_template, request, jsonify
import os
from predict import predict as model_predict 

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route("/")
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict_route():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(image_path)

    # Call model
    result = model_predict(image_path)

    return jsonify({
        "label": result["class"],
        "confidence": result["confidence"],
        "accuracy": result["accuracy"]
    })

if __name__ == "__main__":
    app.run(debug=True)
