from flask import Flask, request, jsonify, render_template
from predict import predict_deepseek
import os

app = Flask(__name__)

# Ensure temp directory exists
os.makedirs('temp', exist_ok=True)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400
        
        file = request.files["image"]
        if file.filename == '':
            return jsonify({"error": "No image selected"}), 400
        
        image_path = os.path.join('temp', "temp.jpg")
        file.save(image_path)
        
        result = predict_deepseek(image_path)
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(port=5000, debug=True)