import os
import uuid
import numpy as np
from flask import Flask, request, jsonify, render_template
from PIL import Image
import tflite_runtime.interpreter as tflite

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MODEL_PATH = os.path.join(BASE_DIR, "models", "cats_vs_dogs_model.tflite")

_interpreter = None
_input_index = None
_output_index = None

def get_interpreter():
    global _interpreter, _input_index, _output_index
    if _interpreter is None:
        _interpreter = tflite.Interpreter(model_path=MODEL_PATH)
        _interpreter.allocate_tensors()
        input_details = _interpreter.get_input_details()
        output_details = _interpreter.get_output_details()
        _input_index = input_details[0]["index"]
        _output_index = output_details[0]["index"]
        print("✅ TFLite cargado:", MODEL_PATH)
    return _interpreter

def preprocess_image(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize((150, 150))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

def predict_image(path: str):
    x = preprocess_image(path)
    interpreter = get_interpreter()
    interpreter.set_tensor(_input_index, x)
    interpreter.invoke()
    pred = float(interpreter.get_tensor(_output_index)[0][0])

    if pred > 0.5:
        return {"class": "dog", "confidence": pred}
    return {"class": "cat", "confidence": 1 - pred}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No se envió ninguna imagen"}), 400
    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "El nombre del archivo está vacío"}), 400

    filename = f"{uuid.uuid4().hex}_{file.filename}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    return jsonify(predict_image(filepath)), 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
