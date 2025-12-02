from utils import db_connect
engine = db_connect()

# your code here
from flask import Flask, request, jsonify, render_template
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# ---------------------------
# Configuración Flask
# ---------------------------
app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------------------------
# Cargar modelo (ruta segura)
# ---------------------------
MODEL_PATH = "../models/cats_vs_dogs_model.h5"
model = load_model(MODEL_PATH)

# ---------------------------
# Función de predicción
# ---------------------------
def predict_image(path):
    # Asegúrate de usar el tamaño que pusiste en tu modelo
    img = load_img(path, target_size=(150, 150))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]

    # Como tu modelo es binario (sigmoid):
    if prediction > 0.5:
        return {
            "class": "dog",
            "confidence": float(prediction)
        }
    else:
        return {
            "class": "cat",
            "confidence": float(1 - prediction)
        }

# ---------------------------
# Endpoint /predict
# ---------------------------
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file received"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # Guardar archivo
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    # Predecir
    result = predict_image(filepath)
    return jsonify(result)

# ---------------------------
# Ejecutar servidor
# ---------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

@app.route("/")
def home():
    return render_template("index.html")