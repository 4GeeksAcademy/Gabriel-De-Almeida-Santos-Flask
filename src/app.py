import os
from flask import Flask, request, jsonify, render_template
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import numpy as np

# -------------------------------------
# Configuración Flask
# -------------------------------------
app = Flask(__name__, template_folder="templates")

# Carpeta para guardar uploads
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -------------------------------------
# Cargar modelo (ruta relativa desde /src)
# -------------------------------------
MODEL_PATH = "../models/cats_vs_dogs_model.h5"

try:
    model = load_model(MODEL_PATH)
    print("Modelo cargado correctamente.")
except Exception as e:
    print("Error cargando el modelo:", e)
    raise e


# -------------------------------------
# Función de predicción
# -------------------------------------
def predict_image(path):
    """Carga una imagen y predice si es perro o gato."""
    img = load_img(path, target_size=(150, 150))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)[0][0]

    if pred > 0.5:
        return {
            "class": "dog",
            "confidence": float(pred)
        }
    else:
        return {
            "class": "cat",
            "confidence": float(1 - pred)
        }


# -------------------------------------
# Página principal (HTML)
# -------------------------------------
@app.route("/")
def home():
    return render_template("index.html")


# -------------------------------------
# Endpoint /predict (POST)
# -------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No se envió ninguna imagen"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "El nombre del archivo está vacío"}), 400

    # Guardar imagen temporalmente
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    # Predecir
    result = predict_image(filepath)
    return jsonify(result)


# -------------------------------------
# Ejecutar servidor
# -------------------------------------
if __name__ == "__main__":
    # MUY IMPORTANTE para Codespaces:
    # host="0.0.0.0" permite que GitHub exponga el puerto 5000
    app.run(host="0.0.0.0", port=5000, debug=True)
