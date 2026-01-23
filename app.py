import os
import uuid
from flask import Flask, request, jsonify, render_template
import numpy as np

from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MODEL_PATH = os.path.join(BASE_DIR, "models", "cats_vs_dogs_model.h5")

# Lazy-load del modelo (se carga solo cuando alguien llama /predict)
_model = None


def get_model():
    """Carga el modelo solo una vez por worker."""
    global _model
    if _model is None:
        _model = load_model(MODEL_PATH)
        print("✅ Modelo cargado correctamente:", MODEL_PATH)
    return _model


def predict_image(path: str):
    """Predice gato/perro a partir de una imagen en disco."""
    img = load_img(path, target_size=(150, 150))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    model = get_model()
    pred = float(model.predict(img_array, verbose=0)[0][0])

    if pred > 0.5:
        return {"class": "dog", "confidence": pred}
    else:
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

    # Evita colisiones y nombres raros
    filename = f"{uuid.uuid4().hex}_{file.filename}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    result = predict_image(filepath)
    return jsonify(result), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
