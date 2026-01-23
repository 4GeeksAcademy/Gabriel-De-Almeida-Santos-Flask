import tensorflow as tf

model = tf.keras.models.load_model("models/cats_vs_dogs_model.h5")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("models/cats_vs_dogs_model.tflite", "wb") as f:
    f.write(tflite_model)

print("Saved models/cats_vs_dogs_model.tflite")
