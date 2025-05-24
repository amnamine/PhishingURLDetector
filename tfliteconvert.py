import tensorflow as tf

# Step 1: Load the Keras .h5 model
model = tf.keras.models.load_model("dnn2.h5")

# Step 2: Convert to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Step 3: Save the converted .tflite model
with open("dnn2.tflite", "wb") as f:
    f.write(tflite_model)

print("Model converted and saved as dnn.tflite")
