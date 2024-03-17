import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps


def load_saved_model(model_path):
    loaded_model = tf.keras.models.load_model(model_path)
    return loaded_model


def preprocess_input_image(image_path):
    input_image = Image.open(image_path)
    input_image = ImageOps.grayscale(input_image)
    input_image = input_image.resize((28, 28))
    input_image = np.array(input_image)
    input_image = input_image / 255.0
    input_image = np.expand_dims(input_image, axis=0)
    return input_image


def predict_with_model(model, input_image):
    prediction = model.predict(input_image)
    predicted_class = np.argmax(prediction)
    return predicted_class


if __name__ == "__main__":
    model_path = "my_model.h5"
    loaded_model = load_saved_model(model_path)
    input_image_path = "input_image.png"
    input_image = preprocess_input_image(input_image_path)
    predicted_class = predict_with_model(loaded_model, input_image)
    print(f"Predicted class: {predicted_class}")
