import tensorflow as tf
import numpy as np
from pathlib import Path
import os


def run_inference(image_path:Path)->str:
    class_names = ['']
    model = tf.keras.model.load_model(Path(__file__).resolve().parents[1] / "models" /"saved_model"/"best_model.keras")
    img =tf.keras.utils.load_img(
    image_path,target_size=model.input_shape)
    img_bytes = tf.keras.utils.img_to_array(
        img
    )
    img_dims = np.expand_dims(img_bytes, axis=0)

    predictions = model.predict(img_dims)
    index = np.argmax(predictions)
    prediction_clas = class_names[index]
    return prediction_clas







