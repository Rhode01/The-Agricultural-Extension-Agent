import tensorflow as tf
import numpy as np
from pathlib import Path
import os


def run_inference(image_path:Path)->str:
    class_names = ['']
    model_dir = Path(__file__).resolve().parents[1] / "models" / "saved_model"
    model_path = model_dir / "best_model.keras"
    model = tf.keras.models.load_model(model_path)

    _, img_h, img_w, _ = model.input_shape
    img = tf.keras.utils.load_img(
        image_path,
        target_size=(img_h, img_w)
    )

    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    preprocess_fn = getattr(
        tf.keras.applications,
        "EfficientNetV2SPreprocessInput",
        None
    )
    if preprocess_fn:
        img_array = preprocess_fn(img_array)
    else:
        img_array = img_array / 255.0

    preds = model.predict(img_array)
    idx = np.argmax(preds, axis=1)[0]
    return class_names[idx]







