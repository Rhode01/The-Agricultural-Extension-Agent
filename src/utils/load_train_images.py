import os
from typing import List
import tensorflow as tf

def clean_train_images(folder_path:str, class_names:List[str]):
    for classname in class_names:
        folder_path = os.path.join("../",folder_path, classname)
        for fname in os.listdir(folder_path):
            file = os.path.join(folder_path,fname)
            try:
                img_bytes = tf.io.read_file(file)
                tf.io.decode_image(img_bytes)
            except Exception as e:
                print(f"Failed to decode an image at {file} deleting the image")
                os.remove(file)
    return 