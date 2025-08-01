import tensorflow as tf
from pathlib import Path
import sys
import os
PROJECT_ROOT = Path(__file__).resolve().parents[3] 
sys.path.insert(0, str(PROJECT_ROOT))  
from src.utils.data_utils import maize_disease_util

def train_maize_classfication_model():
    dataset_path = Path(__file__).resolve().parents[2]/ "data" / "maize_dataset"
    save_model_dir = Path(__file__).resolve().parents[1] / "saved_model"
    if save_model_dir:
        save_model_dir.mkdir(parents=True, exist_ok=True)
    obje = maize_disease_util(dataset_path, save_dir=save_model_dir)
    class_names = obje.list_classname()
    obje.clean_train_images(class_names)
    num_classes = len(class_names)
    model = obje.build_transfer_model(num_classes=num_classes)
    history1, history2 = obje.start_train_loop(model, epochs=25)
    return history1, history2

if __name__ =="__main__":
    print("Application starting .....")
    train_maize_classfication_model()