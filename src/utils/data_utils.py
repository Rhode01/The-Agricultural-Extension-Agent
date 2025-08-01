import tensorflow as tf
from pathlib import Path
import os
from typing import Tuple, List, Optional
from tensorflow.keras import models,callbacks,layers
class maize_disease_util:
    def __init__(
        self,
        dataset_dir: Optional[Path] = None,
        image_size: Tuple[int, int] = (256, 256),
        batch_size: int = 32,
        save_dir: Optional[Path] = None,
        log_dir: Optional[Path] = None,
    ) -> None:
        self.dataset_dir =dataset_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.save_dir = save_dir
        self.log_dir = log_dir
    def data_augmentation(self) -> tf.keras.models.Sequential:
        return models.Sequential(
            [
                layers.RandomFlip("horizontal_and_vertical"),
                layers.RandomRotation(0.2),
                layers.RandomZoom(0.2),
                layers.RandomBrightness(0.2),
                layers.RandomContrast(0.2),
            ]
        )
    def create_train_val_dataset(self, validation_split: float = 0.2, 
            seed: int = 123
    ) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        train_ds = tf.keras.utils.image_dataset_from_directory(
            self.dataset_dir,
            validation_split=validation_split,
            subset="training",
            seed=seed,
            image_size=self.image_size,
            batch_size=self.batch_size,
            label_mode="int",
        )
        val_ds = tf.keras.utils.image_dataset_from_directory(
            self.dataset_dir,
            validation_split=validation_split,
            subset="validation",
            seed=seed,
            image_size=self.image_size,
            batch_size=self.batch_size,
            label_mode="int",
        )
        AUTOTUNE = tf.data.AUTOTUNE
        train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
        return train_ds, val_ds
    def clean_train_images(self,folder_path:Path, class_names:List[str]):
        for classname in class_names:
            folder_path = folder_path/classname
            for fname in os.listdir(folder_path):
                file = os.path.join(folder_path,fname)
                try:
                    img_bytes = tf.io.read_file(file)
                    tf.io.decode_image(img_bytes)
                except Exception as e:
                    print(f"Failed to decode an image at {file} deleting the image")
                    os.remove(file)
        return
    def list_classname(self,dataset_path:Path)-> List[str]:
        class_names = []
        for class_name in os.listdir(dataset_path):
            class_names.append(class_name)
        return class_names

    def build_transfer_model(self,num_classes: int,base_model_name: str = "EfficientNetV2S",
        freeze_backbone: bool = True,
    ) -> tf.keras.Model:
        backbone_cls = getattr(tf.keras.applications, base_model_name)
        backbone = backbone_cls(
            include_top=False,
            weights="imagenet",
            input_shape=(*self.image_size, 3),
            pooling=None,
        )
        self.base_model = backbone 
        if freeze_backbone:
            backbone.trainable = False

        inputs = tf.keras.layers.Input(shape=(*self.image_size, 3))
        x = self.data_augmentation()(inputs)

        preprocess_fn = getattr(tf.keras.applications, f"{base_model_name}PreprocessInput", None)
        if preprocess_fn:
            x = preprocess_fn(x)
        else:
            x = layers.Rescaling(1.0 / 255.0)(x)

        x = backbone(x, training=not freeze_backbone)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(256, activation="relu")(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(num_classes, activation="softmax")(x)

        model = tf.keras.Model(inputs, outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-4),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        return model
    def get_callbacks(self) -> List[callbacks.Callback]:
        return [
            callbacks.EarlyStopping(
                monitor="val_accuracy", patience=5, restore_best_weights=True,mode='max', verbose=1),
            callbacks.ModelCheckpoint(
                filepath=str(self.save_dir / "best_model.keras"),
                monitor="val_accuracy",
                save_best_only=True,
                verbose=1,
                mode="max"
            )]
    def start_train_loop(self,model:tf.keras.models.Model,epochs:int =20):
        train_ds, val_ds = self.create_train_val_dataset()
        callbacks = self.get_callbacks()
        version_one = model.fit(
            train_ds,
            validation_data = val_ds,
            epochs = epochs,
            callbacks = callbacks
        )
        for layer in self.base_model.layers[-20:]:
            layer.trainable = True

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        history2 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=10,
            callbacks=callbacks
        )
        model.save(self.save_dir / "model_finetuned.keras")
        print(" Saved fine-tuned model to model_finetuned.keras")
        return version_one, history2
                    
    