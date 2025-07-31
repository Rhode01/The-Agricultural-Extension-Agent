import tensorflow as tf
from pathlib import Path
import os
from typing import Tuple, List, Optional
from tensorflow.keras import models,callbacks,layers
class maize_disease_util:
    def __init__(
        self,
        dataset_dir: Path,
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
    def build_transfer_model(self,num_classes: int,base_model_name: str = "EfficientNetB0",
        freeze_backbone: bool = True,
    ) -> tf.keras.Model:
        backbone_cls = getattr(tf.keras.applications, base_model_name)
        backbone = backbone_cls(
            include_top=False,
            weights="imagenet",
            input_shape=(*self.image_size, 3),
            pooling=None,
        )
        if freeze_backbone:
            backbone.trainable = False

        inputs = tf.keras.Input(shape=(*self.image_size, 3))
        x = self.data_augmentation()(inputs)

        preprocess_fn = getattr(tf.keras.applications, f"{base_model_name}PreprocessInput", None)
        if preprocess_fn:
            x = preprocess_fn(x)
        else:
            x = layers.Rescaling(1.0 / 255)(x)

        x = backbone(x, training=not freeze_backbone)
        x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(
            256, activation="relu", kernel_regularizer=regularizers.l2(), name="dense_1"
        )(x)
        x = layers.Dropout(0.3, name="dropout_dense_1")(x)
        outputs = layers.Dense(num_classes, activation="softmax", name="output_layer")(x)

        model = tf.keras.Model(inputs, outputs, name=f"{base_model_name}_classifier")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-4),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        model.summary()
        return model
    def get_callbacks(self) -> List[callbacks.Callback]:
        return [
            callbacks.EarlyStopping(
                monitor="val_accuracy", patience=5, restore_best_weights=True
            ),
            callbacks.ModelCheckpoint(
                filepath=str(self.save_dir / "best_model.keras"),
                monitor="val_accuracy",
                save_best_only=True,
                verbose=1,
            )]
    def start_training(self):
        model = self.build_finetune_model()
        model.fit(self.create_train_val_dataset(),
                    epochs =20,
                    verbose=1,
                    validation_data = validation_dataset,
                    callbacks =self.add_callsbacks())
                    
    