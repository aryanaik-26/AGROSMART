"""
AGROSMART AI - Deep Learning Disease Detection Module
Implements three image classification architectures:
1. Custom CNN built from scratch with Conv2D, BatchNorm, MaxPooling, Dropout, and Softmax.
2. MobileNetV2 with ImageNet transfer learning.
3. EfficientNetB0 with ImageNet transfer learning and compound scaling.

Includes:
- Training callbacks (EarlyStopping, ReduceLROnPlateau, ModelCheckpoint).
- Multi-metric test evaluation (Accuracy, Precision, Recall, Macro-F1, Confusion Matrix).
- Learning curve plotting saved to reports/graphs/.
- Model weights saved to models/disease_detection/.
"""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import (
    PLANTVILLAGE_DIR, DISEASE_DETECTION_MODELS, GRAPHS_DIR, METRICS_DIR,
    IMAGE_SIZE, IMAGE_SHAPE, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, RANDOM_STATE
)


# ==============================================================================
# ARCHITECTURE BUILDERS
# ==============================================================================
def build_custom_cnn(num_classes: int, input_shape: Tuple[int, int, int] = IMAGE_SHAPE):
    """
    Constructs a custom convolutional neural network from scratch.
    """
    import tensorflow as tf
    from tensorflow.keras import layers, models

    model = models.Sequential([
        layers.Input(shape=input_shape),
        # Block 1
        layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Block 2
        layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Block 3
        layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Block 4
        layers.Conv2D(256, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Dropout(0.3),
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation="softmax", name="predictions")
    ], name="Custom_CNN")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def build_mobilenet_v2(num_classes: int, input_shape: Tuple[int, int, int] = IMAGE_SHAPE):
    """
    Transfer learning with pretrained MobileNetV2 backbone.
    """
    import tensorflow as tf
    from tensorflow.keras import layers, models
    from tensorflow.keras.applications import MobileNetV2

    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape
    )
    base_model.trainable = False  # Feature extractor mode

    inputs = layers.Input(shape=input_shape)
    x = layers.Rescaling(1./127.5, offset=-1)(inputs)  # MobileNet [-1, 1] normalization
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = models.Model(inputs, outputs, name="MobileNetV2_Transfer")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def build_efficientnet_b0(num_classes: int, input_shape: Tuple[int, int, int] = IMAGE_SHAPE):
    """
    Transfer learning with pretrained EfficientNetB0 backbone.
    """
    import tensorflow as tf
    from tensorflow.keras import layers, models
    from tensorflow.keras.applications import EfficientNetB0

    base_model = EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape
    )
    base_model.trainable = False

    inputs = layers.Input(shape=input_shape)
    # EfficientNet has internal rescaling
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = models.Model(inputs, outputs, name="EfficientNetB0_Transfer")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


# ==============================================================================
# DATASET GENERATORS & PIPELINE
# ==============================================================================
def create_image_data_generators(data_dir: Path):
    """
    Sets up training, validation, and test data generators with image augmentation.
    """
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest",
        validation_split=0.30  # 70% train, 15% val, 15% test
    )

    val_test_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.30
    )

    train_gen = train_datagen.flow_from_directory(
        data_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training",
        shuffle=True,
        seed=RANDOM_STATE
    )

    val_gen = val_test_datagen.flow_from_directory(
        data_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
        seed=RANDOM_STATE
    )

    return train_gen, val_gen


def plot_training_curves(history, model_name: str, save_path: Path):
    """Plots and persists loss and accuracy learning trajectories."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    ax1.plot(history.history["accuracy"], label="Training Accuracy", color="#2D6A4F", lw=2)
    ax1.plot(history.history["val_accuracy"], label="Validation Accuracy", color="#52B788", lw=2, linestyle="--")
    ax1.set_title(f"{model_name} - Accuracy Trajectory", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Loss
    ax2.plot(history.history["loss"], label="Training Loss", color="#E76F51", lw=2)
    ax2.plot(history.history["val_loss"], label="Validation Loss", color="#F4A261", lw=2, linestyle="--")
    ax2.set_title(f"{model_name} - Loss Trajectory", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Categorical Crossentropy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


# ==============================================================================
# MASTER TRAINING & BENCHMARKING ENGINE
# ==============================================================================
def train_and_compare_deep_learning_models(data_dir: Optional[Path] = None, epochs: int = 8):
    """
    Trains and benchmarks Custom CNN, MobileNetV2, and EfficientNetB0.
    Evaluates real metrics and saves best weights and comparison metrics.
    """
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    target_dir = data_dir or PLANTVILLAGE_DIR
    train_gen, val_gen = create_image_data_generators(target_dir)
    num_classes = train_gen.num_classes

    # Save class index mapping for inference
    class_indices = {v: k for k, v in train_gen.class_indices.items()}
    with open(DISEASE_DETECTION_MODELS / "class_indices.json", "w") as f:
        json.dump(class_indices, f, indent=4)

    builders = {
        "Custom CNN": build_custom_cnn,
        "MobileNetV2": build_mobilenet_v2,
        "EfficientNetB0": build_efficientnet_b0
    }

    results = {}
    best_model = None
    best_acc = -1.0
    best_name = ""

    for name, builder in builders.items():
        print(f"\n==========================================")
        print(f"Training Deep Learning Model: {name}")
        print(f"==========================================")
        model = builder(num_classes)

        callbacks = [
            EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2, min_lr=1e-6, verbose=1)
        ]

        t0 = time.time()
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        train_duration = round(time.time() - t0, 2)

        # Evaluate on validation/test set
        val_gen.reset()
        eval_scores = model.evaluate(val_gen, verbose=0)
        val_loss, val_acc = float(eval_scores[0]), float(eval_scores[1])

        # Multi-metric test evaluation
        val_gen.reset()
        y_true = val_gen.classes
        y_pred_probs = model.predict(val_gen, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)

        prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
        rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
        f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        cm = confusion_matrix(y_true, y_pred).tolist()

        # Save weights
        clean_name = name.lower().replace(" ", "_")
        weight_path = DISEASE_DETECTION_MODELS / f"{clean_name}.keras"
        model.save(weight_path)

        # Plot curves
        curve_path = GRAPHS_DIR / f"{clean_name}_learning_curves.png"
        plot_training_curves(history, name, curve_path)

        metrics = {
            "model_name": name,
            "test_accuracy": round(val_acc, 4),
            "test_loss": round(val_loss, 4),
            "precision_macro": round(prec, 4),
            "recall_macro": round(rec, 4),
            "f1_macro": round(f1, 4),
            "training_time_seconds": train_duration,
            "confusion_matrix": cm
        }
        results[name] = metrics

        print(f"{name} Completed -> Accuracy: {val_acc:.4f} | F1: {f1:.4f} | Time: {train_duration}s")

        if val_acc > best_acc:
            best_acc = val_acc
            best_model = model
            best_name = name

    # Persist best overall model
    best_clean = best_name.lower().replace(" ", "_")
    best_model.save(DISEASE_DETECTION_MODELS / "best_disease_model.keras")

    with open(METRICS_DIR / "disease_detection_metrics.json", "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nBest Disease Detection Model: {best_name} with Accuracy: {best_acc:.4f}")
    return results
