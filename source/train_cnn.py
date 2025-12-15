# -*- coding: utf-8 -*-
"""
Treina uma CNN simples (Keras) para o problema de semáforos.

Mantém o mesmo input 30x30 usado no SVM para poder comparar e reutilizar SafeML.
Salva:
- Modelo: models/traffic_light_cnn.h5
- Cache para SafeML: artifacts/train_data_cnn.npz (flatten + classes)
"""

import sys
from pathlib import Path

import numpy as np
from tensorflow.keras import layers, models, utils
from sklearn.metrics import classification_report, accuracy_score

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

import config
from data_utils import load_split


def build_cnn(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(128, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def main():
    print(f"Lendo dados de {config.DATASET_ROOT}")
    X_train_flat, y_train_str, classes = load_split(
        "train",
        max_per_class=config.MAX_PER_CLASS_TRAIN,
        return_paths=False,
    )
    X_test_flat, y_test_str, _ = load_split(
        "test",
        max_per_class=None,
        return_paths=False,
    )

    # reshape para CNN
    H, W = config.IMG_SIZE
    X_train = X_train_flat.reshape((-1, H, W, 3))
    X_test = X_test_flat.reshape((-1, H, W, 3))

    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_train = np.array([class_to_idx[c] for c in y_train_str])
    y_test = np.array([class_to_idx[c] for c in y_test_str])

    y_train_cat = utils.to_categorical(y_train, num_classes=len(classes))
    y_test_cat = utils.to_categorical(y_test, num_classes=len(classes))

    print(f"Train samples: {len(y_train)}, Test samples: {len(y_test)}")
    model = build_cnn(input_shape=(H, W, 3), num_classes=len(classes))

    model.fit(
        X_train,
        y_train_cat,
        epochs=15,
        batch_size=32,
        validation_data=(X_test, y_test_cat),
        verbose=2,
    )

    y_pred = model.predict(X_test, verbose=0)
    y_pred_labels = np.argmax(y_pred, axis=1)
    acc = accuracy_score(y_test, y_pred_labels)
    print(f"Acurácia (test): {acc:.4f}")
    print(classification_report(y_test, y_pred_labels, target_names=classes))

    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save(config.MODEL_PATH_CNN)
    print(f"Modelo CNN salvo em: {config.MODEL_PATH_CNN}")

    # Salva cache para SafeML (flatten + classes)
    config.ARTIFACTS_DIR_CNN.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        config.TRAIN_DATA_PATH_CNN,
        X_train=X_train_flat,
        y_train=y_train_str,
        classes=np.array(classes),
    )
    print(f"Cache CNN para SafeML salvo em: {config.TRAIN_DATA_PATH_CNN}")


if __name__ == "__main__":
    main()
