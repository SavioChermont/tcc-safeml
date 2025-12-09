# -*- coding: utf-8 -*-
"""
Treina um classificador simples (scikit-learn) usando o dataset recortado.

Estrutura esperada:
    dataset/train/<classe>/*.jpg
    dataset/test/<classe>/*.jpg

Modelo: pipeline StandardScaler + LinearSVC sobre vetores de pixels
        (imagens redimensionadas para 30x30 e achatadas).
"""
import sys
from pathlib import Path

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score
import joblib

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

import config
from data_utils import load_split


def train_and_eval():
    """Treina o modelo e avalia no conjunto de teste."""
    print(f"Lendo dados de {config.DATASET_ROOT}")
    X_train, y_train, classes = load_split(
        "train",
        max_per_class=config.MAX_PER_CLASS_TRAIN,
    )
    X_test, y_test, _ = load_split(
        "test",
        max_per_class=None,  # avalia com tudo no teste
    )

    print(f"Train samples: {len(y_train)}, Test samples: {len(y_test)}")

    model = make_pipeline(
        StandardScaler(with_mean=False),  # funciona bem com vetores densos e mantém compatibilidade com matriz esparsa
        LinearSVC(),
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"Acurácia (test): {acc:.4f}")
    print("Relatório de classificação:")
    print(classification_report(y_test, y_pred, labels=classes))

    config.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, config.MODEL_PATH)
    print(f"Modelo salvo em: {config.MODEL_PATH}")


if __name__ == "__main__":
    train_and_eval()
