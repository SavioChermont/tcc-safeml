# -*- coding: utf-8 -*-
"""
Avalia o classificador treinado (scikit-learn) no conjunto de teste.
Mostra acurácia global e contagem de corretas/erradas por classe.
"""
import sys
from pathlib import Path

import joblib
from sklearn.metrics import accuracy_score

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

import config
from data_utils import load_split


def main():
    print(f"Carregando modelo de: {config.MODEL_PATH}")
    model = joblib.load(config.MODEL_PATH)

    print(f"Lendo teste em: {config.DATASET_ROOT / 'test'}")
    X_test, y_test, classes = load_split(
        "test",
        max_per_class=None,
    )
    print(f"Test samples: {len(y_test)}; classes: {classes}")

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Acurácia (test): {acc:.4f}")

    # Contagem por classe
    print("Por classe:")
    for cls in classes:
        cls_mask = y_test == cls
        cls_total = int(cls_mask.sum())
        cls_correct = int(((y_pred == cls) & cls_mask).sum())
        cls_wrong = cls_total - cls_correct
        print(f"  {cls}: corretas={cls_correct}, erradas={cls_wrong}, total={cls_total}")

    total_correct = int((y_pred == y_test).sum())
    total_wrong = len(y_test) - total_correct
    print(f"Total corretas: {total_correct}, erradas: {total_wrong}")


if __name__ == "__main__":
    main()
