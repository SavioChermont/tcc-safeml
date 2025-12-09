# -*- coding: utf-8 -*-
"""
Treina um classificador simples (scikit-learn) usando o dataset recortado.

Estrutura esperada:
    dataset/train/<classe>/*.jpg
    dataset/test/<classe>/*.jpg

Modelo: pipeline StandardScaler + LinearSVC sobre vetores de pixels
        (imagens redimensionadas para 30x30 e achatadas).
"""

from pathlib import Path
import numpy as np
from PIL import Image
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score
import joblib
import numpy as np

# Caminhos e parâmetros
REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = REPO_ROOT / "dataset"
IMG_SIZE = (30, 30)
MODEL_PATH = REPO_ROOT / "models" / "traffic_light_svm.joblib"

# Classes a usar no treino. Se vazio, usa todas encontradas.
CLASS_WHITELIST = {"go", "stop", "warning"}  # ajuste para incluir mais se quiser

# Número máximo de amostras por classe (para balancear). None = usa todas.
MAX_PER_CLASS = 3000

# Semente para reprodução no balanceamento
SEED = 43


def load_split(split: str):
    """Carrega imagens de dataset/<split>/<classe> com filtros/balanceamento."""
    split_dir = DATASET_ROOT / split
    X = []
    y = []
    classes = sorted([p.name for p in split_dir.iterdir() if p.is_dir()])
    if CLASS_WHITELIST:
        classes = [c for c in classes if c in CLASS_WHITELIST]

    rng = np.random.default_rng(SEED)

    for cls in classes:
        cls_dir = split_dir / cls
        imgs = list(cls_dir.glob("*.jpg"))

        # Balanceamento simples: limita a quantidade por classe.
        if MAX_PER_CLASS is not None and len(imgs) > MAX_PER_CLASS:
            imgs = rng.choice(imgs, size=MAX_PER_CLASS, replace=False)

        for img_path in imgs:
            with Image.open(img_path).convert("RGB") as img:
                img = img.resize(IMG_SIZE)
                arr = np.array(img, dtype=np.float32) / 255.0
                X.append(arr.flatten())
                y.append(cls)

    X = np.stack(X, axis=0)
    y = np.array(y)
    return X, y, classes


def train_and_eval():
    """Treina o modelo e avalia no conjunto de teste."""
    print(f"Lendo dados de {DATASET_ROOT}")
    X_train, y_train, classes = load_split("train")
    X_test, y_test, _ = load_split("test")

    print(f"Train samples: {len(y_train)}, Test samples: {len(y_test)}")

    model = make_pipeline(
        StandardScaler(with_mean=False),  # mantemos sparse-friendly; funciona com vetores densos também
        LinearSVC()
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"Acurácia (test): {acc:.4f}")
    print("Relatório de classificação:")
    print(classification_report(y_test, y_pred, labels=classes))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Modelo salvo em: {MODEL_PATH}")


if __name__ == "__main__":
    train_and_eval()
