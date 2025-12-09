# -*- coding: utf-8 -*-
"""Configurações compartilhadas do projeto."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = REPO_ROOT / "dataset"
MODEL_DIR = REPO_ROOT / "models"
MODEL_PATH = MODEL_DIR / "traffic_light_svm.joblib"

# Caminho para o dataset original LISA (anotações + frames)
LISA_ROOT = REPO_ROOT.parent / "Codigo" / "OriginalDataSet" / "LISA-dataset"

# Artefatos auxiliares
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
TRAIN_DATA_PATH = ARTIFACTS_DIR / "train_data.npz"

# Imagens redimensionadas para o classificador baseline
IMG_SIZE = (30, 30)

# Classes usadas no experimento atual
CLASS_WHITELIST = {"go", "stop", "warning"}

# Balanceamento opcional por classe (usado no treino); None usa todas
MAX_PER_CLASS_TRAIN = 3000

# Semente para reproduzir amostragem/embaralhamento
SEED = 43

# Amostragem para a análise Wasserstein (limita elementos por conjunto)
WASSERSTEIN_MAX_SAMPLES = 30

# Nível de significância para p-valor (SafeML II)
SAFE_PVAL_ALPHA = 0.05
