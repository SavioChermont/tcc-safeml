# -*- coding: utf-8 -*-
"""Configurações compartilhadas do projeto."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = REPO_ROOT / "dataset"
MODEL_DIR = REPO_ROOT / "models"
MODEL_PATH = MODEL_DIR / "traffic_light_svm.joblib"

# Segundo modelo (CNN)
MODEL_PATH_CNN = MODEL_DIR / "traffic_light_cnn.h5"

# Caminho para o dataset original LISA (anotações + frames)
LISA_ROOT = REPO_ROOT.parent / "Codigo" / "OriginalDataSet" / "LISA-dataset"

# Artefatos auxiliares
ARTIFACTS_DIR = REPO_ROOT / "artifacts"

# Subpastas por modelo
ARTIFACTS_DIR_SVM = ARTIFACTS_DIR / "svm"
ARTIFACTS_DIR_CNN = ARTIFACTS_DIR / "cnn"

# Caches de treino (flatten) por modelo
TRAIN_DATA_PATH = ARTIFACTS_DIR_SVM / "train_data.npz"       # SVM
TRAIN_DATA_PATH_CNN = ARTIFACTS_DIR_CNN / "train_data.npz"   # CNN

# Imagens redimensionadas para o classificador baseline
IMG_SIZE = (30, 30)

# Classes usadas no experimento atual
CLASS_WHITELIST = {"go", "stop", "warning"}

# Balanceamento opcional por classe (usado no treino); None usa todas
MAX_PER_CLASS_TRAIN = 3000

# Semente para reproduzir amostragem/embaralhamento
SEED = 43

# Amostragem para a análise Wasserstein (limita elementos por conjunto)
WASSERSTEIN_MAX_SAMPLES = 15

# Nível de significância para p-valor (SafeML II)
SAFE_PVAL_ALPHA = 0.05

# Limites para acelerar a avaliação com limiar (wd por amostra)
# Subamostra de imagens de treino da classe e de pixels analisados.
WD_TRAIN_LIMIT = 300        # número máximo de exemplos da classe
WD_PIXEL_LIMIT = 300        # número máximo de pixels (de 900)

# Para heatmaps: se True, usa apenas 1 exemplo errado (além do treino amostrado)
HEATMAP_USE_SINGLE_WRONG = True
