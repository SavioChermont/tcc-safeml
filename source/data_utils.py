# -*- coding: utf-8 -*-
"""Utilitários de carga do dataset recortado."""

import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

import config


def load_split(
    split: str,
    max_per_class: Optional[int] = None,
    return_paths: bool = False,
) -> Tuple[np.ndarray, np.ndarray, List[str], Optional[np.ndarray]]:
    """
    Carrega imagens de dataset/<split>/<classe>/*.jpg.

    - Redimensiona para IMG_SIZE de config, normaliza para [0,1] e achata.
    - Filtra classes por CLASS_WHITELIST de config.
    - Limita max_per_class se definido.
    Se return_paths=True, também retorna um array de paths correspondentes.
    """
    split_dir = config.DATASET_ROOT / split
    X: List[np.ndarray] = []
    y: List[str] = []
    paths: List[str] = []

    classes = sorted([p.name for p in split_dir.iterdir() if p.is_dir()])
    if config.CLASS_WHITELIST:
        classes = [c for c in classes if c in set(config.CLASS_WHITELIST)]

    rng = np.random.default_rng(config.SEED)

    for cls in classes:
        cls_dir = split_dir / cls
        imgs = list(cls_dir.glob("*.jpg"))
        if max_per_class is not None and len(imgs) > max_per_class:
            imgs = rng.choice(imgs, size=max_per_class, replace=False)

        for img_path in imgs:
            with Image.open(img_path).convert("RGB") as img:
                arr = np.array(img.resize(config.IMG_SIZE), dtype=np.float32) / 255.0
                X.append(arr.flatten())
                y.append(cls)
                if return_paths:
                    paths.append(str(img_path))

    X_arr = np.stack(X, axis=0)
    y_arr = np.array(y)
    if return_paths:
        paths_arr = np.array(paths)
        return X_arr, y_arr, classes, paths_arr
    else:
        return X_arr, y_arr, classes
