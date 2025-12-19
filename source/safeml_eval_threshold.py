# -*- coding: utf-8 -*-
"""Avalia limiar SafeML em buffer: amostra 10 imagens por classe, prediz com o modelo e compara WD da classe majoritária com o treino."""

import argparse
from collections import Counter
from typing import Dict, List, Optional

import numpy as np
import joblib
from tensorflow.keras.models import load_model

import config
from Wasserstein_Dist_PVal import Wasserstein_Dist_PVal
from data_utils import load_split


def compute_wd_mean_sig_buffer(train_cls: np.ndarray, buffer_cls: np.ndarray) -> float:
    H, W = config.IMG_SIZE
    rng = np.random.default_rng(config.SEED)

    if len(train_cls) > config.WD_TRAIN_LIMIT:
        train_cls = train_cls[rng.choice(len(train_cls), size=config.WD_TRAIN_LIMIT, replace=False)]

    train_imgs = train_cls.reshape((-1, H, W, 3))
    buffer_imgs = buffer_cls.reshape((-1, H, W, 3))

    pixel_indices = np.arange(H * W)
    if len(pixel_indices) > config.WD_PIXEL_LIMIT:
        pixel_indices = rng.choice(pixel_indices, size=config.WD_PIXEL_LIMIT, replace=False)

    wd_means = []
    for ch in range(3):
        train_ch = train_imgs[:, :, :, ch].reshape(train_imgs.shape[0], -1)
        buf_ch = buffer_imgs[:, :, :, ch].reshape(buffer_imgs.shape[0], -1)

        wd_vals = np.zeros(len(pixel_indices), dtype=float)
        pvals = np.ones(len(pixel_indices), dtype=float)
        for j, i in enumerate(pixel_indices):
            pval, wd = Wasserstein_Dist_PVal(train_ch[:, i], buf_ch[:, i])
            wd_vals[j] = wd
            pvals[j] = pval

        sig_mask = pvals < config.SAFE_PVAL_ALPHA
        wd_sig = wd_vals[sig_mask]
        wd_means.append(float(wd_sig.mean()) if len(wd_sig) else 0.0)

    return float(np.mean(wd_means))


def sample_sequential_per_class(
    X: np.ndarray,
    y: np.ndarray,
    paths: np.ndarray,
    classes: List[str],
    filter_substr: Optional[str] = None,
):
    rng = np.random.default_rng(config.SEED)
    if filter_substr:
        mask = np.array([filter_substr in p for p in paths])
        X, y, paths = X[mask], y[mask], paths[mask]

    per_class = 10
    idxs = []
    for cls in classes:
        cls_idx = np.where(y == cls)[0]
        if len(cls_idx) == 0:
            continue
        cls_sorted = cls_idx[np.argsort(paths[cls_idx])]
        take = min(per_class, len(cls_sorted))
        if len(cls_sorted) > take:
            start = int(rng.integers(0, len(cls_sorted) - take))
            sel = cls_sorted[start : start + take]
        else:
            sel = cls_sorted
        idxs.extend(sel.tolist())
    idxs = np.array(idxs)
    return X[idxs], y[idxs], paths[idxs]


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--tag", default="svm")
    parser.add_argument("--thresholds", type=str, default=None)
    parser.add_argument("--only-day", action="store_true")
    parser.add_argument("--only-night", action="store_true")
    args = parser.parse_args()

    # Modelos e caches padrão
    if args.tag == "cnn":
        model_path = config.MODEL_PATH_CNN
        train_cache = config.TRAIN_DATA_PATH_CNN
    else:
        model_path = config.MODEL_PATH
        train_cache = config.TRAIN_DATA_PATH

    data = np.load(train_cache, allow_pickle=True)
    X_train = data["X_train"]
    y_train = data["y_train"]
    classes = list(data["classes"])

    X_test, y_test, _, paths_test = load_split("test", max_per_class=None, return_paths=True)

    filt = "daySequence" if args.only_day else "nightSequence" if args.only_night else None
    X_buf, y_buf, paths_buf = sample_sequential_per_class(
        X_test, y_test, paths_test, classes, filter_substr=filt
    )

    # Carrega modelo para predição
    is_cnn = model_path.suffix == ".h5"
    if is_cnn:
        model = load_model(model_path)
        y_pred_idx = np.argmax(
            model.predict(X_buf.reshape((-1, config.IMG_SIZE[0], config.IMG_SIZE[1], 3)), verbose=0),
            axis=1,
        )
        y_pred = np.array([classes[int(i)] for i in y_pred_idx])
    else:
        model = joblib.load(model_path)
        y_pred = model.predict(X_buf)

    thresholds_map: Dict[str, float] = {}
    if args.thresholds:
        for kv in args.thresholds.split(","):
            if "=" in kv:
                k, v = kv.split("=")
                thresholds_map[k.strip()] = float(v)

    # Avalia por classe (buffer de 10 imagens verdadeiras dessa classe)
    for cls in classes:
        buf_mask = y_buf == cls
        if not np.any(buf_mask):
            continue
        X_cls = X_buf[buf_mask]
        preds_cls = y_pred[buf_mask]
        paths_cls = paths_buf[buf_mask]
        majority_pred = Counter(preds_cls).most_common(1)[0][0]

        train_cls = X_train[y_train == majority_pred]
        wd_mean_sig = compute_wd_mean_sig_buffer(train_cls, X_cls)
        thr = thresholds_map.get(majority_pred, 0.2)
        accepted = wd_mean_sig <= thr
        print(
            f"True class {cls}: pred_majority={majority_pred} "
            f"wd_mean_sig={wd_mean_sig:.4f} limiar={thr:.4f} accept={accepted}"
        )
        for p in paths_cls[:2]:
            print(f"  {cls}: {p}")


if __name__ == "__main__":
    main()
