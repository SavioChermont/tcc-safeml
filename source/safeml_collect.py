# -*- coding: utf-8 -*-
"""
Coleta métricas SafeML II (Wasserstein + p-valor) e salva em artifacts/<tag>/safeml_results.npz.
- Carrega modelo (svm ou cnn) e cache de treino padrão.
- Prediz o teste, separa acertos/erros por classe.
- Calcula WD/p-valor por pixel/canal entre treino e erros (e acertos como baseline).
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import joblib
from tensorflow.keras.models import load_model

from Wasserstein_Dist_PVal import Wasserstein_Dist_PVal

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

import config
from data_utils import load_split


def compute_wd_maps(train_set: np.ndarray, target_set: np.ndarray) -> Optional[Tuple[List[np.ndarray], List[np.ndarray]]]:
    if len(target_set) == 0 or len(train_set) == 0:
        return None
    H, W = config.IMG_SIZE
    train_imgs = train_set.reshape((-1, H, W, 3))
    target_imgs = target_set.reshape((-1, H, W, 3))
    wd_maps: List[np.ndarray] = []
    wd_sig_maps: List[np.ndarray] = []
    for ch in range(3):
        train_ch = train_imgs[:, :, :, ch].reshape(train_imgs.shape[0], -1)
        target_ch = target_imgs[:, :, :, ch].reshape(target_imgs.shape[0], -1)
        wd_vals = np.zeros(train_ch.shape[1], dtype=float)
        pvals = np.ones(train_ch.shape[1], dtype=float)
        for i in range(train_ch.shape[1]):
            pval, wd = Wasserstein_Dist_PVal(train_ch[:, i], target_ch[:, i])
            wd_vals[i] = wd
            pvals[i] = pval
        sig_mask = pvals < config.SAFE_PVAL_ALPHA
        wd_sig = wd_vals.copy()
        wd_sig[~sig_mask] = 0.0
        wd_maps.append(wd_vals.reshape(H, W))
        wd_sig_maps.append(wd_sig.reshape(H, W))
    return wd_maps, wd_sig_maps


def summarize_wd(train_set: np.ndarray, target_set: np.ndarray, rng: np.random.Generator) -> Dict[str, float]:
    if len(target_set) == 0 or len(train_set) == 0:
        return {"channels": [], "features_total": 0}
    max_n = config.WASSERSTEIN_MAX_SAMPLES
    if max_n is not None:
        if len(train_set) > max_n:
            train_set = train_set[rng.choice(len(train_set), size=max_n, replace=False)]
        if len(target_set) > max_n:
            target_set = target_set[rng.choice(len(target_set), size=max_n, replace=False)]
    H, W = config.IMG_SIZE
    train_imgs = train_set.reshape((-1, H, W, 3))
    target_imgs = target_set.reshape((-1, H, W, 3))
    channel_stats = []
    for ch in range(3):
        train_ch = train_imgs[:, :, :, ch].reshape(train_imgs.shape[0], -1)
        target_ch = target_imgs[:, :, :, ch].reshape(target_imgs.shape[0], -1)
        wd_vals = []
        pvals = []
        for i in range(train_ch.shape[1]):
            pval, wd = Wasserstein_Dist_PVal(train_ch[:, i], target_ch[:, i])
            wd_vals.append(wd)
            pvals.append(pval)
        wd_arr = np.array(wd_vals)
        pvals_arr = np.array(pvals)
        sig_mask = pvals_arr < config.SAFE_PVAL_ALPHA
        wd_sig = wd_arr[sig_mask]
        sig_count = int(sig_mask.sum())
        wd_mean_sig = float(wd_sig.mean()) if len(wd_sig) else 0.0
        channel_stats.append(
            {
                "channel": ch,
                "wd_mean_sig": wd_mean_sig,
                "sig_count": sig_count,
                "features": train_ch.shape[1],
            }
        )
    return {"channels": channel_stats, "features_total": 3 * train_ch.shape[1]}


def sample_set(arr: np.ndarray, paths: Optional[np.ndarray], max_n: Optional[int], rng: np.random.Generator):
    if max_n is not None and len(arr) > max_n:
        idx = rng.choice(len(arr), size=max_n, replace=False)
        arr_s = arr[idx]
        paths_s = paths[idx] if paths is not None else None
    else:
        arr_s = arr
        paths_s = paths
    return arr_s, paths_s


def sample_day_night_balanced(
    arr: np.ndarray,
    paths: Optional[np.ndarray],
    max_n: Optional[int],
    rng: np.random.Generator,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if paths is None or max_n is None or len(arr) <= max_n:
        return arr, paths
    day_mask = np.array(["daySequence" in p for p in paths])
    night_mask = np.array(["nightSequence" in p for p in paths])
    arr_day, paths_day = arr[day_mask], paths[day_mask]
    arr_night, paths_night = arr[night_mask], paths[night_mask]
    half = max_n // 2
    sel_day = rng.choice(len(arr_day), size=min(len(arr_day), half), replace=False) if len(arr_day) else np.array([], dtype=int)
    sel_night = rng.choice(len(arr_night), size=min(len(arr_night), half), replace=False) if len(arr_night) else np.array([], dtype=int)
    idx_chosen = set()
    chosen_arr = []
    chosen_paths = []
    for idx in sel_day:
        idx_chosen.add(int(np.where(day_mask)[0][idx]))
        chosen_arr.append(arr_day[idx])
        chosen_paths.append(paths_day[idx])
    for idx in sel_night:
        idx_chosen.add(int(np.where(night_mask)[0][idx]))
        chosen_arr.append(arr_night[idx])
        chosen_paths.append(paths_night[idx])
    if len(chosen_arr) < max_n:
        remaining_pool = [i for i in range(len(arr)) if i not in idx_chosen]
        extra = rng.choice(remaining_pool, size=min(len(remaining_pool), max_n - len(chosen_arr)), replace=False)
        for i in extra:
            chosen_arr.append(arr[i])
            chosen_paths.append(paths[i])
    return np.array(chosen_arr), np.array(chosen_paths) if paths is not None else None


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--tag", default="svm")
    parser.add_argument("--only-day", action="store_true")
    parser.add_argument("--only-night", action="store_true")
    args = parser.parse_args()

    if args.tag == "cnn":
        model_path = config.MODEL_PATH_CNN
        train_cache = config.TRAIN_DATA_PATH_CNN
    else:
        model_path = config.MODEL_PATH
        train_cache = config.TRAIN_DATA_PATH

    out_dir = config.ARTIFACTS_DIR / (args.tag or "")
    out_path = out_dir / "safeml_results.npz"

    print("Carregando dados para coleta SafeML (Wasserstein)...")
    data = np.load(train_cache, allow_pickle=True)
    X_train = data["X_train"]
    y_train = data["y_train"]
    classes = list(data["classes"])
    print(f"Cache de treino carregado: {len(y_train)} amostras (cache: {train_cache}).")

    X_test, y_test, _, paths_test = load_split("test", max_per_class=None, return_paths=True)

    if args.only_day and args.only_night:
        raise SystemExit("Use apenas um filtro: --only-day ou --only-night.")
    if args.only_day:
        mask = np.array(["daySequence" in p for p in paths_test])
        X_test, y_test, paths_test = X_test[mask], y_test[mask], paths_test[mask]
    elif args.only_night:
        mask = np.array(["nightSequence" in p for p in paths_test])
        X_test, y_test, paths_test = X_test[mask], y_test[mask], paths_test[mask]

    print(f"Teste: {len(y_test)} amostras.")

    is_cnn = model_path.suffix == ".h5"
    if is_cnn:
        model = load_model(model_path)
        predict_fn = lambda X: np.argmax(
            model.predict(X.reshape((-1, config.IMG_SIZE[0], config.IMG_SIZE[1], 3)), verbose=0), axis=1
        )
    else:
        model = joblib.load(model_path)
        predict_fn = model.predict
    y_pred = predict_fn(X_test)
    if is_cnn:
        y_pred = np.array([classes[int(i)] for i in y_pred])

    rng = np.random.default_rng(config.SEED)
    out_dir.mkdir(parents=True, exist_ok=True)
    balance_day_night = not args.only_day and not args.only_night

    results = {}

    for cls in classes:
        train_cls = X_train[y_train == cls]
        wrong_cls_mask = (y_test == cls) & (y_pred != cls)
        wrong_cls = X_test[wrong_cls_mask]
        wrong_paths_cls = paths_test[wrong_cls_mask] if paths_test is not None else None
        correct_cls_mask = (y_test == cls) & (y_pred == cls)
        correct_cls = X_test[correct_cls_mask]
        correct_paths_cls = paths_test[correct_cls_mask] if paths_test is not None else None

        train_sample, _ = sample_set(train_cls, None, config.WASSERSTEIN_MAX_SAMPLES, rng)
        if balance_day_night:
            wrong_sample, wrong_paths_sample = sample_day_night_balanced(
                wrong_cls, wrong_paths_cls, config.WASSERSTEIN_MAX_SAMPLES, rng
            )
            correct_sample, _ = sample_day_night_balanced(
                correct_cls, correct_paths_cls, config.WASSERSTEIN_MAX_SAMPLES, rng
            )
        else:
            wrong_sample, wrong_paths_sample = sample_set(
                wrong_cls, wrong_paths_cls, config.WASSERSTEIN_MAX_SAMPLES, rng
            )
            correct_sample, _ = sample_set(
                correct_cls, correct_paths_cls, config.WASSERSTEIN_MAX_SAMPLES, rng
            )

        summary_wrong = summarize_wd(train_sample, wrong_sample, rng)
        summary_correct = summarize_wd(train_sample, correct_sample, rng)

        wd_maps_wrong, wd_sig_maps_wrong = (compute_wd_maps(train_sample, wrong_sample) or (None, None))

        results[cls] = {
            "train_count": int(train_cls.shape[0]),
            "wrong_count": int(wrong_cls.shape[0]),
            "correct_count": int(correct_cls.shape[0]),
            "wrong_summary": summary_wrong,
            "correct_summary": summary_correct,
            "wrong_paths_used": wrong_paths_sample,
            "wd_maps_wrong": wd_maps_wrong,
            "wd_sig_maps_wrong": wd_sig_maps_wrong,
        }

        print(
            f"Classe {cls}: train={train_cls.shape[0]}, "
            f"wrong={wrong_cls.shape[0]}, correct={correct_cls.shape[0]}"
        )
        print("  Erros:")
        for ch_stat in summary_wrong["channels"]:
            print(
                f"  Canal {ch_stat['channel']}: wd_mean_sig={ch_stat['wd_mean_sig']:.6f}, "
                f"sig_count={ch_stat['sig_count']}/{ch_stat['features']}"
            )
        print("  Acertos (baseline):")
        for ch_stat in summary_correct["channels"]:
            print(
                f"  Canal {ch_stat['channel']}: wd_mean_sig={ch_stat['wd_mean_sig']:.6f}, "
                f"sig_count={ch_stat['sig_count']}/{ch_stat['features']}"
            )
        if wrong_paths_sample is not None:
            print("  Arquivos usados (erros amostrados):")
            for p in wrong_paths_sample:
                print(f"    {p}")

    np.savez_compressed(out_path, results=results, classes=np.array(classes))
    print(f"Resultados SafeML salvos em: {out_path}")


if __name__ == "__main__":
    main()
