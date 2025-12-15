# -*- coding: utf-8 -*-
"""
Coleta métricas SafeML II (Wasserstein + p-valor) e salva para uso posterior
(heatmaps/limiares) sem recalcular.

Fluxo:
- Carrega cache de treino (train_data.npz) e teste.
- Prediz no teste.
- Para cada classe: amostra até WASSERSTEIN_MAX_SAMPLES de treino/erros/acertos.
- Calcula WD/p-valor por pixel/canal (R,G,B) entre treino e erros (e também acertos).
- Salva estatísticas e mapas (wd_maps e wd_sig_maps) em artifacts/safeml_results.npz.

Depois use `safeml_heatmaps.py` para gerar os heatmaps a partir desse arquivo.
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


def compute_wd_maps(
    train_set: np.ndarray, target_set: np.ndarray
) -> Optional[Tuple[List[np.ndarray], List[np.ndarray]]]:
    """Retorna listas de wd_maps e wd_sig_maps (3 canais, shape 30x30)."""
    if len(target_set) == 0 or len(train_set) == 0:
        return None

    H, W = config.IMG_SIZE
    train_imgs = train_set.reshape((-1, H, W, 3))
    target_imgs = target_set.reshape((-1, H, W, 3))

    wd_maps: List[np.ndarray] = []
    wd_sig_maps: List[np.ndarray] = []

    for ch in range(3):
        train_ch = train_imgs[:, :, :, ch].reshape(train_imgs.shape[0], -1)  # (N, 900)
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
    """Estatísticas resumidas (média dos WDs significativos) por canal."""
    if len(target_set) == 0 or len(train_set) == 0:
        return {"channels": [], "features_total": 0}

    max_n = config.WASSERSTEIN_MAX_SAMPLES
    if max_n is not None:
        if len(train_set) > max_n:
            idx = rng.choice(len(train_set), size=max_n, replace=False)
            train_set = train_set[idx]
        if len(target_set) > max_n:
            idx = rng.choice(len(target_set), size=max_n, replace=False)
            target_set = target_set[idx]

    H, W = config.IMG_SIZE
    train_imgs = train_set.reshape((-1, H, W, 3))
    target_imgs = target_set.reshape((-1, H, W, 3))

    channel_stats = []
    for ch in range(3):
        train_ch = train_imgs[:, :, :, ch].reshape(train_imgs.shape[0], -1)
        target_ch = target_imgs[:, :, :, ch].reshape(target_imgs.shape[0], -1)
        wd_vals: List[float] = []
        pvals: List[float] = []
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
    """Subamostra array (e paths correspondentes) até max_n."""
    if max_n is not None and len(arr) > max_n:
        idx = rng.choice(len(arr), size=max_n, replace=False)
        arr_s = arr[idx]
        paths_s = paths[idx] if paths is not None else None
    else:
        arr_s = arr
        paths_s = paths
    return arr_s, paths_s


def main():
    parser = argparse.ArgumentParser(description="Coleta SafeML (Wasserstein + p-valor)")
    parser.add_argument("--model-path", default=None, help="Caminho do modelo (joblib ou .h5). Se omitido, usa o padrão da tag.")
    parser.add_argument("--train-cache", default=None, help="Cache de treino (npz) com X_train/y_train/classes. Se omitido, usa o padrão da tag.")
    parser.add_argument("--tag", default="svm", help="Tag do modelo (ex.: svm, cnn) para escolher caminhos padrão e subpasta de saída.")
    parser.add_argument("--output", default=None, help="Arquivo de saída .npz (opcional; sobrescreve --tag)")
    args = parser.parse_args()

    # Define caminhos padrão por tag
    if args.tag == "cnn":
        default_model = config.MODEL_PATH_CNN
        default_cache = config.TRAIN_DATA_PATH_CNN
    else:
        default_model = config.MODEL_PATH
        default_cache = config.TRAIN_DATA_PATH

    model_path = Path(args.model_path) if args.model_path else default_model
    train_cache = Path(args.train_cache) if args.train_cache else default_cache
    if not model_path.is_absolute():
        model_path = config.REPO_ROOT / model_path
    if not train_cache.is_absolute():
        train_cache = config.REPO_ROOT / train_cache

    if args.output:
        out_path = Path(args.output)
        if not out_path.is_absolute():
            out_path = config.REPO_ROOT / out_path
        out_dir = out_path.parent
    else:
        out_dir = config.ARTIFACTS_DIR / (args.tag or "")
        out_path = out_dir / "safeml_results.npz"

    print("Carregando dados para coleta SafeML (Wasserstein)...")
    if not train_cache.exists():
        raise SystemExit("Cache de treino não encontrado. Rode o treinamento antes.")

    data = np.load(train_cache, allow_pickle=True)
    X_train = data["X_train"]
    y_train = data["y_train"]
    classes = list(data["classes"])
    print(f"Cache de treino carregado: {len(y_train)} amostras (cache: {train_cache}).")

    X_test, y_test, _, paths_test = load_split("test", max_per_class=None, return_paths=True)
    print(f"Teste: {len(y_test)} amostras.")

    # Carrega modelo (joblib ou .h5)
    is_cnn = model_path.suffix == ".h5"
    if is_cnn:
        model = load_model(model_path)
        predict_fn = lambda X: np.argmax(model.predict(X.reshape((-1, config.IMG_SIZE[0], config.IMG_SIZE[1], 3)), verbose=0), axis=1)
    else:
        model = joblib.load(model_path)
        predict_fn = model.predict
    y_pred = predict_fn(X_test)
    if is_cnn:
        y_pred = np.array([classes[int(i)] for i in y_pred])

    rng = np.random.default_rng(config.SEED)
    out_dir.mkdir(parents=True, exist_ok=True)

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
        wrong_sample, wrong_paths_sample = sample_set(wrong_cls, wrong_paths_cls, config.WASSERSTEIN_MAX_SAMPLES, rng)
        correct_sample, _ = sample_set(correct_cls, correct_paths_cls, config.WASSERSTEIN_MAX_SAMPLES, rng)

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

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, results=results, classes=np.array(classes))
    print(f"Resultados SafeML salvos em: {out_path}")


if __name__ == "__main__":
    main()
