# -*- coding: utf-8 -*-
"""
Análise de confiança usando Wasserstein Distance (SafeML-like).

Espelha o case do professor, mas focando só em Wasserstein:
- Usa a função Wasserstein_Dist_PVal da biblioteca SafeML (mesma do notebook).
- Compara treino (referência) vs. exemplos de teste errados daquela classe.
- Limita a amostragem por classe (config.WASSERSTEIN_MAX_SAMPLES) para acelerar,
  e usa até 30 exemplos como no case.
"""

import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import joblib
from Wasserstein_Dist_PVal import Wasserstein_Dist_PVal

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

import config
from data_utils import load_split


def summarize_wd(train_set: np.ndarray, target_set: np.ndarray, rng: np.random.Generator) -> Dict[str, float]:
    """
    Calcula Wasserstein por feature e retorna estatísticas resumo usando
    Wasserstein_Dist_PVal (SafeML) entre treino e um conjunto alvo (erros ou acertos).
    """
    if len(target_set) == 0 or len(train_set) == 0:
        return {
            "wd_mean_all": 0.0,
            "wd_p95_all": 0.0,
            "wd_mean_sig": 0.0,
            "wd_p95_sig": 0.0,
            "sig_frac": 0.0,
            "features": 0,
        }

    # Amostragem para limitar custo (espelha o uso de ~30 amostras no case)
    max_n = config.WASSERSTEIN_MAX_SAMPLES
    if max_n is not None:
        if len(train_set) > max_n:
            idx = rng.choice(len(train_set), size=max_n, replace=False)
            train_set = train_set[idx]
        if len(target_set) > max_n:
            idx = rng.choice(len(target_set), size=max_n, replace=False)
            target_set = target_set[idx]

    # Segue o padrão do professor: calcula por canal (3 x 900) em vez de vetor único 2700
    H, W = config.IMG_SIZE
    train_imgs = train_set.reshape((-1, H, W, 3))
    target_imgs = target_set.reshape((-1, H, W, 3))

    channel_stats = []
    for ch in range(3):
        train_ch = train_imgs[:, :, :, ch].reshape(train_imgs.shape[0], -1)  # (N, 900)
        target_ch = target_imgs[:, :, :, ch].reshape(target_imgs.shape[0], -1)
        n_features = train_ch.shape[1]
        wd_vals: List[float] = []
        pvals: List[float] = []
        for i in range(n_features):
            pval, wd = Wasserstein_Dist_PVal(train_ch[:, i], target_ch[:, i])
            wd_vals.append(wd)
            pvals.append(pval)

        wd_arr = np.array(wd_vals)
        pvals_arr = np.array(pvals)

        # SafeML II: manter apenas distâncias com p < alpha
        sig_mask = pvals_arr < config.SAFE_PVAL_ALPHA
        wd_sig = wd_arr[sig_mask]
        sig_count = int(sig_mask.sum())

        if len(wd_sig) == 0:
            wd_mean_sig = 0.0
        else:
            wd_mean_sig = float(wd_sig.mean())

        channel_stats.append(
            {
                "channel": ch,
                "wd_mean_sig": wd_mean_sig,
                "sig_count": sig_count,
                "features": n_features,
            }
        )

    return {
        "channels": channel_stats,
        "features_total": 3 * n_features,
    }


def main():
    print("Carregando dados para análise SafeML (Wasserstein)...")
    if not config.TRAIN_DATA_PATH.exists():
        raise SystemExit(
            f"Arquivo de treino não encontrado em {config.TRAIN_DATA_PATH}. "
            "Rode primeiro `python source/train_classifier.py` para gerar o cache."
        )

    data = np.load(config.TRAIN_DATA_PATH, allow_pickle=True)
    X_train = data["X_train"]
    y_train = data["y_train"]
    classes = list(data["classes"])
    print(f"Carregado treino salvo em {config.TRAIN_DATA_PATH}")

    X_test, y_test, _ = load_split("test", max_per_class=None)

    print(f"Treino: {len(y_train)} amostras | Teste: {len(y_test)} amostras")

    model = joblib.load(config.MODEL_PATH)
    y_pred = model.predict(X_test)

    results = {}
    rng = np.random.default_rng(config.SEED)

    for cls in classes:
        train_cls = X_train[y_train == cls]
        wrong_cls_mask = (y_test == cls) & (y_pred != cls)
        wrong_cls = X_test[wrong_cls_mask]
        correct_cls_mask = (y_test == cls) & (y_pred == cls)
        correct_cls = X_test[correct_cls_mask]

        summary_wrong = summarize_wd(train_cls, wrong_cls, rng)
        summary_correct = summarize_wd(train_cls, correct_cls, rng)
        results[cls] = {
            "train_count": int(train_cls.shape[0]),
            "wrong_count": int(wrong_cls.shape[0]),
            "correct_count": int(correct_cls.shape[0]),
            "wrong_summary": summary_wrong,
            "correct_summary": summary_correct,
        }

        print(f"Classe {cls}: train={train_cls.shape[0]}, wrong={wrong_cls.shape[0]}, correct={correct_cls.shape[0]}")
        print("  Erros:")
        for ch_stat in summary_wrong["channels"]:
            ch = ch_stat["channel"]
            print(
                f"  Canal {ch}: wd_mean_sig={ch_stat['wd_mean_sig']:.6f}, "
                f"sig_count={ch_stat['sig_count']}/{ch_stat['features']}"
            )
        print("  Acertos (baseline para limiar):")
        for ch_stat in summary_correct["channels"]:
            ch = ch_stat["channel"]
            print(
                f"  Canal {ch}: wd_mean_sig={ch_stat['wd_mean_sig']:.6f}, "
                f"sig_count={ch_stat['sig_count']}/{ch_stat['features']}"
            )

    artifacts_dir = config.REPO_ROOT / "artifacts"
    artifacts_dir.mkdir(exist_ok=True, parents=True)
    np.savez_compressed(
        artifacts_dir / "wasserstein_results.npz",
        results=results,
        classes=np.array(classes),
    )
    print(f"Resultados salvos em: {artifacts_dir / 'wasserstein_results.npz'}")


if __name__ == "__main__":
    main()
