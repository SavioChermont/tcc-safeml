# -*- coding: utf-8 -*-
"""
Gera heatmaps SafeML II a partir de um arquivo de resultados (.npz) produzido pelo safeml_collect.py.
Não recalcula Wasserstein; apenas carrega os mapas e plota.

Use o argumento --results-path para apontar para o arquivo desejado (SVM ou CNN).
"""

import sys
import argparse
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

import config


def _normalize_for_plot(m: np.ndarray) -> np.ndarray:
    """Normaliza o mapa para [0,1], usando percentil 99 para reduzir outliers."""
    m = m.copy()
    m[m < 0] = 0
    if np.any(m > 0):
        vmax = float(np.percentile(m[m > 0], 99.0))
        if vmax > 0:
            m = np.clip(m / vmax, 0.0, 1.0)
    return m


def main():
    parser = argparse.ArgumentParser(description="Gera heatmaps SafeML II a partir de safeml_results.npz")
    parser.add_argument(
        "--results-path",
        default=str(config.ARTIFACTS_DIR / "safeml_results.npz"),
        help="Arquivo .npz gerado pelo safeml_collect.py",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Diretório de saída para os PNGs; default é a pasta do results-path",
    )
    args = parser.parse_args()

    results_path = Path(args.results_path)
    if not results_path.is_absolute():
        results_path = config.REPO_ROOT / results_path

    if not results_path.exists():
        raise SystemExit("Arquivo safeml_results.npz não encontrado. Rode safeml_collect.py primeiro.")

    data = np.load(results_path, allow_pickle=True)
    results = data["results"].item()  # dict por classe
    classes = data["classes"]

    out_dir = Path(args.output_dir) if args.output_dir else results_path.parent
    if not out_dir.is_absolute():
        out_dir = config.REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    channel_names = ["R", "G", "B"]

    for cls in classes:
        cls_results = results.item().get(cls) if isinstance(results, np.ndarray) else results.get(cls)
        if cls_results is None:
            continue
        wd_maps = cls_results.get("wd_maps_wrong")
        wd_sig_maps = cls_results.get("wd_sig_maps_wrong")
        if wd_maps is None or wd_sig_maps is None:
            print(f"Sem mapas para a classe {cls}, pulando.")
            continue

        # Painel 1x2 agregado (média dos 3 canais)
        wd_stack = np.stack(wd_maps, axis=-1)
        wd_sig_stack = np.stack(wd_sig_maps, axis=-1)
        wd_agg = wd_stack.mean(axis=-1)
        wd_sig_agg = wd_sig_stack.mean(axis=-1)

        wd_agg_n = _normalize_for_plot(wd_agg)
        wd_sig_agg_n = _normalize_for_plot(wd_sig_agg)

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        im1 = axes[0].imshow(wd_agg_n, cmap="hot", interpolation="bilinear")
        axes[0].set_title(f"{cls} - WD (todos)")
        axes[0].axis("off")
        fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

        im2 = axes[1].imshow(wd_sig_agg_n, cmap="hot", interpolation="bilinear")
        axes[1].set_title(f"{cls} - WD (p < {config.SAFE_PVAL_ALPHA})")
        axes[1].axis("off")
        fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

        out_path_panel = out_dir / f"{cls}_safeMLII_panel.png"
        plt.tight_layout()
        plt.savefig(out_path_panel, dpi=300, bbox_inches="tight")
        plt.close(fig)

        # Também salva por canal, se quiser inspecionar
        for ch in range(3):
            wd_ch_n = _normalize_for_plot(wd_maps[ch])
            wd_sig_ch_n = _normalize_for_plot(wd_sig_maps[ch])
            for name, data, title in [
                ("wd_all", wd_ch_n, f"{cls} - {channel_names[ch]} - WD"),
                ("wd_sig", wd_sig_ch_n, f"{cls} - {channel_names[ch]} - WD p<{config.SAFE_PVAL_ALPHA}"),
            ]:
                plt.figure(figsize=(4, 4))
                plt.imshow(data, cmap="hot", interpolation="bilinear")
                plt.axis("off")
                plt.title(title)
                out_path = out_dir / f"{cls}_{channel_names[ch]}_{name}.png"
                plt.savefig(out_path, dpi=200, bbox_inches="tight")
                plt.close()

        print(f"Painéis/heatmaps salvos para a classe {cls} em {out_dir}")


if __name__ == "__main__":
    main()
