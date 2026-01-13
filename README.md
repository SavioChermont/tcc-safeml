# TCC – Semáforos com SafeML (LISA Dataset)

Pipeline completo (dataset LISA recortado, SVM e CNN, SafeML II com Wasserstein):

## Pré‑requisitos
- Python 3.10+ (testado em Windows)
- `pip install -r requirements.txt` (scikit‑learn, tensorflow/keras, pillow, matplotlib, numpy)
- Estrutura do dataset LISA original em `../Codigo/OriginalDataSet/LISA-dataset` (ajuste em `source/config.py` se necessário).

## Estrutura principal
- `source/lisa_dataset_builder.py`: recorta o LISA (anotações CSV) em `dataset/train` e `dataset/test` (classes go, goForward, goLeft, stop, stopLeft, warning, warningLeft). Para treino/avaliação, usamos a whitelist `{go, stop, warning}`.
- `source/data_utils.py`: `load_split` carrega/normaliza imagens 30×30 RGB, whitelista classes e opcionalmente limita por classe.
- `source/train_svm.py`: treina LinearSVC (sklearn); salva modelo em `models/traffic_light_svm.joblib` e cache `artifacts/svm/train_data.npz`.
- `source/train_cnn.py`: treina CNN simples (Keras) mantendo input 30×30×3; salva `models/traffic_light_cnn.h5` e cache `artifacts/cnn/train_data.npz`.
- SafeML II:
  - `source/Wasserstein_Dist_PVal.py`: implementação de WD + bootstrap p‑valor.
  - `source/safeml_collect.py`: calcula WD/p‑valor por pixel/canal entre treino e erros/acertos; salva `artifacts/<tag>/safeml_results.npz` e prints de estatísticas.
  - `source/safeml_heatmaps.py`: gera heatmaps a partir do `safeml_results.npz`.
  - `source/safeml_eval_threshold.py`: amostra 10 imgs/ classe, prediz e calcula WD médio vs treino da classe majoritária para testar limiares.

## Pipeline do zero
1) Recortar o LISA:
```
python source/lisa_dataset_builder.py
```
2) Treinar SVM:
```
python source/train_svm.py
```
3) Treinar CNN:
```
python source/train_cnn.py
```
4) Coletar SafeML (WD/p‑valor):
```
python source/safeml_collect.py --tag svm          # completo
python source/safeml_collect.py --tag svm --only-day
python source/safeml_collect.py --tag svm --only-night
python source/safeml_collect.py --tag cnn          # completo
python source/safeml_collect.py --tag cnn --only-day
python source/safeml_collect.py --tag cnn --only-night
```
5) Gerar heatmaps:
```
python source/safeml_heatmaps.py --results-path artifacts/svm/safeml_results.npz
python source/safeml_heatmaps.py --results-path artifacts/cnn/safeml_results.npz
```
6) Avaliar limiar (ex.: SVM dia):
```
python source/safeml_eval_threshold.py --tag svm --only-day --thresholds go=0.209,stop=0.202,warning=0.232
python source/safeml_eval_threshold.py --tag svm --only-night --thresholds go=0.209,stop=0.202,warning=0.232
python source/safeml_eval_threshold.py --tag cnn --only-day --thresholds go=...,stop=...,warning=...
python source/safeml_eval_threshold.py --tag cnn --only-night --thresholds go=...,stop=...,warning=...
```

## Configurações chave (`source/config.py`)
- `LISA_ROOT`: caminho para o dataset original com anotações/frames.
- `CLASS_WHITELIST = {"go", "stop", "warning"}`.
- `IMG_SIZE = (30, 30)`, normalização em [0,1]; SVM usa flatten (2700), CNN usa 30×30×3.
- `MAX_PER_CLASS_TRAIN = 3000` (limita por classe no treino).
  - SafeML: `SAFE_PVAL_ALPHA = 0.05`, `WASSERSTEIN_MAX_SAMPLES = 15`, `WD_TRAIN_LIMIT`, `WD_PIXEL_LIMIT` para aceleração.
- Caminhos: modelos em `models/`, caches/artefatos em `artifacts/svm` e `artifacts/cnn`.

## Saídas principais
- Modelos: `models/traffic_light_svm.joblib`, `models/traffic_light_cnn.h5`.
- Caches SafeML: `artifacts/svm/train_data.npz`, `artifacts/cnn/train_data.npz`.
- Coleta SafeML: `artifacts/<tag>/safeml_results.npz` + prints de `wd_mean_sig`/`sig_count` por canal.
- Heatmaps: PNGs em `artifacts/<tag>/` (mapas agregados e por canal, WD e WD p<alpha).

## Observações
- `safeml_collect.py` balanceia amostras dia/noite (metade/metade) quando não há filtros.
- `safeml_heatmaps.py` normaliza mapas para visualização usando percentil 99 (cores são relativas ao próprio mapa).
- `safeml_eval_threshold.py` não recalcula mapas; usa treino + um buffer de 10 imagens por classe para comparar WD médio vs. limiar informado.
