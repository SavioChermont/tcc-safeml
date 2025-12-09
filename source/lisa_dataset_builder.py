# -*- coding: utf-8 -*-
"""
Recorta o LISA Traffic Light dataset em uma estrutura simples:
    dataset/train/<classe>/*.jpg  (dayTrain + nightTrain)
    dataset/test/<classe>/*.jpg   (daySequence1/2 + nightSequence1/2)

Conta apenas imagens realmente salvas (sem duplicar ou sobrescrever).
"""

import sys
from pathlib import Path
import csv
import os
import shutil
from typing import Dict, Iterable
from collections import defaultdict

from PIL import Image

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

import config

# Classes de semáforo no LISA
CLASS_NAMES = [
    "go",
    "goForward",
    "goLeft",
    "stop",
    "stopLeft",
    "warning",
    "warningLeft",
]

# Caminhos base
LISA_ROOT = config.LISA_ROOT
ANNOTATIONS_ROOT = LISA_ROOT / "Annotations" / "Annotations"
OUTPUT_ROOT = config.DATASET_ROOT


def resolve_image_path(rest: str) -> Path:
    """Resolve o caminho real do frame a partir do fragmento após o prefixo."""
    basename = os.path.basename(rest)
    clip = rest.split("--", 1)[0]

    if clip.startswith("dayClip"):
        return LISA_ROOT / "dayTrain" / "dayTrain" / clip / "frames" / basename
    if clip.startswith("nightClip"):
        return LISA_ROOT / "nightTrain" / "nightTrain" / clip / "frames" / basename
    if clip.startswith("daySequence1"):
        return LISA_ROOT / "daySequence1" / "daySequence1" / "frames" / basename
    if clip.startswith("daySequence2"):
        return LISA_ROOT / "daySequence2" / "daySequence2" / "frames" / basename
    if clip.startswith("nightSequence1"):
        return LISA_ROOT / "nightSequence1" / "nightSequence1" / "frames" / basename
    if clip.startswith("nightSequence2"):
        return LISA_ROOT / "nightSequence2" / "nightSequence2" / "frames" / basename

    raise ValueError(f"Clip desconhecido em: {rest}")


def iter_annotations(csv_paths: Iterable[Path]) -> Iterable[Dict[str, str]]:
    """Itera sobre linhas de anotações em vários CSVs."""
    for csv_path in csv_paths:
        with csv_path.open(newline="") as f:
            reader = csv.DictReader(f, delimiter=";")
            for row in reader:
                yield row


def crop_and_save(
    row: Dict[str, str],
    split: str,
    class_counts: Dict[str, int],
    stem_counts: Dict[str, int],
) -> None:
    """Recorta uma caixa e salva em dataset/<split>/<classe>/."""
    tag = row["Annotation tag"]
    if tag not in CLASS_NAMES:
        return

    _, rest = row["Filename"].split("/", 1)
    img_path = resolve_image_path(rest)
    if not img_path.exists():
        return

    x1 = int(float(row["Upper left corner X"]))
    y1 = int(float(row["Upper left corner Y"]))
    x2 = int(float(row["Lower right corner X"]))
    y2 = int(float(row["Lower right corner Y"]))
    if x2 <= x1 or y2 <= y1:
        return

    out_dir = OUTPUT_ROOT / split / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    # Nome único: mantém o stem do frame e numera múltiplas boxes nesse frame com __2, __3, ...
    stem = os.path.basename(rest).replace("--", "-")
    stem_no_ext, _ = os.path.splitext(stem)
    idx = stem_counts[stem_no_ext]
    if idx == 0:
        filename = f"{stem_no_ext}.jpg"
    else:
        filename = f"{stem_no_ext}__{idx+1}.jpg"
    out_path = out_dir / filename

    with Image.open(img_path).convert("RGB") as img:
        crop = img.crop((x1, y1, x2, y2))
        crop.save(out_path, format="JPEG")

    stem_counts[stem_no_ext] = idx + 1
    class_counts[tag] = class_counts.get(tag, 0) + 1


def build_dataset() -> Dict[str, int]:
    """Processa todas as fontes e retorna contagem salva por classe."""
    class_counts: Dict[str, int] = {}
    stem_counts: Dict[str, int] = defaultdict(int)

    sources = [
        ("dayTrain", "train"),
        ("nightTrain", "train"),
        ("daySequence1", "test"),
        ("daySequence2", "test"),
        ("nightSequence1", "test"),
        ("nightSequence2", "test"),
    ]

    for folder, split in sources:
        csv_files = (ANNOTATIONS_ROOT / folder).glob("**/frameAnnotationsBOX.csv")
        for row in iter_annotations(csv_files):
            crop_and_save(row, split=split, class_counts=class_counts, stem_counts=stem_counts)

    return class_counts


def clean_output() -> None:
    """Remove a pasta dataset/ gerada."""
    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)


def main():
    print(f"Anotações em: {ANNOTATIONS_ROOT}")
    print(f"Saída em: {OUTPUT_ROOT}")
    counters = build_dataset()
    total = sum(counters.values())
    print("Imagens salvas por classe (train+test):")
    for cls in CLASS_NAMES:
        print(f"  {cls}: {counters.get(cls, 0)}")
    print(f"Total salvo: {total}")


if __name__ == "__main__":
    main()
