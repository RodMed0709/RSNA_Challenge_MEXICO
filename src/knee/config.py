import os
from pathlib import Path


LABELS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]

DATA_ROOT = Path(os.environ.get("RSNA_DATA", "/workspace/data"))
RAW_ROOT = DATA_ROOT / "raw"
CACHE_ROOT = DATA_ROOT / "cache"
LABELS_ROOT = DATA_ROOT / "labels"
MODELS_ROOT = Path("/workspace/models")
RESULTS_ROOT = Path("/workspace/results")
PLANE_TO_ID = {"Sagittal": 0, "Coronal": 1, "Axial": 2}


def user_dir() -> Path:
    user = os.environ.get("RSNA_USER")
    if not user:
        raise RuntimeError("Define RSNA_USER con tu nombre antes de continuar, por ejemplo: export RSNA_USER=ana")
    return Path("/workspace/users") / user
