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
USERS_ROOT = Path(os.environ.get("RSNA_USERS", "/workspace/users"))
REPO_ROOT = Path(__file__).resolve().parents[2]
SPLITS_ROOT = REPO_ROOT / "SPLITS"
PLANE_TO_ID = {"Sagittal": 0, "Coronal": 1, "Axial": 2}


def current_user() -> str:
    user = os.environ.get("RSNA_USER")
    if not user and REPO_ROOT.parent == USERS_ROOT:
        user = REPO_ROOT.name
    if not user:
        raise RuntimeError("Clona el repo en /workspace/users/<tu_nombre> o define RSNA_USER, por ejemplo: export RSNA_USER=ana")
    os.environ["RSNA_USER"] = user
    return user
