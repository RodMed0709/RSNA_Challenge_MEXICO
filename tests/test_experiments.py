import json
import math

import pandas as pd
import pytest

from src.knee import experiments
from src.knee.splits import load_split


def test_canonical_split_is_valid():
    split = load_split("split_v1_seed42")
    assert (split.is_gold == 1).sum() == 58
    assert set(split.fold) == {-2, -1, 0, 1, 2, 3, 4}
    sizes = split[split.fold >= 0].fold.value_counts()
    assert sizes.max() / sizes.min() < 1.05


@pytest.fixture
def team(tmp_path, monkeypatch):
    repo = tmp_path / "users" / "ana"
    (repo / "experiments" / "_plantilla").mkdir(parents=True)
    (repo / "experiments" / "_plantilla" / "SPEC.md").write_text("# expNNN_tag <autor> <padre> <fecha> <idea>\n")
    parent = repo / "experiments" / "exp003_ref"
    parent.mkdir()
    notebook = {"cells": [{"cell_type": "code", "metadata": {"tags": ["parameters"]}, "outputs": [], "execution_count": 1,
                           "source": ['EXP = "exp003_ref"\n', 'PADRE = "exp002"\n', "SMOKE = False\n"]}],
                "metadata": {}, "nbformat": 4, "nbformat_minor": 5}
    (parent / "exp003_ref.ipynb").write_text(json.dumps(notebook))
    (repo / "docs").mkdir()
    (repo / "docs" / "EXPERIMENTOS.md").write_text("| exp | tag |\n|---|---|\n| exp003 | ref |\n")
    other = tmp_path / "users" / "jose" / "experiments" / "exp007_lora"
    other.mkdir(parents=True)
    monkeypatch.setattr(experiments, "REPO_ROOT", repo)
    monkeypatch.setattr(experiments, "USERS_ROOT", tmp_path / "users")
    monkeypatch.setattr(experiments, "CLAIMS_ROOT", tmp_path / "users" / ".claims")
    monkeypatch.setenv("RSNA_USER", "ana")
    return repo


def test_new_experiment_takes_next_global_number(team):
    first = experiments.nuevo_experimento("DINOv2 base", padre="exp003", idea="backbone base")
    second = experiments.nuevo_experimento("more slices", padre="exp003")
    assert first.name == "exp008_dinov2_base"
    assert second.name == "exp009_more_slices"
    params = "".join(json.loads((first / "exp008_dinov2_base.ipynb").read_text())["cells"][0]["source"])
    assert 'EXP = "exp008_dinov2_base"' in params and 'PADRE = "exp003"' in params and "SMOKE = True" in params
    assert "ana" in (first / "SPEC.md").read_text()
    assert "| exp008 | dinov2_base | ana | exp003 |" in (team / "docs" / "EXPERIMENTOS.md").read_text()


def test_claim_survives_race(team):
    (experiments.CLAIMS_ROOT / "exp008_dinov2_base").mkdir(parents=True)
    assert experiments.nuevo_experimento("x", padre="exp003").name == "exp009_x"


def test_table_and_weights(team, tmp_path):
    for user, exp, auc in [("ana", "exp008_a", 0.83), ("jose", "exp007_lora", 0.84)]:
        results = tmp_path / "users" / user / "results" / exp
        experiments.guardar_metricas(results, {"author": user, "per_fold": {"0": {"pseudo_auc": auc, "ckpt_path": f"/w/{exp}.pt"}},
                                               "gold_auc": 0.78, "decision_auto": "promising"})
    table = experiments.tabla_experimentos()
    assert list(table.exp) == ["exp007_lora", "exp008_a"]
    assert experiments.pesos("exp008") == ["/w/exp008_a.pt"]


def test_decision_rules():
    assert experiments.decision(0.01, -0.03) == "killed"
    assert experiments.decision(0.006, 0.0) == "promising"
    assert experiments.decision(0.001, 0.0) == "neutral"
    assert experiments.decision(-0.01, 0.0) == "didnt"
    assert experiments.decision(None, None) == "sin_padre"
    assert experiments.decision(math.nan, None) == "sin_padre"
