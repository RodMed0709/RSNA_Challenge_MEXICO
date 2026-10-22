import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from analyze_stress import (
    atomic_write_public_metrics,
    macro_auc_filtered,
    paired_mean_delta,
    render_verdict,
    select_hard_cases,
    stability_metrics,
    validate_public_metrics,
    update_registry_text,
)
from infer_stress import (
    INFERENCE_CONDITIONS,
    PLAUSIBLE_CONDITIONS,
    PUBLISHED_CONDITIONS,
    SEVERE_CONDITIONS,
    derive_published_predictions,
    deterministic_seed,
    load_panel_config,
    resolve_model_runtime,
    select_cohort_indices,
    slice_plan,
    validate_panel_config,
    writable_array,
)


class ConditionContractTests(unittest.TestCase):
    def test_validate_only_cli_reports_frozen_inventory(self):
        here = Path(__file__).resolve().parent
        completed = subprocess.run(
            [sys.executable, str(here / "infer_stress.py"), "--config", str(here / "panel_config.json"), "--validate-only"],
            check=True,
            capture_output=True,
            text=True,
        )
        summary = json.loads(completed.stdout)
        self.assertEqual({"F": 10, "R": 18, "S": 17}, summary["panel_counts"])

    def test_frozen_config_has_expected_panels_and_existing_artifacts(self):
        config = load_panel_config(Path(__file__).with_name("panel_config.json"))
        summary = validate_panel_config(config, check_paths=True)
        self.assertEqual({"R": 18, "F": 10, "S": 17}, summary["panel_counts"])
        self.assertEqual(3, summary["derived_ensembles"])
        self.assertEqual(45, summary["unique_model_ids"])

    def test_runtime_resolves_224_slots_without_mutating_config(self):
        model = {"id": "student", "res": 224, "use_slots": [0, 1, 2], "fold": 0}
        resolved = resolve_model_runtime(model, {"res": 224, "params": {"USE_SLOTS": [0, 1, 2]}})
        self.assertEqual(224, resolved["res"])
        self.assertEqual((0, 1, 2), resolved["use_slots"])
        self.assertEqual(24, resolved["k_eval"])
        self.assertEqual([0, 1, 2], model["use_slots"])

    def test_memmap_volume_is_copied_to_writable_storage_before_torch(self):
        source = np.arange(8, dtype=np.uint8).reshape(2, 2, 2)
        source.flags.writeable = False
        result = writable_array(source)
        self.assertTrue(result.flags.writeable)
        self.assertTrue(result.flags.c_contiguous)
        np.testing.assert_array_equal(source, result)
        self.assertFalse(np.shares_memory(source, result))

    def test_cohort_selection_keeps_target_fold_and_gold_separate(self):
        folds = np.array([0, 1, -1, 0, -2])
        is_gold = np.array([0, 0, 1, 0, 0], dtype=bool)
        val, gold = select_cohort_indices(folds, is_gold, target_fold=0)
        self.assertEqual([0, 3], val.tolist())
        self.assertEqual([2], gold.tolist())

    def test_condition_catalog_keeps_plausible_and_severe_disjoint(self):
        self.assertEqual(16, len(PUBLISHED_CONDITIONS))
        self.assertEqual(16, len(INFERENCE_CONDITIONS))
        self.assertEqual({"ref", "tta_mirror"}, set(PUBLISHED_CONDITIONS[:2]))
        self.assertTrue(set(PLAUSIBLE_CONDITIONS).isdisjoint(SEVERE_CONDITIONS))
        self.assertEqual(
            set(PUBLISHED_CONDITIONS) - {"tta_mirror"},
            set(INFERENCE_CONDITIONS) - {"mirror"},
        )

    def test_seed_is_stable_per_uid_and_condition(self):
        first = deterministic_seed("1.2.840.10008.1", "gauss2")
        self.assertEqual(1273542455, first)
        self.assertEqual(first, deterministic_seed("1.2.840.10008.1", "gauss2"))
        self.assertNotEqual(first, deterministic_seed("1.2.840.10008.1", "gauss5"))

    def test_slice_plan_subsamples_each_slot_without_crossing_boundaries(self):
        indices, sizes = slice_plan("cortes50", [6, 4])
        self.assertEqual([3, 3], sizes)
        self.assertEqual([0, 2, 5, 6, 8, 9], indices.tolist())

    def test_tta_is_mean_of_ref_and_mirror_and_hides_raw_mirror(self):
        raw = {
            "ref": np.array([[0.2, 0.8]], dtype=np.float32),
            "mirror": np.array([[0.6, 0.4]], dtype=np.float32),
            "gauss2": np.array([[0.1, 0.7]], dtype=np.float32),
        }
        published = derive_published_predictions(raw)
        np.testing.assert_allclose([[0.4, 0.6]], published["tta_mirror"], rtol=0, atol=1e-7)
        self.assertNotIn("mirror", published)
        self.assertIs(raw["gauss2"], published["gauss2"])


class AggregateMetricTests(unittest.TestCase):
    def test_verdict_renders_only_aggregate_metric_values(self):
        metrics = {
            "coverage": {"complete": True, "completed": 45, "planned": 45},
            "reproduction_gate": {"auc": 0.8681754489, "delta": -0.0000022061, "passed": True},
            "hypotheses": {"H1": {"rho": 0.25}, "H2": {"confirmed": 3, "tested": 5}, "H3": {"delta_bce_ref": -0.0123}},
            "runtime": {"gpu_hours": 8.5, "estimated_runpod_cost_usd": 6.12},
        }
        verdict = render_verdict(metrics)
        self.assertIn("worked", verdict)
        self.assertIn("0.868175", verdict)
        self.assertIn("-0.012300", verdict)
        self.assertNotIn("StudyInstanceUID", verdict)

    def test_registry_update_changes_only_exp091_row(self):
        original = (
            "header\n"
            "| exp090 | old | rodrigo | exp074 | x | worked | | | | ok |\n"
            "| exp091 | stress_v1 | rodrigo | exp040 | stress | spec | | | | |\n"
        )
        updated = update_registry_text(original, "worked", 0.8681754489, "H1 rho=0.250; coverage 45/45")
        self.assertIn("| exp090 | old |", updated)
        self.assertIn("| exp091 | stress_v1 | rodrigo | exp040 | stress | worked | 0.8682 (gate ref)", updated)
        self.assertEqual(1, updated.count("| exp091 |"))

    def test_initialize_cli_writes_versionable_partial_metrics(self):
        here = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory() as directory:
            completed = subprocess.run(
                [
                    sys.executable,
                    str(here / "analyze_stress.py"),
                    "--config",
                    str(here / "panel_config.json"),
                    "--results",
                    directory,
                    "--initialize",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            metrics = json.loads((Path(directory) / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual("partial", metrics["phase"])
            self.assertEqual({"F": 10, "R": 18, "S": 17}, metrics["planned"])
            self.assertIn("metrics.json", completed.stdout)

    def test_macro_auc_excludes_labels_below_class_count_gate(self):
        y = np.array(
            [
                [0, 0],
                [0, 0],
                [1, 0],
                [1, 1],
            ],
            dtype=float,
        )
        p = np.array(
            [
                [0.1, 0.1],
                [0.2, 0.2],
                [0.8, 0.3],
                [0.9, 0.9],
            ],
            dtype=float,
        )
        result = macro_auc_filtered(y, p, min_class_count=2)
        self.assertEqual(1, result["n_labels"])
        self.assertEqual([2, 2], result["class_counts"][0])
        self.assertAlmostEqual(1.0, result["auc"], places=12)

    def test_stability_uses_mad_and_mean_per_label_spearman(self):
        ref = np.array([[0.1, 0.8], [0.4, 0.2], [0.9, 0.5]], dtype=float)
        cond = np.array([[0.2, 0.7], [0.5, 0.3], [0.8, 0.4]], dtype=float)
        result = stability_metrics(ref, cond)
        self.assertAlmostEqual(0.1, result["mad"], places=12)
        self.assertAlmostEqual(1.0, result["rho"], places=12)
        self.assertEqual(3, result["n"])

    def test_hard_cases_use_masked_bce_and_top_fraction(self):
        y = np.array([[1.0, 0.0], [1.0, np.nan], [0.0, 1.0], [0.5, 1.0]])
        p = np.array([[0.9, 0.1], [0.2, 0.9], [0.1, 0.8], [0.9, 0.2]])
        selected, losses = select_hard_cases(y, p, fraction=0.5)
        self.assertEqual([1, 3], selected.tolist())
        self.assertTrue(losses[1] > losses[0])
        self.assertTrue(losses[3] > losses[2])

    def test_paired_delta_reports_literal_difference_and_ci(self):
        result = paired_mean_delta(np.array([1.0, 2.0, 3.0]), np.array([0.5, 2.5, 4.0]), bootstrap=200, seed=42)
        self.assertAlmostEqual(1.0 / 3.0, result["delta"], places=12)
        self.assertLessEqual(result["ci95"][0], result["delta"])
        self.assertGreaterEqual(result["ci95"][1], result["delta"])

    def test_public_metrics_reject_uid_and_small_published_cell(self):
        valid = {"schema_version": 1, "cell": {"n": 20, "mad": 0.1}}
        validate_public_metrics(valid)
        with self.assertRaisesRegex(ValueError, "UID"):
            validate_public_metrics({"StudyInstanceUID": ["1.2.3"]})
        with self.assertRaisesRegex(ValueError, "n >= 20"):
            validate_public_metrics({"cell": {"n": 19, "mad": 0.1}})

    def test_public_metrics_roundtrip_contains_no_nonfinite_numbers(self):
        payload = {"schema_version": 1, "cell": {"n": 20, "mad": 0.0, "rho": 1.0}}
        validate_public_metrics(payload)
        text = json.dumps(payload, allow_nan=False)
        self.assertNotIn("NaN", text)
        self.assertFalse(any(math.isnan(v) for v in payload["cell"].values() if isinstance(v, float)))

    def test_atomic_public_write_replaces_valid_json(self):
        payload = {"schema_version": 1, "cell": {"n": 20, "mad": 0.125}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.json"
            path.write_text('{"stale": true}', encoding="utf-8")
            atomic_write_public_metrics(path, payload)
            self.assertEqual(payload, json.loads(path.read_text(encoding="utf-8")))
            self.assertFalse(path.with_suffix(".tmp").exists())


if __name__ == "__main__":
    unittest.main()
