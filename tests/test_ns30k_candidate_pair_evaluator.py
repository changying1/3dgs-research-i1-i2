from __future__ import annotations

import contextlib
import copy
import csv
import io
import json
from pathlib import Path
import sys
from unittest import mock

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from script import evaluate_ns30k_candidate_pair_v1 as evaluator
from script import evaluate_ns30k_i1_schedule_ablation_room_v1 as room_evaluator
from script import evaluate_ns30k_i1_schedule_ablation_v1 as shared


def _audit(group: str) -> dict:
    return {
        "status": "PASS",
        "sampling_records": 30_000,
        "trace_matches_frozen_table": True,
        "same_scene_pairing": {"traces_identical": True},
        "i1_runtime": (
            {"configured": False, "activation_count": 0}
            if group == "B0"
            else {
                "configured": True,
                "activation_count": 8000,
                "post_8000_activation_seen": False,
            }
        ),
    }


def _data(scene: str = "leaves") -> dict:
    names = tuple(f"{index:05d}.png" for index in range(4))
    data = {}
    for iteration in evaluator.NODES:
        for group_index, group in enumerate(evaluator.GROUPS):
            base = iteration / 10_000 + group_index
            aggregate = {metric: base + metric_index for metric_index, metric in enumerate(evaluator.METRICS)}
            details = {
                metric: {name: aggregate[metric] for name in names}
                for metric in evaluator.METRICS
            }
            data[(group, iteration)] = aggregate, details
    return data


def _write_rgb_gt(folder: Path, count: int = 4) -> None:
    for kind in ("renders", "gt"):
        target = folder / kind
        target.mkdir(parents=True)
        for index in range(count):
            (target / f"{index:05d}.png").write_bytes(f"{kind}-{index}".encode("ascii"))


def test_scene_scope_and_report_tree_are_isolated_from_original_reports():
    assert evaluator.SUPPORTED_SCENES == ("leaves", "orchids", "fern", "flower", "trex")
    assert evaluator.GROUPS == ("B0", "End8k")
    assert evaluator.NODES == (5000, 10000, 20000, 30000)
    assert evaluator.report_dir("leaves") == ROOT / "reports/ns30k_v1_pair_evaluation/leaves"
    assert evaluator.report_dir("orchids") == ROOT / "reports/ns30k_v1_pair_evaluation/orchids"
    assert evaluator.REPORT_ROOT != shared.REPORT
    assert evaluator.REPORT_ROOT != room_evaluator.REPORT


def test_render_command_reuses_frozen_manifest_and_original_parameters():
    command = evaluator._render_command("leaves", "End8k", 20_000, "python")
    plan = evaluator.plan_for("leaves", "End8k")
    values = {flag: command[command.index(flag) + 1] for flag in (
        "-s", "-m", "--dataset_name", "--n_views", "--resolution",
        "--sh_degree", "--dropout_factor", "--view_manifest", "--iteration",
    )}
    assert values == {
        "-s": str(evaluator.training.DATA_ROOT / "leaves"),
        "-m": str(plan.output_dir),
        "--dataset_name": "LLFF", "--n_views": "3", "--resolution": "8",
        "--sh_degree": "1", "--dropout_factor": "0.2",
        "--view_manifest": str(plan.manifest), "--iteration": "20000",
    }
    assert command[2] == str(ROOT / "render.py")
    assert "--eval" in command and "--skip_train" in command


def test_real_leaves_preflight_is_read_only_and_does_not_require_orchids():
    before = evaluator.report_dir("leaves").exists()
    result = evaluator.preflight("leaves")
    assert result["status"] == "PASS"
    assert result["scene"] == "leaves"
    assert result["test_view_count"] == 4
    assert result["other_supported_scene_required"] is False
    assert set(result["training_completion_audits"]) == {"B0", "End8k"}
    assert evaluator.report_dir("leaves").exists() is before


@pytest.mark.parametrize("scene", ("leaves", "orchids"))
def test_frozen_legacy_reports_pass_full_read_only_verification(scene):
    result = evaluator.verify_completed_evaluation(scene)
    assert result["status"] == "PASS"
    assert result["mode"] == "VERIFY_COMPLETED_READ_ONLY"


def test_legacy_contract_accepts_only_exact_hash_and_exact_old_audit_schema():
    scene = "leaves"
    record_path = evaluator.report_dir(scene) / "summary/protocol_record.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    current = evaluator.preflight(scene)
    assert evaluator._validate_recorded_preflight_contract(scene, record, current) == (
        "legacy-leaves-orchids-v1"
    )

    bad_hash = copy.deepcopy(record)
    bad_hash["evaluation_code_sha256"]["script/evaluate_ns30k_candidate_pair_v1.py"] = "0" * 64
    with pytest.raises(evaluator.EvaluationError, match="evaluation_code_sha256 mismatch"):
        evaluator._validate_recorded_preflight_contract(scene, bad_hash, current)

    missing_old_field = copy.deepcopy(record)
    del missing_old_field["training_completion_audits"]["B0"]["sampling_records"]
    with pytest.raises(evaluator.EvaluationError, match="legacy B0 training audit schema mismatch"):
        evaluator._validate_recorded_preflight_contract(scene, missing_old_field, current)

    injected_new_field = copy.deepcopy(record)
    injected_new_field["training_completion_audits"]["B0"]["asset_provenance"] = {}
    with pytest.raises(evaluator.EvaluationError, match="legacy B0 training audit schema mismatch"):
        evaluator._validate_recorded_preflight_contract(scene, injected_new_field, current)

    wrong_scene = copy.deepcopy(record)
    with pytest.raises(evaluator.EvaluationError, match="evaluation_code_sha256 mismatch"):
        evaluator._validate_recorded_preflight_contract("fern", wrong_scene, current)


def test_current_contract_rejects_any_missing_audit_field_or_result_change():
    current = evaluator.preflight("leaves")
    record = copy.deepcopy(current)
    assert evaluator._validate_recorded_preflight_contract("leaves", record, current) == "current"

    missing = copy.deepcopy(record)
    del missing["training_completion_audits"]["End8k"]["asset_provenance"]
    with pytest.raises(evaluator.EvaluationError, match="training_completion_audits mismatch"):
        evaluator._validate_recorded_preflight_contract("leaves", missing, current)

    tampered = copy.deepcopy(record)
    tampered["training_completion_audits"]["End8k"]["sampling_records"] = 29_999
    with pytest.raises(evaluator.EvaluationError, match="training_completion_audits mismatch"):
        evaluator._validate_recorded_preflight_contract("leaves", tampered, current)


def test_preflight_fails_closed_on_training_or_pairing_mismatch(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluator, "REPORT_ROOT", tmp_path / "report")
    calls = []

    def invalid(plan):
        calls.append(plan.scene)
        group = "B0" if plan.arm == "B0" else "End8k"
        result = _audit(group)
        result["trace_matches_frozen_table"] = False
        return result

    monkeypatch.setattr(evaluator.training, "verify_completed", invalid)
    with pytest.raises(evaluator.EvaluationError, match="trace/table mismatch"):
        evaluator.preflight("leaves")
    assert calls == ["leaves"]


def test_existing_render_is_hash_validated_and_reused_without_subprocess(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluator, "REPORT_ROOT", tmp_path / "report")
    model = tmp_path / "model"
    source = model / "test/ours_5000"
    _write_rgb_gt(source)
    monkeypatch.setattr(evaluator, "model_path", lambda scene, group: model)
    monkeypatch.setitem(shared.TEST_COUNTS, "leaves", 4)
    with mock.patch.object(shared, "run_logged", side_effect=AssertionError("must reuse")):
        assert evaluator.render_one("leaves", "B0", 5000, "python") == source
    log = evaluator.render_log_path("leaves", "B0", 5000)
    assert "REUSED_EXISTING_RENDER: true" in log.read_text(encoding="utf-8")
    _write = source / "renders/00000.png"
    _write.write_bytes(b"tampered")
    with pytest.raises(evaluator.EvaluationError, match="SHA256 mismatch"):
        evaluator.render_one("leaves", "B0", 5000, "python")


def test_orphan_or_unsuccessful_render_log_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluator, "REPORT_ROOT", tmp_path / "report")
    source = tmp_path / "source"
    _write_rgb_gt(source)
    monkeypatch.setitem(shared.TEST_COUNTS, "leaves", 4)
    log = evaluator.render_log_path("leaves", "B0", 5000)
    log.parent.mkdir(parents=True)
    log.write_text("EXIT_CODE: 7\n", encoding="utf-8")
    with pytest.raises(evaluator.EvaluationError, match="does not prove success"):
        evaluator._validate_render_log(log, source, "leaves")


def test_delta_rows_explicitly_separate_arm_and_temporal_comparisons(monkeypatch):
    monkeypatch.setitem(shared.TEST_COUNTS, "leaves", 4)
    rows = evaluator._delta_rows("leaves", _data())
    assert len(rows) == 72
    arm = [row for row in rows if row["comparison_type"] == "End8k_vs_B0"]
    temporal = [row for row in rows if row["comparison_type"] == "30k_vs_20k"]
    assert len(arm) == 48
    assert len(temporal) == 24
    assert {row["iteration"] for row in arm} == set(evaluator.NODES)
    assert {(row["group"], row["reference_group"]) for row in arm} == {("End8k", "B0")}
    assert {row["group"] for row in temporal} == {"B0", "End8k"}
    assert all(row["iteration"] == 30_000 and row["reference_iteration"] == 20_000 for row in temporal)


def test_summary_is_atomic_hash_bound_reusable_and_tamper_evident(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluator, "REPORT_ROOT", tmp_path / "report")
    data = _data()
    preflight = {
        "models": {"B0": "b0", "End8k": "end8k"},
        "manifest_sha256": "manifest",
        "test_images": [f"{index:05d}.png" for index in range(4)],
        "test_view_count": 4,
        "training_completion_audits": {
            group: {key: None for key in evaluator.CURRENT_TRAINING_AUDIT_FIELDS}
            for group in evaluator.GROUPS
        },
        "evaluation_code_sha256": {"contract": "current-test"},
        "render_configuration": {"resolution": 8},
    }
    with evaluator.scene_profile("leaves"):
        evaluator.write_reports("leaves", data, preflight)
        summary = evaluator.report_dir("leaves") / "summary"
        record = evaluator._validate_summary("leaves", data)
        assert record["comparison_types"] == {
            "End8k_vs_B0": "End8k minus B0 at each common node",
            "30k_vs_20k": "same-arm 30000 minus 20000",
        }
        with mock.patch("uuid.uuid4", side_effect=AssertionError("must reuse summary")):
            evaluator.write_reports("leaves", data, preflight)
        metrics = summary / "metrics_table.csv"
        metrics.write_text(metrics.read_text(encoding="utf-8") + "tamper\n", encoding="utf-8")
        with pytest.raises(evaluator.EvaluationError, match="SHA256 mismatch"):
            evaluator._validate_summary("leaves", data)


def test_stage_reuse_requires_complete_metrics_and_render_logs(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluator, "REPORT_ROOT", tmp_path / "report")
    monkeypatch.setitem(shared.TEST_COUNTS, "leaves", 4)
    source = tmp_path / "source"
    _write_rgb_gt(source)
    stage = evaluator.stage_path("leaves", "B0", 5000)
    _write_rgb_gt(stage / "test/ours_5000")
    aggregate = {metric: 1.0 for metric in evaluator.METRICS}
    details = {
        metric: {f"{index:05d}.png": 1.0 for index in range(4)}
        for metric in evaluator.METRICS
    }
    (stage / "results.json").write_text(json.dumps({"ours_5000": aggregate}), encoding="utf-8")
    (stage / "per_view.json").write_text(json.dumps({"ours_5000": details}), encoding="utf-8")
    with pytest.raises(evaluator.EvaluationError, match="missing metrics log"):
        evaluator._validate_stage("leaves", "B0", 5000, source)


def test_cli_guards_execution_and_preflight_never_starts_subprocess():
    assert evaluator.main(["--scene", "leaves", "--execute"]) == 2
    assert evaluator.main(["--scene", "leaves", evaluator.EXECUTION_AUTHORIZATION_FLAG]) == 2
    with mock.patch.object(evaluator, "preflight", return_value={"status": "PASS"}), \
            mock.patch.object(evaluator.shared.subprocess, "run") as run, \
            contextlib.redirect_stdout(io.StringIO()):
        assert evaluator.main(["--scene", "leaves", "--preflight"]) == 0
    run.assert_not_called()


def test_verify_completed_is_read_only_and_missing_report_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluator, "REPORT_ROOT", tmp_path / "report")
    monkeypatch.setattr(evaluator, "preflight", lambda scene: {"training_completion_audits": {}})
    with pytest.raises((evaluator.EvaluationError, shared.EvaluationError), match="missing RGB/GT"):
        evaluator.verify_completed_evaluation("leaves")


def test_shared_and_room_profiles_restore_original_behavior():
    original = (shared.REPORT, shared.SCENES, shared.FROZEN_REFERENCE_MODEL_DIRS, shared.TEST_COUNTS)
    assert shared.expected_report_rows() == (32, 1008)
    assert room_evaluator.expected_report_rows() == (16, 432)
    assert (shared.REPORT, shared.SCENES, shared.FROZEN_REFERENCE_MODEL_DIRS, shared.TEST_COUNTS) == original
