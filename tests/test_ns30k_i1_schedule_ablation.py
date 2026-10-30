from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from script import run_ns30k_i1_schedule_ablation_v1 as launcher
from script import evaluate_ns30k_i1_schedule_ablation_v1 as evaluator
from script import evaluate_ns30k_i1_schedule_ablation_room_v1 as room_evaluator
from utils.ns30k_sampling import NS30KContractError
from utils.ns30k_schedule_ablation import (
    AUTH_ENV,
    BOUNDARY_ITERATIONS,
    EXPECTED_WEIGHTS,
    VARIANT_ENV,
    validate_ns30k_schedule_ablation_training_configuration,
)
from utils.oe_training import apply_oe_structural_training_loss


ROOT = Path(__file__).resolve().parents[1]


class FakeCache:
    def load_for_camera(self, camera, *, device, dtype):
        return torch.ones((2, 2), device=device, dtype=dtype)


CAMERA = SimpleNamespace(image_height=2, image_width=2)


def schedule_args(variant: str, scene: str = "fortress"):
    command = launcher.build_train_command(launcher.build_plan(scene, variant), "python")
    return launcher._schedule_args_from_command(command)


@pytest.mark.parametrize("variant", ("Strong30k", "Weak30k"))
def test_real_training_loss_path_uses_expected_boundary_weights(variant):
    args = schedule_args(variant)
    for iteration in BOUNDARY_ITERATIONS:
        baseline = torch.tensor(3.0)

        def compute_fn(*unused_args, **unused_kwargs):
            return torch.tensor(2.0), {"valid_pair_count": 1, "active_pair_count": 1}, None

        total, structural, stats = apply_oe_structural_training_loss(
            baseline,
            enabled=args.enable_oe_structural_loss,
            iteration=iteration,
            start_iter=args.oe_structural_start_iter,
            end_iter=args.oe_structural_end_iter,
            weight=args.oe_structural_weight,
            depth_cache=FakeCache(),
            camera=CAMERA,
            rendered_depth=torch.ones((2, 2)),
            gt_rgb=torch.ones((3, 2, 2)),
            weak_continuation=args.i1_b1_weak_continuation,
            weak_start_iter=args.i1_b1_weak_start_iter,
            weak_end_iter=args.i1_b1_weak_end_iter,
            weak_weight=args.i1_b1_weak_weight,
            weak_weight_schedule=args.i1_b1_weak_weight_schedule,
            weak_evidence_gamma=1.0,
            compute_fn=compute_fn,
            validate_fn=lambda depth, height, width: depth,
        )
        expected = EXPECTED_WEIGHTS[variant][iteration]
        assert structural.item() == 2.0
        assert stats["i1_applied_weight"] == pytest.approx(expected)
        assert total.item() == pytest.approx(3.0 + 2.0 * expected)


def test_commands_freeze_all_requested_parameters_and_never_resume():
    for scene in ("fortress", "horns", "room"):
        for variant in ("Strong30k", "Weak30k"):
            plan = launcher.build_plan(scene, variant)
            command = launcher.build_train_command(plan, "python")
            assert command[2] == str(ROOT / "train_ns30k_schedule_ablation.py")
            assert launcher._one_value(command, "--iterations") == "30000"
            assert launcher._one_value(command, "--position_lr_max_steps") == "30000"
            assert launcher._one_value(command, "--dataset_name") == "LLFF"
            assert launcher._one_value(command, "--n_views") == "3"
            assert launcher._one_value(command, "--resolution") == "8"
            assert launcher._one_value(command, "--sh_degree") == "1"
            assert launcher._one_value(command, "--dropout_factor") == "0.2"
            assert launcher._one_value(command, "--sigma_noise") == "0"
            assert launcher._one_value(command, "--opacity_decay_factor") == "0.995"
            assert launcher._one_value(command, "--ns30k_arm") == "ON"
            assert "--no-binocular-consistency" in command
            assert "--disable_training_time_evaluation" in command
            assert "--oe_preserve_baseline_densification_stats" in command
            assert "--start_checkpoint" not in command
            save = command.index("--save_iterations")
            checkpoint = command.index("--checkpoint_iterations")
            assert command[save + 1:checkpoint] == ["5000", "10000", "20000", "30000"]
            assert command[checkpoint + 1:checkpoint + 5] == ["5000", "10000", "20000", "30000"]


def test_variant_specific_command_is_exact():
    strong = launcher.build_train_command(launcher.build_plan("fortress", "Strong30k"), "python")
    weak = launcher.build_train_command(launcher.build_plan("fortress", "Weak30k"), "python")
    assert launcher._one_value(strong, "--oe_structural_end_iter") == "30000"
    assert "--i1_b1_weak_continuation" not in strong
    assert launcher._one_value(weak, "--oe_structural_end_iter") == "8000"
    assert "--i1_b1_weak_continuation" in weak
    assert launcher._one_value(weak, "--i1_b1_weak_start_iter") == "8001"
    assert launcher._one_value(weak, "--i1_b1_weak_end_iter") == "30000"
    assert launcher._one_value(weak, "--i1_b1_weak_weight") == "0.02"
    assert launcher._one_value(weak, "--i1_b1_weak_weight_schedule") == "constant"
    assert launcher._one_value(weak, "--i1_b1_weak_evidence_gamma") == "1.0"
    for variant, command in (("Strong30k", strong), ("Weak30k", weak)):
        audit = launcher._audit_pre8000_equivalence(
            launcher.build_plan("fortress", variant), command, "python"
        )
        assert audit["status"] == "PASS"
        assert audit["same_frozen_train_py_execution"] is True
        assert audit["allowed_schedule_difference_begins_after_iteration"] == 8000


@pytest.mark.parametrize("variant", ("Strong30k", "Weak30k"))
def test_room_reuses_exact_frozen_scene_and_pre8000_configuration(variant):
    plan = launcher.build_plan("room", variant)
    frozen_spec = launcher.frozen.SCENES["room"]
    assert plan.spec is frozen_spec
    assert plan.manifest == ROOT / frozen_spec.manifest
    assert plan.pdc == ROOT / frozen_spec.pdc_relative
    assert plan.cache_dir == ROOT / frozen_spec.cache_dir
    assert plan.sampling_table == ROOT / frozen_spec.sampling_table
    command = launcher.build_train_command(plan, "python")
    assert launcher._one_value(command, "--ns30k_sampling_tape_sha256") == frozen_spec.sampling_sha256
    assert launcher._one_value(command, "--view_manifest") == str(plan.manifest)
    assert launcher._one_value(command, "--suffix") == frozen_spec.pdc_suffix
    assert launcher._one_value(command, "--oe_depth_cache_dir") == str(plan.cache_dir)
    assert launcher._audit_pre8000_equivalence(plan, command, "python")["status"] == "PASS"


def fake_contract_objects(variant="Weak30k", scene="fortress"):
    args = schedule_args(variant)
    vars(args).update({
        "oe_depth_cache_dir": "cache",
        "oe_preserve_baseline_densification_stats": True,
        "oe_structure_gate_mode": "stable_evidence",
        "oe_stable_quantile": 0.8,
        "i1_b1_weak_evidence_gamma": 1.0,
        "binocular_consistency": False,
        "disable_training_time_evaluation": True,
        "start_checkpoint": None,
        "stop_after_checkpoint_iter": None,
        "paired_sampling_tape": "",
        "ns30k_arm": "ON",
        "dropout_factor": 0.2,
        "sigma_noise": 0.0,
        "opacity_decay": True,
        "opacity_decay_factor": 0.995,
        "save_iterations": [5000, 10000, 20000, 30000, 30000],
        "checkpoint_iterations": [5000, 10000, 20000, 30000],
    })
    dataset = SimpleNamespace(
        source_path=f"D:/dataset/nerf_llff_data_colmap/{scene}",
        model_path=f"D:/repo/output/NS30K_I1SCHED_v1_{scene}_{variant}_seed0",
        dataset_name="LLFF", eval=True, n_views=3, resolution=8, sh_degree=1,
        view_manifest="manifest.json", suffix="FrozenPDC", initial_opacity_arm="control",
        initial_opacity_scores="", initial_opacity_score_manifest="",
    )
    opt = SimpleNamespace(
        iterations=30000, position_lr_max_steps=30000,
        position_lr_init=0.00016, position_lr_final=0.0000016,
        lambda_dssim=0.2, densify_from_iter=1000, densify_until_iter=15000,
        densification_interval=100, densify_grad_threshold=0.0002,
        random_background=False,
    )
    table = SimpleNamespace(document={"scene": scene, "pdc_suffix": "FrozenPDC"})
    return dataset, opt, args, table


@pytest.mark.parametrize(
    "mutation,error",
    [
        (lambda a: setattr(a, "i1_b1_weak_weight_schedule", "linear_decay"), "constant"),
        (lambda a: setattr(a, "i1_b1_weak_weight", 0.03), "0.02"),
        (lambda a: setattr(a, "i1_b1_weak_start_iter", 8000), "8001"),
        (lambda a: setattr(a, "start_checkpoint", "chkpnt5000.pth"), "resume"),
    ],
)
def test_protocol_rejects_illegal_schedule_weight_overlap_and_resume(mutation, error):
    dataset, opt, args, table = fake_contract_objects()
    mutation(args)
    with mock.patch.dict(os.environ, {AUTH_ENV: "1", VARIANT_ENV: "Weak30k"}, clear=False):
        with pytest.raises(NS30KContractError, match=error):
            validate_ns30k_schedule_ablation_training_configuration(dataset, opt, args, table)


def test_protocol_rejects_wrong_scene_and_missing_launcher_authorization():
    dataset, opt, args, table = fake_contract_objects(scene="trex")
    with mock.patch.dict(os.environ, {AUTH_ENV: "1", VARIANT_ENV: "Weak30k"}, clear=False):
        with pytest.raises(NS30KContractError, match="fortress, horns, or room"):
            validate_ns30k_schedule_ablation_training_configuration(dataset, opt, args, table)
    dataset, opt, args, table = fake_contract_objects()
    with mock.patch.dict(os.environ, {AUTH_ENV: "", VARIANT_ENV: "Weak30k"}, clear=False):
        with pytest.raises(NS30KContractError, match=AUTH_ENV):
            validate_ns30k_schedule_ablation_training_configuration(dataset, opt, args, table)


@pytest.mark.parametrize("variant", ("Strong30k", "Weak30k"))
def test_room_contract_accepts_only_the_frozen_schedule(variant):
    dataset, opt, args, table = fake_contract_objects(variant=variant, scene="room")
    with mock.patch.dict(os.environ, {AUTH_ENV: "1", VARIANT_ENV: variant}, clear=False):
        assert validate_ns30k_schedule_ablation_training_configuration(
            dataset, opt, args, table
        ) == Path("cache")

    args.oe_structural_end_iter = 7_999
    with mock.patch.dict(os.environ, {AUTH_ENV: "1", VARIANT_ENV: variant}, clear=False):
        with pytest.raises(NS30KContractError):
            validate_ns30k_schedule_ablation_training_configuration(dataset, opt, args, table)


def test_existing_output_directory_fails_before_any_execution(tmp_path):
    (tmp_path / "output").mkdir()
    plan = launcher.build_plan("fortress", "Strong30k", root=tmp_path)
    plan.output_dir.mkdir()
    with pytest.raises(launcher.PreflightError, match="REFUSING"):
        launcher.preflight(plan)


def test_entrypoint_delegates_to_frozen_train_and_frozen_files_have_expected_hashes():
    source = (ROOT / "train_ns30k_schedule_ablation.py").read_text(encoding="utf-8")
    assert "runpy.run_path" in source
    assert 'with_name("train.py")' in source
    expected = {
        "train.py": "878f5c8ac282bf2e00d48255ac6f9311bad215e330c17c9ab3dbca34c6a674b1",
        "utils/ns30k_sampling.py": "801962675a46f5af5d7827036f4349957362755e705c69cfabd94468fdaa0c42",
        "script/run_ns30k_candidate_live_retry.py": "0e9ab2f2b42082d30d542b531f7b9c85a58165e6401c91d40f8bf3bb2b380341",
    }
    for relative, digest in expected.items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == digest


def test_cli_defaults_to_preflight_and_requires_double_execution_opt_in():
    parser = launcher.build_parser()
    args = parser.parse_args(["--scene", "fortress", "--variant", "Strong30k"])
    assert not args.execute and not args.authorize_execution
    assert launcher.main(["--scene", "fortress", "--variant", "Strong30k", "--execute"]) == 2
    assert launcher.main(["--scene", "fortress", "--variant", "Strong30k", "--authorize-execution"]) == 2


def test_evaluation_is_isolated_and_filters_exact_real_rgb_gt_names():
    assert evaluator.REPORT == ROOT / "reports" / "ns30k_i1_schedule_ablation_v1"
    assert evaluator.view_names("fortress") == tuple(f"{i:05d}.png" for i in range(6))
    assert evaluator.view_names("horns") == tuple(f"{i:05d}.png" for i in range(8))
    assert evaluator.model_path("fortress", "B0").name == "NS30K_v1_fortress_B0_seed0_retry1"
    assert evaluator.model_path("horns", "B0").name == "NS30K_v1_horns_B0_seed0"
    assert evaluator.model_path("fortress", "End8k").name == "NS30K_v1_fortress_B0+I1_seed0"
    assert evaluator.model_path("horns", "End8k").name == "NS30K_v1_horns_B0+I1_seed0"
    with pytest.raises(evaluator.EvaluationError, match="unsupported frozen"):
        evaluator.model_path("room", "B0")
    source = (ROOT / "script" / "evaluate_ns30k_i1_schedule_ablation_v1.py").read_text(encoding="utf-8")
    assert 'REPORT / scene / group / str(iteration)' in source
    assert 'stage / "results.json"' in source
    assert 'model / "results.json"' not in source


def test_room_evaluation_has_an_independent_profile_and_expected_row_counts():
    original = (evaluator.REPORT, evaluator.SCENES, evaluator.FROZEN_REFERENCE_MODEL_DIRS)
    assert room_evaluator.REPORT == ROOT / "reports" / "ns30k_i1_schedule_ablation_room_v1"
    assert room_evaluator.model_path("B0").name == "NS30K_v1_room_B0_seed0"
    assert room_evaluator.model_path("End8k").name == "NS30K_v1_room_B0+I1_seed0"
    assert room_evaluator.model_path("Strong30k").name == "NS30K_I1SCHED_v1_room_Strong30k_seed0"
    assert room_evaluator.model_path("Weak30k").name == "NS30K_I1SCHED_v1_room_Weak30k_seed0"
    assert room_evaluator.expected_report_rows() == (16, 432)
    assert (evaluator.REPORT, evaluator.SCENES, evaluator.FROZEN_REFERENCE_MODEL_DIRS) == original
    assert evaluator.expected_report_rows() == (32, 1008)


@pytest.mark.parametrize("scene", ("fortress", "horns"))
@pytest.mark.parametrize("group", ("B0", "End8k"))
def test_formal_reference_paths_checkpoints_and_renders_exist(scene, group):
    model = evaluator.model_path(scene, group)
    expected_name = evaluator.FROZEN_REFERENCE_MODEL_DIRS[(scene, group)]
    assert model.name == expected_name
    assert model.is_dir()
    for iteration in evaluator.NODES:
        checkpoint = model / f"chkpnt{iteration}.pth"
        ply = model / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
        assert checkpoint.is_file() and checkpoint.stat().st_size > 0
        assert ply.is_file() and ply.stat().st_size > 0
        renders, gt = evaluator.validate_rgb_gt(
            model / "test" / f"ours_{iteration}", scene
        )
        real_rgb = [path for path in renders.glob("*.png") if path.stem.isdigit()]
        real_gt = [path for path in gt.glob("*.png") if path.stem.isdigit()]
        assert len(real_rgb) == launcher.TEST_COUNTS[scene]
        assert len(real_gt) == launcher.TEST_COUNTS[scene]


def _activation_log(variant: str, *, corrupt_iteration=None, omit_iteration=None):
    events = [{
        "event": "configuration",
        "diagnostic_interval": launcher.I1_DIAGNOSTIC_INTERVAL,
    }]
    iterations = (*launcher.REQUIRED_LOG_ITERATIONS, 10_000)
    for iteration in iterations:
        if iteration == omit_iteration:
            continue
        phase, weight = launcher._expected_logged_activation(variant, iteration)
        if iteration == corrupt_iteration:
            weight = 0.777
        raw = 2.0
        events.append({
            "event": "activation", "iteration": iteration,
            "oe_struct_loss": raw, "weighted_oe_loss": raw * weight,
            "i1_phase": phase, "i1_applied_weight": weight,
            "i1_evidence_gamma": 1.0,
        })
    return "\n".join(f"[I1] {json.dumps(event, sort_keys=True)}" for event in events)


@pytest.mark.parametrize("variant", ("Strong30k", "Weak30k"))
def test_postflight_audits_every_runtime_activation_and_real_log_nodes(variant):
    observations = launcher.audit_i1_activation_log(_activation_log(variant), variant)
    assert set(launcher.REQUIRED_LOG_ITERATIONS).issubset(observations)
    assert 8_001 in observations  # frozen training path explicitly force-logs this node
    assert 9_000 in observations  # confirmed 1000-iteration diagnostic interval
    with pytest.raises(launcher.PreflightError, match="10000"):
        launcher.audit_i1_activation_log(
            _activation_log(variant, corrupt_iteration=10_000), variant
        )
    with pytest.raises(launcher.PreflightError, match="8001"):
        launcher.audit_i1_activation_log(
            _activation_log(variant, omit_iteration=8_001), variant
        )


def test_frozen_training_source_guarantees_8001_and_9000_logging_contract():
    source = (ROOT / "utils" / "oe_training.py").read_text(encoding="utf-8")
    assert "I1_DIAGNOSTIC_INTERVAL = 1000" in source
    assert "force=int(iteration) in (8001, int(weak_start_iter))" in source
    assert hashlib.sha256((ROOT / "utils" / "oe_training.py").read_bytes()).hexdigest() == (
        "7c7ffd3b817a0b964d601ae807031298d4305566ae1f9930740a308d2eeb0203"
    )


def test_nonfinite_false_json_summary_does_not_trigger_postflight_marker():
    text = '[I1] {"event": "summary", "nonfinite_loss_seen": false}'
    assert launcher.contains_nonfinite_loss_marker(text) is False


def test_nonfinite_true_json_summary_and_event_still_trigger():
    assert launcher.contains_nonfinite_loss_marker(
        '[I1] {"event": "summary", "nonfinite_loss_seen": true}'
    ) is True
    assert launcher.contains_nonfinite_loss_marker(
        '[I1] {"event": "nonfinite_loss", "nonfinite_loss_seen": false}'
    ) is True


@pytest.mark.parametrize(
    "marker",
    ("NONFINITE_LOSS", "NONFINITE_GRADIENT", "NONFINITE_PARAMETER"),
)
def test_independent_nonfinite_status_markers_still_trigger(marker):
    assert launcher.contains_nonfinite_loss_marker(f"status={marker}") is True


@pytest.mark.parametrize("marker", ("nan", "NaN", "inf", "INF", "infinity"))
def test_independent_nan_and_inf_markers_still_trigger(marker):
    assert launcher.contains_nonfinite_loss_marker(f"loss={marker}") is True


def test_completed_evaluation_stage_is_reused_without_overwrite(tmp_path, monkeypatch):
    report = tmp_path / "report"
    monkeypatch.setattr(evaluator, "REPORT", report)
    source = tmp_path / "source"
    stage = report / "fortress" / "B0" / "5000"
    target = stage / "test" / "ours_5000"
    for base in (source, target):
        for kind in ("renders", "gt"):
            folder = base / kind
            folder.mkdir(parents=True)
            for name in evaluator.view_names("fortress"):
                (folder / name).write_bytes((kind + name).encode("ascii"))
    key = "ours_5000"
    aggregate = {metric: 1.0 for metric in evaluator.METRICS}
    details = {
        metric: {name: 1.0 for name in evaluator.view_names("fortress")}
        for metric in evaluator.METRICS
    }
    (stage / "results.json").write_text(json.dumps({key: aggregate}), encoding="utf-8")
    (stage / "per_view.json").write_text(json.dumps({key: details}), encoding="utf-8")
    before = {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in stage.rglob("*") if path.is_file()
    }
    with mock.patch.object(evaluator, "run_logged", side_effect=AssertionError("must reuse")):
        result = evaluator.evaluate_one("fortress", "B0", 5000, source, "python")
    assert result[0] == aggregate
    after = {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in stage.rglob("*") if path.is_file()
    }
    assert after == before


def test_room_completed_stage_reuse_cannot_touch_original_summary(tmp_path, monkeypatch):
    original_report = tmp_path / "original"
    room_report = tmp_path / "room"
    sentinel = original_report / "summary" / "protocol_record.json"
    sentinel.parent.mkdir(parents=True)
    sentinel.write_text('{"status":"PASS","protected":true}', encoding="utf-8")
    sentinel_hash = hashlib.sha256(sentinel.read_bytes()).hexdigest()
    monkeypatch.setattr(room_evaluator, "REPORT", room_report)

    source = tmp_path / "source"
    stage = room_report / "room" / "B0" / "5000"
    target = stage / "test" / "ours_5000"
    for base in (source, target):
        for kind in ("renders", "gt"):
            folder = base / kind
            folder.mkdir(parents=True)
            for name in evaluator.view_names("room"):
                (folder / name).write_bytes((kind + name).encode("ascii"))
    aggregate = {metric: 1.0 for metric in evaluator.METRICS}
    details = {
        metric: {name: 1.0 for name in evaluator.view_names("room")}
        for metric in evaluator.METRICS
    }
    (stage / "results.json").write_text(json.dumps({"ours_5000": aggregate}), encoding="utf-8")
    (stage / "per_view.json").write_text(json.dumps({"ours_5000": details}), encoding="utf-8")

    with room_evaluator.room_profile():
        with mock.patch.object(evaluator, "run_logged", side_effect=AssertionError("must reuse")):
            result = evaluator.evaluate_one("room", "B0", 5000, source, "python")
    assert result[0] == aggregate
    assert hashlib.sha256(sentinel.read_bytes()).hexdigest() == sentinel_hash
