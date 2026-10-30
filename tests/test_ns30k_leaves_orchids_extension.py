from __future__ import annotations

import contextlib
import hashlib
import io
import json
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest import mock

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from script import run_ns30k_candidate_live_retry as launcher
from script import run_ns30k_i1_schedule_ablation_v1 as schedule_launcher
from utils.ns30k_sampling import (
    NS30KSamplingTable,
    canonical_json_bytes,
    generate_sampling_table,
)


NEW_SCENES = {
    "leaves": {
        "manifest": "output/Leaves_3v_FrozenPDC_v1/view_manifest.json",
        "manifest_sha256": "e32abc9067a4d9c5fa067d8fd846e9c929dc3f6bb278e0707ff3bbefd153e2e5",
        "pdc_suffix": "Leaves_3v_FrozenPDC_v1",
        "pdc_sha256": "c6886229ed967fc1f73444d6d351f812ba090223d0c4b8c7bab6e67623022b25",
        "cache_metadata_sha256": "27c792096e8950cea700c40aecb5604b373cb33e5f9a8aca9b87f2f5baa5d1b8",
        "sampling_sha256": "da28a200c77adc02825135d5cc9bf8b786340a1066e26daaacf06a8998ebabe7",
        "content_sha256": "e13a1fba0ef3e9fffeeac034b6a9dabc48d2b38113be7602f2d8f0a1cb30225a",
    },
    "orchids": {
        "manifest": "output/Orchids_3v_FrozenPDC_v1/view_manifest.json",
        "manifest_sha256": "8338e4024825438f27438e334d92d5c46b19fc9fb08efce8e167dbe338d5e1f4",
        "pdc_suffix": "Orchids_3v_FrozenPDC_v1",
        "pdc_sha256": "3c92a3868516b4e9814eecddbb6fca61b95601188548827f7a2962b143917278",
        "cache_metadata_sha256": "580ab5c36f71333ef319b062edd1d17a4fabec480bc3ba66362d4ffde166eafc",
        "sampling_sha256": "ba25d164fe939a598d7aea2d4d79aad3dcd28ea2c7a5cab5b9fce51827a8d491",
        "content_sha256": "54cb113fe95f777e5c8aacdd1afc27d79f9c32aeec4ec2d2f77c0bad7c3480f4",
    },
}

OLD_TAPE_HASHES = {
    "fortress": "716b4202b30a3263c54d8d3acf15327899fb8ff05ff78a5c70775844597b0234",
    "room": "533518ad70725593aede8bf57186897db4d8bd94336efb442a1ed7219ae7a595",
    "horns": "ec47679f320ce04b3be9afcb78f7a046042f4a501fa255ee3f737c18a508595e",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def one_value(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


def i1_line(document: dict) -> str:
    return "[I1] " + json.dumps(document, sort_keys=True)


def valid_end8k_log(*, extra_events=()) -> str:
    parameters = {
        "enabled": True,
        "weight": 0.1,
        "configured_start_iter": 1,
        "configured_end_iter": 8000,
        "gate_mode": "stable_evidence",
        "stable_quantile": 0.8,
        "b1_weak_continuation": False,
        "preserve_baseline_densification_stats": True,
    }
    events = [
        {"event": "configuration", "parameters": parameters,
         "actual_active_start": 1, "actual_active_end": 8000,
         "diagnostic_interval": 1000},
        {"event": "activation", "iteration": 1,
         "i1_applied_weight": 0.1, "i1_phase": "primary"},
        {"event": "activation", "iteration": 8000,
         "i1_applied_weight": 0.1, "i1_phase": "primary"},
        *extra_events,
        {"event": "summary", "actual_activation_count": 8000,
         "nonfinite_loss_seen": False},
    ]
    return "\n".join(i1_line(event) for event in events)


def test_scene_registry_preserves_original_five_scene_order():
    assert tuple(launcher.SCENES)[:5] == ("fortress", "room", "horns", "leaves", "orchids")
    for scene, expected in NEW_SCENES.items():
        spec = launcher.SCENES[scene]
        for field in (
            "manifest", "manifest_sha256", "pdc_suffix", "pdc_sha256",
            "cache_metadata_sha256", "sampling_sha256",
        ):
            assert getattr(spec, field) == expected[field]


@pytest.mark.parametrize("scene", ("leaves", "orchids"))
def test_new_frozen_tape_is_complete_bound_and_deterministic(scene):
    spec = launcher.SCENES[scene]
    expected = NEW_SCENES[scene]
    path = ROOT / spec.sampling_table
    assert sha256(path) == expected["sampling_sha256"]
    table = NS30KSamplingTable(path, spec.sampling_sha256, expected_scene=scene)
    table.verify_assets(ROOT / spec.manifest, ROOT / spec.pdc_relative)
    assert table.document["version"] == "ns30k-sampling-v1"
    assert table.document["seed"] == 0
    assert table.document["entry_count"] == 30_000
    assert table.document["content_sha256"] == expected["content_sha256"]
    assert table.document["manifest_sha256"] == expected["manifest_sha256"]
    assert table.document["pdc_sha256"] == expected["pdc_sha256"]
    assert table.document["entries"][0]["iteration"] == 1
    assert table.document["entries"][-1]["iteration"] == 30_000

    regenerated = generate_sampling_table(
        scene=scene,
        seed=0,
        manifest_path=ROOT / spec.manifest,
        pdc_path=ROOT / spec.pdc_relative,
        pdc_suffix=spec.pdc_suffix,
    )
    assert canonical_json_bytes(regenerated) == path.read_bytes()


def test_original_sampling_algorithm_and_three_tapes_are_unchanged():
    assert sha256(ROOT / "utils/ns30k_sampling.py") == (
        "801962675a46f5af5d7827036f4349957362755e705c69cfabd94468fdaa0c42"
    )
    for scene, expected in OLD_TAPE_HASHES.items():
        assert sha256(ROOT / f"reports/ns30k_frozen_sampling/{scene}_seed0_v1.json") == expected


@pytest.mark.parametrize("scene", tuple(NEW_SCENES))
def test_new_scene_preflight_or_existing_completion_is_read_only_and_passes(scene):
    plan = launcher.build_plan(scene, "B0")
    existed = plan.output_dir.exists()
    if existed:
        result = launcher.verify_completed(plan)
        assert result["status"] == "PASS"
        assert result["sampling_table_sha256"] == NEW_SCENES[scene]["sampling_sha256"]
        assert result["completed_iteration"] == 30_000
    else:
        result = launcher.preflight(plan, python="python")
        assert result["status"] == "PASS"
        assert result["sampling_table_sha256"] == NEW_SCENES[scene]["sampling_sha256"]
        assert result["save_iterations"] == [5000, 10000, 20000, 30000]
        assert result["checkpoint_iterations"] == [5000, 10000, 20000, 30000]
    assert plan.output_dir.exists() is existed


@pytest.mark.parametrize("scene", tuple(launcher.SCENES))
def test_b0_end8k_commands_keep_the_frozen_parameter_contract(scene):
    b0 = launcher.build_train_command(launcher.build_plan(scene, "B0"), "python")
    end8k = launcher.build_train_command(launcher.build_plan(scene, "B0+I1"), "python")
    common = {
        "--iterations": "30000", "--position_lr_init": "0.00016",
        "--position_lr_final": "0.0000016", "--position_lr_max_steps": "30000",
        "--dropout_factor": "0.2", "--sigma_noise": "0",
        "--opacity_decay_factor": "0.995", "--n_views": "3",
        "--resolution": "8", "--sh_degree": "1",
    }
    for command in (b0, end8k):
        for flag, value in common.items():
            assert one_value(command, flag) == value
        assert "--no-binocular-consistency" in command
        assert "--disable_training_time_evaluation" in command
        assert "--i1_b1_weak_continuation" not in command
    assert one_value(b0, "--ns30k_arm") == "OFF"
    assert "--enable_oe_structural_loss" not in b0
    assert one_value(end8k, "--ns30k_arm") == "ON"
    assert one_value(end8k, "--oe_structural_start_iter") == "1"
    assert one_value(end8k, "--oe_structural_end_iter") == "8000"
    assert one_value(end8k, "--oe_structural_weight") == "0.1"


def test_end8k_runtime_boundary_accepts_8000_and_rejects_8001():
    plan = launcher.build_plan("leaves", "B0+I1")
    result = launcher._verify_i1_runtime(plan, valid_end8k_log())
    assert result["activation_count"] == 8000
    assert result["post_8000_activation_seen"] is False

    late = {"event": "activation", "iteration": 8001,
            "i1_applied_weight": 0.1, "i1_phase": "primary"}
    with pytest.raises(launcher.PreflightError, match="outside 1--8000"):
        launcher._verify_i1_runtime(plan, valid_end8k_log(extra_events=(late,)))


def test_b0_runtime_rejects_any_i1_event_and_nonfinite_markers():
    plan = launcher.build_plan("orchids", "B0")
    assert launcher._verify_i1_runtime(plan, "ordinary finite training log")["activation_count"] == 0
    with pytest.raises(launcher.PreflightError, match="B0 train.log contains I1"):
        launcher._verify_i1_runtime(plan, i1_line({"event": "configuration"}))
    assert launcher._contains_nonfinite_loss_marker(
        i1_line({"event": "summary", "nonfinite_loss_seen": False})
    ) is False
    assert launcher._contains_nonfinite_loss_marker("Loss=nan") is True


def test_output_overwrite_and_incomplete_completion_are_fail_closed():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        (root / "output").mkdir()
        plan = launcher.build_plan("leaves", "B0", root=root)
        plan.output_dir.mkdir()
        with pytest.raises(launcher.PreflightError, match="REFUSING to reuse"):
            launcher._validate_output_target(plan)

        fake_table = mock.Mock()
        fake_table.document = {"entries": []}
        fake_table.verify_assets.return_value = None
        with mock.patch.object(launcher, "_validate_critical_code", return_value={}), \
                mock.patch.object(launcher, "require_hash"), \
                mock.patch.object(launcher, "NS30KSamplingTable", return_value=fake_table), \
                mock.patch.object(launcher, "_validate_manifest_and_colmap", return_value=({}, [])), \
                mock.patch.object(launcher, "_validate_cache", return_value={"records": []}):
            with pytest.raises(launcher.PreflightError, match="checkpoint at iteration 5000"):
                launcher.verify_completed(plan)


def test_verify_completed_cli_is_read_only_and_never_launches_training():
    expected = {"status": "PASS", "mode": "VERIFY_COMPLETED_READ_ONLY"}
    with mock.patch.object(launcher, "verify_completed", return_value=expected) as verify, \
            mock.patch.object(launcher.subprocess, "Popen") as popen, \
            contextlib.redirect_stdout(io.StringIO()):
        assert launcher.main([
            "--scene", "leaves", "--arm", "B0", "--verify-completed"
        ]) == 0
    verify.assert_called_once()
    popen.assert_not_called()


def test_logged_command_audit_rejects_any_parameter_change():
    plan = launcher.build_plan("leaves", "B0+I1")
    command = launcher.build_train_command(plan, "python")
    launcher._verify_logged_command(plan, "COMMAND: " + shlex.join(command))
    command[command.index("--oe_structural_end_iter") + 1] = "8001"
    with pytest.raises(launcher.PreflightError, match="does not match"):
        launcher._verify_logged_command(plan, "COMMAND: " + shlex.join(command))


def test_cross_launcher_hash_and_old_schedule_scope_regression():
    assert sha256(ROOT / "script/run_ns30k_candidate_live_retry.py") == (
        schedule_launcher.FROZEN_LAUNCHER_SHA256
    )
    assert schedule_launcher.SUPPORTED_SCENES == ("fortress", "horns", "room")
    assert schedule_launcher.SUPPORTED_VARIANTS == ("Strong30k", "Weak30k")
    for scene in schedule_launcher.SUPPORTED_SCENES:
        for variant in schedule_launcher.SUPPORTED_VARIANTS:
            command = schedule_launcher.build_train_command(
                schedule_launcher.build_plan(scene, variant), "python"
            )
            assert one_value(command, "--iterations") == "30000"
            assert one_value(command, "--ns30k_arm") == "ON"
            assert "--save_iterations" in command
            assert "--checkpoint_iterations" in command
