from __future__ import annotations

import contextlib
import csv
import dataclasses
import io
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from script import run_ns30k_candidate as launcher
from utils.ns30k_sampling import NS30KContractError, NS30KSamplingTable


def one_value(command, flag):
    return command[command.index(flag) + 1]


class NS30KCandidateCommandTests(unittest.TestCase):
    def test_all_six_scene_arm_commands_are_complete_and_frozen(self):
        for scene in launcher.SCENES:
            for arm in launcher.ARMS:
                with self.subTest(scene=scene, arm=arm):
                    plan = launcher.build_plan(scene, arm)
                    command = launcher.build_train_command(plan, python="python")
                    spec = launcher.SCENES[scene]
                    self.assertEqual(command[:3], ["python", "-u", str(ROOT / "train.py")])
                    self.assertEqual(one_value(command, "-s"), str(launcher.DATA_ROOT / scene))
                    self.assertEqual(one_value(command, "-m"), str(plan.output_dir))
                    self.assertEqual(one_value(command, "--view_manifest"), str(ROOT / spec.manifest))
                    self.assertEqual(one_value(command, "--suffix"), spec.pdc_suffix)
                    self.assertEqual(one_value(command, "--ns30k_sampling_tape"), str(ROOT / spec.sampling_table))
                    self.assertEqual(one_value(command, "--ns30k_sampling_tape_sha256"), spec.sampling_sha256)
                    expected_values = {
                        "--iterations": "30000",
                        "--position_lr_init": "0.00016",
                        "--position_lr_final": "0.0000016",
                        "--position_lr_max_steps": "30000",
                        "--dropout_factor": "0.2",
                        "--sigma_noise": "0",
                        "--opacity_decay_factor": "0.995",
                    }
                    for flag, expected in expected_values.items():
                        self.assertEqual(one_value(command, flag), expected)
                    save_index = command.index("--save_iterations")
                    checkpoint_index = command.index("--checkpoint_iterations")
                    self.assertEqual(command[save_index + 1:checkpoint_index], ["5000", "10000", "20000", "30000"])
                    self.assertEqual(command[checkpoint_index + 1:checkpoint_index + 5], ["5000", "10000", "20000", "30000"])
                    for flag in ("--eval", "--no-binocular-consistency", "--disable_training_time_evaluation"):
                        self.assertIn(flag, command)
                    for flag in ("--binocular_consistency", "--random_background", "--start_checkpoint", "--i1_b1_weak_continuation"):
                        self.assertNotIn(flag, command)
                    self.assertIn(str(plan.train_log), launcher.format_logged_command(plan, "python"))
                    if arm == "B0":
                        self.assertEqual(one_value(command, "--ns30k_arm"), "OFF")
                        self.assertNotIn("--enable_oe_structural_loss", command)
                        self.assertNotIn("--oe_preserve_baseline_densification_stats", command)
                        self.assertNotIn("--oe_depth_cache_dir", command)
                    else:
                        self.assertEqual(one_value(command, "--ns30k_arm"), "ON")
                        self.assertIn("--enable_oe_structural_loss", command)
                        self.assertIn("--oe_preserve_baseline_densification_stats", command)
                        expected_i1 = {
                            "--oe_structural_start_iter": "1",
                            "--oe_structural_end_iter": "8000",
                            "--oe_structural_weight": "0.1",
                            "--oe_structure_gate_mode": "stable_evidence",
                            "--oe_stable_quantile": "0.8",
                            "--oe_depth_cache_dir": str(ROOT / spec.cache_dir),
                        }
                        for flag, expected in expected_i1.items():
                            self.assertEqual(one_value(command, flag), expected)

    def test_off_and_on_share_exactly_one_scene_sampling_table(self):
        for scene in launcher.SCENES:
            with self.subTest(scene=scene):
                off = launcher.build_plan(scene, "B0")
                on = launcher.build_plan(scene, "B0+I1")
                self.assertEqual(off.sampling_table, on.sampling_table)
                self.assertEqual(off.spec.sampling_sha256, on.spec.sampling_sha256)
                self.assertEqual(off.manifest, on.manifest)
                self.assertEqual(off.pdc, on.pdc)

    def test_scenes_do_not_mix_any_frozen_asset(self):
        plans = [launcher.build_plan(scene, "B0+I1") for scene in launcher.SCENES]
        for field in ("manifest", "pdc", "cache_dir", "sampling_table"):
            self.assertEqual(len({getattr(plan, field) for plan in plans}), 3)
        for field in ("manifest_sha256", "pdc_sha256", "cache_metadata_sha256", "sampling_sha256"):
            self.assertEqual(len({getattr(plan.spec, field) for plan in plans}), 3)


class NS30KCandidatePreflightTests(unittest.TestCase):
    def test_real_read_only_preflight_passes_for_each_scene(self):
        for scene in launcher.SCENES:
            with self.subTest(scene=scene):
                plan = launcher.build_plan(scene, "B0")
                self.assertFalse(plan.output_dir.exists())
                result = launcher.preflight(plan, python="python")
                self.assertEqual(result["status"], "PASS")
                self.assertEqual(result["scene"], scene)
                self.assertEqual(result["midas_record_count"], 3)
                self.assertEqual(len(result["train_images"]), 3)
                self.assertFalse(set(result["train_images"]) & set(result["test_images"]))
                self.assertFalse(plan.output_dir.exists())

    def test_tampered_frozen_asset_hashes_are_refused(self):
        for field in ("sampling_sha256", "manifest_sha256", "pdc_sha256", "cache_metadata_sha256"):
            with self.subTest(field=field):
                plan = launcher.build_plan("fortress", "B0")
                bad_spec = dataclasses.replace(plan.spec, **{field: "0" * 64})
                with self.assertRaisesRegex((launcher.PreflightError, launcher.NS30KContractError), "SHA256|hash"):
                    launcher.preflight(dataclasses.replace(plan, spec=bad_spec), python="python")

    def test_tampered_critical_code_hash_is_refused(self):
        with mock.patch.object(launcher, "CRITICAL_CODE_SHA256", {"train.py": "0" * 64}):
            with self.assertRaisesRegex(launcher.PreflightError, "critical code train.py SHA256 mismatch"):
                launcher.preflight(launcher.build_plan("fortress", "B0"), python="python")

    def test_wrong_data_root_is_refused_before_asset_use(self):
        with tempfile.TemporaryDirectory() as temporary:
            plan = launcher.build_plan("fortress", "B0", data_root=Path(temporary))
            with self.assertRaisesRegex(launcher.PreflightError, "data root is frozen"):
                launcher.preflight(plan, python="python")

    def test_any_existing_output_directory_is_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "output").mkdir()
            plan = launcher.build_plan("fortress", "B0", root=root)
            plan.output_dir.mkdir()
            (plan.output_dir / "historical.txt").write_text("old", encoding="utf-8")
            with self.assertRaisesRegex(launcher.PreflightError, "REFUSING to reuse"):
                launcher._validate_output_target(plan)


class NS30KCandidateExecutionGuardTests(unittest.TestCase):
    def test_default_mode_never_starts_a_subprocess(self):
        output = io.StringIO()
        with mock.patch.object(launcher, "preflight", return_value={"status": "PASS"}), \
                mock.patch.object(launcher.subprocess, "Popen") as popen, \
                contextlib.redirect_stdout(output):
            code = launcher.main(["--scene", "fortress", "--arm", "B0"])
        self.assertEqual(code, 0)
        popen.assert_not_called()
        self.assertIn("no training process started", output.getvalue())

    def test_unauthorized_execute_is_refused_before_preflight_or_subprocess(self):
        error = io.StringIO()
        with mock.patch.object(launcher, "preflight") as preflight_mock, \
                mock.patch.object(launcher.subprocess, "Popen") as popen, \
                contextlib.redirect_stderr(error):
            code = launcher.main(["--scene", "fortress", "--arm", "B0", "--execute"])
        self.assertEqual(code, 2)
        preflight_mock.assert_not_called()
        popen.assert_not_called()
        self.assertIn(launcher.EXECUTION_AUTHORIZATION_FLAG, error.getvalue())

    def test_authorization_marker_alone_is_refused(self):
        with mock.patch.object(launcher, "preflight") as preflight_mock, contextlib.redirect_stderr(io.StringIO()):
            code = launcher.main([
                "--scene", "fortress", "--arm", "B0", launcher.EXECUTION_AUTHORIZATION_FLAG
            ])
        self.assertEqual(code, 2)
        preflight_mock.assert_not_called()

    def test_execute_claims_directory_and_records_failed_exit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "output").mkdir()
            plan = launcher.build_plan("fortress", "B0", root=root)
            command = ["fake-python", "-u", "fake-train.py"]

            class FakeProcess:
                def wait(self):
                    return 7

            with mock.patch.object(
                launcher, "preflight", return_value={"status": "PASS", "training_command": command}
            ), mock.patch.object(launcher.subprocess, "Popen", return_value=FakeProcess()) as popen:
                self.assertEqual(
                    launcher.execute(plan, python="fake-python", formal_authorized=True), 7
                )
            self.assertTrue(plan.output_dir.is_dir())
            text = plan.train_log.read_text(encoding="utf-8")
            self.assertIn("PREFLIGHT: PASS", text)
            self.assertIn("EXIT_CODE: 7", text)
            self.assertIn("STATUS: FAILED", text)
            self.assertNotIn("STATUS: SUCCESS", text)
            self.assertIs(popen.call_args.kwargs["stderr"], subprocess.STDOUT)
            with self.assertRaisesRegex(launcher.PreflightError, "REFUSING to reuse"):
                launcher._validate_output_target(plan)

    def test_failed_preflight_creates_no_output_or_log(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "output").mkdir()
            plan = launcher.build_plan("fortress", "B0", root=root)
            with mock.patch.object(launcher, "preflight", side_effect=launcher.PreflightError("blocked")):
                with self.assertRaisesRegex(launcher.PreflightError, "blocked"):
                    launcher.execute(plan, python="fake-python", formal_authorized=True)
            self.assertFalse(plan.output_dir.exists())
            self.assertFalse(plan.train_log.exists())


class NS30KCandidateSmokeTests(unittest.TestCase):
    def test_smoke_commands_are_isolated_without_changing_formal_commands(self):
        for arm in launcher.ARMS:
            with self.subTest(arm=arm):
                formal = launcher.build_plan("fortress", arm)
                smoke = launcher.build_plan("fortress", arm, smoke=True)
                formal_command = launcher.build_train_command(formal, "python")
                smoke_command = launcher.build_train_command(smoke, "python")
                self.assertNotIn("SMOKE", str(formal.output_dir))
                self.assertIn("SMOKE", str(smoke.output_dir))
                self.assertNotEqual(formal.output_dir, smoke.output_dir)
                self.assertNotIn("--ns30k_smoke", formal_command)
                normalized_formal = list(formal_command)
                normalized_smoke = list(smoke_command[:-1])
                normalized_formal[normalized_formal.index(str(formal.output_dir))] = "<OUTPUT>"
                normalized_smoke[normalized_smoke.index(str(smoke.output_dir))] = "<OUTPUT>"
                self.assertEqual(normalized_smoke, normalized_formal)
                self.assertEqual(smoke_command[-1], "--ns30k_smoke")
                self.assertEqual(one_value(smoke_command, "--iterations"), "30000")
                self.assertEqual(one_value(smoke_command, "--position_lr_max_steps"), "30000")
                self.assertEqual(smoke.sampling_table, formal.sampling_table)
                self.assertEqual(smoke.spec.sampling_sha256, formal.spec.sampling_sha256)

    def test_smoke_is_restricted_to_fortress(self):
        for scene in ("room", "horns"):
            with self.subTest(scene=scene):
                with self.assertRaisesRegex(launcher.PreflightError, "restricted to Fortress"):
                    launcher.build_plan(scene, "B0", smoke=True)

    def test_smoke_read_only_preflight_passes_for_both_arms(self):
        for arm in launcher.ARMS:
            with self.subTest(arm=arm):
                plan = launcher.build_plan("fortress", arm, smoke=True)
                result = launcher.preflight(plan, python="python")
                self.assertEqual(result["mode"], "SMOKE_PREFLIGHT_ONLY")
                self.assertIs(result["SMOKE_ONLY"], True)
                self.assertIs(result["NOT_FORMAL_RESULT"], True)
                self.assertEqual(result["smoke_stop_after_normal_iteration"], 10)
                self.assertEqual(result["configured_iterations_remain"], 30_000)
                self.assertFalse(plan.output_dir.exists())

    def test_first_ten_records_are_partial_and_formal_assert_complete_still_rejects(self):
        spec = launcher.SCENES["fortress"]
        table = NS30KSamplingTable(
            ROOT / spec.sampling_table,
            spec.sampling_sha256,
            expected_scene="fortress",
        )
        observed = [table.consume(iteration) for iteration in range(1, 11)]
        self.assertEqual(table.consumed_count, 10)
        self.assertEqual([item["iteration"] for item in observed], list(range(1, 11)))
        with self.assertRaisesRegex(NS30KContractError, "incomplete"):
            table.assert_complete()

    def test_unauthorized_smoke_execution_is_refused_at_cli_and_execute_boundary(self):
        error = io.StringIO()
        with mock.patch.object(launcher, "preflight") as preflight_mock, \
                contextlib.redirect_stderr(error):
            code = launcher.main([
                "--scene", "fortress", "--arm", "B0", "--smoke", "--execute-smoke"
            ])
        self.assertEqual(code, 2)
        preflight_mock.assert_not_called()
        self.assertIn(launcher.SMOKE_EXECUTION_AUTHORIZATION_FLAG, error.getvalue())
        plan = launcher.build_plan("fortress", "B0", smoke=True)
        with self.assertRaisesRegex(launcher.PreflightError, "independent authorization"):
            launcher.execute(plan, python="fake-python")

    def test_failed_smoke_keeps_log_exit_code_and_fail_audit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "output").mkdir()
            plan = launcher.build_plan("fortress", "B0", root=root, smoke=True)
            command = ["fake-python", "-u", "fake-train.py", "--ns30k_smoke"]

            class FakeProcess:
                def wait(self):
                    return 7

            with mock.patch.object(
                launcher, "preflight", return_value={"status": "PASS", "training_command": command}
            ), mock.patch.object(launcher.subprocess, "Popen", return_value=FakeProcess()) as popen:
                code = launcher.execute(plan, python="fake-python", smoke_authorized=True)
            self.assertEqual(code, 7)
            self.assertTrue(plan.train_log.is_file())
            log = plan.train_log.read_text(encoding="utf-8")
            self.assertIn("EXIT_CODE: 7", log)
            self.assertIn("STATUS: SMOKE_FAIL", log)
            result = __import__("json").loads(
                (plan.output_dir / "ns30k_smoke_result.json").read_text(encoding="utf-8")
            )
            self.assertEqual(result["status"], "SMOKE_FAIL")
            self.assertEqual(result["process_exit_code"], 7)
            self.assertIs(result["process_started"], True)
            self.assertIsNone(result["actual_completed_training_iterations"])
            self.assertEqual(result["completed_iteration_status"], "UNVERIFIED")
            self.assertEqual(popen.call_args.kwargs["env"][launcher.SMOKE_AUTH_ENV], "1")

    def test_six_trace_rows_do_not_claim_six_completed_training_iterations(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "output").mkdir()
            plan = launcher.build_plan("fortress", "B0", root=root, smoke=True)
            plan.output_dir.mkdir()
            trace = plan.output_dir / "ns30k_sampling_trace.csv"
            with trace.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=("iteration", "camera_image_name"))
                writer.writeheader()
                for iteration in range(1, 7):
                    writer.writerow({"iteration": iteration, "camera_image_name": "IMG"})
            self.assertIs(launcher._finalize_smoke_result(
                plan, 9, process_started=True
            ), False)
            result = __import__("json").loads(
                (plan.output_dir / "ns30k_smoke_result.json").read_text(encoding="utf-8")
            )
            self.assertEqual(result["sampling_trace_rows_written"], 6)
            self.assertIsNone(result["actual_completed_training_iterations"])
            self.assertEqual(result["completed_iteration_status"], "UNVERIFIED")
            self.assertEqual(result["status"], "SMOKE_FAIL")

    def test_popen_failure_keeps_original_exception_and_writes_failure_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "output").mkdir()
            plan = launcher.build_plan("fortress", "B0", root=root, smoke=True)
            command = ["missing-python", "-u", "fake-train.py", "--ns30k_smoke"]
            launch_error = FileNotFoundError("mock child launch failure")
            with mock.patch.object(
                launcher, "preflight", return_value={"status": "PASS", "training_command": command}
            ), mock.patch.object(launcher.subprocess, "Popen", side_effect=launch_error):
                with self.assertRaises(FileNotFoundError) as raised:
                    launcher.execute(plan, python="missing-python", smoke_authorized=True)
            self.assertIs(raised.exception, launch_error)
            log = plan.train_log.read_text(encoding="utf-8")
            self.assertIn("LAUNCH_ERROR: FileNotFoundError: mock child launch failure", log)
            self.assertIn("STATUS: SMOKE_FAIL", log)
            result = __import__("json").loads(
                (plan.output_dir / "ns30k_smoke_result.json").read_text(encoding="utf-8")
            )
            self.assertEqual(result["status"], "SMOKE_FAIL")
            self.assertIs(result["process_started"], False)
            self.assertIsNone(result["process_exit_code"])
            self.assertIsNone(result["actual_completed_training_iterations"])
            self.assertEqual(result["completed_iteration_status"], "UNVERIFIED")
            self.assertEqual(result["sampling_trace_rows_written"], 0)

    def test_successful_smoke_audit_is_checked_before_launcher_pass(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "output").mkdir()
            plan = launcher.build_plan("fortress", "B0", root=root, smoke=True)
            command = ["fake-python", "-u", "fake-train.py", "--ns30k_smoke"]

            class FakeProcess:
                def wait(self_inner):
                    initial = {
                        "arm": "OFF", "scene": "fortress", "initialization_seed": 0,
                        "manifest_sha256": "m", "pdc_sha256": "p",
                        "sampling_table_file_sha256": "s", "initial_gaussian_count": 3,
                        "initial_gaussian_model_sha256": "abc", "train_py_sha256": "t",
                        "SMOKE_ONLY": True, "NOT_FORMAL_RESULT": True,
                    }
                    result = {
                        "audit_version": "ns30k-smoke-result-v1",
                        "status": "SMOKE_PASS", "SMOKE_ONLY": True,
                        "NOT_FORMAL_RESULT": True,
                        "actual_completed_training_iterations": 10,
                        "sampling_records_consumed": 10,
                        "camera_sequence_matches_sampling_table": True,
                        "nonfinite_loss_seen": False,
                        "i1_structural_loss_observation_matches_arm": True,
                        "initial_gaussian_model_sha256": "abc",
                    }
                    import json
                    (plan.output_dir / "ns30k_initial_state_audit.json").write_text(
                        json.dumps(initial), encoding="utf-8"
                    )
                    (plan.output_dir / "ns30k_smoke_result.json").write_text(
                        json.dumps(result), encoding="utf-8"
                    )
                    with (plan.output_dir / "ns30k_sampling_trace.csv").open(
                        "w", encoding="utf-8", newline=""
                    ) as handle:
                        writer = csv.DictWriter(handle, fieldnames=("iteration",))
                        writer.writeheader()
                        for iteration in range(1, 11):
                            writer.writerow({"iteration": iteration})
                    return 0

            with mock.patch.object(
                launcher, "preflight", return_value={"status": "PASS", "training_command": command}
            ), mock.patch.object(launcher.subprocess, "Popen", return_value=FakeProcess()):
                code = launcher.execute(plan, python="fake-python", smoke_authorized=True)
            self.assertEqual(code, 0)
            result = __import__("json").loads(
                (plan.output_dir / "ns30k_smoke_result.json").read_text(encoding="utf-8")
            )
            self.assertEqual(result["status"], "SMOKE_PASS")
            self.assertEqual(result["process_exit_code"], 0)
            self.assertIs(result["process_started"], True)
            self.assertEqual(result["sampling_trace_rows_written"], 10)
            self.assertEqual(result["actual_completed_training_iterations"], 10)
            self.assertEqual(result["completed_iteration_status"], "VERIFIED_BY_TRAINING_REPORT")
            self.assertEqual(result["initialization_pairing"]["status"], "UNVERIFIED_OTHER_ARM_NOT_AVAILABLE")

    def test_initial_state_pair_requires_identical_hash_and_frozen_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            off_path, on_path = root / "off.json", root / "on.json"
            common = {
                "scene": "fortress", "initialization_seed": 0,
                "manifest_sha256": "m", "pdc_sha256": "p",
                "sampling_table_file_sha256": "s", "initial_gaussian_count": 42,
                "initial_gaussian_model_sha256": "same", "train_py_sha256": "t",
                "SMOKE_ONLY": True, "NOT_FORMAL_RESULT": True,
            }
            import json
            off_path.write_text(json.dumps({**common, "arm": "OFF"}), encoding="utf-8")
            on_path.write_text(json.dumps({**common, "arm": "ON"}), encoding="utf-8")
            valid = launcher.compare_smoke_initial_state_audits(off_path, on_path)
            self.assertIs(valid["eligible_as_same_initialization_pair"], True)
            on_path.write_text(json.dumps({**common, "arm": "ON", "initial_gaussian_model_sha256": "different"}), encoding="utf-8")
            invalid = launcher.compare_smoke_initial_state_audits(off_path, on_path)
            self.assertIs(invalid["eligible_as_same_initialization_pair"], False)

    def test_train_source_keeps_formal_completion_and_read_only_initial_hashing(self):
        source = (ROOT / "train.py").read_text(encoding="utf-8")
        self.assertIn("else:\n            ns30k_table.assert_complete()", source)
        self.assertIn("loop_end = NS30K_SMOKE_ITERATIONS", source)
        self.assertIn("ns30k_table.consumed_count == NS30K_SMOKE_ITERATIONS", source)
        self.assertIn('"SMOKE_ONLY": True', source)
        start = source.index("def write_ns30k_initial_state_audit")
        end = source.index("\ndef _validate_resume_device_consistency", start)
        helper = source[start:end]
        self.assertEqual(helper.count("gaussian_model_state_sha256(gaussians)"), 2)
        self.assertNotIn("manual_seed", helper)
        self.assertNotIn("optimizer.step", helper)
        self.assertNotIn("requires_grad", helper)


class NS30KCandidateScopeTests(unittest.TestCase):
    def test_historical_formal_launcher_and_protocol_are_unchanged(self):
        expected = {
            ROOT / "script/run_llff_30k_formal.py":
                "47400f05a6bec9acccaf2a62f15349acc7afd1a0c9e237037f9b282ef47d643e",
            ROOT / "script/llff_30k_formal_protocol.json":
                "ab2464cc4129596af37ff30aa42a40f0a1a37b690e2c2e35dcaab058ca32ddd1",
        }
        self.assertEqual({path: launcher.sha256_file(path) for path in expected}, expected)


if __name__ == "__main__":
    unittest.main()
