"""Guarded launcher for the NS30K-I1-Schedule-v1 ablation.

The default mode is a read-only preflight.  GPU training requires both
--execute and --authorize-execution.  Existing output directories are never
reused.
"""

from __future__ import annotations

import argparse
import codecs
import csv
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
from types import SimpleNamespace
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from script import run_ns30k_candidate_live_retry as frozen  # noqa: E402
from utils.ns30k_sampling import NS30KContractError, NS30KSamplingTable  # noqa: E402
from utils.ns30k_schedule_ablation import (  # noqa: E402
    AUTH_ENV,
    BOUNDARY_ITERATIONS,
    EXPECTED_WEIGHTS,
    PROTOCOL_VERSION,
    SUPPORTED_SCENES,
    SUPPORTED_VARIANTS,
    VARIANT_ENV,
    schedule_snapshot,
)


DATA_ROOT = frozen.DATA_ROOT
SAVE_NODES = frozen.SAVE_NODES
EXECUTION_AUTHORIZATION_FLAG = "--authorize-execution"
FROZEN_LAUNCHER_SHA256 = "0e9ab2f2b42082d30d542b531f7b9c85a58165e6401c91d40f8bf3bb2b380341"
TEST_COUNTS = {"fortress": 6, "horns": 8, "room": 6}
I1_DIAGNOSTIC_INTERVAL = 1_000
REQUIRED_LOG_ITERATIONS = (1, 8_000, 8_001, 9_000, 20_000, 30_000)
TRACE_FIELDS = (
    "iteration", "camera_image_name", "actual_camera_image_name",
    "dropout_seed", "densification_seed", "background_seed",
    "opacity_noise_seed", "sampling_file_sha256", "sampling_content_sha256",
    "entry_sha256", "dropout_pairing_level",
)


class PreflightError(RuntimeError):
    pass


@dataclass(frozen=True)
class RunPlan:
    root: Path
    data_root: Path
    scene: str
    variant: str
    output_dir: Path
    train_log: Path
    manifest: Path
    pdc: Path
    cache_dir: Path
    sampling_table: Path
    spec: frozen.SceneSpec


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_plan(scene: str, variant: str, *, root: Path = ROOT, data_root: Path = DATA_ROOT) -> RunPlan:
    scene = str(scene).lower()
    if scene not in SUPPORTED_SCENES:
        raise PreflightError(f"unsupported scene: {scene}")
    if variant not in SUPPORTED_VARIANTS:
        raise PreflightError(f"unsupported variant: {variant}")
    root = Path(root).resolve()
    data_root = Path(data_root).resolve()
    spec = frozen.SCENES[scene]
    output_dir = root / "output" / f"NS30K_I1SCHED_v1_{scene}_{variant}_seed0"
    return RunPlan(
        root=root,
        data_root=data_root,
        scene=scene,
        variant=variant,
        output_dir=output_dir,
        train_log=output_dir / "train.log",
        manifest=root / spec.manifest,
        pdc=root / spec.pdc_relative,
        cache_dir=root / spec.cache_dir,
        sampling_table=root / spec.sampling_table,
        spec=spec,
    )


def build_train_command(plan: RunPlan, python: str = sys.executable) -> list[str]:
    command = [
        str(python), "-u", str(plan.root / "train_ns30k_schedule_ablation.py"),
        "-s", str(plan.data_root / plan.scene),
        "-m", str(plan.output_dir),
        "--eval", "--dataset_name", "LLFF", "--n_views", "3",
        "--resolution", "8", "--sh_degree", "1",
        "--view_manifest", str(plan.manifest),
        "--suffix", plan.spec.pdc_suffix,
        "--iterations", "30000",
        "--position_lr_init", "0.00016",
        "--position_lr_final", "0.0000016",
        "--position_lr_max_steps", "30000",
        "--dropout_factor", "0.2", "--sigma_noise", "0",
        "--opacity_decay", "--opacity_decay_factor", "0.995",
        "--no-binocular-consistency", "--disable_training_time_evaluation",
        "--ns30k_sampling_tape", str(plan.sampling_table),
        "--ns30k_sampling_tape_sha256", plan.spec.sampling_sha256,
        "--ns30k_arm", "ON",
        "--save_iterations", *map(str, SAVE_NODES),
        "--checkpoint_iterations", *map(str, SAVE_NODES),
        "--enable_oe_structural_loss",
        "--oe_structural_start_iter", "1",
        "--oe_structural_weight", "0.1",
        "--oe_structure_gate_mode", "stable_evidence",
        "--oe_stable_quantile", "0.8",
        "--oe_depth_cache_dir", str(plan.cache_dir),
        "--oe_preserve_baseline_densification_stats",
    ]
    if plan.variant == "Strong30k":
        command.extend(["--oe_structural_end_iter", "30000"])
    else:
        command.extend([
            "--oe_structural_end_iter", "8000",
            "--i1_b1_weak_continuation",
            "--i1_b1_weak_start_iter", "8001",
            "--i1_b1_weak_end_iter", "30000",
            "--i1_b1_weak_weight", "0.02",
            "--i1_b1_weak_weight_schedule", "constant",
            "--i1_b1_weak_evidence_gamma", "1.0",
        ])
    return command


def _one_value(command: list[str], flag: str) -> str:
    index = command.index(flag)
    return command[index + 1]


def _schedule_args_from_command(command: list[str]) -> SimpleNamespace:
    return SimpleNamespace(
        enable_oe_structural_loss="--enable_oe_structural_loss" in command,
        oe_structural_start_iter=int(_one_value(command, "--oe_structural_start_iter")),
        oe_structural_end_iter=int(_one_value(command, "--oe_structural_end_iter")),
        oe_structural_weight=float(_one_value(command, "--oe_structural_weight")),
        i1_b1_weak_continuation="--i1_b1_weak_continuation" in command,
        i1_b1_weak_start_iter=int(_one_value(command, "--i1_b1_weak_start_iter")) if "--i1_b1_weak_start_iter" in command else 8001,
        i1_b1_weak_end_iter=int(_one_value(command, "--i1_b1_weak_end_iter")) if "--i1_b1_weak_end_iter" in command else 30000,
        i1_b1_weak_weight=float(_one_value(command, "--i1_b1_weak_weight")) if "--i1_b1_weak_weight" in command else 0.02,
        i1_b1_weak_weight_schedule=_one_value(command, "--i1_b1_weak_weight_schedule") if "--i1_b1_weak_weight_schedule" in command else "constant",
    )


def _audit_pre8000_equivalence(plan: RunPlan, command: list[str], python: str) -> dict:
    legacy_plan = frozen.build_plan(plan.scene, "B0+I1", root=plan.root, data_root=plan.data_root)
    legacy = frozen.build_train_command(legacy_plan, python)
    value_flags = (
        "--dataset_name", "--n_views", "--resolution", "--sh_degree",
        "--view_manifest", "--suffix", "--iterations", "--position_lr_init",
        "--position_lr_final", "--position_lr_max_steps", "--dropout_factor",
        "--sigma_noise", "--opacity_decay_factor", "--ns30k_sampling_tape",
        "--ns30k_sampling_tape_sha256", "--ns30k_arm",
        "--oe_structural_start_iter", "--oe_structural_weight",
        "--oe_structure_gate_mode", "--oe_stable_quantile", "--oe_depth_cache_dir",
    )
    mismatches = [
        flag for flag in value_flags
        if _one_value(command, flag) != _one_value(legacy, flag)
    ]
    boolean_flags = (
        "--eval", "--opacity_decay", "--no-binocular-consistency",
        "--disable_training_time_evaluation", "--enable_oe_structural_loss",
        "--oe_preserve_baseline_densification_stats",
    )
    mismatches.extend(
        flag for flag in boolean_flags if (flag in command) != (flag in legacy)
    )
    if mismatches:
        raise PreflightError(f"iterations 1--8000 differ from frozen End8k configuration: {mismatches}")
    return {
        "status": "PASS",
        "same_frozen_train_py_execution": True,
        "same_sampling_manifest_pdc_midas": True,
        "same_invariant_cli_flags": list(value_flags + boolean_flags),
        "allowed_schedule_difference_begins_after_iteration": 8000,
    }


def _validate_entrypoint(root: Path) -> dict[str, str]:
    entry = root / "train_ns30k_schedule_ablation.py"
    protocol = root / "utils" / "ns30k_schedule_ablation.py"
    for path in (entry, protocol):
        if not path.is_file() or path.stat().st_size == 0:
            raise PreflightError(f"missing schedule code: {path}")
    source = entry.read_text(encoding="utf-8")
    required = (
        "ns30k_sampling.validate_ns30k_training_configuration",
        "validate_ns30k_schedule_ablation_training_configuration",
        "runpy.run_path",
        'with_name("train.py")',
    )
    if any(item not in source for item in required):
        raise PreflightError("schedule entry point no longer delegates exactly to frozen train.py")
    return {str(path.relative_to(root)).replace("\\", "/"): sha256_file(path) for path in (entry, protocol)}


def preflight(plan: RunPlan, *, python: str = sys.executable) -> dict:
    if plan.data_root != DATA_ROOT.resolve():
        raise PreflightError(f"data root is frozen to {DATA_ROOT.resolve()}")
    if plan.output_dir.exists():
        raise PreflightError(f"REFUSING to reuse existing output directory: {plan.output_dir}")
    if not plan.output_dir.parent.is_dir():
        raise PreflightError(f"output parent does not exist: {plan.output_dir.parent}")
    if sha256_file(plan.root / "script" / "run_ns30k_candidate_live_retry.py") != FROZEN_LAUNCHER_SHA256:
        raise PreflightError("frozen NS30K launcher SHA256 mismatch")

    frozen._validate_frozen_data_root(plan.data_root)
    frozen_plan = frozen.build_plan(plan.scene, "B0+I1", root=plan.root, data_root=plan.data_root)
    code_hashes = frozen._validate_critical_code(frozen_plan)
    table = NS30KSamplingTable(plan.sampling_table, plan.spec.sampling_sha256, expected_scene=plan.scene)
    frozen.require_hash(plan.manifest, plan.spec.manifest_sha256, "view Manifest")
    frozen.require_hash(plan.pdc, plan.spec.pdc_sha256, "frozen PDC")
    table.verify_assets(plan.manifest, plan.pdc)
    manifest, colmap_records = frozen._validate_manifest_and_colmap(frozen_plan)
    cache = frozen._validate_cache(frozen_plan, manifest)
    command = build_train_command(plan, python)
    actual_weights = schedule_snapshot(_schedule_args_from_command(command))
    expected_weights = {str(k): v for k, v in EXPECTED_WEIGHTS[plan.variant].items()}
    if actual_weights != expected_weights:
        raise PreflightError(f"training-path schedule mismatch: {actual_weights} != {expected_weights}")
    if "--start_checkpoint" in command:
        raise PreflightError("checkpoint resume is forbidden")
    pre8000 = _audit_pre8000_equivalence(plan, command, python)

    return {
        "status": "PASS",
        "mode": "PREFLIGHT_ONLY",
        "protocol": PROTOCOL_VERSION,
        "scene": plan.scene,
        "variant": plan.variant,
        "data_root": str(plan.data_root),
        "output_dir": str(plan.output_dir),
        "train_log": str(plan.train_log),
        "training_environment": {AUTH_ENV: "1", VARIANT_ENV: plan.variant},
        "training_command": command,
        "logged_command": shlex.join(command),
        "effective_weights_from_training_path": actual_weights,
        "pre8000_end8k_equivalence": pre8000,
        "save_iterations": list(SAVE_NODES),
        "checkpoint_iterations": list(SAVE_NODES),
        "checkpoint_resume": False,
        "sampling_table_sha256": table.file_sha256,
        "sampling_content_sha256": table.table_id,
        "manifest_sha256": sha256_file(plan.manifest),
        "pdc_sha256": sha256_file(plan.pdc),
        "midas_metadata_sha256": sha256_file(plan.cache_dir / "oe_depth_cache.json"),
        "train_images": list(manifest["selected_train_images"]),
        "test_images": list(manifest["test_images"]),
        "expected_test_image_count": TEST_COUNTS[plan.scene],
        "colmap_image_count": len(colmap_records),
        "midas_record_count": len(cache["records"]),
        "frozen_critical_code_sha256": code_hashes,
        "schedule_code_sha256": _validate_entrypoint(plan.root),
    }


def _read_trace(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 30_000:
        raise PreflightError(f"sampling trace must have 30000 records, got {len(rows)}")
    for iteration, row in enumerate(rows, 1):
        if row.get("iteration") != str(iteration) or row.get("camera_image_name") != row.get("actual_camera_image_name"):
            raise PreflightError(f"invalid sampling trace at iteration {iteration}")
    return rows


def _expected_logged_activation(variant: str, iteration: int) -> tuple[str, float]:
    if variant == "Strong30k":
        return "primary", 0.1
    if variant == "Weak30k" and int(iteration) <= 8_000:
        return "primary", 0.1
    if variant == "Weak30k":
        return "b1_weak_continuation", 0.02
    raise PreflightError(f"unsupported variant in log audit: {variant}")


def audit_i1_activation_log(text: str, variant: str) -> dict[int, dict]:
    """Validate every emitted runtime activation, not a printed expectation."""
    observations: dict[int, dict] = {}
    configuration = None
    for match in re.finditer(r"\[I1\]\s+(\{[^\r\n]+\})", text):
        try:
            event = json.loads(match.group(1))
        except json.JSONDecodeError as exc:
            raise PreflightError(f"malformed I1 JSON event in train.log: {exc}") from exc
        if event.get("event") == "configuration":
            configuration = event
            continue
        if event.get("event") != "activation":
            continue
        try:
            iteration = int(event["iteration"])
            applied_weight = float(event["i1_applied_weight"])
            raw_loss = float(event["oe_struct_loss"])
            weighted_loss = float(event["weighted_oe_loss"])
            evidence_gamma = float(event["i1_evidence_gamma"])
        except (KeyError, TypeError, ValueError) as exc:
            raise PreflightError(f"incomplete I1 activation event: {event}") from exc
        if iteration in observations:
            raise PreflightError(f"duplicate I1 activation event at iteration {iteration}")
        if not 1 <= iteration <= 30_000:
            raise PreflightError(f"I1 activation iteration outside 1--30000: {iteration}")
        expected_phase, expected_weight = _expected_logged_activation(variant, iteration)
        if event.get("i1_phase") != expected_phase:
            raise PreflightError(
                f"logged I1 phase mismatch at {iteration}: "
                f"expected={expected_phase}, actual={event.get('i1_phase')}"
            )
        if not math.isclose(applied_weight, expected_weight, rel_tol=0.0, abs_tol=1e-12):
            raise PreflightError(
                f"logged effective weight mismatch at {iteration}: "
                f"expected={expected_weight}, actual={applied_weight}"
            )
        if not all(math.isfinite(value) for value in (raw_loss, weighted_loss, evidence_gamma)):
            raise PreflightError(f"non-finite I1 activation value at iteration {iteration}")
        if not math.isclose(
            weighted_loss, raw_loss * applied_weight,
            rel_tol=1e-5, abs_tol=1e-8,
        ):
            raise PreflightError(f"I1 weighted loss does not equal raw loss * weight at {iteration}")
        if not math.isclose(evidence_gamma, 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise PreflightError(f"I1 evidence gamma must be 1.0 at iteration {iteration}")
        observations[iteration] = event

    if configuration is None:
        raise PreflightError("train.log lacks the runtime I1 configuration event")
    if int(configuration.get("diagnostic_interval", -1)) != I1_DIAGNOSTIC_INTERVAL:
        raise PreflightError(
            f"unexpected runtime I1 diagnostic interval: "
            f"{configuration.get('diagnostic_interval')}"
        )
    missing = [iteration for iteration in REQUIRED_LOG_ITERATIONS if iteration not in observations]
    if missing:
        raise PreflightError(f"train.log lacks required runtime I1 activation events: {missing}")
    return observations


def contains_nonfinite_loss_marker(text: str) -> bool:
    """Recognize actual non-finite evidence without matching JSON false values."""
    return bool(
        re.search(r'"event"\s*:\s*"nonfinite_loss"', text)
        or re.search(r'"nonfinite_loss_seen"\s*:\s*true', text, re.IGNORECASE)
        or re.search(
            r"(?i)(\bnan\b|\binf(?:inity)?\b|\bNONFINITE_(?:LOSS|GRADIENT|PARAMETER)\b)",
            text,
        )
    )


def verify_completed(plan: RunPlan) -> dict:
    if not plan.output_dir.is_dir():
        raise PreflightError(f"completed output does not exist: {plan.output_dir}")
    artifacts = {}
    for iteration in SAVE_NODES:
        checkpoint = plan.output_dir / f"chkpnt{iteration}.pth"
        ply = plan.output_dir / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
        for label, path in (("checkpoint", checkpoint), ("ply", ply)):
            if not path.is_file() or path.stat().st_size == 0:
                raise PreflightError(f"missing {label} at iteration {iteration}: {path}")
        artifacts[str(iteration)] = {"checkpoint": str(checkpoint), "ply": str(ply)}

    rows = _read_trace(plan.output_dir / "ns30k_sampling_trace.csv")
    end8k_trace = plan.root / "output" / f"NS30K_v1_{plan.scene}_B0+I1_seed0" / "ns30k_sampling_trace.csv"
    reference = _read_trace(end8k_trace)
    for index, (left, right) in enumerate(zip(rows, reference), 1):
        if any(left.get(field) != right.get(field) for field in TRACE_FIELDS):
            raise PreflightError(f"sampling trace differs from same-scene End8k at iteration {index}")

    text = plan.train_log.read_text(encoding="utf-8", errors="replace")
    if contains_nonfinite_loss_marker(text):
        raise PreflightError("train.log contains a non-finite loss marker")
    observations = audit_i1_activation_log(text, plan.variant)
    if "Training complete." not in text:
        raise PreflightError("train.log does not prove normal 30000-iteration completion")
    return {
        "status": "PASS",
        "protocol": PROTOCOL_VERSION,
        "scene": plan.scene,
        "variant": plan.variant,
        "completed_iteration": 30_000,
        "artifacts": artifacts,
        "sampling_records": len(rows),
        "sampling_aligned_with_end8k": True,
        "runtime_activation_records_checked": len(observations),
        "required_runtime_activation_iterations": list(REQUIRED_LOG_ITERATIONS),
        "logged_activation_weights": {
            str(key): float(value["i1_applied_weight"])
            for key, value in sorted(observations.items())
        },
        "nonfinite_loss_seen": False,
    }


def execute(plan: RunPlan, *, python: str, authorized: bool, live_output: bool) -> int:
    if not authorized:
        raise PreflightError(f"execution requires {EXECUTION_AUTHORIZATION_FLAG}")
    evidence = preflight(plan, python=python)
    plan.output_dir.mkdir(exist_ok=False)
    command = list(evidence["training_command"])
    env = os.environ.copy()
    env.update({
        AUTH_ENV: "1", VARIANT_ENV: plan.variant,
        "PYTHONUNBUFFERED": "1", "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8",
    })
    with plan.train_log.open("x", encoding="utf-8", newline="") as log:
        log.write(f"PROTOCOL: {PROTOCOL_VERSION}\nVARIANT: {plan.variant}\n")
        log.write(f"ENVIRONMENT: {AUTH_ENV}=1 {VARIANT_ENV}={plan.variant}\n")
        log.write(f"COMMAND: {shlex.join(command)}\nPREFLIGHT: PASS\nLIVE_OUTPUT: {str(live_output).lower()}\n\n")
        log.flush()
        process = subprocess.Popen(
            command, cwd=str(plan.root), env=env,
            stdout=subprocess.PIPE if live_output else log,
            stderr=subprocess.STDOUT, bufsize=0,
        )
        if live_output:
            decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
            assert process.stdout is not None
            while True:
                chunk = process.stdout.read(4096)
                if not chunk:
                    break
                message = decoder.decode(chunk)
                if message:
                    log.write(message); log.flush(); sys.stdout.write(message); sys.stdout.flush()
            tail = decoder.decode(b"", final=True)
            if tail:
                log.write(tail); log.flush(); sys.stdout.write(tail); sys.stdout.flush()
            process.stdout.close()
        code = process.wait()
        log.write(f"\nEXIT_CODE: {code}\nSTATUS: {'SUCCESS' if code == 0 else 'FAILED'}\n")
    if code == 0:
        result = verify_completed(plan)
        audit_path = plan.output_dir / "ns30k_i1_schedule_postflight.json"
        with audit_path.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
            handle.write("\n")
    return code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True, choices=SUPPORTED_SCENES)
    parser.add_argument("--variant", required=True, choices=SUPPORTED_VARIANTS)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight", action="store_true", help="read-only preflight (default)")
    mode.add_argument("--execute", action="store_true", help="run one guarded 30k training")
    mode.add_argument("--verify-completed", action="store_true", help="read-only audit of an existing completed run")
    parser.add_argument(EXECUTION_AUTHORIZATION_FLAG, action="store_true")
    parser.add_argument("--live-output", action="store_true")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--python", default=sys.executable)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.execute != args.authorize_execution:
        print(f"FAIL: {EXECUTION_AUTHORIZATION_FLAG} is valid and required only with --execute", file=sys.stderr)
        return 2
    try:
        plan = build_plan(args.scene, args.variant, data_root=args.data_root)
        if args.verify_completed:
            result = verify_completed(plan)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        result = preflight(plan, python=args.python)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if not args.execute:
            print("PASS: read-only preflight complete; no training process started")
            return 0
        print("STATUS: EXECUTION_AUTHORIZED; starting exactly one 30k training process")
        return execute(plan, python=args.python, authorized=True, live_output=args.live_output)
    except (PreflightError, frozen.PreflightError, NS30KContractError, OSError, ValueError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
