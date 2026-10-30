# -*- coding: utf-8 -*-
"""Guarded one-run launcher for the frozen No-Stereo 30k candidate protocol.

The default and ``--preflight`` modes are read-only.  Training can only be
started when both ``--execute`` and ``--authorize-execution`` are supplied.
One invocation always describes exactly one scene and one arm.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import os
import shlex
import struct
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llff_view_protocol import (  # noqa: E402
    ViewProtocolError,
    load_view_manifest,
    read_and_validate_colmap_scene,
)
from utils.ns30k_sampling import (  # noqa: E402
    NS30KContractError,
    NS30KSamplingTable,
)


DATA_ROOT = Path(r"D:\dataset\nerf_llff_data_colmap")
SAVE_NODES = (5_000, 10_000, 20_000, 30_000)
ARMS = ("B0", "B0+I1")
EXECUTION_AUTHORIZATION_FLAG = "--authorize-execution"
SMOKE_EXECUTION_AUTHORIZATION_FLAG = "--authorize-smoke-execution"
SMOKE_AUTH_ENV = "NS30K_SMOKE_GPU_AUTHORIZED"
EXPECTED_MIDAS_PREPROCESSING = {
    "input_rgb_range": "[0,1]",
    "normalization": "(img - 0.5) / 0.5",
    "network_size_hw": [384, 512],
    "resize_to_network": "bicubic_align_corners_false",
    "resize_to_training_image": "bicubic_align_corners_false",
    "official_dpt_transform_called": False,
    "source_image_resize": "PIL.Image.resize default (matches baseline PILtoTorch)",
}


class PreflightError(RuntimeError):
    """Raised when the frozen candidate contract cannot be proven."""


@dataclass(frozen=True)
class SceneSpec:
    scene: str
    manifest: str
    manifest_sha256: str
    pdc_suffix: str
    pdc_sha256: str
    cache_dir: str
    cache_metadata_sha256: str
    sampling_table: str
    sampling_sha256: str

    @property
    def pdc_relative(self) -> str:
        return f"keypoints_to_3d/LLFF_{self.pdc_suffix}/{self.scene}_keypoints_to_3d.ply"


SCENES: Mapping[str, SceneSpec] = {
    "fortress": SceneSpec(
        "fortress",
        "output/Fortress_3v_FrozenPDC_v1/view_manifest.json",
        "8d30c970680d26c94276b4aff04fcd2b8c1020fd3df4d51c44e454144f7402d2",
        "Fortress_3v_FrozenPDC_v1",
        "1b0d97e68e6065411314c9cddd3fca263ba76122a9ab37546ee44548d1e17944",
        "output/I1_Fortress_3v_MiDaS_DPT_Hybrid_Cache_v2",
        "1aff05afd0975597c2df85ae175ba940c5ce8ff97a952c8bed6a860373940fdd",
        "reports/ns30k_frozen_sampling/fortress_seed0_v1.json",
        "716b4202b30a3263c54d8d3acf15327899fb8ff05ff78a5c70775844597b0234",
    ),
    "room": SceneSpec(
        "room",
        "output/Room_3v_FrozenPDC_v1/view_manifest.json",
        "1f102eeed48762164e3c708acc48dfea6fefdd32ec19da2848e8b2d919c4e4dd",
        "Room_3v_FrozenPDC_v1",
        "172d524b56b3749fc3e64985dcd058ac64630d94c8fa349240c78842e9e1938b",
        "output/I1_Room_3v_MiDaS_DPT_Hybrid_Cache_v2",
        "3c44216b6e861659fd19dff7cf82447b06c7206473fd623efad1d4b14a045453",
        "reports/ns30k_frozen_sampling/room_seed0_v1.json",
        "533518ad70725593aede8bf57186897db4d8bd94336efb442a1ed7219ae7a595",
    ),
    "horns": SceneSpec(
        "horns",
        "output/A3_A3_Horns_3v_Baseline_5k_FrozenPDC/view_manifest.json",
        "a43d544c48f4767369ab37cf9ff82106f1d64365cb0e03d129559d297108586b",
        "A3_A3_Horns_3v_Baseline_5k_FrozenPDC",
        "971e1836b1fe88cba1a34d9f42c553ad75e2444e9313a75611b7cd6d4d023c7d",
        "output/I1_Horns_3v_MiDaS_DPT_Hybrid_Cache_v2",
        "c9828316d252c4e17ad7e09386453e04986caca6325794df998518819944ddfc",
        "reports/ns30k_frozen_sampling/horns_seed0_v1.json",
        "ec47679f320ce04b3be9afcb78f7a046042f4a501fa255ee3f737c18a508595e",
    ),
}


# These hashes intentionally freeze the training entry point, the NS30K
# contract, and code that controls view selection, I1, densification, RNG,
# scene construction, Gaussian state, and rendering.
CRITICAL_CODE_SHA256: Mapping[str, str] = {
    "train.py": "878f5c8ac282bf2e00d48255ac6f9311bad215e330c17c9ab3dbca34c6a674b1",
    "utils/ns30k_sampling.py": "801962675a46f5af5d7827036f4349957362755e705c69cfabd94468fdaa0c42",
    "llff_view_protocol.py": "581a8ce5996c2d52bc69682b2f3ddbbeb149902ac60a59fb6637fd88d8c49d8f",
    "utils/oe_depth_cache.py": "99ad0c2b945d4529a12f49368846d26c8a5571803245a794c921bb8bd50c6801",
    "utils/evidence_structural_loss.py": "c8c4adc76926242dd8bf561ae508bd92ebe5b6aa436c44c1dd938bec3c495275",
    "utils/oe_training.py": "7c7ffd3b817a0b964d601ae807031298d4305566ae1f9930740a308d2eeb0203",
    "utils/oe_densification.py": "0f41d7b1425d210ad4ab2abe566a3b817140285b079fa2a4f4809fcc2aae92d1",
    "utils/general_utils.py": "9d67f784af2a9860d5405a676d2792caeb834c7a06e543171f07956190dc159c",
    "arguments/__init__.py": "e2194bb2ac1ec6c62caa6e6caf88f34ab86a33dd6e2cadd58d04b8b027c7dfe8",
    "scene/__init__.py": "0c4d15d9cefbf4afde7625dfc9df13dd84038c6079f9895161ebead3dac0304d",
    "scene/dataset_readers.py": "f83e6a9e8002aff35d7a29db981c96fe5caa3cf12b8a13a89bdfa8c0ce8ba125",
    "scene/gaussian_model.py": "12eccba17e73aae89665072638bf7a3ada574fd09391b8175b0e1731b3c57b80",
    "gaussian_renderer/__init__.py": "044e7a8904f5f121e6d374f0d0572f404ff6d4ad83fd1e4f96bb07c9e41d4ee1",
}


@dataclass(frozen=True)
class RunPlan:
    root: Path
    data_root: Path
    scene: str
    arm: str
    spec: SceneSpec
    output_dir: Path
    train_log: Path
    manifest: Path
    pdc: Path
    cache_dir: Path
    sampling_table: Path
    smoke: bool


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_hash(path: Path, expected: str, label: str) -> str:
    if not path.is_file() or path.stat().st_size == 0:
        raise PreflightError(f"missing or empty {label}: {path}")
    actual = sha256_file(path)
    if actual.lower() != str(expected).lower():
        raise PreflightError(
            f"{label} SHA256 mismatch: {path}; expected={expected}; actual={actual}"
        )
    return actual


def build_plan(
    scene: str,
    arm: str,
    *,
    root: Path = ROOT,
    data_root: Path = DATA_ROOT,
    smoke: bool = False,
) -> RunPlan:
    scene_key = str(scene).lower()
    if scene_key not in SCENES:
        raise PreflightError(f"unsupported scene: {scene}")
    if arm not in ARMS:
        raise PreflightError(f"unsupported arm: {arm}")
    if smoke and scene_key != "fortress":
        raise PreflightError("NS30K Smoke is currently restricted to Fortress")
    root = Path(root).resolve()
    data_root = Path(data_root).resolve()
    spec = SCENES[scene_key]
    output_name = (
        f"NS30K_SMOKE_v1_{scene_key}_{arm}_seed0"
        if smoke else f"NS30K_v1_{scene_key}_{arm}_seed0"
    )
    output_dir = root / "output" / output_name
    return RunPlan(
        root=root,
        data_root=data_root,
        scene=scene_key,
        arm=arm,
        spec=spec,
        output_dir=output_dir,
        train_log=output_dir / "train.log",
        manifest=root / spec.manifest,
        pdc=root / spec.pdc_relative,
        cache_dir=root / spec.cache_dir,
        sampling_table=root / spec.sampling_table,
        smoke=bool(smoke),
    )


def build_train_command(plan: RunPlan, python: str = sys.executable) -> list[str]:
    command = [
        str(python), "-u", str(plan.root / "train.py"),
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
        "--ns30k_arm", "ON" if plan.arm == "B0+I1" else "OFF",
        "--save_iterations", *map(str, SAVE_NODES),
        "--checkpoint_iterations", *map(str, SAVE_NODES),
    ]
    if plan.arm == "B0+I1":
        command.extend([
            "--enable_oe_structural_loss",
            "--oe_structural_start_iter", "1",
            "--oe_structural_end_iter", "8000",
            "--oe_structural_weight", "0.1",
            "--oe_structure_gate_mode", "stable_evidence",
            "--oe_stable_quantile", "0.8",
            "--oe_depth_cache_dir", str(plan.cache_dir),
            "--oe_preserve_baseline_densification_stats",
        ])
    if plan.smoke:
        command.append("--ns30k_smoke")
    return command


def format_logged_command(plan: RunPlan, python: str = sys.executable) -> str:
    """Human-readable command plus the launcher's complete log destination."""
    return f"{shlex.join(build_train_command(plan, python))}  [stdout+stderr -> {plan.train_log}]"


def compare_smoke_initial_state_audits(off_path: Path, on_path: Path) -> dict:
    """Recognize a paired initialization only when all frozen identities match."""
    try:
        off = json.loads(Path(off_path).read_text(encoding="utf-8"))
        on = json.loads(Path(on_path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PreflightError(f"cannot read Smoke initialization audits: {exc}") from exc
    required_equal = (
        "scene", "initialization_seed", "manifest_sha256", "pdc_sha256",
        "sampling_table_file_sha256", "initial_gaussian_count",
        "initial_gaussian_model_sha256", "train_py_sha256",
    )
    tags_valid = (
        off.get("arm") == "OFF" and on.get("arm") == "ON"
        and off.get("SMOKE_ONLY") is True and on.get("SMOKE_ONLY") is True
        and off.get("NOT_FORMAL_RESULT") is True
        and on.get("NOT_FORMAL_RESULT") is True
    )
    matching = tags_valid and all(off.get(key) == on.get(key) for key in required_equal)
    return {
        "status": "VALID_INITIALIZATION_PAIR" if matching else "INVALID_INITIALIZATION_PAIR",
        "eligible_as_same_initialization_pair": matching,
        "off_initial_gaussian_model_sha256": off.get("initial_gaussian_model_sha256"),
        "on_initial_gaussian_model_sha256": on.get("initial_gaussian_model_sha256"),
        "compared_fields": list(required_equal),
    }


def _validate_frozen_data_root(data_root: Path) -> None:
    if data_root.resolve() != DATA_ROOT.resolve():
        raise PreflightError(
            f"data root is frozen to {DATA_ROOT.resolve()}, got {data_root.resolve()}"
        )


def _validate_output_target(plan: RunPlan) -> None:
    if plan.output_dir.exists():
        raise PreflightError(
            f"REFUSING to reuse existing output directory: {plan.output_dir}"
        )
    if not plan.output_dir.parent.is_dir():
        raise PreflightError(f"output parent does not exist: {plan.output_dir.parent}")


def _validate_critical_code(plan: RunPlan) -> dict[str, str]:
    return {
        relative: require_hash(plan.root / relative, expected, f"critical code {relative}")
        for relative, expected in CRITICAL_CODE_SHA256.items()
    }


def _validate_manifest_and_colmap(plan: RunPlan) -> tuple[dict, list]:
    require_hash(plan.manifest, plan.spec.manifest_sha256, "view Manifest")
    records = read_and_validate_colmap_scene(plan.data_root / plan.scene)
    manifest = load_view_manifest(
        plan.manifest,
        current_image_names=[record.image_name for record in records],
        expected_dataset="LLFF",
        expected_scene=plan.scene,
        expected_view_count=3,
    )
    points = plan.data_root / plan.scene / "sparse" / "0" / "points3D.bin"
    if not points.is_file() or points.stat().st_size == 0:
        raise PreflightError(f"missing or empty COLMAP points file: {points}")
    return manifest, records


def _validate_npy_depth(path: Path, expected_shape: tuple[int, int]) -> None:
    """Strict, dependency-free validation of one ordinary floating-point NPY."""
    with path.open("rb") as handle:
        if handle.read(6) != b"\x93NUMPY":
            raise PreflightError(f"invalid NPY magic: {path}")
        major, minor = struct.unpack("BB", handle.read(2))
        if (major, minor) == (1, 0):
            header_length = struct.unpack("<H", handle.read(2))[0]
            encoding = "latin1"
        elif major in (2, 3):
            header_length = struct.unpack("<I", handle.read(4))[0]
            encoding = "utf-8" if major == 3 else "latin1"
        else:
            raise PreflightError(f"unsupported NPY version {major}.{minor}: {path}")
        try:
            header = ast.literal_eval(handle.read(header_length).decode(encoding).strip())
        except (UnicodeError, SyntaxError, ValueError) as exc:
            raise PreflightError(f"invalid NPY header: {path}: {exc}") from exc
        if not isinstance(header, dict):
            raise PreflightError(f"invalid NPY header object: {path}")
        if bool(header.get("fortran_order")):
            raise PreflightError(f"Fortran-order depth arrays are forbidden: {path}")
        shape = header.get("shape")
        if tuple(shape) != expected_shape:
            raise PreflightError(
                f"depth shape mismatch: {path}; expected={expected_shape}; actual={shape}"
            )
        descriptor = header.get("descr")
        formats = {
            "<f4": ("<f", 4), ">f4": (">f", 4), "=f4": ("=f", 4), "|f4": ("=f", 4),
            "<f8": ("<d", 8), ">f8": (">d", 8), "=f8": ("=d", 8), "|f8": ("=d", 8),
        }
        if descriptor not in formats:
            raise PreflightError(f"depth NPY must be float32/float64, got {descriptor!r}: {path}")
        unpack_format, item_size = formats[descriptor]
        expected_bytes = math.prod(expected_shape) * item_size
        payload = handle.read()
        if len(payload) != expected_bytes:
            raise PreflightError(
                f"depth payload size mismatch: {path}; expected={expected_bytes}; actual={len(payload)}"
            )
        if any(not math.isfinite(value[0]) for value in struct.iter_unpack(unpack_format, payload)):
            raise PreflightError(f"depth array contains non-finite values: {path}")


def _validate_cache(plan: RunPlan, manifest: Mapping) -> dict:
    """Validate the production cache contract without importing CUDA libraries."""
    metadata_path = plan.cache_dir / "oe_depth_cache.json"
    require_hash(metadata_path, plan.spec.cache_metadata_sha256, "MiDaS metadata")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PreflightError(f"cannot parse MiDaS metadata {metadata_path}: {exc}") from exc
    records = metadata.get("records")
    selected = manifest.get("selected_train_images")
    if (
        metadata.get("cache_version") != "oe-depth-cache-v2"
        or metadata.get("model_type") != "DPT_Hybrid"
        or metadata.get("preprocessing") != EXPECTED_MIDAS_PREPROCESSING
        or metadata.get("source_view_manifest_sha256") != plan.spec.manifest_sha256
        or metadata.get("selected_train_images") != selected
        or Path(str(metadata.get("source_images_dir", ""))).resolve()
        != (plan.data_root / plan.scene / "images").resolve()
        or not isinstance(records, list)
        or len(records) != 3
    ):
        raise PreflightError("MiDaS cache header is not bound to the frozen scene/Manifest")
    if not isinstance(records, list) or len(records) != 3:
        raise PreflightError("MiDaS metadata must contain exactly three records")
    by_name = {}
    for record in records:
        try:
            image_name = str(record["image_name"])
            source_image = str(record["source_image"])
            height, width = int(record["height"]), int(record["width"])
            depth_name = str(record["depth_file"])
            source_hash = str(record["source_sha256"])
            depth_hash = str(record["depth_sha256"])
        except (KeyError, TypeError, ValueError) as exc:
            raise PreflightError(f"malformed MiDaS cache record: {record!r}") from exc
        if image_name != source_image or image_name in by_name or image_name not in selected:
            raise PreflightError(f"MiDaS record is not one unique frozen training image: {image_name!r}")
        if Path(depth_name).name != depth_name or depth_name in ("", ".", ".."):
            raise PreflightError(f"unsafe MiDaS depth filename: {depth_name!r}")
        source_path = plan.data_root / plan.scene / "images" / source_image
        depth_path = plan.cache_dir / depth_name
        require_hash(source_path, source_hash, f"MiDaS source image {source_image}")
        require_hash(depth_path, depth_hash, f"MiDaS depth {depth_name}")
        _validate_npy_depth(depth_path, (height, width))
        by_name[image_name] = record
    if list(by_name) != list(selected):
        raise PreflightError("MiDaS record set/order differs from the frozen training images")
    return metadata


def preflight(plan: RunPlan, *, python: str = sys.executable) -> dict:
    """Run the complete read-only preflight without creating experiment files."""
    _validate_frozen_data_root(plan.data_root)
    _validate_output_target(plan)
    code_hashes = _validate_critical_code(plan)
    table = NS30KSamplingTable(
        plan.sampling_table,
        plan.spec.sampling_sha256,
        expected_scene=plan.scene,
    )
    require_hash(plan.manifest, plan.spec.manifest_sha256, "view Manifest")
    require_hash(plan.pdc, plan.spec.pdc_sha256, "frozen PDC")
    table.verify_assets(plan.manifest, plan.pdc)
    manifest, colmap_records = _validate_manifest_and_colmap(plan)
    cache = _validate_cache(plan, manifest)
    return {
        "status": "PASS",
        "mode": "SMOKE_PREFLIGHT_ONLY" if plan.smoke else "PREFLIGHT_ONLY",
        "SMOKE_ONLY": bool(plan.smoke),
        "NOT_FORMAL_RESULT": bool(plan.smoke),
        "scene": plan.scene,
        "arm": plan.arm,
        "data_root": str(plan.data_root),
        "output_dir": str(plan.output_dir),
        "train_log": str(plan.train_log),
        "training_command": build_train_command(plan, python),
        "logged_command": format_logged_command(plan, python),
        "save_iterations": list(SAVE_NODES),
        "checkpoint_iterations": list(SAVE_NODES),
        "sampling_table_sha256": table.file_sha256,
        "sampling_content_sha256": table.table_id,
        "manifest_sha256": sha256_file(plan.manifest),
        "pdc_sha256": sha256_file(plan.pdc),
        "midas_metadata_sha256": sha256_file(plan.cache_dir / "oe_depth_cache.json"),
        "train_images": list(manifest["selected_train_images"]),
        "test_images": list(manifest["test_images"]),
        "colmap_image_count": len(colmap_records),
        "midas_record_count": len(cache["records"]),
        "critical_code_sha256": code_hashes,
        "training_time_evaluation": False,
        "binocular_consistency": False,
        "checkpoint_resume": False,
        "smoke_stop_after_normal_iteration": 10 if plan.smoke else None,
        "configured_iterations_remain": 30_000,
    }


def _smoke_trace_row_count(plan: RunPlan) -> int:
    trace_path = plan.output_dir / "ns30k_sampling_trace.csv"
    if not trace_path.is_file():
        return 0
    with trace_path.open("r", encoding="utf-8", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def _finalize_smoke_result(
    plan: RunPlan,
    process_exit_code: int | None,
    *,
    process_started: bool = True,
) -> bool:
    result_path = plan.output_dir / "ns30k_smoke_result.json"
    initial_path = plan.output_dir / "ns30k_initial_state_audit.json"
    normal_training_report_loaded = False
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if not isinstance(result, dict):
            raise ValueError("result is not an object")
        normal_training_report_loaded = (
            result.get("audit_version") == "ns30k-smoke-result-v1"
            and isinstance(result.get("actual_completed_training_iterations"), int)
            and isinstance(result.get("sampling_records_consumed"), int)
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
        initial = {}
        if initial_path.is_file():
            try:
                initial = json.loads(initial_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                initial = {}
        result = {
            "audit_version": "ns30k-smoke-result-v1",
            "status": "SMOKE_FAIL",
            "SMOKE_ONLY": True,
            "NOT_FORMAL_RESULT": True,
            "scene": plan.scene,
            "arm": "ON" if plan.arm == "B0+I1" else "OFF",
            "actual_completed_training_iterations": None,
            "completed_iteration_status": "UNVERIFIED",
            "sampling_records_consumed": None,
            "initial_gaussian_model_sha256": initial.get(
                "initial_gaussian_model_sha256"
            ),
            "limitation": (
                "The 10-iteration Smoke validates only the initial training path; "
                "it is not a complete 30k result and does not validate densification."
            ),
        }
    sampling_trace_rows_written = _smoke_trace_row_count(plan)
    if normal_training_report_loaded:
        result["completed_iteration_status"] = "VERIFIED_BY_TRAINING_REPORT"
    else:
        result["actual_completed_training_iterations"] = None
        result["completed_iteration_status"] = "UNVERIFIED"
        result["sampling_records_consumed"] = None
    result["sampling_trace_rows_written"] = sampling_trace_rows_written
    formal_artifacts = [
        str(path) for path in (
            list(plan.output_dir.glob("chkpnt*.pth"))
            + list((plan.output_dir / "point_cloud").glob("iteration_*"))
        )
    ]
    audit_ok = (
        process_started
        and process_exit_code == 0
        and normal_training_report_loaded
        and result.get("status") == "SMOKE_PASS"
        and result.get("SMOKE_ONLY") is True
        and result.get("NOT_FORMAL_RESULT") is True
        and result.get("actual_completed_training_iterations") == 10
        and result.get("sampling_records_consumed") == 10
        and result.get("sampling_trace_rows_written") == 10
        and result.get("completed_iteration_status") == "VERIFIED_BY_TRAINING_REPORT"
        and result.get("camera_sequence_matches_sampling_table") is True
        and result.get("nonfinite_loss_seen") is False
        and result.get("i1_structural_loss_observation_matches_arm") is True
        and bool(result.get("initial_gaussian_model_sha256"))
        and not formal_artifacts
    )
    result["process_started"] = bool(process_started)
    result["process_exit_code"] = (
        int(process_exit_code) if process_exit_code is not None else None
    )
    result["formal_checkpoint_or_ply_artifacts"] = formal_artifacts
    result["status"] = "SMOKE_PASS" if audit_ok else "SMOKE_FAIL"
    other_arm = "B0+I1" if plan.arm == "B0" else "B0"
    other_audit = (
        plan.root / "output" /
        f"NS30K_SMOKE_v1_{plan.scene}_{other_arm}_seed0" /
        "ns30k_initial_state_audit.json"
    )
    current_audit = plan.output_dir / "ns30k_initial_state_audit.json"
    result["initialization_pairing"] = {
        "status": "UNVERIFIED_OTHER_ARM_NOT_AVAILABLE",
        "eligible_as_same_initialization_pair": False,
    }
    if current_audit.is_file() and other_audit.is_file():
        off_audit, on_audit = (
            (current_audit, other_audit) if plan.arm == "B0" else (other_audit, current_audit)
        )
        try:
            result["initialization_pairing"] = compare_smoke_initial_state_audits(
                off_audit, on_audit
            )
        except PreflightError as exc:
            result["initialization_pairing"] = {
                "status": "INVALID_INITIALIZATION_PAIR",
                "eligible_as_same_initialization_pair": False,
                "error": str(exc),
            }
    temporary = result_path.with_suffix(result_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, result_path)
    return audit_ok


def execute(
    plan: RunPlan,
    *,
    python: str = sys.executable,
    formal_authorized: bool = False,
    smoke_authorized: bool = False,
) -> int:
    """Execute one authorized run after a fresh full preflight.

    Authorization is checked here as well as at the CLI boundary.  This
    function deliberately performs preflight again immediately before
    atomically claiming the output path.
    """
    if plan.smoke:
        if not smoke_authorized or formal_authorized:
            raise PreflightError("Smoke execution lacks its independent authorization")
    elif not formal_authorized or smoke_authorized:
        raise PreflightError("formal execution lacks its independent authorization")
    evidence = preflight(plan, python=python)
    plan.output_dir.mkdir(exist_ok=False)
    command = list(evidence["training_command"])
    env = os.environ.copy()
    env.update({"PYTHONUNBUFFERED": "1", "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"})
    if plan.smoke:
        env[SMOKE_AUTH_ENV] = "1"
    with plan.train_log.open("x", encoding="utf-8", newline="\n") as log:
        log.write(f"COMMAND: {shlex.join(command)}\n")
        log.write(f"LOG_PATH: {plan.train_log}\n")
        log.write("PREFLIGHT: PASS\n")
        if plan.smoke:
            log.write("SMOKE_ONLY: true\nNOT_FORMAL_RESULT: true\n")
        log.write("\n")
        log.flush()
        try:
            process = subprocess.Popen(
                command,
                cwd=str(plan.root),
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
            return_code = process.wait()
        except BaseException as exc:
            log.write(f"\nLAUNCH_ERROR: {type(exc).__name__}: {exc}\n")
            log.write("EXIT_CODE: NOT_STARTED\n")
            if plan.smoke:
                try:
                    _finalize_smoke_result(
                        plan, None, process_started=False
                    )
                    log.write("STATUS: SMOKE_FAIL\n")
                except BaseException as report_exc:
                    log.write(
                        "SMOKE_REPORT_ERROR: {}: {}\n".format(
                            type(report_exc).__name__, report_exc
                        )
                    )
            else:
                log.write("STATUS: FAILED\n")
            log.flush()
            raise
        effective_code = return_code
        if plan.smoke and not _finalize_smoke_result(
            plan, return_code, process_started=True
        ):
            effective_code = return_code if return_code != 0 else 3
        log.write(f"\nEXIT_CODE: {return_code}\n")
        if plan.smoke:
            log.write("STATUS: SMOKE_PASS\n" if effective_code == 0 else "STATUS: SMOKE_FAIL\n")
        else:
            log.write("STATUS: SUCCESS\n" if effective_code == 0 else "STATUS: FAILED\n")
    return effective_code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True, choices=tuple(SCENES))
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument(
        "--smoke", action="store_true",
        help="prepare the isolated Fortress 10-iteration SMOKE_ONLY path",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight", action="store_true", help="explicit read-only preflight (also the default)")
    mode.add_argument("--execute", action="store_true", help="start exactly one training process after all checks")
    mode.add_argument(
        "--execute-smoke", action="store_true",
        help="start exactly one isolated Smoke process after all checks",
    )
    parser.add_argument(
        EXECUTION_AUTHORIZATION_FLAG,
        action="store_true",
        help="independent explicit authorization required together with --execute",
    )
    parser.add_argument(
        SMOKE_EXECUTION_AUTHORIZATION_FLAG,
        action="store_true",
        help="independent Smoke authorization required with --execute-smoke",
    )
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--python", default=sys.executable)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.execute and not args.authorize_execution:
        print(
            f"FAIL: --execute is refused without {EXECUTION_AUTHORIZATION_FLAG}",
            file=sys.stderr,
        )
        return 2
    if args.authorize_execution and not args.execute:
        print(
            f"FAIL: {EXECUTION_AUTHORIZATION_FLAG} is only valid together with --execute",
            file=sys.stderr,
        )
        return 2
    if args.execute and args.smoke:
        print("FAIL: Smoke must use --execute-smoke, never formal --execute", file=sys.stderr)
        return 2
    if args.execute_smoke and (not args.smoke or not args.authorize_smoke_execution):
        print(
            f"FAIL: --execute-smoke requires --smoke and {SMOKE_EXECUTION_AUTHORIZATION_FLAG}",
            file=sys.stderr,
        )
        return 2
    if args.authorize_smoke_execution and not args.execute_smoke:
        print(
            f"FAIL: {SMOKE_EXECUTION_AUTHORIZATION_FLAG} is only valid with --execute-smoke",
            file=sys.stderr,
        )
        return 2
    try:
        plan = build_plan(
            args.scene, args.arm, data_root=args.data_root, smoke=args.smoke
        )
        if args.execute:
            result = preflight(plan, python=args.python)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            print("STATUS: EXECUTION_AUTHORIZED; starting exactly one training process")
            return execute(plan, python=args.python, formal_authorized=True)
        if args.execute_smoke:
            result = preflight(plan, python=args.python)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            print("STATUS: SMOKE_EXECUTION_AUTHORIZED; starting one SMOKE_ONLY process")
            return execute(plan, python=args.python, smoke_authorized=True)
        result = preflight(plan, python=args.python)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print("PASS: read-only preflight complete; no training process started")
        return 0
    except (PreflightError, NS30KContractError, ViewProtocolError, OSError, ValueError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
