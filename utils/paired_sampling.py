"""Fail-closed sampling tape and audit logging for D2-P7-B paired continuations.

This module is deliberately independent of the I1 arm.  A tape contains only
pre-registered external random inputs; it never reads losses or evaluation
metrics.  CUDA tensors are created by callers, using the per-purpose seeds in
each consumed entry.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from pathlib import Path


TAPE_VERSION = "d2-p7b-sampling-tape-v1"
START_ITERATION = 20_001
END_ITERATION = 21_000
PURPOSES = (
    "dropout", "binocular_render", "opacity_noise", "background", "densification",
)


class SamplingContractError(RuntimeError):
    """Raised whenever a frozen pairing contract cannot be proven."""


def save_phase(*, paired_mode: bool, iteration: int, save_iterations) -> str | None:
    """Return the frozen PLY save phase without touching training mathematics."""
    if int(iteration) not in {int(value) for value in save_iterations}:
        return None
    return "post_optimizer_step" if paired_mode else "pre_optimizer_step"


def gaussian_model_state_sha256(gaussians) -> str:
    """Digest the Gaussian parameters serialized by PLY/checkpoint capture.

    This is a runtime audit primitive, not an input to training.  It deliberately
    excludes optimizer state because PLY does not contain optimizer tensors.
    """
    digest = hashlib.sha256()
    digest.update(str(int(gaussians.active_sh_degree)).encode("ascii"))
    for name in ("_xyz", "_features_dc", "_features_rest", "_opacity", "_scaling", "_rotation"):
        tensor = getattr(gaussians, name).detach().contiguous().cpu()
        digest.update(name.encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(canonical_json_bytes(list(tensor.shape)))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def canonical_json_bytes(value) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _entry_digest(entry: dict) -> str:
    return sha256_bytes(canonical_json_bytes(entry))


def generate_sampling_tape(scene: str, camera_image_names, seed: int) -> dict:
    """Create the complete 20001--21000 tape without consulting model output."""
    names = [str(name) for name in camera_image_names]
    if not names or len(set(names)) != len(names):
        raise SamplingContractError("camera_image_names must be non-empty and unique")
    rng = random.Random(int(seed))
    entries = []
    for cursor, iteration in enumerate(range(START_ITERATION, END_ITERATION + 1)):
        camera_index = rng.randrange(len(names))
        entry = {
            "cursor": cursor,
            "iteration": iteration,
            "camera_id": f"train:{camera_index}",
            "camera_image_name": names[camera_index],
            "binocular_unit": rng.random(),
            "binocular_sign": -1 if rng.getrandbits(1) == 0 else 1,
            "seeds": {purpose: rng.randrange(0, 2**63 - 1) for purpose in PURPOSES},
        }
        entry["entry_sha256"] = _entry_digest(entry)
        entries.append(entry)
    body = {
        "version": TAPE_VERSION,
        "scene": str(scene).lower(),
        "seed": int(seed),
        "start_iteration": START_ITERATION,
        "end_iteration": END_ITERATION,
        "entry_count": len(entries),
        "camera_population": [
            {"camera_id": f"train:{index}", "camera_image_name": name}
            for index, name in enumerate(names)
        ],
        "random_stream_policy": "independent per-iteration per-purpose seeds",
        "dropout_pairing_level": "RULE_AND_SEED; ELEMENTWISE_ONLY_WHEN_SHAPES_MATCH",
        "entries": entries,
    }
    body["content_sha256"] = sha256_bytes(canonical_json_bytes(body))
    return body


def write_sampling_tape(path: Path | str, tape: dict) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(tape)
    path.write_bytes(payload)
    return sha256_bytes(payload)


def validate_tape_document(tape: dict, *, expected_scene: str | None = None) -> None:
    if not isinstance(tape, dict) or tape.get("version") != TAPE_VERSION:
        raise SamplingContractError("unsupported sampling tape version")
    if expected_scene is not None and tape.get("scene") != str(expected_scene).lower():
        raise SamplingContractError("sampling tape scene mismatch")
    if (tape.get("start_iteration"), tape.get("end_iteration")) != (START_ITERATION, END_ITERATION):
        raise SamplingContractError("sampling tape iteration range mismatch")
    entries = tape.get("entries")
    if not isinstance(entries, list) or len(entries) != END_ITERATION - START_ITERATION + 1:
        raise SamplingContractError("sampling tape has missing or extra entries")
    if tape.get("entry_count") != len(entries):
        raise SamplingContractError("sampling tape entry_count mismatch")
    content_hash = tape.get("content_sha256")
    unhashed = dict(tape)
    unhashed.pop("content_sha256", None)
    if content_hash != sha256_bytes(canonical_json_bytes(unhashed)):
        raise SamplingContractError("sampling tape content hash mismatch")
    seen = set()
    for cursor, entry in enumerate(entries):
        expected_iteration = START_ITERATION + cursor
        if entry.get("cursor") != cursor or entry.get("iteration") != expected_iteration:
            raise SamplingContractError("sampling tape is missing, duplicated, or out of order")
        key = (entry.get("cursor"), entry.get("iteration"))
        if key in seen:
            raise SamplingContractError("sampling tape contains duplicate entries")
        seen.add(key)
        unhashed_entry = dict(entry)
        digest = unhashed_entry.pop("entry_sha256", None)
        if digest != _entry_digest(unhashed_entry):
            raise SamplingContractError(f"sampling tape entry digest mismatch at {expected_iteration}")
        if not entry.get("camera_image_name") or not entry.get("camera_id"):
            raise SamplingContractError("sampling tape camera identity is incomplete")
        if entry.get("binocular_sign") not in (-1, 1):
            raise SamplingContractError("invalid binocular sign")
        unit = entry.get("binocular_unit")
        if not isinstance(unit, (int, float)) or not math.isfinite(unit) or not 0.0 <= unit < 1.0:
            raise SamplingContractError("invalid binocular unit variate")
        seeds = entry.get("seeds")
        if not isinstance(seeds, dict) or set(seeds) != set(PURPOSES):
            raise SamplingContractError("sampling tape seed purposes mismatch")
        if any(not isinstance(value, int) or value < 0 or value >= 2**63 for value in seeds.values()):
            raise SamplingContractError("sampling tape seed outside signed 64-bit range")


class SamplingTape:
    """Sequential, exactly-once consumer for one immutable tape file."""

    def __init__(self, path: Path | str, expected_sha256: str, *, expected_scene: str | None = None):
        self.path = Path(path)
        if not self.path.is_file():
            raise SamplingContractError(f"sampling tape does not exist: {self.path}")
        self.file_sha256 = sha256_file(self.path)
        if not expected_sha256 or self.file_sha256.lower() != str(expected_sha256).lower():
            raise SamplingContractError("sampling tape SHA256 mismatch")
        try:
            self.document = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise SamplingContractError(f"cannot read sampling tape: {exc}") from exc
        validate_tape_document(self.document, expected_scene=expected_scene)
        self.tape_id = self.document["content_sha256"]
        self._next_cursor = 0

    def consume(self, iteration: int) -> dict:
        if self._next_cursor >= len(self.document["entries"]):
            raise SamplingContractError("sampling tape consumed beyond its end")
        entry = self.document["entries"][self._next_cursor]
        if int(iteration) != entry["iteration"]:
            raise SamplingContractError(
                f"sampling tape expected iteration {entry['iteration']}, got {iteration}"
            )
        self._next_cursor += 1
        return dict(entry)

    @property
    def consumed_count(self) -> int:
        return self._next_cursor

    def assert_complete(self) -> None:
        if self._next_cursor != len(self.document["entries"]):
            raise SamplingContractError(
                f"sampling tape incomplete: consumed {self._next_cursor} of {len(self.document['entries'])}"
            )


def dropout_pairing_level(shape_a, shape_b) -> str:
    """Never claim elementwise identity for unequal dynamic Gaussian shapes."""
    return "ELEMENTWISE_ELIGIBLE" if tuple(shape_a) == tuple(shape_b) else "RULE_AND_SEED_ONLY"


def tensor_summary(tensor) -> dict:
    detached = tensor.detach().contiguous().cpu()
    raw = detached.numpy().tobytes()
    result = {"shape": list(detached.shape), "sha256": sha256_bytes(raw)}
    if str(detached.dtype) == "torch.bool":
        result["true_count"] = int(detached.sum().item())
        result["false_count"] = int(detached.numel() - detached.sum().item())
    return result


class PairedRunLogger:
    TRACE_FIELDS = (
        "iteration", "camera_id", "camera_uid_observed", "camera_image_name",
        "tape_id", "tape_file_sha256", "tape_cursor", "tape_entry_sha256",
        "dropout_seed", "dropout_input_shape", "dropout_input_sha256", "dropout_input_pairing_level",
        "dropout_mask_shape", "dropout_mask_sha256", "dropout_mask_true_count",
        "binocular_unit", "binocular_sign", "binocular_translation",
        "planned_fields", "observed_fields",
    )
    METRIC_FIELDS = (
        "iteration", "rgb_loss", "binocular_loss", "i1_raw_loss",
        "i1_applied_weight", "i1_weighted_loss", "total_loss", "gaussian_count",
        "optimizer_step_executed", "finite_status", "field_provenance",
    )
    EVENT_FIELDS = (
        "iteration", "gaussians_before", "clone_added", "split_parents_selected",
        "split_children_created", "split_parents_deleted", "opacity_prune_deleted",
        "size_prune_deleted", "prune_deleted_total", "gaussians_after", "field_provenance",
    )

    def __init__(self, output_dir: Path | str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.trace_path = self.output_dir / "sampling_trace.csv"
        self.metrics_path = self.output_dir / "training_step_metrics.csv"
        self.events_path = self.output_dir / "gaussian_events.csv"
        self.metric_count = 0
        self.optimizer_step_count = 0
        self.i1_weights = set()
        self.nonfinite_count = 0
        self._initialize(self.trace_path, self.TRACE_FIELDS)
        self._initialize(self.metrics_path, self.METRIC_FIELDS)
        self._initialize(self.events_path, self.EVENT_FIELDS)

    @staticmethod
    def _initialize(path: Path, fields) -> None:
        if path.exists() and path.stat().st_size:
            raise SamplingContractError(f"refusing to append to existing audit log: {path}")
        with path.open("w", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=fields).writeheader()

    @staticmethod
    def _append(path: Path, fields, row: dict) -> None:
        with path.open("a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore").writerow(row)

    def record_trace(self, row: dict) -> None:
        self._append(self.trace_path, self.TRACE_FIELDS, row)

    def record_metrics(self, row: dict) -> None:
        self._append(self.metrics_path, self.METRIC_FIELDS, row)
        self.metric_count += 1
        if row.get("optimizer_step_executed") is True:
            self.optimizer_step_count += 1
        try:
            self.i1_weights.add(float(row.get("i1_applied_weight", 0.0)))
        except (TypeError, ValueError):
            self.i1_weights.add("INVALID")
        if row.get("finite_status") != "FINITE":
            self.nonfinite_count += 1

    def metrics_summary(self) -> dict:
        weights = sorted(self.i1_weights, key=str)
        return {
            "metric_row_count": self.metric_count,
            "optimizer_step_count": self.optimizer_step_count,
            "observed_i1_weights": weights,
            "nonfinite_count": self.nonfinite_count,
        }

    def record_event(self, row: dict) -> None:
        payload = dict(row)
        payload["field_provenance"] = "observed_runtime"
        self._append(self.events_path, self.EVENT_FIELDS, payload)
