"""Independent 1--30000 random-input protocol for No-Stereo paired runs.

This module is CPU-only.  It freezes camera identities and independent seeds;
it never creates random tensors or claims elementwise Dropout-mask identity.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
from pathlib import Path


PROTOCOL_VERSION = "ns30k-sampling-v1"
START_ITERATION = 1
END_ITERATION = 30_000
SEED_PURPOSES = ("dropout", "densification", "background", "opacity_noise")
DROPOUT_PAIRING_LEVEL = "RULE_AND_SEED_ONLY; DYNAMIC_GAUSSIAN_SHAPES_MAY_DIFFER"
FROZEN_SAVE_NODES = (5_000, 10_000, 20_000, 30_000)


class NS30KContractError(RuntimeError):
    """Raised when an NS-30k frozen-input contract cannot be proven."""


def canonical_json_bytes(value) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: str | Path, description: str) -> dict:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise NS30KContractError(f"cannot read {description}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise NS30KContractError(f"{description} must contain a JSON object")
    return value


def _manifest_contract(manifest: dict, expected_scene: str | None = None) -> tuple[str, list[str], list[str]]:
    scene = str(manifest.get("scene", "")).lower()
    selected = manifest.get("selected_train_images")
    pdc_images = manifest.get("pdc_initialization_images")
    test_images = manifest.get("test_images")
    if manifest.get("dataset") != "LLFF" or manifest.get("view_count") != 3:
        raise NS30KContractError("NS-30k requires a frozen three-view LLFF manifest")
    if expected_scene is not None and scene != str(expected_scene).lower():
        raise NS30KContractError("manifest scene does not match the requested scene")
    if (
        not isinstance(selected, list)
        or len(selected) != 3
        or len(set(selected)) != 3
        or any(not isinstance(name, str) or not name for name in selected)
    ):
        raise NS30KContractError("manifest must contain exactly three unique training images")
    if pdc_images != selected:
        raise NS30KContractError("manifest PDC images do not exactly match training images")
    if not isinstance(test_images, list) or set(selected) & set(test_images):
        raise NS30KContractError("manifest leaks test images into the training/PDC set")
    camera_names = [Path(name).stem for name in selected]
    if any(not name for name in camera_names) or len(set(camera_names)) != 3:
        raise NS30KContractError("training image names do not map to three unique camera names")
    return scene, list(selected), camera_names


def _entry_digest(entry: dict) -> str:
    return sha256_bytes(canonical_json_bytes(entry))


def _content_digest(document: dict) -> str:
    unhashed = dict(document)
    unhashed.pop("content_sha256", None)
    return sha256_bytes(canonical_json_bytes(unhashed))


def generate_sampling_table(
    *,
    scene: str,
    seed: int,
    manifest_path: str | Path,
    pdc_path: str | Path,
    pdc_suffix: str,
) -> dict:
    """Generate a deterministic full-run table using one isolated Random instance."""
    manifest_path = Path(manifest_path)
    pdc_path = Path(pdc_path)
    manifest = _read_json(manifest_path, "view manifest")
    normalized_scene, selected_images, camera_names = _manifest_contract(
        manifest, expected_scene=scene
    )
    if not pdc_suffix:
        raise NS30KContractError("a frozen PDC suffix is required")
    if not pdc_path.is_file():
        raise NS30KContractError(f"frozen PDC file does not exist: {pdc_path}")

    rng = random.Random(int(seed))
    entries = []
    for iteration in range(START_ITERATION, END_ITERATION + 1):
        entry = {
            "iteration": iteration,
            "camera_image_name": camera_names[rng.randrange(len(camera_names))],
            "seeds": {
                purpose: rng.randrange(0, 2**63)
                for purpose in SEED_PURPOSES
            },
        }
        entry["entry_sha256"] = _entry_digest(entry)
        entries.append(entry)

    document = {
        "version": PROTOCOL_VERSION,
        "scene": normalized_scene,
        "seed": int(seed),
        "start_iteration": START_ITERATION,
        "end_iteration": END_ITERATION,
        "entry_count": len(entries),
        "manifest_sha256": sha256_file(manifest_path),
        "manifest_train_images": selected_images,
        "training_camera_image_names": camera_names,
        "pdc_suffix": str(pdc_suffix),
        "pdc_sha256": sha256_file(pdc_path),
        "seed_purposes": list(SEED_PURPOSES),
        "random_stream_policy": "isolated Python random.Random(seed); independent per-iteration seeds",
        "dropout_pairing_level": DROPOUT_PAIRING_LEVEL,
        "entries": entries,
    }
    document["content_sha256"] = _content_digest(document)
    validate_sampling_document(document)
    return document


def validate_sampling_document(document: dict, *, expected_scene: str | None = None) -> None:
    if not isinstance(document, dict) or document.get("version") != PROTOCOL_VERSION:
        raise NS30KContractError("unsupported NS-30k sampling protocol version")
    if expected_scene is not None and document.get("scene") != str(expected_scene).lower():
        raise NS30KContractError("NS-30k sampling scene mismatch")
    if (document.get("start_iteration"), document.get("end_iteration")) != (
        START_ITERATION,
        END_ITERATION,
    ):
        raise NS30KContractError("NS-30k iteration range mismatch")
    if document.get("seed_purposes") != list(SEED_PURPOSES):
        raise NS30KContractError("NS-30k seed-purpose contract mismatch")
    if document.get("dropout_pairing_level") != DROPOUT_PAIRING_LEVEL:
        raise NS30KContractError("NS-30k Dropout pairing claim is invalid")
    camera_names = document.get("training_camera_image_names")
    if (
        not isinstance(camera_names, list)
        or len(camera_names) != 3
        or len(set(camera_names)) != 3
    ):
        raise NS30KContractError("NS-30k camera population must contain three unique names")
    if not document.get("manifest_sha256") or not document.get("pdc_sha256"):
        raise NS30KContractError("NS-30k asset hashes are incomplete")
    if document.get("content_sha256") != _content_digest(document):
        raise NS30KContractError("NS-30k content SHA256 mismatch")

    entries = document.get("entries")
    expected_count = END_ITERATION - START_ITERATION + 1
    if not isinstance(entries, list) or len(entries) != expected_count:
        raise NS30KContractError("NS-30k table has missing or extra records")
    if document.get("entry_count") != expected_count:
        raise NS30KContractError("NS-30k entry_count mismatch")
    seen_iterations = set()
    for offset, entry in enumerate(entries):
        expected_iteration = START_ITERATION + offset
        if not isinstance(entry, dict) or entry.get("iteration") != expected_iteration:
            raise NS30KContractError("NS-30k records are missing, duplicated, or out of order")
        if expected_iteration in seen_iterations:
            raise NS30KContractError("NS-30k contains a duplicate iteration")
        seen_iterations.add(expected_iteration)
        if entry.get("camera_image_name") not in camera_names:
            raise NS30KContractError("NS-30k record contains a non-training camera")
        seeds = entry.get("seeds")
        if not isinstance(seeds, dict) or set(seeds) != set(SEED_PURPOSES):
            raise NS30KContractError("NS-30k record seed fields are invalid")
        if any(
            not isinstance(value, int) or isinstance(value, bool) or not 0 <= value < 2**63
            for value in seeds.values()
        ):
            raise NS30KContractError("NS-30k seed is outside the signed 64-bit range")
        unhashed = dict(entry)
        entry_hash = unhashed.pop("entry_sha256", None)
        if entry_hash != _entry_digest(unhashed):
            raise NS30KContractError(
                f"NS-30k record SHA256 mismatch at iteration {expected_iteration}"
            )


def write_sampling_table(path: str | Path, document: dict) -> str:
    validate_sampling_document(document)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(document)
    try:
        with path.open("xb") as handle:
            handle.write(payload)
    except FileExistsError as exc:
        raise NS30KContractError(f"refusing to overwrite sampling table: {path}") from exc
    return sha256_bytes(payload)


class NS30KSamplingTable:
    """Strict sequential consumer for one immutable NS-30k table file."""

    def __init__(
        self,
        path: str | Path,
        expected_sha256: str,
        *,
        expected_scene: str | None = None,
    ):
        self.path = Path(path)
        if not self.path.is_file():
            raise NS30KContractError(f"NS-30k sampling table does not exist: {self.path}")
        self.file_sha256 = sha256_file(self.path)
        if not expected_sha256 or self.file_sha256.lower() != str(expected_sha256).lower():
            raise NS30KContractError("NS-30k sampling file SHA256 mismatch")
        self.document = _read_json(self.path, "NS-30k sampling table")
        validate_sampling_document(self.document, expected_scene=expected_scene)
        self.table_id = self.document["content_sha256"]
        self._next_offset = 0

    def verify_assets(self, manifest_path: str | Path, pdc_path: str | Path) -> None:
        manifest_path = Path(manifest_path)
        pdc_path = Path(pdc_path)
        if not manifest_path.is_file() or sha256_file(manifest_path) != self.document["manifest_sha256"]:
            raise NS30KContractError("frozen Manifest SHA256 mismatch")
        if not pdc_path.is_file() or sha256_file(pdc_path) != self.document["pdc_sha256"]:
            raise NS30KContractError("frozen PDC SHA256 mismatch")
        manifest = _read_json(manifest_path, "view manifest")
        scene, selected, camera_names = _manifest_contract(
            manifest, expected_scene=self.document["scene"]
        )
        if scene != self.document["scene"]:
            raise NS30KContractError("frozen Manifest scene mismatch")
        if selected != self.document["manifest_train_images"]:
            raise NS30KContractError("frozen Manifest training images mismatch")
        if camera_names != self.document["training_camera_image_names"]:
            raise NS30KContractError("frozen Manifest camera-name mapping mismatch")

    def consume(self, iteration: int) -> dict:
        entries = self.document["entries"]
        if self._next_offset >= len(entries):
            raise NS30KContractError("NS-30k sampling table consumed beyond iteration 30000")
        entry = entries[self._next_offset]
        if int(iteration) != entry["iteration"]:
            raise NS30KContractError(
                f"NS-30k expected iteration {entry['iteration']}, got {iteration}"
            )
        self._next_offset += 1
        return {**entry, "seeds": dict(entry["seeds"])}

    @property
    def consumed_count(self) -> int:
        return self._next_offset

    def assert_complete(self) -> None:
        if self._next_offset != len(self.document["entries"]):
            raise NS30KContractError(
                f"NS-30k sampling table incomplete: consumed {self._next_offset} "
                f"of {len(self.document['entries'])}"
            )


class NS30KTraceLogger:
    """Write only scalar/identity provenance; never copy Gaussian tensors to CPU."""

    FIELDS = (
        "iteration",
        "camera_image_name",
        "actual_camera_image_name",
        "dropout_seed",
        "densification_seed",
        "background_seed",
        "opacity_noise_seed",
        "sampling_file_sha256",
        "sampling_content_sha256",
        "entry_sha256",
        "dropout_pairing_level",
    )

    def __init__(self, output_dir: str | Path):
        self.path = Path(output_dir) / "ns30k_sampling_trace.csv"
        if self.path.exists() and self.path.stat().st_size:
            raise NS30KContractError(f"refusing to overwrite NS-30k trace: {self.path}")
        with self.path.open("w", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=self.FIELDS).writeheader()

    def record(self, entry: dict, actual_camera_image_name: str, table: NS30KSamplingTable) -> None:
        actual = str(actual_camera_image_name)
        if actual != entry["camera_image_name"]:
            raise NS30KContractError(
                f"NS-30k camera mismatch: planned {entry['camera_image_name']}, observed {actual}"
            )
        seeds = entry["seeds"]
        row = {
            "iteration": entry["iteration"],
            "camera_image_name": entry["camera_image_name"],
            "actual_camera_image_name": actual,
            "dropout_seed": seeds["dropout"],
            "densification_seed": seeds["densification"],
            "background_seed": seeds["background"],
            "opacity_noise_seed": seeds["opacity_noise"],
            "sampling_file_sha256": table.file_sha256,
            "sampling_content_sha256": table.table_id,
            "entry_sha256": entry["entry_sha256"],
            "dropout_pairing_level": DROPOUT_PAIRING_LEVEL,
        }
        with self.path.open("a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=self.FIELDS).writerow(row)


def ns30k_pdc_path(dataset, scene: str) -> Path:
    suffix = str(getattr(dataset, "suffix", "") or "")
    dataset_name = str(getattr(dataset, "dataset_name", "") or "")
    return (
        Path("keypoints_to_3d")
        / f"{dataset_name}_{suffix}"
        / f"{str(scene).lower()}_keypoints_to_3d.ply"
    )


def validate_ns30k_output_directory(model_path: str | Path | None) -> None:
    """Allow a new/empty output or a launcher-created directory containing only train.log."""
    if not model_path:
        return
    output = Path(model_path)
    if not output.exists():
        return
    if not output.is_dir():
        raise NS30KContractError("NS-30k model_path exists but is not a directory")
    entries = list(output.iterdir())
    unexpected = [entry.name for entry in entries if entry.name != "train.log"]
    if unexpected:
        raise NS30KContractError(
            "NS-30k output directory contains historical experiment artifacts: "
            + ", ".join(sorted(unexpected))
        )
    train_logs = [entry for entry in entries if entry.name == "train.log"]
    if train_logs and not train_logs[0].is_file():
        raise NS30KContractError("NS-30k output train.log placeholder is not a file")


def _matches_float(value, expected: float) -> bool:
    try:
        return math.isclose(float(value), expected, rel_tol=0.0, abs_tol=1e-15)
    except (TypeError, ValueError):
        return False


def _save_nodes_are_frozen(save_iterations, checkpoint_iterations) -> bool:
    try:
        saves = [int(value) for value in save_iterations]
        checkpoints = [int(value) for value in checkpoint_iterations]
    except (TypeError, ValueError):
        return False
    accepted_saves = (
        list(FROZEN_SAVE_NODES),
        [*FROZEN_SAVE_NODES, END_ITERATION],
    )
    return saves in accepted_saves and checkpoints == list(FROZEN_SAVE_NODES)


def validate_ns30k_training_configuration(dataset, opt, args, table: NS30KSamplingTable) -> Path:
    """Fail before output creation if the formal NS-30k contract is incomplete."""
    scene = os.path.basename(os.path.normpath(str(dataset.source_path))).lower()
    failures = []

    def require(condition, message):
        if not condition:
            failures.append(message)

    require(int(opt.iterations) == END_ITERATION, "iterations must be exactly 30000")
    require(
        int(opt.position_lr_max_steps) == END_ITERATION,
        "position_lr_max_steps must be exactly 30000",
    )
    require(_matches_float(opt.position_lr_init, 0.00016), "position_lr_init must be 0.00016")
    require(
        _matches_float(opt.position_lr_final, 0.0000016),
        "position_lr_final must be 0.0000016",
    )
    require(_matches_float(opt.lambda_dssim, 0.2), "lambda_dssim must be 0.2")
    require(int(opt.densify_from_iter) == 1000, "densify_from_iter must be 1000")
    require(
        int(opt.densify_until_iter) == 15_000,
        "initial densify_until_iter must be 15000",
    )
    require(
        int(opt.densification_interval) == 100,
        "densification_interval must be 100",
    )
    require(
        _matches_float(opt.densify_grad_threshold, 0.0002),
        "densify_grad_threshold must be 0.0002",
    )
    require(opt.random_background is False, "random_background must be False")
    require(args.binocular_consistency is False, "--no-binocular-consistency is required")
    require(
        args.disable_training_time_evaluation is True,
        "--disable_training_time_evaluation is required",
    )
    require(dataset.dataset_name == "LLFF", "dataset_name must be LLFF")
    require(dataset.eval is True, "--eval is required")
    require(int(dataset.n_views) == 3, "n_views must be exactly 3")
    require(int(dataset.resolution) == 8, "resolution must be exactly 8")
    require(int(dataset.sh_degree) == 1, "sh_degree must be exactly 1")
    require(bool(dataset.view_manifest), "an explicit frozen view_manifest is required")
    require(bool(dataset.suffix), "an explicit frozen PDC suffix is required")
    require(not args.start_checkpoint, "historical checkpoint resume is forbidden")
    require(args.stop_after_checkpoint_iter is None, "early checkpoint stop is forbidden")
    require(
        getattr(dataset, "initial_opacity_arm", "control") == "control",
        "initial_opacity_arm must be control",
    )
    require(
        not getattr(dataset, "initial_opacity_scores", "")
        and not getattr(dataset, "initial_opacity_score_manifest", ""),
        "Control must not use opacity score inputs",
    )
    require(not args.i1_b1_weak_continuation, "I1 weak continuation is forbidden")
    require(not args.paired_sampling_tape, "NS-30k and legacy P7 sampling are mutually exclusive")
    require(args.ns30k_arm in ("OFF", "ON"), "ns30k_arm must be OFF or ON")
    require(_matches_float(args.dropout_factor, 0.2), "dropout_factor must be 0.2")
    require(_matches_float(args.sigma_noise, 0.0), "sigma_noise must be 0.0")
    require(args.opacity_decay is True, "opacity_decay must be True")
    require(
        _matches_float(args.opacity_decay_factor, 0.995),
        "opacity_decay_factor must be 0.995",
    )
    require(
        _save_nodes_are_frozen(args.save_iterations, args.checkpoint_iterations),
        "save/checkpoint nodes must be exactly 5000,10000,20000,30000; "
        "only one auto-appended duplicate final 30000 is allowed for saves",
    )
    require(scene == table.document["scene"], "dataset scene does not match sampling table")
    require(dataset.suffix == table.document["pdc_suffix"], "PDC suffix mismatch")

    if args.ns30k_arm == "ON":
        require(args.enable_oe_structural_loss is True, "ON arm must enable I1")
        require(
            bool(args.oe_depth_cache_dir),
            "ON arm must provide an explicit validated I1 depth-cache directory",
        )
        require(
            args.oe_preserve_baseline_densification_stats is True,
            "ON arm must preserve baseline densification statistics",
        )
        require(int(args.oe_structural_start_iter) == 1, "ON I1 start must be 1")
        require(int(args.oe_structural_end_iter) == 8000, "ON I1 end must be 8000")
        require(math.isclose(float(args.oe_structural_weight), 0.1), "ON I1 weight must be 0.1")
        require(args.oe_structure_gate_mode == "stable_evidence", "ON gate must be stable_evidence")
        require(math.isclose(float(args.oe_stable_quantile), 0.8), "ON quantile must be 0.8")
    elif args.ns30k_arm == "OFF":
        require(args.enable_oe_structural_loss is False, "OFF arm must not enable I1")
        require(
            args.oe_preserve_baseline_densification_stats is False,
            "OFF arm must not enable baseline densification-stat preservation",
        )

    if failures:
        raise NS30KContractError("invalid NS-30k configuration: " + "; ".join(failures))

    pdc_path = ns30k_pdc_path(dataset, scene)
    table.verify_assets(dataset.view_manifest, pdc_path)
    validate_ns30k_output_directory(getattr(dataset, "model_path", ""))
    return pdc_path


def dropout_pairing_level(_shape_a=None, _shape_b=None) -> str:
    """Dynamic Gaussian counts prohibit an unconditional elementwise claim."""
    return DROPOUT_PAIRING_LEVEL


def _main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Generate a frozen NS-30k sampling table")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--pdc", required=True)
    parser.add_argument("--pdc-suffix", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    document = generate_sampling_table(
        scene=args.scene,
        seed=args.seed,
        manifest_path=args.manifest,
        pdc_path=args.pdc,
        pdc_suffix=args.pdc_suffix,
    )
    file_sha256 = write_sampling_table(args.output, document)
    print(json.dumps({
        "output": str(Path(args.output)),
        "file_sha256": file_sha256,
        "content_sha256": document["content_sha256"],
        "entry_count": document["entry_count"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
