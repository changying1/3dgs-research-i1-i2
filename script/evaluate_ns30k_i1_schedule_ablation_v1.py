"""Isolated four-arm evaluation for NS30K-I1-Schedule-v1.

Default operation is read-only preflight.  Evaluation requires --execute and
--authorize-execution and refuses any existing report directory.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import uuid


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from script import run_ns30k_i1_schedule_ablation_v1 as schedule  # noqa: E402


REPORT = ROOT / "reports" / "ns30k_i1_schedule_ablation_v1"
NODES = (5_000, 10_000, 20_000, 30_000)
SCENES = ("fortress", "horns")
GROUPS = ("B0", "End8k", "Strong30k", "Weak30k")
METRICS = ("PSNR", "SSIM", "LPIPS")
TEST_COUNTS = dict(schedule.TEST_COUNTS)
EVAL_CODE_SHA256 = {
    "render.py": "a5f1a0e754d5d2eb95a79cbfed3bbcec1445da08ab07f865bbe0746578ca223a",
    "metrics.py": "c614e742c13eafa25c1d6d78a86d09b865cb35f1cf9340e404552362f85e660f",
}
FROZEN_REFERENCE_MODEL_DIRS = {
    ("fortress", "B0"): "NS30K_v1_fortress_B0_seed0_retry1",
    ("fortress", "End8k"): "NS30K_v1_fortress_B0+I1_seed0",
    ("horns", "B0"): "NS30K_v1_horns_B0_seed0",
    ("horns", "End8k"): "NS30K_v1_horns_B0+I1_seed0",
}


class EvaluationError(RuntimeError):
    pass


@contextmanager
def evaluation_profile(
    *, report: Path, scenes: tuple[str, ...], references: dict,
    test_counts: dict[str, int] | None = None,
):
    """Temporarily select an isolated evaluation cohort.

    The Room entry point uses this profile while the original Fortress/Horns
    defaults and report directory remain unchanged for all other invocations.
    """
    global REPORT, SCENES, FROZEN_REFERENCE_MODEL_DIRS, TEST_COUNTS
    previous = REPORT, SCENES, FROZEN_REFERENCE_MODEL_DIRS, TEST_COUNTS
    REPORT = Path(report)
    SCENES = tuple(scenes)
    FROZEN_REFERENCE_MODEL_DIRS = dict(references)
    if test_counts is not None:
        TEST_COUNTS = dict(test_counts)
    try:
        yield
    finally:
        REPORT, SCENES, FROZEN_REFERENCE_MODEL_DIRS, TEST_COUNTS = previous


def expected_report_rows() -> tuple[int, int]:
    metrics_rows = len(SCENES) * len(NODES) * len(GROUPS)
    delta_rows = (
        len(NODES)
        * 3  # End8k, Strong30k, Weak30k
        * 2  # B0 and End8k references
        * len(METRICS)
        * sum(TEST_COUNTS[scene] for scene in SCENES)
    )
    return metrics_rows, delta_rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise EvaluationError(message)


def model_path(scene: str, group: str) -> Path:
    if group in ("B0", "End8k"):
        key = (scene, group)
        require(key in FROZEN_REFERENCE_MODEL_DIRS, f"unsupported frozen reference model: {key}")
        return ROOT / "output" / FROZEN_REFERENCE_MODEL_DIRS[key]
    require(scene in SCENES and group in ("Strong30k", "Weak30k"), f"unsupported evaluation model: {scene}/{group}")
    return schedule.build_plan(scene, group).output_dir


def view_names(scene: str) -> tuple[str, ...]:
    return tuple(f"{index:05d}.png" for index in range(TEST_COUNTS[scene]))


def validate_rgb_gt(folder: Path, scene: str) -> tuple[Path, Path]:
    renders, gt = folder / "renders", folder / "gt"
    expected = set(view_names(scene))
    require(renders.is_dir() and gt.is_dir(), f"missing RGB/GT directory: {folder}")
    actual_rgb = {path.name for path in renders.glob("*.png") if path.stem.isdigit()}
    actual_gt = {path.name for path in gt.glob("*.png") if path.stem.isdigit()}
    require(actual_rgb == expected and actual_gt == expected, f"{scene} requires exactly {len(expected)} paired real RGB/GT images: {folder}")
    for name in expected:
        require((renders / name).stat().st_size > 0 and (gt / name).stat().st_size > 0, f"empty RGB/GT file: {name}")
    return renders, gt


def preflight() -> dict:
    hashes = {}
    for relative, expected in EVAL_CODE_SHA256.items():
        path = ROOT / relative
        actual = sha256(path)
        require(actual == expected, f"frozen evaluation code mismatch: {relative}")
        hashes[relative] = actual
    models = {}
    reusable_stages = []
    incomplete_work_directories = []
    for scene in SCENES:
        for group in GROUPS:
            model = model_path(scene, group)
            require(model.is_dir(), f"missing completed model: {model}")
            for iteration in NODES:
                require((model / f"chkpnt{iteration}.pth").is_file(), f"missing checkpoint: {model}/{iteration}")
                require((model / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply").is_file(), f"missing PLY: {model}/{iteration}")
            if group in ("Strong30k", "Weak30k"):
                schedule.verify_completed(schedule.build_plan(scene, group))
            else:
                # Frozen old groups must already have renders; evaluation never writes into them.
                for iteration in NODES:
                    validate_rgb_gt(model / "test" / f"ours_{iteration}", scene)
            models[f"{scene}/{group}"] = str(model)
            if REPORT.is_dir():
                for iteration in NODES:
                    stage = REPORT / scene / group / str(iteration)
                    source = model / "test" / f"ours_{iteration}"
                    if stage.exists():
                        validate_completed_stage(stage, source, scene, iteration)
                        reusable_stages.append(str(stage))
    if REPORT.is_dir():
        incomplete_work_directories = [
            str(path) for path in REPORT.rglob("*.work_*") if path.is_dir()
        ]
    return {
        "status": "PASS", "mode": "PREFLIGHT_ONLY",
        "protocol": schedule.PROTOCOL_VERSION,
        "report": str(REPORT), "nodes": list(NODES), "groups": list(GROUPS),
        "test_counts": {scene: TEST_COUNTS[scene] for scene in SCENES},
        "models": models,
        "evaluation_code_sha256": hashes,
        "old_groups_read_only": True,
        "existing_completed_stages_reusable": reusable_stages,
        "incomplete_work_directories_ignored_not_overwritten": incomplete_work_directories,
    }


def run_logged(command: list[str], log_path: Path) -> None:
    require(not log_path.exists(), f"refusing existing log: {log_path}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("x", encoding="utf-8", newline="") as log:
        log.write("COMMAND: " + subprocess.list2cmdline(command) + "\n\n")
        log.flush()
        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
        log.write(f"\nEXIT_CODE: {result.returncode}\n")
    require(result.returncode == 0, f"subprocess failed; see {log_path}")


def render_new(scene: str, group: str, iteration: int, python: str) -> Path:
    model = model_path(scene, group)
    source = model / "test" / f"ours_{iteration}"
    if source.exists():
        validate_rgb_gt(source, scene)
        return source
    require(group in ("Strong30k", "Weak30k"), "old B0/End8k renders are read-only and must pre-exist")
    plan = schedule.build_plan(scene, group)
    command = [
        python, "-u", str(ROOT / "render.py"),
        "-s", str(schedule.DATA_ROOT / scene), "-m", str(model),
        "--eval", "--dataset_name", "LLFF", "--n_views", "3",
        "--resolution", "8", "--sh_degree", "1", "--dropout_factor", "0.2",
        "--view_manifest", str(plan.manifest), "--iteration", str(iteration), "--skip_train",
    ]
    run_logged(command, model / f"render_{iteration}.log")
    validate_rgb_gt(source, scene)
    return source


def load_metrics(stage: Path, scene: str, iteration: int) -> tuple[dict, dict]:
    target = stage / "test" / f"ours_{iteration}"
    validate_rgb_gt(target, scene)
    results_path, per_view_path = stage / "results.json", stage / "per_view.json"
    require(results_path.is_file() and results_path.stat().st_size > 0, f"missing results.json: {stage}")
    require(per_view_path.is_file() and per_view_path.stat().st_size > 0, f"missing per_view.json: {stage}")
    results = json.loads(results_path.read_text(encoding="utf-8-sig"))
    per_view = json.loads(per_view_path.read_text(encoding="utf-8-sig"))
    key = f"ours_{iteration}"
    require(set(results) == {key} and set(per_view) == {key}, f"invalid metric method key: {stage}")
    aggregate, details = results[key], per_view[key]
    require(set(aggregate) == set(METRICS) and set(details) == set(METRICS), f"incomplete metrics: {stage}")
    expected = set(view_names(scene))
    for metric in METRICS:
        require(math.isfinite(float(aggregate[metric])), f"non-finite aggregate metric: {stage}/{metric}")
        require(set(details[metric]) == expected, f"auxiliary/missing images in per-view metric: {stage}/{metric}")
        values = [float(details[metric][name]) for name in view_names(scene)]
        require(all(math.isfinite(value) for value in values), f"non-finite per-view metric: {stage}/{metric}")
        require(abs(sum(values) / len(values) - float(aggregate[metric])) <= 1e-4, f"aggregate/per-view mismatch: {stage}/{metric}")
    return aggregate, details


def validate_completed_stage(
    stage: Path, source: Path, scene: str, iteration: int
) -> tuple[dict, dict]:
    source_rgb, source_gt = validate_rgb_gt(source, scene)
    target = stage / "test" / f"ours_{iteration}"
    target_rgb, target_gt = validate_rgb_gt(target, scene)
    for name in view_names(scene):
        require(sha256(source_rgb / name) == sha256(target_rgb / name), f"staged RGB differs from source: {stage}/{name}")
        require(sha256(source_gt / name) == sha256(target_gt / name), f"staged GT differs from source: {stage}/{name}")
    return load_metrics(stage, scene, iteration)


def evaluate_one(scene: str, group: str, iteration: int, source: Path, python: str) -> tuple[dict, dict]:
    stage = REPORT / scene / group / str(iteration)
    if stage.exists():
        return validate_completed_stage(stage, source, scene, iteration)

    work = stage.with_name(f"{stage.name}.work_{uuid.uuid4().hex}")
    require(not work.exists(), f"refusing existing evaluation work directory: {work}")
    target = work / "test" / f"ours_{iteration}"
    source_rgb, source_gt = validate_rgb_gt(source, scene)
    for kind, original in (("renders", source_rgb), ("gt", source_gt)):
        destination = target / kind
        destination.mkdir(parents=True, exist_ok=False)
        for name in view_names(scene):
            shutil.copy2(original / name, destination / name)
            require(sha256(original / name) == sha256(destination / name), "isolated image copy mismatch")
    validate_rgb_gt(target, scene)
    run_logged(
        [python, "-u", str(ROOT / "metrics.py"), "--model_paths", str(work), "--dataset_name", "LLFF"],
        work / "metrics.log",
    )
    data = load_metrics(work, scene, iteration)
    work.rename(stage)
    return validate_completed_stage(stage, source, scene, iteration)


def write_reports(data: dict, preflight_record: dict) -> None:
    summary_dir = REPORT / "summary"
    expected_metrics_rows, expected_delta_rows = expected_report_rows()
    if summary_dir.exists():
        for name in ("metrics_table.csv", "per_view_deltas.csv", "protocol_record.json"):
            path = summary_dir / name
            require(path.is_file() and path.stat().st_size > 0, f"incomplete existing summary: {path}")
        record = json.loads((summary_dir / "protocol_record.json").read_text(encoding="utf-8"))
        require(record.get("status") == "PASS", "existing summary protocol status is not PASS")
        require(record.get("protocol") == schedule.PROTOCOL_VERSION, "existing summary protocol mismatch")
        with (summary_dir / "metrics_table.csv").open("r", encoding="utf-8", newline="") as handle:
            require(
                len(list(csv.DictReader(handle))) == expected_metrics_rows,
                f"existing metrics table must contain {expected_metrics_rows} rows",
            )
        with (summary_dir / "per_view_deltas.csv").open("r", encoding="utf-8", newline="") as handle:
            require(
                len(list(csv.DictReader(handle))) == expected_delta_rows,
                f"existing per-view delta table must contain {expected_delta_rows} rows",
            )
        return
    work = REPORT / f"summary.work_{uuid.uuid4().hex}"
    require(not work.exists(), f"refusing existing summary work directory: {work}")
    work.mkdir(parents=True, exist_ok=False)
    summary_path = work / "metrics_table.csv"
    with summary_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("scene", "group", "iteration", *METRICS))
        writer.writeheader()
        for scene in SCENES:
            for iteration in NODES:
                for group in GROUPS:
                    aggregate, _ = data[(scene, group, iteration)]
                    writer.writerow({"scene": scene, "group": group, "iteration": iteration, **aggregate})

    delta_path = work / "per_view_deltas.csv"
    fields = ("scene", "group", "reference", "iteration", "view", "metric", "value", "reference_value", "delta")
    with delta_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for scene in SCENES:
            for iteration in NODES:
                for group in ("End8k", "Strong30k", "Weak30k"):
                    for reference in ("B0", "End8k"):
                        _, values = data[(scene, group, iteration)]
                        _, baseline = data[(scene, reference, iteration)]
                        for metric in METRICS:
                            for view in view_names(scene):
                                value, ref = float(values[metric][view]), float(baseline[metric][view])
                                writer.writerow({
                                    "scene": scene, "group": group, "reference": reference,
                                    "iteration": iteration, "view": view, "metric": metric,
                                    "value": value, "reference_value": ref, "delta": value - ref,
                                })

    record = dict(preflight_record)
    record.update({
        "mode": "COMPLETED_EVALUATION", "status": "PASS",
        "metrics_table": str(summary_dir / "metrics_table.csv"),
        "per_view_deltas": str(summary_dir / "per_view_deltas.csv"),
        "rgb_gt_filter": "numeric PNG names only; exact scene count; renders and GT names identical",
        "original_model_results_overwritten": False,
    })
    with (work / "protocol_record.json").open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
        handle.write("\n")
    work.rename(summary_dir)


def execute(*, python: str, authorized: bool) -> None:
    require(authorized, "evaluation requires --authorize-execution")
    evidence = preflight()
    REPORT.mkdir(parents=True, exist_ok=True)
    data = {}
    for scene in SCENES:
        for iteration in NODES:
            for group in GROUPS:
                source = render_new(scene, group, iteration, python)
                data[(scene, group, iteration)] = evaluate_one(scene, group, iteration, source, python)
    write_reports(data, evidence)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--authorize-execution", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args(argv)
    if args.execute != args.authorize_execution:
        print("FAIL: --execute and --authorize-execution must be supplied together", file=sys.stderr)
        return 2
    try:
        evidence = preflight()
        print(json.dumps(evidence, indent=2))
        if args.execute:
            execute(python=args.python, authorized=True)
        else:
            print("PASS: read-only evaluation preflight; no render or metrics process started")
        return 0
    except (EvaluationError, schedule.PreflightError, OSError, ValueError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
