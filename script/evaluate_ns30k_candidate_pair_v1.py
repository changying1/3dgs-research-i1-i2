"""Isolated NS30K-v1 B0/End8k evaluator for extended LLFF scenes.

The default mode is a read-only preflight. Rendering and metric computation
require both ``--execute`` and ``--authorize-execution``. Each scene has an
independent report subtree, so an unavailable scene never blocks another scene.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import uuid


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from script import evaluate_ns30k_i1_schedule_ablation_v1 as shared  # noqa: E402
from script import run_ns30k_candidate_live_retry as training  # noqa: E402


PROTOCOL_VERSION = "NS30K-v1-B0-End8k-Evaluation-v1"
SUPPORTED_SCENES = ("leaves", "orchids", "fern", "flower", "trex")
GROUPS = ("B0", "End8k")
GROUP_ARMS = {"B0": "B0", "End8k": "B0+I1"}
NODES = (5_000, 10_000, 20_000, 30_000)
METRICS = shared.METRICS
REPORT_ROOT = ROOT / "reports" / "ns30k_v1_pair_evaluation"
EXECUTION_AUTHORIZATION_FLAG = "--authorize-execution"

# Leaves and Orchids were finalized before ``asset_provenance`` was added to
# the training completion audit.  Their immutable reports identify that
# historical contract with the complete evaluation-code hash set below; the
# evaluator hash alone is deliberately not enough to select the legacy path.
LEGACY_REPORT_SCENES = frozenset(("leaves", "orchids"))
LEGACY_EVALUATION_CODE_SHA256 = {
    "metrics.py": "c614e742c13eafa25c1d6d78a86d09b865cb35f1cf9340e404552362f85e660f",
    "render.py": "a5f1a0e754d5d2eb95a79cbfed3bbcec1445da08ab07f865bbe0746578ca223a",
    "script/evaluate_ns30k_candidate_pair_v1.py":
        "63b4514d66901d03649a044e2364038fbb5b58b2d2c136f4d9596549c8cac776",
    "script/evaluate_ns30k_i1_schedule_ablation_v1.py":
        "4049d98533e3b7965111c1443c297169c39b7205e71d517324697e7785e3eb31",
}
LEGACY_TRAINING_AUDIT_FIELDS = frozenset((
    "arm", "artifacts", "colmap_image_count", "completed_iteration",
    "critical_code_sha256", "i1_runtime", "manifest_sha256",
    "midas_metadata_sha256", "midas_record_count", "mode",
    "nonfinite_loss_seen", "pdc_sha256", "protocol", "same_scene_pairing",
    "sampling_content_sha256", "sampling_records", "sampling_table_sha256",
    "scene", "status", "trace_matches_frozen_table",
))
CURRENT_TRAINING_AUDIT_FIELDS = (
    LEGACY_TRAINING_AUDIT_FIELDS | {"asset_provenance"}
)


class EvaluationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise EvaluationError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def report_dir(scene: str) -> Path:
    require(scene in SUPPORTED_SCENES, f"unsupported evaluation scene: {scene}")
    return REPORT_ROOT / scene


def plan_for(scene: str, group: str) -> training.RunPlan:
    require(scene in SUPPORTED_SCENES, f"unsupported evaluation scene: {scene}")
    require(group in GROUPS, f"unsupported evaluation group: {group}")
    return training.build_plan(scene, GROUP_ARMS[group])


def model_path(scene: str, group: str) -> Path:
    return plan_for(scene, group).output_dir


def _manifest(scene: str) -> dict:
    plan = plan_for(scene, "B0")
    try:
        document = json.loads(plan.manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"cannot read frozen manifest: {plan.manifest}: {exc}") from exc
    tests = document.get("test_images")
    require(
        document.get("scene") == scene and isinstance(tests, list) and tests,
        f"invalid frozen test split for {scene}",
    )
    require(len(tests) == len(set(tests)), f"duplicate frozen test images for {scene}")
    return document


def test_count(scene: str) -> int:
    return len(_manifest(scene)["test_images"])


@contextmanager
def scene_profile(scene: str):
    references = {
        (scene, group): model_path(scene, group).name
        for group in GROUPS
    }
    with shared.evaluation_profile(
        report=REPORT_ROOT,
        scenes=(scene,),
        references=references,
        test_counts={scene: test_count(scene)},
    ):
        yield


def render_log_path(scene: str, group: str, iteration: int) -> Path:
    return report_dir(scene) / group / f"render_{iteration}.log"


def stage_path(scene: str, group: str, iteration: int) -> Path:
    return REPORT_ROOT / scene / group / str(iteration)


def _render_output_audit(source: Path, scene: str) -> dict:
    renders, gt = shared.validate_rgb_gt(source, scene)
    return {
        "source": str(source),
        "renders_sha256": {
            name: sha256(renders / name) for name in shared.view_names(scene)
        },
        "gt_sha256": {
            name: sha256(gt / name) for name in shared.view_names(scene)
        },
    }


def _append_render_output_audit(path: Path, source: Path, scene: str) -> None:
    record = _render_output_audit(source, scene)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write("RENDER_OUTPUT_AUDIT: " + json.dumps(record, sort_keys=True) + "\n")


def _validate_render_log(path: Path, source: Path, scene: str) -> None:
    require(path.is_file() and path.stat().st_size > 0, f"missing complete render log: {path}")
    text = path.read_text(encoding="utf-8", errors="replace")
    completed = "EXIT_CODE: 0" in text or "REUSED_EXISTING_RENDER: true" in text
    require(completed, f"render log does not prove success or validated reuse: {path}")
    audit_lines = [
        line[len("RENDER_OUTPUT_AUDIT: "):]
        for line in text.splitlines()
        if line.startswith("RENDER_OUTPUT_AUDIT: ")
    ]
    require(len(audit_lines) == 1, f"render log must contain one output audit: {path}")
    try:
        recorded = json.loads(audit_lines[0])
    except json.JSONDecodeError as exc:
        raise EvaluationError(f"malformed render output audit: {path}") from exc
    current = _render_output_audit(source, scene)
    require(recorded == current, f"render output SHA256 mismatch: {path}")


def _write_render_reuse_log(path: Path, source: Path, scene: str) -> None:
    require(not path.exists(), f"refusing existing render log: {path}")
    shared.validate_rgb_gt(source, scene)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write("REUSED_EXISTING_RENDER: true\n")
        handle.write("RENDER_PROCESS_STARTED: false\n")
    _append_render_output_audit(path, source, scene)


def _render_command(scene: str, group: str, iteration: int, python: str) -> list[str]:
    plan = plan_for(scene, group)
    return [
        python, "-u", str(ROOT / "render.py"),
        "-s", str(training.DATA_ROOT / scene), "-m", str(plan.output_dir),
        "--eval", "--dataset_name", "LLFF", "--n_views", "3",
        "--resolution", "8", "--sh_degree", "1", "--dropout_factor", "0.2",
        "--view_manifest", str(plan.manifest), "--iteration", str(iteration),
        "--skip_train",
    ]


def render_one(scene: str, group: str, iteration: int, python: str) -> Path:
    source = model_path(scene, group) / "test" / f"ours_{iteration}"
    log = render_log_path(scene, group, iteration)
    if source.exists():
        shared.validate_rgb_gt(source, scene)
        if not log.exists():
            _write_render_reuse_log(log, source, scene)
        _validate_render_log(log, source, scene)
        return source
    require(not log.exists(), f"orphan render log would be overwritten: {log}")
    shared.run_logged(_render_command(scene, group, iteration, python), log)
    shared.validate_rgb_gt(source, scene)
    _append_render_output_audit(log, source, scene)
    _validate_render_log(log, source, scene)
    return source


def _validate_stage(scene: str, group: str, iteration: int, source: Path) -> tuple[dict, dict]:
    stage = stage_path(scene, group, iteration)
    aggregate, details = shared.validate_completed_stage(stage, source, scene, iteration)
    metrics_log = stage / "metrics.log"
    require(metrics_log.is_file() and metrics_log.stat().st_size > 0, f"missing metrics log: {metrics_log}")
    text = metrics_log.read_text(encoding="utf-8", errors="replace")
    require("EXIT_CODE: 0" in text, f"metrics log does not prove success: {metrics_log}")
    _validate_render_log(render_log_path(scene, group, iteration), source, scene)
    return aggregate, details


def evaluate_one(
    scene: str, group: str, iteration: int, source: Path, python: str
) -> tuple[dict, dict]:
    stage = stage_path(scene, group, iteration)
    existed = stage.exists()
    data = shared.evaluate_one(scene, group, iteration, source, python)
    require(stage.exists(), f"evaluation stage was not finalized: {stage}")
    if existed:
        return _validate_stage(scene, group, iteration, source)
    return _validate_stage(scene, group, iteration, source)


def _metrics_rows(scene: str, data: dict) -> list[dict]:
    rows = []
    for iteration in NODES:
        for group in GROUPS:
            aggregate, _ = data[(group, iteration)]
            rows.append({
                "scene": scene, "group": group, "iteration": iteration,
                **{metric: float(aggregate[metric]) for metric in METRICS},
            })
    return rows


def _delta_rows(scene: str, data: dict) -> list[dict]:
    rows = []

    # End8k relative to B0 at the same node.
    for iteration in NODES:
        _, values = data[("End8k", iteration)]
        _, baseline = data[("B0", iteration)]
        for metric in METRICS:
            for view in shared.view_names(scene):
                value = float(values[metric][view])
                reference = float(baseline[metric][view])
                rows.append({
                    "scene": scene,
                    "comparison_type": "End8k_vs_B0",
                    "group": "End8k", "reference_group": "B0",
                    "iteration": iteration, "reference_iteration": iteration,
                    "view": view, "metric": metric,
                    "value": value, "reference_value": reference,
                    "delta": value - reference,
                })

    # Within-arm change from 20k to 30k.
    for group in GROUPS:
        _, values = data[(group, 30_000)]
        _, baseline = data[(group, 20_000)]
        for metric in METRICS:
            for view in shared.view_names(scene):
                value = float(values[metric][view])
                reference = float(baseline[metric][view])
                rows.append({
                    "scene": scene,
                    "comparison_type": "30k_vs_20k",
                    "group": group, "reference_group": group,
                    "iteration": 30_000, "reference_iteration": 20_000,
                    "view": view, "metric": metric,
                    "value": value, "reference_value": reference,
                    "delta": value - reference,
                })
    return rows


def expected_report_rows(scene: str) -> tuple[int, int]:
    count = test_count(scene)
    return len(GROUPS) * len(NODES), (len(NODES) + len(GROUPS)) * len(METRICS) * count


def _read_csv(path: Path) -> list[dict[str, str]]:
    require(path.is_file() and path.stat().st_size > 0, f"missing report table: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _validate_recorded_preflight_contract(
    scene: str, record: dict, expected_preflight: dict
) -> str:
    """Validate either the exact current contract or the one frozen legacy contract."""
    recorded_hashes = record.get("evaluation_code_sha256")
    current_hashes = expected_preflight.get("evaluation_code_sha256")
    if recorded_hashes == current_hashes:
        contract = "current"
    elif (
        scene in LEGACY_REPORT_SCENES
        and recorded_hashes == LEGACY_EVALUATION_CODE_SHA256
    ):
        contract = "legacy-leaves-orchids-v1"
    else:
        raise EvaluationError("protocol record evaluation_code_sha256 mismatch")

    for key in (
        "models", "manifest_sha256", "test_images", "test_view_count",
        "render_configuration",
    ):
        require(record.get(key) == expected_preflight.get(key), f"protocol record {key} mismatch")

    recorded_audits = record.get("training_completion_audits")
    current_audits = expected_preflight.get("training_completion_audits")
    require(isinstance(recorded_audits, dict), "protocol record training_completion_audits mismatch")
    require(isinstance(current_audits, dict), "current training_completion_audits are invalid")
    require(set(recorded_audits) == set(GROUPS), "protocol record training_completion_audits mismatch")
    require(set(current_audits) == set(GROUPS), "current training_completion_audits are invalid")

    for group in GROUPS:
        recorded = recorded_audits[group]
        current = current_audits[group]
        require(isinstance(recorded, dict), f"protocol record {group} training audit is invalid")
        require(isinstance(current, dict), f"current {group} training audit is invalid")
        if contract == "current":
            require(
                set(current) == CURRENT_TRAINING_AUDIT_FIELDS,
                f"current {group} training audit schema mismatch",
            )
            require(recorded == current, "protocol record training_completion_audits mismatch")
        else:
            require(
                set(current) == CURRENT_TRAINING_AUDIT_FIELDS,
                f"current {group} training audit schema mismatch",
            )
            require(
                set(recorded) == LEGACY_TRAINING_AUDIT_FIELDS,
                f"legacy {group} training audit schema mismatch",
            )
            projected = {key: current[key] for key in LEGACY_TRAINING_AUDIT_FIELDS}
            require(recorded == projected, "protocol record training_completion_audits mismatch")
    return contract


def _validate_summary(scene: str, data: dict, expected_preflight: dict | None = None) -> dict:
    summary = report_dir(scene) / "summary"
    metrics_path = summary / "metrics_table.csv"
    delta_path = summary / "per_view_deltas.csv"
    record_path = summary / "protocol_record.json"
    require(record_path.is_file() and record_path.stat().st_size > 0, f"missing protocol record: {record_path}")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    require(record.get("status") == "PASS", "protocol record status is not PASS")
    require(record.get("protocol") == PROTOCOL_VERSION, "protocol record version mismatch")
    require(record.get("scene") == scene, "protocol record scene mismatch")
    require(record.get("groups") == list(GROUPS), "protocol record group mismatch")
    require(record.get("nodes") == list(NODES), "protocol record node mismatch")
    if expected_preflight is not None:
        _validate_recorded_preflight_contract(scene, record, expected_preflight)
    hashes = record.get("summary_artifact_sha256")
    require(isinstance(hashes, dict), "protocol record lacks summary hashes")
    require(hashes.get("metrics_table.csv") == sha256(metrics_path), "metrics table SHA256 mismatch")
    require(hashes.get("per_view_deltas.csv") == sha256(delta_path), "delta table SHA256 mismatch")

    expected_metrics, expected_deltas = expected_report_rows(scene)
    metric_rows, delta_rows = _read_csv(metrics_path), _read_csv(delta_path)
    require(len(metric_rows) == expected_metrics, f"metrics table must contain {expected_metrics} rows")
    require(len(delta_rows) == expected_deltas, f"delta table must contain {expected_deltas} rows")
    require(
        {row["comparison_type"] for row in delta_rows} == {"End8k_vs_B0", "30k_vs_20k"},
        "delta table does not distinguish both required comparison types",
    )

    expected_metric_rows = _metrics_rows(scene, data)
    for actual, expected in zip(metric_rows, expected_metric_rows):
        require(actual["scene"] == expected["scene"], "metrics table scene/order mismatch")
        require(actual["group"] == expected["group"], "metrics table group/order mismatch")
        require(int(actual["iteration"]) == expected["iteration"], "metrics table iteration mismatch")
        for metric in METRICS:
            require(
                math.isclose(float(actual[metric]), expected[metric], rel_tol=0.0, abs_tol=1e-12),
                f"metrics table value mismatch: {metric}",
            )
    expected_delta_rows = _delta_rows(scene, data)
    string_fields = (
        "scene", "comparison_type", "group", "reference_group", "view", "metric",
    )
    integer_fields = ("iteration", "reference_iteration")
    numeric_fields = ("value", "reference_value", "delta")
    for actual, expected in zip(delta_rows, expected_delta_rows):
        for field in string_fields:
            require(actual[field] == str(expected[field]), f"delta table {field} mismatch")
        for field in integer_fields:
            require(int(actual[field]) == int(expected[field]), f"delta table {field} mismatch")
        for field in numeric_fields:
            require(
                math.isclose(float(actual[field]), float(expected[field]), rel_tol=0.0, abs_tol=1e-12),
                f"delta table {field} mismatch",
            )
    return record


def write_reports(scene: str, data: dict, preflight_record: dict) -> None:
    summary = report_dir(scene) / "summary"
    if summary.exists():
        _validate_summary(scene, data, preflight_record)
        return
    work = summary.with_name(f"summary.work_{uuid.uuid4().hex}")
    require(not work.exists(), f"refusing existing summary work directory: {work}")
    work.mkdir(parents=True, exist_ok=False)

    metrics_path = work / "metrics_table.csv"
    metric_fields = ("scene", "group", "iteration", *METRICS)
    with metrics_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=metric_fields)
        writer.writeheader()
        writer.writerows(_metrics_rows(scene, data))

    delta_path = work / "per_view_deltas.csv"
    delta_fields = (
        "scene", "comparison_type", "group", "reference_group",
        "iteration", "reference_iteration", "view", "metric",
        "value", "reference_value", "delta",
    )
    with delta_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=delta_fields)
        writer.writeheader()
        writer.writerows(_delta_rows(scene, data))

    record = dict(preflight_record)
    record.update({
        "status": "PASS",
        "mode": "COMPLETED_EVALUATION",
        "protocol": PROTOCOL_VERSION,
        "scene": scene,
        "groups": list(GROUPS),
        "nodes": list(NODES),
        "comparison_types": {
            "End8k_vs_B0": "End8k minus B0 at each common node",
            "30k_vs_20k": "same-arm 30000 minus 20000",
        },
        "metrics_table": str(summary / "metrics_table.csv"),
        "per_view_deltas": str(summary / "per_view_deltas.csv"),
        "summary_artifact_sha256": {
            "metrics_table.csv": sha256(metrics_path),
            "per_view_deltas.csv": sha256(delta_path),
        },
        "render_logs": [
            str(render_log_path(scene, group, iteration))
            for iteration in NODES for group in GROUPS
        ],
        "metrics_logs": [
            str(stage_path(scene, group, iteration) / "metrics.log")
            for iteration in NODES for group in GROUPS
        ],
        "original_models_or_renders_overwritten": False,
    })
    with (work / "protocol_record.json").open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
        handle.write("\n")
    work.rename(summary)


def _evaluation_code_hashes() -> dict[str, str]:
    hashes = {}
    for relative, expected in shared.EVAL_CODE_SHA256.items():
        path = ROOT / relative
        actual = sha256(path)
        require(actual == expected, f"frozen evaluation code mismatch: {relative}")
        hashes[relative] = actual
    hashes["script/evaluate_ns30k_i1_schedule_ablation_v1.py"] = sha256(
        ROOT / "script/evaluate_ns30k_i1_schedule_ablation_v1.py"
    )
    hashes["script/evaluate_ns30k_candidate_pair_v1.py"] = sha256(Path(__file__))
    return hashes


def preflight(scene: str) -> dict:
    require(scene in SUPPORTED_SCENES, f"unsupported evaluation scene: {scene}")
    evaluation_hashes = _evaluation_code_hashes()
    manifest = _manifest(scene)
    count = len(manifest["test_images"])
    audits = {}
    models = {}
    existing_renders = []
    reusable_stages = []
    incomplete_work = []

    with scene_profile(scene):
        for group in GROUPS:
            plan = plan_for(scene, group)
            audit = training.verify_completed(plan)
            require(audit.get("status") == "PASS", f"training completion audit failed: {scene}/{group}")
            require(audit.get("sampling_records") == 30_000, f"incomplete sampling trace: {scene}/{group}")
            require(audit.get("trace_matches_frozen_table") is True, f"trace/table mismatch: {scene}/{group}")
            require(
                audit.get("same_scene_pairing", {}).get("traces_identical") is True,
                f"B0/End8k trace pairing is not proven: {scene}/{group}",
            )
            if group == "B0":
                require(audit.get("i1_runtime", {}).get("configured") is False, "B0 enabled I1")
            else:
                runtime = audit.get("i1_runtime", {})
                require(runtime.get("activation_count") == 8000, "End8k activation count mismatch")
                require(runtime.get("post_8000_activation_seen") is False, "End8k remained active after 8000")
            audits[group] = audit
            models[group] = str(plan.output_dir)

            for iteration in NODES:
                source = plan.output_dir / "test" / f"ours_{iteration}"
                log = render_log_path(scene, group, iteration)
                stage = stage_path(scene, group, iteration)
                if source.exists():
                    shared.validate_rgb_gt(source, scene)
                    existing_renders.append(str(source))
                if log.exists():
                    require(source.exists(), f"orphan render log without render output: {log}")
                    _validate_render_log(log, source, scene)
                if stage.exists():
                    require(source.exists(), f"completed stage lacks source render: {stage}")
                    _validate_stage(scene, group, iteration, source)
                    reusable_stages.append(str(stage))

        scene_report = report_dir(scene)
        if scene_report.is_dir():
            incomplete_work = [
                str(path) for path in scene_report.rglob("*.work_*") if path.is_dir()
            ]

    return {
        "status": "PASS",
        "mode": "PREFLIGHT_ONLY",
        "protocol": PROTOCOL_VERSION,
        "scene": scene,
        "groups": list(GROUPS),
        "nodes": list(NODES),
        "report": str(report_dir(scene)),
        "models": models,
        "manifest": str(plan_for(scene, "B0").manifest),
        "manifest_sha256": sha256(plan_for(scene, "B0").manifest),
        "test_images": list(manifest["test_images"]),
        "test_view_count": count,
        "training_completion_audits": audits,
        "evaluation_code_sha256": evaluation_hashes,
        "render_configuration": {
            "resolution": 8, "sh_degree": 1, "dropout_factor": 0.2,
            "dataset_name": "LLFF", "n_views": 3,
        },
        "existing_validated_renders": existing_renders,
        "existing_completed_stages_reusable": reusable_stages,
        "incomplete_work_directories_ignored_not_overwritten": incomplete_work,
        "other_supported_scene_required": False,
        "original_three_scene_reports_touched": False,
    }


def verify_completed_evaluation(scene: str) -> dict:
    evidence = preflight(scene)
    data = {}
    with scene_profile(scene):
        for iteration in NODES:
            for group in GROUPS:
                source = model_path(scene, group) / "test" / f"ours_{iteration}"
                shared.validate_rgb_gt(source, scene)
                data[(group, iteration)] = _validate_stage(scene, group, iteration, source)
        record = _validate_summary(scene, data, evidence)
    return {
        "status": "PASS",
        "mode": "VERIFY_COMPLETED_READ_ONLY",
        "protocol": PROTOCOL_VERSION,
        "scene": scene,
        "report": str(report_dir(scene)),
        "metrics_rows": expected_report_rows(scene)[0],
        "per_view_delta_rows": expected_report_rows(scene)[1],
        "comparison_types": record["comparison_types"],
        "training_completion_audits": evidence["training_completion_audits"],
    }


def execute(scene: str, *, python: str, authorized: bool) -> dict:
    require(authorized, f"evaluation requires {EXECUTION_AUTHORIZATION_FLAG}")
    evidence = preflight(scene)
    report_dir(scene).mkdir(parents=True, exist_ok=True)
    data = {}
    with scene_profile(scene):
        for iteration in NODES:
            for group in GROUPS:
                source = render_one(scene, group, iteration, python)
                data[(group, iteration)] = evaluate_one(
                    scene, group, iteration, source, python
                )
        write_reports(scene, data, evidence)
    return verify_completed_evaluation(scene)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True, choices=SUPPORTED_SCENES)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--verify-completed", action="store_true")
    parser.add_argument(EXECUTION_AUTHORIZATION_FLAG, action="store_true")
    parser.add_argument("--python", default=sys.executable)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.execute != args.authorize_execution:
        print(
            f"FAIL: --execute and {EXECUTION_AUTHORIZATION_FLAG} must be supplied together",
            file=sys.stderr,
        )
        return 2
    try:
        if args.verify_completed:
            result = verify_completed_evaluation(args.scene)
        elif args.execute:
            result = execute(args.scene, python=args.python, authorized=True)
        else:
            result = preflight(args.scene)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if not args.execute and not args.verify_completed:
            print("PASS: read-only evaluation preflight; no render or metrics process started")
        return 0
    except (
        EvaluationError, shared.EvaluationError, training.PreflightError,
        training.NS30KContractError, OSError, ValueError,
    ) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
