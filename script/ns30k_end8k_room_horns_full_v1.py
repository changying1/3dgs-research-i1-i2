# -*- coding: utf-8 -*-
"""Frozen NS30K End8k multi-scene paired training + four-node evaluation.

Usage in repository root (Windows PowerShell):
    python -u script/ns30k_end8k_room_horns_full_v1.py --preflight
    python -u script/ns30k_end8k_room_horns_full_v1.py --execute --authorize-execution

Or after all four training runs finished (e.g. after a separate batch):
    python -u script/ns30k_end8k_room_horns_full_v1.py --evaluate-only

Targets only Room/Horns, seed0, 3 views, fixed sampling and frozen PDC.
The established guarded per-run launcher is the *only* training entry point.
Nothing overwrites original training models, logs, renders, or original metrics.
All evaluations are isolated in a new report tree.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "script" / "run_ns30k_candidate_live_retry.py"
EXPECTED_RUNNER_HASH = "7f27627e0172575c366d2251e28908990a2d0f1ddeb08c887d4c35524687630f"
EVAL_CODE = {
    "render.py": "a5f1a0e754d5d2eb95a79cbfed3bbcec1445da08ab07f865bbe0746578ca223a",
    "metrics.py": "c614e742c13eafa25c1d6d78a86d09b865cb35f1cf9340e404552362f85e660f",
}
SCENES = {
    "room": {
        "manifest": "output/Room_3v_FrozenPDC_v1/view_manifest.json",
        "manifest_hash": "1f102eeed48762164e3c708acc48dfea6fefdd32ec19da2848e8b2d919c4e4dd",
        "pdc": "keypoints_to_3d/LLFF_Room_3v_FrozenPDC_v1/room_keypoints_to_3d.ply",
        "pdc_hash": "172d524b56b3749fc3e64985dcd058ac64630d94c8fa349240c78842e9e1938b",
        "tape": "reports/ns30k_frozen_sampling/room_seed0_v1.json",
        "tape_hash": "533518ad70725593aede8bf57186897db4d8bd94336efb442a1ed7219ae7a595",
    },
    "horns": {
        "manifest": "output/A3_A3_Horns_3v_Baseline_5k_FrozenPDC/view_manifest.json",
        "manifest_hash": "a43d544c48f4767369ab37cf9ff82106f1d64365cb0e03d129559d297108586b",
        "pdc": "keypoints_to_3d/LLFF_A3_A3_Horns_3v_Baseline_5k_FrozenPDC/horns_keypoints_to_3d.ply",
        "pdc_hash": "971e1836b1fe88cba1a34d9f42c553ad75e2444e9313a75611b7cd6d4d023c7d",
        "tape": "reports/ns30k_frozen_sampling/horns_seed0_v1.json",
        "tape_hash": "ec47679f320ce04b3be9afcb78f7a046042f4a501fa255ee3f737c18a508595e",
    },
}
ARMS = ("B0", "B0+I1")
ITERATIONS = (5000, 10000, 20000, 30000)
METRICS = ("PSNR", "SSIM", "LPIPS")
VIEW_NAMES = tuple(f"{i:05d}.png" for i in range(6))
TRACE_FIELDS = (
    "iteration", "camera_image_name", "actual_camera_image_name",
    "dropout_seed", "densification_seed", "background_seed", "opacity_noise_seed",
    "sampling_file_sha256", "sampling_content_sha256", "entry_sha256", "dropout_pairing_level",
)
REPORT = ROOT / "reports" / "ns30k_eval_v1" / "end8k_room_horns_4node_pair_v1"
DATA_ROOT = Path(r"D:\dataset\nerf_llff_data_colmap")


class AuditError(RuntimeError):
    pass


def require(condition: bool, description: str) -> None:
    if not condition:
        raise AuditError(description)


def log_step(text: str) -> None:
    print(f"\n[END8K CROSS-SCENE] {text}", flush=True)


def nonempty(path: Path) -> None:
    require(path.is_file() and path.stat().st_size > 0, f"Missing/empty: {path}")


def sha256(path: Path) -> str:
    nonempty(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def frozen_hash(path: Path, expected: str) -> None:
    actual = sha256(path)
    require(actual == expected, f"Frozen SHA256 mismatch: {path}\nexpected {expected}\nactual   {actual}")


def run_path(scene: str, arm: str) -> Path:
    return ROOT / "output" / f"NS30K_v1_{scene}_{arm}_seed0"


def subprocess_command(command: list[str], logfile: Path | None = None) -> None:
    """Stream stdout+stderr and optionally save to exclusively created local logfile."""
    log_step("RUN: " + subprocess.list2cmdline(command))
    output = None
    if logfile is not None:
        require(not logfile.exists(), f"Refusing to overwrite log: {logfile}")
        output = logfile.open("x", encoding="utf-8", newline="")
        output.write("COMMAND: " + subprocess.list2cmdline(command) + "\n")
        output.flush()
    proc = None
    try:
        proc = subprocess.Popen(
            command, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"},
            bufsize=0,
        )
        import codecs
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        assert proc.stdout is not None
        while True:
            chunk = proc.stdout.read(4096)
            if not chunk:
                break
            text = decoder.decode(chunk)
            if text:
                print(text, end="", flush=True)
                if output:
                    output.write(text)
                    output.flush()
        last = decoder.decode(b"", final=True)
        if last:
            print(last, end="", flush=True)
            if output:
                output.write(last)
        ret = proc.wait()
        if output:
            output.write(f"\nEXIT_CODE: {ret}\n")
        require(ret == 0, f"Process failed, exit={ret}, command={command}; log={logfile}")
    except BaseException:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        raise
    finally:
        if proc is not None and proc.stdout is not None:
            proc.stdout.close()
        if output:
            output.close()


def check_global_code() -> None:
    require(Path.cwd().resolve() == ROOT, f"Run from repository root: {ROOT}")
    frozen_hash(RUNNER, EXPECTED_RUNNER_HASH)
    for name, expected in EVAL_CODE.items():
        frozen_hash(ROOT / name, expected)
    for scene, spec in SCENES.items():
        frozen_hash(ROOT / spec["manifest"], spec["manifest_hash"])
        frozen_hash(ROOT / spec["pdc"], spec["pdc_hash"])
        frozen_hash(ROOT / spec["tape"], spec["tape_hash"])
        require((DATA_ROOT / scene).is_dir(), f"Missing dataset scene: {DATA_ROOT / scene}")


def run_read_only_preflights(*, require_fresh: bool) -> None:
    check_global_code()
    if require_fresh:
        require(not REPORT.exists(), f"Report already exists: {REPORT}")
    for scene in SCENES:
        for arm in ARMS:
            p = run_path(scene, arm)
            if require_fresh:
                require(not p.exists(), f"Training output already exists, refusing overwrite: {p}")
            cmd = [sys.executable, "-u", str(RUNNER), "--scene", scene,
                   "--arm", arm, "--preflight"]
            result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
            require(result.returncode == 0, f"Preflight failed for {scene}/{arm}:\n{result.stdout}\n{result.stderr}")
            require('"status": "PASS"' in result.stdout and
                    '"binocular_consistency": false' in result.stdout and
                    '"training_time_evaluation": false' in result.stdout,
                    f"Unexpected preflight result for {scene}/{arm}:\n{result.stdout}")
            require('"save_iterations": [' in result.stdout, f"Missing 4-point save schedule {scene}/{arm}")
            require('"output_dir": "' in result.stdout, f"Missing planned output for {scene}/{arm}")
            log_step(f"Preflight {scene}/{arm}: PASS")


def run_all_training() -> None:
    # Do all four *read-only* preflights before any GPU training runs.
    run_read_only_preflights(require_fresh=True)
    for scene in SCENES:
        for arm in ARMS:
            # Each guarded runner repeats preflight, refuses output overwrites,
            # writes a complete train.log, and streams stdout/stderr live.
            command = [sys.executable, "-u", str(RUNNER), "--scene", scene,
                       "--arm", arm, "--live-output", "--execute", "--authorize-execution"]
            subprocess_command(command)
            check_train_output(scene, arm)
            log_step(f"TRAINING COMPLETE: {scene}/{arm}")
        check_trace_pair(scene)


def check_train_output(scene: str, arm: str) -> None:
    run = run_path(scene, arm)
    require(run.is_dir(), f"Missing trained run: {run}")
    for iteration in ITERATIONS:
        nonempty(run / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply")
        nonempty(run / f"chkpnt{iteration}.pth")
    log = run / "train.log"
    nonempty(log)
    with log.open("rb") as stream:
        command_line = stream.readline().decode("utf-8", "replace")
        stream.seek(max(0, log.stat().st_size - 8192))
        tail = stream.read().decode("utf-8", "replace")
    for fragment in (
        "--no-binocular-consistency", "--ns30k_sampling_tape",
        "--iterations 30000", "--n_views 3", "--dropout_factor 0.2",
        "--opacity_decay_factor 0.995", "--disable_training_time_evaluation",
    ):
        require(fragment in command_line, f"Train command mismatch {scene}/{arm}: {fragment}")
    if arm == "B0":
        require("--ns30k_arm OFF" in command_line and "--enable_oe_structural_loss" not in command_line,
                f"B0 configuration mismatch: {scene}")
    else:
        for fragment in (
            "--ns30k_arm ON", "--enable_oe_structural_loss",
            "--oe_structural_start_iter 1", "--oe_structural_end_iter 8000",
            "--oe_structural_weight 0.1", "--oe_stable_quantile 0.8",
            "--oe_preserve_baseline_densification_stats",
        ):
            require(fragment in command_line, f"I1 End8k configuration mismatch {scene}: {fragment}")
        require('"actual_activation_count": 8000' in tail, f"Missing 8k I1 activation count: {scene}")
        require('"nonfinite_loss_seen": false' in tail, f"Nonfinite I1 loss observed: {scene}")
    require("EXIT_CODE: 0" in tail and "STATUS: SUCCESS" in tail, f"Train failed: {scene}/{arm}")
    nonempty(run / "ns30k_sampling_trace.csv")


def check_trace_pair(scene: str) -> None:
    traces = []
    try:
        for arm in ARMS:
            stream = (run_path(scene, arm) / "ns30k_sampling_trace.csv").open(
                "r", encoding="utf-8-sig", newline="")
            reader = csv.DictReader(stream)
            require(reader.fieldnames is not None and set(reader.fieldnames) == set(TRACE_FIELDS),
                    f"Invalid sampling trace columns for {scene}/{arm}")
            traces.append((stream, reader))
        for iteration in range(1, 30001):
            left = next(traces[0][1], None)
            right = next(traces[1][1], None)
            require(left is not None and right is not None,
                    f"Missing trace record {scene} iteration {iteration}")
            for row in (left, right):
                require(int(row["iteration"]) == iteration, f"Invalid trace iteration {scene}/{iteration}")
                require(row["camera_image_name"] == row["actual_camera_image_name"],
                        f"Actual camera mismatch {scene}/{iteration}")
                require(row["sampling_file_sha256"] == SCENES[scene]["tape_hash"],
                        f"Wrong sampling tape {scene}/{iteration}")
            for key in TRACE_FIELDS:
                require(left[key] == right[key], f"Pair not aligned {scene}/{iteration}/{key}")
        require(next(traces[0][1], None) is None and next(traces[1][1], None) is None,
                f"Trace exceeds 30000 rows: {scene}")
    finally:
        for stream, _ in traces:
            stream.close()
    log_step(f"PAIR TRACE PASS: {scene} all 30000 samples identical")


def method_dir(scene: str, arm: str, iteration: int) -> Path:
    return run_path(scene, arm) / "test" / f"ours_{iteration}"


def check_images(folder: Path) -> tuple[Path, Path]:
    renders, gt = folder / "renders", folder / "gt"
    require(renders.is_dir() and gt.is_dir(), f"Missing render/GT directory: {folder}")
    rgb_names = {p.name for p in renders.glob("*.png") if p.stem.isdigit()}
    gt_names = {p.name for p in gt.glob("*.png")}
    require(rgb_names == set(VIEW_NAMES) and gt_names == set(VIEW_NAMES),
            f"Expected six exactly paired RGB/GT images in {folder}, got RGB={sorted(rgb_names)} GT={sorted(gt_names)}")
    for name in VIEW_NAMES:
        nonempty(renders / name)
        nonempty(gt / name)
    return renders, gt


def render_one(scene: str, arm: str, iteration: int) -> Path:
    source = method_dir(scene, arm, iteration)
    run = run_path(scene, arm)
    if source.exists():
        check_images(source)
        log_step(f"REUSE existing validated render: {scene}/{arm}/{iteration}")
        return source
    log_path = run / f"render_{iteration}.log"
    require(not log_path.exists(), f"Render log exists with missing render output: {log_path}")
    command = [
        sys.executable, "-u", str(ROOT / "render.py"),
        "-s", str(DATA_ROOT / scene), "-m", str(run),
        "--eval", "--dataset_name", "LLFF", "--n_views", "3",
        "--resolution", "8", "--sh_degree", "1", "--dropout_factor", "0.2",
        "--view_manifest", str(ROOT / SCENES[scene]["manifest"]),
        "--iteration", str(iteration), "--skip_train",
    ]
    subprocess_command(command, log_path)
    check_images(source)
    return source


def isolated_eval_stage(scene: str, arm: str, iteration: int, source: Path) -> Path:
    stage = REPORT / scene / ("B0_I1" if arm == "B0+I1" else arm) / str(iteration)
    target = stage / "test" / f"ours_{iteration}"
    render_dir, gt_dir = check_images(source)
    for sub, original in (("renders", render_dir), ("gt", gt_dir)):
        new_dir = target / sub
        new_dir.mkdir(parents=True, exist_ok=False)
        for name in VIEW_NAMES:
            original_file = original / name
            destination = new_dir / name
            shutil.copy2(original_file, destination)
            require(sha256(original_file) == sha256(destination),
                    f"Image copy mismatch: {original_file}")
    check_images(target)
    return stage


def evaluate_one(scene: str, arm: str, iteration: int, source: Path) -> tuple[dict, dict]:
    stage = isolated_eval_stage(scene, arm, iteration, source)
    command = [sys.executable, "-u", str(ROOT / "metrics.py"),
               "--model_paths", str(stage), "--dataset_name", "LLFF"]
    subprocess_command(command, stage / "metrics.log")
    result_path, perview_path = stage / "results.json", stage / "per_view.json"
    nonempty(result_path)
    nonempty(perview_path)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    view = json.loads(perview_path.read_text(encoding="utf-8"))
    key = f"ours_{iteration}"
    require(set(result) == {key} and set(view) == {key},
            f"Wrong/missing metric method at {scene}/{arm}/{iteration}")
    agg, details = result[key], view[key]
    require(set(agg) == set(METRICS) and set(details) == set(METRICS),
            f"Metric keys incomplete at {scene}/{arm}/{iteration}")
    for metric in METRICS:
        require(isinstance(agg[metric], (int, float)) and math.isfinite(agg[metric]),
                f"Nonfinite mean {scene}/{arm}/{iteration}/{metric}")
        row = details[metric]
        require(set(row) == set(VIEW_NAMES),
                f"Six per-view records missing {scene}/{arm}/{iteration}/{metric}")
        require(all(isinstance(v, (int, float)) and math.isfinite(v) for v in row.values()),
                f"Nonfinite view metric {scene}/{arm}/{iteration}/{metric}")
        require(abs(agg[metric] - sum(row.values()) / 6) <= 1e-4,
                f"Mean != per-view mean {scene}/{arm}/{iteration}/{metric}")
    log_step(f"METRIC PASS {scene}/{arm}/{iteration}: " + " ".join(
        f"{metric}={agg[metric]:.6f}" for metric in METRICS))
    return agg, details


def produce_reports(results: dict) -> None:
    paired, perview = [], []
    for scene in SCENES:
        for iteration in ITERATIONS:
            b0, bv = results[(scene, "B0", iteration)]
            on, ov = results[(scene, "B0+I1", iteration)]
            row = {"scene": scene, "iteration": iteration}
            for metric in METRICS:
                change = on[metric] - b0[metric]
                row[f"B0_{metric}"] = b0[metric]
                row[f"I1_{metric}"] = on[metric]
                row[f"delta_{metric}"] = change
                row[f"improved_{metric}"] = change < 0 if metric == "LPIPS" else change > 0
                row[f"wins_{metric}_of_6"] = sum(
                    (ov[metric][name] < bv[metric][name]) if metric == "LPIPS"
                    else (ov[metric][name] > bv[metric][name]) for name in VIEW_NAMES)
            paired.append(row)
            for name in VIEW_NAMES:
                detail = {"scene": scene, "iteration": iteration, "view": name}
                for metric in METRICS:
                    detail[f"B0_{metric}"] = bv[metric][name]
                    detail[f"I1_{metric}"] = ov[metric][name]
                    detail[f"delta_{metric}"] = ov[metric][name] - bv[metric][name]
                perview.append(detail)
    for filename, rows in (("paired_summary.csv", paired), ("per_view_deltas.csv", perview)):
        path = REPORT / filename
        with path.open("x", encoding="utf-8-sig", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    summary = {
        "protocol": "NS30K_v1_no_stereo_seed0_3v_frozenPDC_I1End8k",
        "scene_set": list(SCENES),
        "iterations": list(ITERATIONS),
        "training_baseline": "Binocular3DGS + Gaussian Dropout (stereo consistency disabled in candidate)",
        "arms": {scene: {arm: str(run_path(scene, arm)) for arm in ARMS} for scene in SCENES},
        "I1_schedule": {"active_start": 1, "active_end": 8000, "weight": 0.1, "weak_continuation": False},
        "view_filenames": list(VIEW_NAMES),
        "expected_code_sha256": {"runner": EXPECTED_RUNNER_HASH, **EVAL_CODE},
        "scene_provenance": {scene: {k: v for k, v in spec.items()} for scene, spec in SCENES.items()},
        "note": "Room/Horns are two cross-scene checks; single seed0; not a multi-seed SOTA claim",
        "results": paired,
    }
    with (REPORT / "summary.json").open("x", encoding="utf-8") as output:
        json.dump(summary, output, ensure_ascii=False, indent=2)
    log_step("ALL ROOM + HORNS RESULTS")
    for row in paired:
        print(f"{row['scene']:7} {row['iteration']:5}: "
              f"PSNR {row['delta_PSNR']:+.4f}dB | "
              f"SSIM {row['delta_SSIM']:+.6f} | LPIPS {row['delta_LPIPS']:+.6f} "
              f"| view wins {row['wins_PSNR_of_6']}/{row['wins_SSIM_of_6']}/{row['wins_LPIPS_of_6']}", flush=True)
    print(f"\nRESULTS: {REPORT}\n"
          "FILES: paired_summary.csv, per_view_deltas.csv, summary.json\n"
          "NS30K END8K ROOM HORNS CROSS-SCENE: PASS", flush=True)


def evaluate_all() -> None:
    check_global_code()
    require(not REPORT.exists(), f"Report directory already exists, refusing overwrite: {REPORT}")
    for scene in SCENES:
        for arm in ARMS:
            check_train_output(scene, arm)
            for iteration in ITERATIONS:
                p = method_dir(scene, arm, iteration)
                if p.exists():
                    check_images(p)
                else:
                    require(not (run_path(scene, arm) / f"render_{iteration}.log").exists(),
                            f"Render log exists but model render folder missing: {scene}/{arm}/{iteration}")
        check_trace_pair(scene)
    REPORT.mkdir(parents=True, exist_ok=False)
    results = {}
    for scene in SCENES:
        reference_gt_hashes = None
        for iteration in ITERATIONS:
            for arm in ARMS:
                source = render_one(scene, arm, iteration)
                _, gt = check_images(source)
                hashes = tuple(sha256(gt / name) for name in VIEW_NAMES)
                if reference_gt_hashes is None:
                    reference_gt_hashes = hashes
                require(hashes == reference_gt_hashes,
                        f"GT images not identical for same scene: {scene}/{arm}/{iteration}")
                results[(scene, arm, iteration)] = evaluate_one(scene, arm, iteration, source)
    produce_reports(results)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true", help="read-only full 4-run preflight")
    mode.add_argument("--execute", action="store_true", help="run all 4 training jobs then 16 checkpoint evaluations")
    mode.add_argument("--evaluate-only", action="store_true", help="evaluate completed 4 runs (never retrain)")
    parser.add_argument("--authorize-execution", action="store_true", help="required to execute 4 training jobs")
    args = parser.parse_args()
    if args.execute != args.authorize_execution:
        raise AuditError("--execute and --authorize-execution must be used together")
    if args.preflight:
        run_read_only_preflights(require_fresh=True)
        print("NS30K END8K CROSS-SCENE PREFLIGHT: PASS (no GPU run)", flush=True)
    elif args.execute:
        run_all_training()
        evaluate_all()
    elif args.evaluate_only:
        evaluate_all()


if __name__ == "__main__":
    try:
        main()
    except (AuditError, OSError, ValueError, KeyError, RuntimeError) as exc:
        print(f"\nNS30K END8K CROSS-SCENE: FAIL\n{type(exc).__name__}: {exc}\n"
              "Do not delete existing runs or report. Inspect error before resuming.", file=sys.stderr)
        raise SystemExit(1)
