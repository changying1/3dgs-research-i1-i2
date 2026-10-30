# -*- coding: utf-8 -*-
"""One-shot, non-overwriting paired 4-checkpoint Fortress NS30k evaluation.

Run from the Binocular3DGS + Dropout repository root:
    python -u script/ns30k_fortress_batch_eval_v1.py

- No training or checkpoint modifications.
- Reuses existing complete 5000 render products, refuses incomplete products.
- Renders absent 10000/20000/30000 test products for B0 and B0+I1.
- Evaluates all eight checkpoints in isolated reports staging areas, so the
  model-root results.json / per_view.json are never overwritten.
- Checks frozen protocol, paired train sampling, RGB/GT content, per-view
  metric completeness, and exported summary and deltas.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path.cwd().resolve()
DATA_SOURCE = Path(r"D:\dataset\nerf_llff_data_colmap\fortress")
ARMS = {
    "B0": ROOT / "output" / "NS30K_v1_fortress_B0_seed0_retry1",
    "B0_I1": ROOT / "output" / "NS30K_v1_fortress_B0+I1_seed0",
}
ITERATIONS = (5000, 10000, 20000, 30000)
VIEWS = tuple(f"{i:05d}.png" for i in range(6))
REPORT = ROOT / "reports" / "ns30k_eval_v1" / "fortress_4node_pair_v1"
MANIFEST = ROOT / "output" / "Fortress_3v_FrozenPDC_v1" / "view_manifest.json"
PDC = ROOT / "keypoints_to_3d" / "LLFF_Fortress_3v_FrozenPDC_v1" / "fortress_keypoints_to_3d.ply"
SAMPLING_TAPE = ROOT / "reports" / "ns30k_frozen_sampling" / "fortress_seed0_v1.json"
EXPECTED_HASHES = {
    "render.py": "a5f1a0e754d5d2eb95a79cbfed3bbcec1445da08ab07f865bbe0746578ca223a",
    "metrics.py": "c614e742c13eafa25c1d6d78a86d09b865cb35f1cf9340e404552362f85e660f",
}
EXPECTED_MANIFEST_HASH = "8d30c970680d26c94276b4aff04fcd2b8c1020fd3df4d51c44e454144f7402d2"
EXPECTED_PDC_HASH = "1b0d97e68e6065411314c9cddd3fca263ba76122a9ab37546ee44548d1e17944"
EXPECTED_TAPE_HASH = "716b4202b30a3263c54d8d3acf15327899fb8ff05ff78a5c70775844597b0234"
TRACE_FIELDS = (
    "iteration", "camera_image_name", "actual_camera_image_name",
    "dropout_seed", "densification_seed", "background_seed",
    "opacity_noise_seed", "sampling_file_sha256", "sampling_content_sha256",
    "entry_sha256", "dropout_pairing_level",
)
METRICS = ("PSNR", "SSIM", "LPIPS")


class AuditError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def nonempty(path: Path) -> None:
    require(path.is_file() and path.stat().st_size > 0, f"Missing/empty file: {path}")


def sha256(path: Path) -> str:
    nonempty(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_hash(path: Path, expected: str) -> None:
    actual = sha256(path)
    require(actual == expected, f"SHA256 mismatch: {path}\nexpected={expected}\nactual={actual}")


def print_step(message: str) -> None:
    print(f"\n[NS30K-EVAL] {message}", flush=True)


def saved_point_files(run: Path) -> None:
    for iteration in ITERATIONS:
        nonempty(run / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply")
        nonempty(run / f"chkpnt{iteration}.pth")


def check_train_log(arm: str, run: Path) -> None:
    log = run / "train.log"
    nonempty(log)
    with log.open("rb") as handle:
        start = handle.readline().decode("utf-8", "replace")
        handle.seek(max(0, log.stat().st_size - 8192))
        tail = handle.read().decode("utf-8", "replace")
    require(start.startswith("COMMAND:"), f"Training command absent: {arm}")
    for flag in ("--no-binocular-consistency", "--dataset_name LLFF", "--n_views 3",
                 "--resolution 8", "--dropout_factor 0.2", "--iterations 30000",
                 "--ns30k_sampling_tape", "--view_manifest"):
        require(flag in start, f"Training configuration mismatch: {arm} missing {flag}")
    if arm == "B0":
        require("--ns30k_arm OFF" in start and "--enable_oe_structural_loss" not in start,
                "B0 arm is not the I1-OFF control")
    else:
        for flag in ("--ns30k_arm ON", "--enable_oe_structural_loss",
                     "--oe_structural_start_iter 1", "--oe_structural_end_iter 8000"):
            require(flag in start, f"B0+I1 arm config mismatch: {flag}")
        require('"actual_activation_count": 8000' in tail and
                '"nonfinite_loss_seen": false' in tail,
                "I1 activation or non-finite loss summary mismatch")
    require("EXIT_CODE: 0" in tail and "STATUS: SUCCESS" in tail,
            f"Training was not successful: {arm}")


def check_sampling_pair() -> None:
    paths = [run / "ns30k_sampling_trace.csv" for run in ARMS.values()]
    readers = []
    handles = []
    try:
        for path in paths:
            nonempty(path)
            stream = path.open("r", encoding="utf-8-sig", newline="")
            handles.append(stream)
            reader = csv.DictReader(stream)
            require(reader.fieldnames is not None and
                    set(reader.fieldnames) == set(TRACE_FIELDS),
                    f"Invalid sampling trace columns: {path}")
            readers.append(reader)
        for iteration in range(1, 30001):
            rows = [next(r, None) for r in readers]
            require(all(row is not None for row in rows),
                    f"Sampling trace incomplete at iteration {iteration}")
            a, b = rows
            for row in rows:
                require(int(row["iteration"]) == iteration,
                        f"Sampling trace iteration mismatch at {iteration}")
                require(row["camera_image_name"] == row["actual_camera_image_name"],
                        f"Sampled/actual camera mismatch at {iteration}")
                require(row["sampling_file_sha256"] == EXPECTED_TAPE_HASH,
                        f"Unexpected tape identity at {iteration}")
            for field in TRACE_FIELDS:
                require(a[field] == b[field],
                        f"B0/B0+I1 sampling mismatch: iteration={iteration} field={field}")
        require(all(next(r, None) is None for r in readers),
                "Sampling traces contain more than 30000 rows")
    finally:
        for stream in handles:
            stream.close()
    print_step("30,000 paired sampling records: PASS")


def check_images(method_dir: Path) -> tuple[Path, Path]:
    renders = method_dir / "renders"
    gt = method_dir / "gt"
    require(renders.is_dir() and gt.is_dir(), f"Incomplete render directory: {method_dir}")
    render_names = {p.name for p in renders.glob("*.png") if p.stem.isdigit()}
    gt_names = {p.name for p in gt.glob("*.png")}
    require(render_names == set(VIEWS) and gt_names == set(VIEWS),
            f"RGB/GT filename set mismatch: {method_dir}; RGB={sorted(render_names)}, GT={sorted(gt_names)}")
    for name in VIEWS:
        nonempty(renders / name)
        nonempty(gt / name)
    return renders, gt


def snapshot_5000_inputs() -> None:
    """Do not overwrite existing root metrics; copy them into the new report."""
    for arm, run in ARMS.items():
        dest = REPORT / arm / "original_5000_snapshot"
        dest.mkdir(parents=True, exist_ok=False)
        for filename in ("results.json", "per_view.json"):
            src = run / filename
            nonempty(src)
            dst = dest / filename
            shutil.copy2(src, dst)
            require(sha256(src) == sha256(dst), f"Snapshot verification failed: {src}")
        print_step(f"{arm} original 5000 metrics snapshot: PASS")


def stream_command(cmd: list[str], log_path: Path) -> None:
    """Live combined output to console and a new non-overwriting UTF-8 log."""
    print_step("COMMAND: " + subprocess.list2cmdline(cmd))
    with log_path.open("x", encoding="utf-8", newline="") as log:
        log.write("COMMAND: " + subprocess.list2cmdline(cmd) + "\n")
        log.flush()
        child = subprocess.Popen(cmd, cwd=str(ROOT), stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, bufsize=0)
        try:
            assert child.stdout is not None
            import codecs
            decoder = codecs.getincrementaldecoder("utf-8")("replace")
            while True:
                data = child.stdout.read(8192)
                if not data:
                    break
                rendered = decoder.decode(data)
                if rendered:
                    sys.stdout.write(rendered)
                    sys.stdout.flush()
                    log.write(rendered)
                    log.flush()
            pending = decoder.decode(b"", final=True)
            if pending:
                sys.stdout.write(pending)
                log.write(pending)
            ret = child.wait()
        except BaseException:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
            log.write("\nSTATUS: INTERRUPTED_OR_ERROR\n")
            raise
        log.write(f"\nEXIT_CODE: {ret}\n")
        log.flush()
    require(ret == 0, f"Subprocess failed, exit={ret}: {cmd[2]}; log={log_path}")


def render_or_reuse(arm: str, iteration: int) -> Path:
    run = ARMS[arm]
    method_dir = run / "test" / f"ours_{iteration}"
    render_log = run / f"render_{iteration}.log"
    if method_dir.exists():
        require(method_dir.is_dir(), f"Render output is not a directory: {method_dir}")
        # Never overwrite existing outputs: reuse only when full six-view audit passes.
        check_images(method_dir)
        print_step(f"{arm} iteration={iteration}: existing RGB/GT reused (no overwrite)")
        return method_dir
    require(not render_log.exists(),
            f"Render log already exists, refusing overwrite: {render_log}")
    print_step(f"Rendering {arm} @ {iteration} (6 test views)")
    cmd = [
        sys.executable, "-u", str(ROOT / "render.py"),
        "-s", str(DATA_SOURCE), "-m", str(run),
        "--eval", "--dataset_name", "LLFF", "--n_views", "3",
        "--resolution", "8", "--sh_degree", "1", "--dropout_factor", "0.2",
        "--view_manifest", str(MANIFEST), "--iteration", str(iteration),
        "--skip_train",
    ]
    stream_command(cmd, render_log)
    check_images(method_dir)
    return method_dir


def copy_rgb_gt_to_isolated_stage(arm: str, iteration: int, method_dir: Path) -> Path:
    stage = REPORT / arm / str(iteration)
    target = stage / "test" / f"ours_{iteration}"
    r_dir = target / "renders"
    g_dir = target / "gt"
    r_dir.mkdir(parents=True, exist_ok=False)
    g_dir.mkdir(parents=True, exist_ok=False)
    renders, gt = check_images(method_dir)
    for name in VIEWS:
        for source, dest_dir in ((renders / name, r_dir), (gt / name, g_dir)):
            dest = dest_dir / name
            shutil.copy2(source, dest)
            require(sha256(source) == sha256(dest), f"Copy SHA256 mismatch: {source}")
    check_images(target)
    return stage


def validate_metrics(stage: Path, iteration: int) -> tuple[dict, dict]:
    results_path = stage / "results.json"
    views_path = stage / "per_view.json"
    nonempty(results_path)
    nonempty(views_path)
    with results_path.open(encoding="utf-8") as file:
        results = json.load(file)
    with views_path.open(encoding="utf-8") as file:
        per_view = json.load(file)
    key = f"ours_{iteration}"
    require(set(results) == {key} and set(per_view) == {key},
            f"Unexpected metric JSON keys for iteration={iteration}")
    totals = results[key]
    per_views = per_view[key]
    require(set(totals) == set(METRICS) and set(per_views) == set(METRICS),
            f"Missing metric(s) for iteration={iteration}")
    for metric in METRICS:
        value = totals[metric]
        require(isinstance(value, (float, int)) and math.isfinite(value),
                f"Non-finite aggregate: iteration={iteration} {metric}")
        values = per_views[metric]
        require(set(values) == set(VIEWS),
                f"Missing per-view metrics: iteration={iteration} {metric}")
        require(all(isinstance(v, (float, int)) and math.isfinite(v) for v in values.values()),
                f"Non-finite view metric: iteration={iteration} {metric}")
        arithmetic_mean = sum(values.values()) / 6
        require(abs(arithmetic_mean - value) <= 1e-4,
                f"Aggregate != six-view mean: iteration={iteration} {metric}")
    return totals, per_views


def compare_5000_with_original(arm: str, calculated: dict) -> None:
    path = REPORT / arm / "original_5000_snapshot" / "results.json"
    with path.open(encoding="utf-8") as file:
        original = json.load(file)["ours_5000"]
    for metric in METRICS:
        require(abs(calculated[metric] - original[metric]) <= 1e-4,
                f"5000 metrics changed versus original {arm}: {metric}; "
                f"original={original[metric]} recomputed={calculated[metric]}")
    print_step(f"{arm} 5000 recomputation matches original results: PASS")


def evaluate_one(arm: str, iteration: int, method_dir: Path) -> tuple[dict, dict]:
    stage = copy_rgb_gt_to_isolated_stage(arm, iteration, method_dir)
    print_step(f"Evaluating {arm} @ {iteration} in isolated directory")
    cmd = [sys.executable, "-u", str(ROOT / "metrics.py"),
           "--model_paths", str(stage), "--dataset_name", "LLFF"]
    stream_command(cmd, stage / "metrics.log")
    totals, views = validate_metrics(stage, iteration)
    if iteration == 5000:
        compare_5000_with_original(arm, totals)
    print(f"[PASS] {arm} {iteration}: PSNR={totals['PSNR']:.6f} "
          f"SSIM={totals['SSIM']:.6f} LPIPS={totals['LPIPS']:.6f}", flush=True)
    return totals, views


def save_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("x", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_reports(data: dict) -> None:
    summary_rows = []
    detail_rows = []
    for iteration in ITERATIONS:
        b0, b0_views = data[("B0", iteration)]
        on, on_views = data[("B0_I1", iteration)]
        row = {"scene": "fortress", "iteration": iteration}
        for metric in METRICS:
            delta = on[metric] - b0[metric]
            row[f"B0_{metric}"] = b0[metric]
            row[f"I1_{metric}"] = on[metric]
            row[f"delta_{metric}"] = delta
            row[f"improved_{metric}"] = delta > 0 if metric != "LPIPS" else delta < 0
            wins = sum((on_views[metric][v] > b0_views[metric][v]) if metric != "LPIPS"
                       else (on_views[metric][v] < b0_views[metric][v]) for v in VIEWS)
            row[f"wins_{metric}_of_6"] = wins
        summary_rows.append(row)
        for view in VIEWS:
            item = {"scene": "fortress", "iteration": iteration, "view": view}
            for metric in METRICS:
                a = b0_views[metric][view]
                b = on_views[metric][view]
                item[f"B0_{metric}"] = a
                item[f"I1_{metric}"] = b
                item[f"delta_{metric}"] = b - a
            detail_rows.append(item)
    save_csv(REPORT / "paired_summary.csv", list(summary_rows[0]), summary_rows)
    save_csv(REPORT / "per_view_deltas.csv", list(detail_rows[0]), detail_rows)
    audit = {
        "scene": "fortress", "protocol": "NS30K_v1_no_stereo_seed0_3v_frozenPDC",
        "iterations": list(ITERATIONS), "test_view_filenames": list(VIEWS),
        "training_baseline": "Binocular3DGS + Gaussian Dropout (official baseline; stereo disabled for this candidate)",
        "arms": {arm: str(path) for arm, path in ARMS.items()},
        "render_sha256": EXPECTED_HASHES["render.py"],
        "metrics_sha256": EXPECTED_HASHES["metrics.py"],
        "manifest_sha256": EXPECTED_MANIFEST_HASH,
        "sampling_tape_sha256": EXPECTED_TAPE_HASH,
        "pdc_sha256": EXPECTED_PDC_HASH,
        "note": "Single-scene and single-seed exploratory comparison; no SOTA/generalization claim.",
        "metric_direction": {"PSNR": "higher", "SSIM": "higher", "LPIPS": "lower"},
        "results": summary_rows,
    }
    with (REPORT / "summary.json").open("x", encoding="utf-8") as handle:
        json.dump(audit, handle, ensure_ascii=False, indent=2)
    print_step("ALL FOUR ITERATIONS — PAIRED RESULT SUMMARY")
    print(" iter | B0_PSNR I1_PSNR   ΔPSNR | B0_SSIM I1_SSIM    ΔSSIM | B0_LPIPS I1_LPIPS  ΔLPIPS")
    for r in summary_rows:
        print(f"{r['iteration']:5} | {r['B0_PSNR']:7.4f} {r['I1_PSNR']:7.4f} {r['delta_PSNR']:+8.4f} | "
              f"{r['B0_SSIM']:.6f} {r['I1_SSIM']:.6f} {r['delta_SSIM']:+.6f} | "
              f"{r['B0_LPIPS']:.6f} {r['I1_LPIPS']:.6f} {r['delta_LPIPS']:+.6f}")
    print("\nPer-view improvement wins (out of 6):")
    for r in summary_rows:
        print(f"  {r['iteration']}: PSNR {r['wins_PSNR_of_6']}/6, "
              f"SSIM {r['wins_SSIM_of_6']}/6, LPIPS {r['wins_LPIPS_of_6']}/6")
    print(f"\nREPORT: {REPORT}")
    print("FILES: paired_summary.csv, per_view_deltas.csv, summary.json")
    print("NS30K FORTRESS BATCH EVAL: PASS", flush=True)


def preflight() -> None:
    require((ROOT / "render.py").is_file() and (ROOT / "metrics.py").is_file(),
            "Run script from Binocular3DGS + Dropout repository root")
    for script, digest in EXPECTED_HASHES.items():
        check_hash(ROOT / script, digest)
    require(DATA_SOURCE.is_dir(), f"Dataset not found: {DATA_SOURCE}")
    check_hash(MANIFEST, EXPECTED_MANIFEST_HASH)
    check_hash(PDC, EXPECTED_PDC_HASH)
    check_hash(SAMPLING_TAPE, EXPECTED_TAPE_HASH)
    require(not REPORT.exists(),
            f"The archive already exists; refusing to overwrite: {REPORT}")
    for arm, run in ARMS.items():
        require(run.is_dir(), f"Model output missing: {run}")
        saved_point_files(run)
        check_train_log(arm, run)
        check_images(run / "test" / "ours_5000")
        for filename in ("results.json", "per_view.json"):
            nonempty(run / filename)
        for iteration in ITERATIONS[1:]:
            method_dir = run / "test" / f"ours_{iteration}"
            if method_dir.exists():
                check_images(method_dir)  # Reuse only if complete and nonempty.
            else:
                require(not (run / f"render_{iteration}.log").exists(),
                        f"Render log exists but render product missing: {run}, {iteration}")
    check_sampling_pair()
    print_step("GLOBAL PREFLIGHT: PASS (no existing model/metric outputs will be overwritten)")


def main() -> None:
    preflight()
    REPORT.mkdir(parents=True, exist_ok=False)
    snapshot_5000_inputs()
    paired_data = {}
    gt_reference = None
    for iteration in ITERATIONS:
        for arm in ARMS:
            method_dir = render_or_reuse(arm, iteration)
            _, gt = check_images(method_dir)
            current_gt = [sha256(gt / name) for name in VIEWS]
            if gt_reference is None:
                gt_reference = current_gt
            require(current_gt == gt_reference,
                    f"GT images are not byte-identical with frozen 5000 B0 GT: {arm} @ {iteration}")
            paired_data[(arm, iteration)] = evaluate_one(arm, iteration, method_dir)
    build_reports(paired_data)


if __name__ == "__main__":
    try:
        main()
    except (AuditError, FileExistsError, OSError, KeyError, ValueError, RuntimeError) as exc:
        print(f"\nNS30K FORTRESS BATCH EVAL: FAIL\n{type(exc).__name__}: {exc}", file=sys.stderr)
        print("Existing results preserved. Do not delete/overwrite; inspect the failure.", file=sys.stderr)
        raise SystemExit(1)
