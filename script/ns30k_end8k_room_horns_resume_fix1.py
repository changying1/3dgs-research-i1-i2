# -*- coding: utf-8 -*-
"""Non-destructive recovery after Room=6 / Horns=8 render-view mismatch.

Place beside ns30k_end8k_room_horns_full_v1.py in the repository's script/ directory.
From repository root:
    python -u script/ns30k_end8k_room_horns_resume_fix1.py --preflight
    python -u script/ns30k_end8k_room_horns_resume_fix1.py --execute

Never trains. Reuses the completed Room 4x2 metric stages, reuses Horns B0 5k
render, and only renders/evaluates remaining Horns stages in a separate new
report directory. Then produces a three-scene report using the completed
Fortress archive, without modifying the original partial batch report.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / 'script'
OLD_SCRIPT = SCRIPT_DIR / 'ns30k_end8k_room_horns_full_v1.py'
OLD_HASH = 'b4b4fd0d599062c5759361caa30f5ae24a83b5d397cae5b384d63357be9da4ca'
OLD_REPORT = ROOT / 'reports/ns30k_eval_v1/end8k_room_horns_4node_pair_v1'
FORTRESS_REPORT = ROOT / 'reports/ns30k_eval_v1/fortress_4node_pair_v1'
NEW_REPORT = ROOT / 'reports/ns30k_eval_v1/end8k_room_horns_resume_fix1'
ITERATIONS = (5000, 10000, 20000, 30000)
METRICS = ('PSNR', 'SSIM', 'LPIPS')
SCENE_VIEWS = {'room': tuple(f'{i:05d}.png' for i in range(6)),
               'horns': tuple(f'{i:05d}.png' for i in range(8)),
               'fortress': tuple(f'{i:05d}.png' for i in range(6))}
ARMS = ('B0', 'B0+I1')


class AuditError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def nonempty(p: Path) -> None:
    require(p.is_file() and p.stat().st_size > 0, f'Missing or empty file: {p}')


def digest(p: Path) -> str:
    nonempty(p)
    h = hashlib.sha256()
    with p.open('rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def log(message: str) -> None:
    print(f'[NS30K FIX1] {message}', flush=True)


def load_v1():
    require(digest(OLD_SCRIPT) == OLD_HASH,
            f'Original batch runner changed. Refuse to resume: {OLD_SCRIPT}')
    spec = importlib.util.spec_from_file_location('ns30k_end8k_original_frozen', OLD_SCRIPT)
    require(spec is not None and spec.loader is not None, 'Cannot load frozen original launcher')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    require(mod.ROOT == ROOT, 'Unexpected original runner root')
    return mod


def images(folder: Path, scene: str):
    rgb = folder / 'renders'
    gt = folder / 'gt'
    require(rgb.is_dir() and gt.is_dir(), f'Missing render or GT directory: {folder}')
    expected = set(SCENE_VIEWS[scene])
    actual_rgb = {p.name for p in rgb.glob('*.png') if p.stem.isdigit()}
    actual_gt = {p.name for p in gt.glob('*.png')}
    require(actual_rgb == expected and actual_gt == expected,
            f'{scene}: expected exactly {len(expected)} RGB/GT images: {folder}; '
            f'RGB={sorted(actual_rgb)} GT={sorted(actual_gt)}')
    for name in expected:
        nonempty(rgb / name)
        nonempty(gt / name)
    return rgb, gt


def load_metrics(stage: Path, scene: str, iteration: int):
    images(stage / 'test' / f'ours_{iteration}', scene)
    result_file = stage / 'results.json'
    view_file = stage / 'per_view.json'
    nonempty(result_file)
    nonempty(view_file)
    results = json.loads(result_file.read_text(encoding='utf-8-sig'))
    views = json.loads(view_file.read_text(encoding='utf-8-sig'))
    key = f'ours_{iteration}'
    require(set(results) == {key} and set(views) == {key}, f'Incorrect method key: {stage}')
    agg, per = results[key], views[key]
    require(set(agg) == set(METRICS) and set(per) == set(METRICS), f'Incomplete metrics: {stage}')
    expected = set(SCENE_VIEWS[scene])
    for metric in METRICS:
        require(isinstance(agg[metric], (int, float)) and math.isfinite(agg[metric]),
                f'Nonfinite mean: {stage}/{metric}')
        require(set(per[metric]) == expected, f'Wrong per-view count: {stage}/{metric}')
        require(all(isinstance(v, (int, float)) and math.isfinite(v)
                    for v in per[metric].values()), f'Nonfinite per-view: {stage}/{metric}')
        require(abs(sum(per[metric].values()) / len(expected) - agg[metric]) < 1e-4,
                f'Invalid per-view mean: {stage}/{metric}')
    return agg, per


def verify_same_images(source: Path, stage: Path, scene: str, iteration: int) -> None:
    src_rgb, src_gt = images(source, scene)
    dst_rgb, dst_gt = images(stage / 'test' / f'ours_{iteration}', scene)
    for name in SCENE_VIEWS[scene]:
        require(digest(src_rgb / name) == digest(dst_rgb / name),
                f'Archived render differs from original: {scene}/{iteration}/{name}')
        require(digest(src_gt / name) == digest(dst_gt / name),
                f'Archived GT differs from original: {scene}/{iteration}/{name}')


def room_stage(arm: str, iteration: int) -> Path:
    return OLD_REPORT / 'room' / ('B0_I1' if arm == 'B0+I1' else arm) / str(iteration)


def horns_stage(arm: str, iteration: int) -> Path:
    return NEW_REPORT / 'horns' / ('B0_I1' if arm == 'B0+I1' else arm) / str(iteration)


def source_folder(v1, scene: str, arm: str, iteration: int) -> Path:
    return v1.run_path(scene, arm) / 'test' / f'ours_{iteration}'


def validate_room_all(v1):
    data = {}
    gt_reference = None
    for iteration in ITERATIONS:
        for arm in ARMS:
            source = source_folder(v1, 'room', arm, iteration)
            stage = room_stage(arm, iteration)
            verify_same_images(source, stage, 'room', iteration)
            _, gt_dir = images(source, 'room')
            hashes = tuple(digest(gt_dir / name) for name in SCENE_VIEWS['room'])
            if gt_reference is None:
                gt_reference = hashes
            require(hashes == gt_reference, f'Room GT mismatch: {arm}/{iteration}')
            data[('room', arm, iteration)] = load_metrics(stage, 'room', iteration)
    log('Room: 8 prior metrics stages verified and reused; no rerender or re-evaluation')
    return data


def validate_fortress_archive():
    summary_file = FORTRESS_REPORT / 'summary.json'
    detail_file = FORTRESS_REPORT / 'per_view_deltas.csv'
    nonempty(summary_file)
    nonempty(detail_file)
    summary = json.loads(summary_file.read_text(encoding='utf-8-sig'))
    require(summary.get('scene') == 'fortress', 'Wrong Fortress archive scene')
    require(summary.get('iterations') == list(ITERATIONS), 'Wrong Fortress iterations')
    require(set(summary.get('test_view_filenames', [])) == set(SCENE_VIEWS['fortress']),
            'Wrong Fortress test view protocol')
    require(summary.get('render_sha256') == 'a5f1a0e754d5d2eb95a79cbfed3bbcec1445da08ab07f865bbe0746578ca223a'
            and summary.get('metrics_sha256') == 'c614e742c13eafa25c1d6d78a86d09b865cb35f1cf9340e404552362f85e660f',
            'Wrong Fortress evaluation code identity')
    paired = summary.get('results', [])
    require(len(paired) == 4, 'Fortress summary must have four rows')
    detail = {}
    with detail_file.open('r', encoding='utf-8-sig', newline='') as handle:
        for row in csv.DictReader(handle):
            require(row['scene'] == 'fortress', 'Unexpected Fortress detail scene')
            key = (int(row['iteration']), row['view'])
            require(key not in detail, f'Duplicate Fortress per-view: {key}')
            detail[key] = row
    require(len(detail) == 24, 'Fortress archive must have 24 detailed views')
    results = {}
    for row in paired:
        n = int(row['iteration'])
        require(n in ITERATIONS, f'Invalid Fortress checkpoint {n}')
        per0, per1, avg0, avg1 = {}, {}, {}, {}
        for metric in METRICS:
            v0, v1 = {}, {}
            for name in SCENE_VIEWS['fortress']:
                item = detail[(n, name)]
                a = float(item[f'B0_{metric}'])
                b = float(item[f'I1_{metric}'])
                require(math.isfinite(a) and math.isfinite(b) and
                        abs((b - a) - float(item[f'delta_{metric}'])) < 1e-5,
                        f'Invalid Fortress detail: {n}/{name}/{metric}')
                v0[name], v1[name] = a, b
            a, b = float(row[f'B0_{metric}']), float(row[f'I1_{metric}'])
            require(abs(a - sum(v0.values()) / 6) < 1e-4 and
                    abs(b - sum(v1.values()) / 6) < 1e-4 and
                    abs((b - a) - float(row[f'delta_{metric}'])) < 1e-5,
                    f'Fortress mean disagrees with per-view: {n}/{metric}')
            wins = sum((v1[name] < v0[name]) if metric == 'LPIPS'
                       else (v1[name] > v0[name]) for name in SCENE_VIEWS['fortress'])
            require(wins == int(row[f'wins_{metric}_of_6']),
                    f'Fortress wins mismatch: {n}/{metric}')
            per0[metric], per1[metric] = v0, v1
            avg0[metric], avg1[metric] = a, b
        require(('fortress', 'B0', n) not in results, f'Duplicate Fortress iteration: {n}')
        results[('fortress', 'B0', n)] = (avg0, per0)
        results[('fortress', 'B0+I1', n)] = (avg1, per1)
    require(len(results) == 8, 'Fortress archive missing a checkpoint')
    log('Fortress: 4 existing paired checkpoints verified and reused')
    return results


def verify_existing_new_stage(v1, arm: str, iteration: int) -> None:
    stage = horns_stage(arm, iteration)
    if stage.exists():
        source = source_folder(v1, 'horns', arm, iteration)
        verify_same_images(source, stage, 'horns', iteration)
        load_metrics(stage, 'horns', iteration)  # incomplete stage => stop; never delete


def preflight(v1):
    require(Path.cwd().resolve() == ROOT, f'Run from repository root: {ROOT}')
    v1.check_global_code()
    for scene in ('room', 'horns'):
        for arm in ARMS:
            v1.check_train_output(scene, arm)
        v1.check_trace_pair(scene)
    room_data = validate_room_all(v1)
    fortress_data = validate_fortress_archive()
    for out in ('paired_summary.csv', 'per_view_deltas.csv', 'summary.json',
                'paired_summary_3scenes.csv', 'per_view_deltas_3scenes.csv',
                'mean_by_iteration_3scenes.csv', 'summary_3scenes.json'):
        require(not (NEW_REPORT / out).exists(), f'Final report already exists: {NEW_REPORT / out}')
    for iteration in ITERATIONS:
        for arm in ARMS:
            source = source_folder(v1, 'horns', arm, iteration)
            if source.exists():
                images(source, 'horns')
            else:
                logfile = v1.run_path('horns', arm) / f'render_{iteration}.log'
                require(not logfile.exists(), f'Render log exists without complete RGB/GT: {logfile}')
            verify_existing_new_stage(v1, arm, iteration)
    log('PREFLIGHT PASS: 4 complete trains; Room=6, Horns=8, Fortress=6; old work preserved')
    return room_data, fortress_data


def render_horns(v1, arm: str, iteration: int) -> Path:
    source = source_folder(v1, 'horns', arm, iteration)
    if source.exists():
        images(source, 'horns')
        log(f'Reusing Horns render {arm}/{iteration} ({len(SCENE_VIEWS["horns"])} views)')
        return source
    run = v1.run_path('horns', arm)
    logfile = run / f'render_{iteration}.log'
    require(not logfile.exists(), f'Render logfile already exists: {logfile}')
    command = [sys.executable, '-u', str(ROOT / 'render.py'),
               '-s', str(v1.DATA_ROOT / 'horns'), '-m', str(run),
               '--eval', '--dataset_name', 'LLFF', '--n_views', '3',
               '--resolution', '8', '--sh_degree', '1', '--dropout_factor', '0.2',
               '--view_manifest', str(ROOT / v1.SCENES['horns']['manifest']),
               '--iteration', str(iteration), '--skip_train']
    v1.subprocess_command(command, logfile)
    images(source, 'horns')
    return source


def evaluate_horns(v1, arm: str, iteration: int, source: Path):
    stage = horns_stage(arm, iteration)
    if stage.exists():
        verify_same_images(source, stage, 'horns', iteration)
        data = load_metrics(stage, 'horns', iteration)
        log(f'Reusing completed Horns metrics {arm}/{iteration}')
        return data
    render_dir, gt_dir = images(source, 'horns')
    target = stage / 'test' / f'ours_{iteration}'
    for kind, original in (('renders', render_dir), ('gt', gt_dir)):
        destination = target / kind
        destination.mkdir(parents=True, exist_ok=False)
        for name in SCENE_VIEWS['horns']:
            p = destination / name
            require(not p.exists(), f'Refusing file overwrite: {p}')
            shutil.copy2(original / name, p)
            require(digest(original / name) == digest(p), f'Failed isolated copy: {p}')
    verify_same_images(source, stage, 'horns', iteration)
    command = [sys.executable, '-u', str(ROOT / 'metrics.py'),
               '--model_paths', str(stage), '--dataset_name', 'LLFF']
    v1.subprocess_command(command, stage / 'metrics.log')
    data = load_metrics(stage, 'horns', iteration)
    log(f'Horns METRICS PASS: {arm}/{iteration}: ' +
        ' '.join(f'{metric}={data[0][metric]:.6f}' for metric in METRICS))
    return data


def compile_rows(data, scenes):
    paired, detail = [], []
    for scene in scenes:
        views = SCENE_VIEWS[scene]
        for iteration in ITERATIONS:
            b0, bv = data[(scene, 'B0', iteration)]
            on, ov = data[(scene, 'B0+I1', iteration)]
            row = {'scene': scene, 'iteration': iteration, 'n_test_views': len(views)}
            for metric in METRICS:
                delta = on[metric] - b0[metric]
                row.update({f'B0_{metric}': b0[metric], f'I1_{metric}': on[metric],
                            f'delta_{metric}': delta,
                            f'improved_{metric}': delta < 0 if metric == 'LPIPS' else delta > 0,
                            f'wins_{metric}': sum(
                                (ov[metric][name] < bv[metric][name]) if metric == 'LPIPS'
                                else (ov[metric][name] > bv[metric][name]) for name in views)})
            paired.append(row)
            for name in views:
                item = {'scene': scene, 'iteration': iteration, 'view': name}
                for metric in METRICS:
                    item.update({f'B0_{metric}': bv[metric][name],
                                 f'I1_{metric}': ov[metric][name],
                                 f'delta_{metric}': ov[metric][name] - bv[metric][name]})
                detail.append(item)
    return paired, detail


def save_csv(path, rows):
    require(rows, f'No report rows: {path}')
    with path.open('x', encoding='utf-8-sig', newline='') as stream:
        w = csv.DictWriter(stream, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def save_json(path, obj):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(obj, stream, ensure_ascii=False, indent=2)


def produce_final(v1, room_data, fortress_data, horns_data):
    results = {**fortress_data, **room_data, **horns_data}
    paired2, detail2 = compile_rows(results, ('room', 'horns'))
    paired3, detail3 = compile_rows(results, ('fortress', 'room', 'horns'))
    require(len(paired2) == 8 and len(detail2) == 56, 'Wrong Room/Horns report cardinality')
    require(len(paired3) == 12 and len(detail3) == 80, 'Wrong three-scene report cardinality')
    means = []
    for iteration in ITERATIONS:
        selected = [r for r in paired3 if r['iteration'] == iteration]
        require(len(selected) == 3, 'Expected three scenes per checkpoint')
        meanrow = {'iteration': iteration, 'n_scenes': 3}
        for metric in METRICS:
            for prefix in ('B0_', 'I1_', 'delta_'):
                key = prefix + metric
                meanrow[f'mean_{key}'] = sum(r[key] for r in selected) / 3
            meanrow[f'improved_scenes_{metric}'] = sum(bool(r[f'improved_{metric}']) for r in selected)
        means.append(meanrow)
    NEW_REPORT.mkdir(parents=True, exist_ok=True)
    save_csv(NEW_REPORT / 'paired_summary.csv', paired2)
    save_csv(NEW_REPORT / 'per_view_deltas.csv', detail2)
    save_json(NEW_REPORT / 'summary.json', {
        'protocol': 'NS30K_v1_no_stereo_seed0_3v_frozenPDC_I1End8k',
        'scenes': ['room', 'horns'], 'scene_test_counts': {'room': 6, 'horns': 8},
        'iterations': list(ITERATIONS), 'results': paired2,
        'room_source': str(OLD_REPORT), 'horns_new_stages': str(NEW_REPORT / 'horns'),
        'note': 'One seed; room stages reused, horns recovered; no retraining',
    })
    save_csv(NEW_REPORT / 'paired_summary_3scenes.csv', paired3)
    save_csv(NEW_REPORT / 'per_view_deltas_3scenes.csv', detail3)
    save_csv(NEW_REPORT / 'mean_by_iteration_3scenes.csv', means)
    save_json(NEW_REPORT / 'summary_3scenes.json', {
        'protocol': 'NS30K_v1_no_stereo_seed0_3v_frozenPDC_I1End8k',
        'scenes': ['fortress', 'room', 'horns'],
        'test_view_counts': {'fortress': 6, 'room': 6, 'horns': 8},
        'aggregation': 'Equal-weight scene means, not pooled-test-view means',
        'iterations': list(ITERATIONS), 'results': paired3,
        'scene_means': means,
        'source_reports': {'fortress': str(FORTRESS_REPORT), 'room': str(OLD_REPORT),
                           'horns': str(NEW_REPORT / 'horns')},
        'render_sha256': v1.EVAL_CODE['render.py'],
        'metrics_sha256': v1.EVAL_CODE['metrics.py'],
        'note': '3 scenes, one seed, no stereo consistency. Not the official full-stereo baseline.',
    })
    log('===== COMPLETED 3-SCENE 4-NODE PAIRED RESULT =====')
    for r in paired3:
        log(f"{r['scene']:9} {r['iteration']:5} n={r['n_test_views']}: "
            f"ΔPSNR={r['delta_PSNR']:+.4f} "
            f"ΔSSIM={r['delta_SSIM']:+.6f} "
            f"ΔLPIPS={r['delta_LPIPS']:+.6f}")
    log('THREE-SCENE REPORTS: ' + str(NEW_REPORT))
    print('NS30K END8K CROSS-SCENE FIX1: PASS', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--preflight', action='store_true')
    mode.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    v1 = load_v1()
    room_data, fortress_data = preflight(v1)
    if args.preflight:
        return
    horns_data = {}
    ref_gt = None
    for iteration in ITERATIONS:
        for arm in ARMS:
            source = render_horns(v1, arm, iteration)
            _, gt = images(source, 'horns')
            hashes = tuple(digest(gt / name) for name in SCENE_VIEWS['horns'])
            if ref_gt is None:
                ref_gt = hashes
            require(hashes == ref_gt, f'Horns GT changed across arms/iterations: {arm}/{iteration}')
            horns_data[('horns', arm, iteration)] = evaluate_horns(v1, arm, iteration, source)
    produce_final(v1, room_data, fortress_data, horns_data)


if __name__ == '__main__':
    try:
        main()
    except (AuditError, OSError, ValueError, KeyError, RuntimeError, IndexError, AssertionError) as exc:
        print('NS30K END8K CROSS-SCENE FIX1: FAIL\n'
              f'{type(exc).__name__}: {exc}\n'
              'Original trained runs and old evaluation outputs have NOT been deleted.',
              file=sys.stderr)
        raise SystemExit(1)
