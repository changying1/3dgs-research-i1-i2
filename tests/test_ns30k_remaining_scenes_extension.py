from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from script import evaluate_ns30k_candidate_pair_v1 as evaluator
from script import run_ns30k_candidate_live_retry as launcher
from utils.ns30k_sampling import canonical_json_bytes, generate_sampling_table


NEW_SCENES = {
    "fern": {
        "manifest_sha256": "47542b30e71b6735b2a35bf027f468bec94002f829b9356ffee708ff8895df5a",
        "pdc_sha256": "db637e49a7e18ce70235ada5e9363df2531c5a1b91ccc70b4299af48edd29a83",
        "cache_sha256": "139326c1d6f876dc55d387a6758e9bb6948378d5ac9d489f4d601e65b4ad9843",
        "sampling_sha256": "34e98a484a0513cd260a75eaf2f5e3400bfa41f65751a57d0d6aea08af0b0ee8",
        "content_sha256": "a07d2f826fd37b8508e9692fb29d0b10b733b068063db34b3ed09687da332d05",
        "test_count": 3,
    },
    "flower": {
        "manifest_sha256": "fdabbc6dcf4cc49e32fb4c055a0a2f0ef7c7455c9610814edf9232dcad792b6e",
        "pdc_sha256": "0bad102dd2a8af06465927da9a0dec61654f3d55498286acd76497837762b751",
        "cache_sha256": "3bc9a8fbfae94e5f02d07618da9194ebfd5589c6e97998a6a89230e3171d82cd",
        "sampling_sha256": "35b29382876b82d9b75069af20e754fc25ca30274fea9e9ea1082e74526265b9",
        "content_sha256": "afc1b5299c6171c17d4a11cad537104dcd8a944f7def67f7374c059fb410d62a",
        "test_count": 5,
    },
    "trex": {
        "manifest_sha256": "ae86e0ee71805cf04ac6b7bcea0bb0302aa1d08f41ddbe057308eabb2a6cf881",
        "pdc_sha256": "14da7f12510582dabb347228a1bbc800a3e563566de6cfd55448fd93e5d34844",
        "cache_sha256": "9090c99a6c08dbb205664b2a1a488c8b03f7339fedfd7d66a67f591a6caac27a",
        "sampling_sha256": "9b4e4de7abcfa35464043b655ceafc46ff586e400c9ce66ee546213bcd644263",
        "content_sha256": "efb01d38f0170a1da8ad89ad6dd1b98abdd844f4be1b2bca408acad5417ce431",
        "test_count": 7,
    },
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def one_value(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


def test_remaining_scene_registry_uses_real_dataset_names_and_formal_assets():
    assert tuple(launcher.SCENES) == (
        "fortress", "room", "horns", "leaves", "orchids", "fern", "flower", "trex"
    )
    assert evaluator.SUPPORTED_SCENES[-3:] == ("fern", "flower", "trex")
    for scene, expected in NEW_SCENES.items():
        spec = launcher.SCENES[scene]
        assert (launcher.DATA_ROOT / scene).is_dir()
        assert spec.manifest_sha256 == expected["manifest_sha256"]
        assert spec.pdc_sha256 == expected["pdc_sha256"]
        assert spec.cache_metadata_sha256 == expected["cache_sha256"]
        assert spec.sampling_sha256 == expected["sampling_sha256"]
        manifest = json.loads((ROOT / spec.manifest).read_text(encoding="utf-8"))
        assert manifest["scene"] == scene
        assert len(manifest["test_images"]) == expected["test_count"]


@pytest.mark.parametrize("scene", tuple(NEW_SCENES))
def test_new_sampling_tape_is_complete_asset_bound_and_deterministic(scene):
    spec = launcher.SCENES[scene]
    expected = NEW_SCENES[scene]
    path = ROOT / spec.sampling_table
    document = json.loads(path.read_text(encoding="utf-8"))
    assert sha256(path) == expected["sampling_sha256"]
    assert document["seed"] == 0
    assert document["entry_count"] == 30_000
    assert document["content_sha256"] == expected["content_sha256"]
    assert document["manifest_sha256"] == expected["manifest_sha256"]
    assert document["pdc_sha256"] == expected["pdc_sha256"]
    regenerated = generate_sampling_table(
        scene=scene,
        seed=0,
        manifest_path=ROOT / spec.manifest,
        pdc_path=ROOT / spec.pdc_relative,
        pdc_suffix=spec.pdc_suffix,
    )
    assert canonical_json_bytes(regenerated) == path.read_bytes()


@pytest.mark.parametrize("scene", tuple(NEW_SCENES))
def test_new_scene_commands_preserve_b0_end8k_contract(scene):
    b0 = launcher.build_train_command(launcher.build_plan(scene, "B0"), "python")
    end8k = launcher.build_train_command(launcher.build_plan(scene, "B0+I1"), "python")
    for command in (b0, end8k):
        assert one_value(command, "-s") == str(launcher.DATA_ROOT / scene)
        assert one_value(command, "--iterations") == "30000"
        assert one_value(command, "--resolution") == "8"
        assert one_value(command, "--sh_degree") == "1"
        assert one_value(command, "--dropout_factor") == "0.2"
        assert "--no-binocular-consistency" in command
        assert command[command.index("--save_iterations") + 1:command.index("--checkpoint_iterations")] == [
            "5000", "10000", "20000", "30000"
        ]
    assert one_value(b0, "--ns30k_arm") == "OFF"
    assert "--enable_oe_structural_loss" not in b0
    assert one_value(end8k, "--ns30k_arm") == "ON"
    assert one_value(end8k, "--oe_structural_weight") == "0.1"
    assert one_value(end8k, "--oe_structural_start_iter") == "1"
    assert one_value(end8k, "--oe_structural_end_iter") == "8000"


@pytest.mark.parametrize("scene", tuple(NEW_SCENES))
def test_asset_provenance_is_current_scene_bound_and_excludes_test_views(scene):
    plan = launcher.build_plan(scene, "B0")
    manifest = json.loads(plan.manifest.read_text(encoding="utf-8"))
    record = launcher._validate_asset_record(plan, manifest)
    assert record is not None
    assert record["legacy_fsgs_asset_reused"] is False
    assert record["selected_train_images"] == manifest["selected_train_images"]
    assert set(record["selected_train_images"]).isdisjoint(record["test_images"])
    cache = launcher._validate_cache(plan, manifest)
    assert cache["selected_train_images"] == manifest["selected_train_images"]
    assert {item["image_name"] for item in cache["records"]}.isdisjoint(manifest["test_images"])


def test_asset_provenance_hash_mismatch_and_cross_scene_pollution_fail_closed():
    plan = launcher.build_plan("fern", "B0")
    manifest = json.loads(plan.manifest.read_text(encoding="utf-8"))
    bad_hash = replace(plan.spec, asset_record_sha256="0" * 64)
    with pytest.raises(launcher.PreflightError, match="SHA256 mismatch"):
        launcher._validate_asset_record(replace(plan, spec=bad_hash), manifest)

    flower = launcher.SCENES["flower"]
    crossed = replace(
        plan.spec,
        asset_record=flower.asset_record,
        asset_record_sha256=flower.asset_record_sha256,
    )
    with pytest.raises(launcher.PreflightError, match="not bound"):
        launcher._validate_asset_record(replace(plan, spec=crossed), manifest)


def test_missing_asset_and_duplicate_output_fail_closed(tmp_path):
    plan = launcher.build_plan("fern", "B0", root=tmp_path)
    (tmp_path / "output").mkdir()
    with pytest.raises(launcher.PreflightError, match="missing or empty view Manifest"):
        launcher._validate_manifest_and_colmap(plan)
    plan.output_dir.mkdir()
    with pytest.raises(launcher.PreflightError, match="REFUSING to reuse"):
        launcher._validate_output_target(plan)


def test_evaluator_row_counts_follow_each_manifest_instead_of_a_fixed_four():
    assert evaluator.expected_report_rows("fern") == (8, 54)
    assert evaluator.expected_report_rows("flower") == (8, 90)
    assert evaluator.expected_report_rows("trex") == (8, 126)
