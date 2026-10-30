import argparse
import ast
import copy
import json
import os
import random
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

from utils.ns30k_sampling import (
    DROPOUT_PAIRING_LEVEL,
    END_ITERATION,
    NS30KContractError,
    NS30KSamplingTable,
    NS30KTraceLogger,
    PROTOCOL_VERSION,
    SEED_PURPOSES,
    START_ITERATION,
    _content_digest,
    dropout_pairing_level,
    generate_sampling_table,
    sha256_file,
    validate_ns30k_training_configuration,
    validate_ns30k_output_directory,
    validate_sampling_document,
    write_sampling_table,
)


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "train.py"
LEGACY_P7_PATH = ROOT / "utils" / "paired_sampling.py"
LEGACY_P7_SHA256 = "58fec5decc4e66f81143fe0fdf9c098610a9d8efb47000e0bfae42a7d4df597a"


@contextmanager
def _working_directory(path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _train_tree():
    return ast.parse(TRAIN_PATH.read_text(encoding="utf-8"), filename=str(TRAIN_PATH))


def _main_body():
    for node in _train_tree().body:
        if isinstance(node, ast.If) and ast.unparse(node.test) == "__name__ == '__main__'":
            return node.body
    raise AssertionError("train.py has no __main__ block")


def _is_direct_registration(node):
    if isinstance(node, ast.Assign):
        return (
            len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "binocular_group"
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and node.value.func.attr == "add_mutually_exclusive_group"
        )
    if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
        return False
    function = node.value.func
    return (
        isinstance(function, ast.Attribute)
        and isinstance(function.value, ast.Name)
        and function.attr == "add_argument"
        and function.value.id in {"parser", "binocular_group"}
    )


def _train_cli_parser():
    parser = argparse.ArgumentParser(prog="train.py")
    namespace = {"parser": parser}
    statements = [node for node in _main_body() if _is_direct_registration(node)]
    module = ast.fix_missing_locations(ast.Module(body=statements, type_ignores=[]))
    exec(compile(module, str(TRAIN_PATH), "exec"), namespace)
    return parser


class NS30KSamplingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.scene = "fern"
        cls.suffix = "Fern_3v_FrozenPDC_NS30K"
        cls.manifest_path = cls.root / "view_manifest.json"
        cls.manifest = {
            "dataset": "LLFF",
            "scene": cls.scene,
            "view_count": 3,
            "selected_train_images": ["IMG_01.JPG", "IMG_09.JPG", "IMG_17.JPG"],
            "pdc_initialization_images": ["IMG_01.JPG", "IMG_09.JPG", "IMG_17.JPG"],
            "test_images": ["IMG_00.JPG", "IMG_08.JPG"],
        }
        cls.manifest_path.write_text(
            json.dumps(cls.manifest, indent=2) + "\n", encoding="utf-8"
        )
        cls.pdc_path = (
            cls.root
            / "keypoints_to_3d"
            / f"LLFF_{cls.suffix}"
            / f"{cls.scene}_keypoints_to_3d.ply"
        )
        cls.pdc_path.parent.mkdir(parents=True)
        cls.pdc_path.write_bytes(b"ply\nformat ascii 1.0\nelement vertex 0\nend_header\n")
        cls.document = generate_sampling_table(
            scene=cls.scene,
            seed=20261027,
            manifest_path=cls.manifest_path,
            pdc_path=cls.pdc_path,
            pdc_suffix=cls.suffix,
        )
        cls.table_path = cls.root / "ns30k_sampling.json"
        cls.table_file_sha256 = write_sampling_table(cls.table_path, cls.document)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def _table(self):
        return NS30KSamplingTable(
            self.table_path,
            self.table_file_sha256,
            expected_scene=self.scene,
        )

    def _dataset(self):
        return SimpleNamespace(
            source_path=str(self.root / self.scene),
            dataset_name="LLFF",
            eval=True,
            n_views=3,
            view_manifest=str(self.manifest_path),
            suffix=self.suffix,
            initial_opacity_arm="control",
            initial_opacity_scores="",
            initial_opacity_score_manifest="",
            resolution=8,
            sh_degree=1,
            model_path=str(self.root / "unused_output_candidate"),
        )

    @staticmethod
    def _opt():
        return SimpleNamespace(
            iterations=30_000,
            position_lr_max_steps=30_000,
            position_lr_init=0.00016,
            position_lr_final=0.0000016,
            lambda_dssim=0.2,
            densify_from_iter=1000,
            densify_until_iter=15_000,
            densification_interval=100,
            densify_grad_threshold=0.0002,
            random_background=False,
        )

    @staticmethod
    def _args(arm):
        return SimpleNamespace(
            binocular_consistency=False,
            disable_training_time_evaluation=True,
            start_checkpoint=None,
            stop_after_checkpoint_iter=None,
            i1_b1_weak_continuation=False,
            paired_sampling_tape="",
            ns30k_arm=arm,
            enable_oe_structural_loss=arm == "ON",
            oe_depth_cache_dir="frozen_depth_cache" if arm == "ON" else "",
            oe_preserve_baseline_densification_stats=arm == "ON",
            oe_structural_start_iter=1 if arm == "ON" else 0,
            oe_structural_end_iter=8000 if arm == "ON" else 30_000,
            oe_structural_weight=0.1 if arm == "ON" else 0.0,
            oe_structure_gate_mode="stable_evidence",
            oe_stable_quantile=0.8,
            dropout_factor=0.2,
            sigma_noise=0.0,
            opacity_decay=True,
            opacity_decay_factor=0.995,
            save_iterations=[5_000, 10_000, 20_000, 30_000, 30_000],
            checkpoint_iterations=[5_000, 10_000, 20_000, 30_000],
        )

    def test_protocol_is_deterministic_and_independent_of_global_random_calls(self):
        random.seed(9)
        for _ in range(500):
            random.random()
        regenerated = generate_sampling_table(
            scene=self.scene,
            seed=20261027,
            manifest_path=self.manifest_path,
            pdc_path=self.pdc_path,
            pdc_suffix=self.suffix,
        )
        self.assertEqual(regenerated, self.document)
        self.assertEqual(regenerated["version"], PROTOCOL_VERSION)
        regenerated_path = self.root / "deterministic_copy.json"
        regenerated_sha256 = write_sampling_table(regenerated_path, regenerated)
        self.assertEqual(regenerated_path.read_bytes(), self.table_path.read_bytes())
        self.assertEqual(regenerated_sha256, self.table_file_sha256)

    def test_table_covers_every_iteration_exactly_once_and_only_training_cameras(self):
        entries = self.document["entries"]
        self.assertEqual(len(entries), 30_000)
        self.assertEqual(entries[0]["iteration"], START_ITERATION)
        self.assertEqual(entries[-1]["iteration"], END_ITERATION)
        self.assertEqual(
            [entry["iteration"] for entry in entries],
            list(range(START_ITERATION, END_ITERATION + 1)),
        )
        allowed = set(self.document["training_camera_image_names"])
        self.assertEqual(allowed, {"IMG_01", "IMG_09", "IMG_17"})
        self.assertTrue(all(entry["camera_image_name"] in allowed for entry in entries))
        self.assertTrue(
            all(set(entry["seeds"]) == set(SEED_PURPOSES) for entry in entries)
        )
        table = self._table()
        for iteration in range(START_ITERATION, END_ITERATION + 1):
            self.assertEqual(table.consume(iteration)["iteration"], iteration)
        table.assert_complete()
        with self.assertRaises(NS30KContractError):
            table.consume(END_ITERATION + 1)

    def test_missing_duplicate_and_out_of_order_records_are_rejected(self):
        mutations = []
        missing = copy.deepcopy(self.document)
        missing["entries"].pop()
        missing["content_sha256"] = _content_digest(missing)
        mutations.append(missing)

        duplicate = copy.deepcopy(self.document)
        duplicate["entries"].append(copy.deepcopy(duplicate["entries"][-1]))
        duplicate["content_sha256"] = _content_digest(duplicate)
        mutations.append(duplicate)

        out_of_order = copy.deepcopy(self.document)
        out_of_order["entries"][0], out_of_order["entries"][1] = (
            out_of_order["entries"][1],
            out_of_order["entries"][0],
        )
        out_of_order["content_sha256"] = _content_digest(out_of_order)
        mutations.append(out_of_order)

        bad_entry_hash = copy.deepcopy(self.document)
        bad_entry_hash["entries"][99]["seeds"]["dropout"] += 1
        bad_entry_hash["content_sha256"] = _content_digest(bad_entry_hash)
        mutations.append(bad_entry_hash)

        for document in mutations:
            with self.subTest(entry_count=len(document["entries"])):
                with self.assertRaises(NS30KContractError):
                    validate_sampling_document(document)

    def test_tampered_table_and_modified_manifest_are_rejected(self):
        tampered_path = self.root / "tampered.json"
        payload = self.table_path.read_bytes().replace(b'"seed":20261027', b'"seed":20261028')
        tampered_path.write_bytes(payload)
        with self.assertRaisesRegex(NS30KContractError, "file SHA256 mismatch"):
            NS30KSamplingTable(tampered_path, self.table_file_sha256)

        changed_manifest = self.root / "changed_manifest.json"
        changed = dict(self.manifest)
        changed["source_commit"] = "tampered"
        changed_manifest.write_text(json.dumps(changed), encoding="utf-8")
        with self.assertRaisesRegex(NS30KContractError, "Manifest SHA256 mismatch"):
            self._table().verify_assets(changed_manifest, self.pdc_path)

    def test_off_and_on_arms_accept_the_same_arm_independent_table(self):
        with _working_directory(self.root):
            off_pdc = validate_ns30k_training_configuration(
                self._dataset(), self._opt(), self._args("OFF"), self._table()
            ).resolve()
            on_pdc = validate_ns30k_training_configuration(
                self._dataset(), self._opt(), self._args("ON"), self._table()
            ).resolve()
            expected_pdc = self.pdc_path.resolve()
        self.assertEqual(off_pdc, expected_pdc)
        self.assertEqual(on_pdc, expected_pdc)
        self.assertNotIn("arm", self.document)

    def test_formal_configuration_rejects_every_required_contract_drift(self):
        cases = []
        bad_opt = self._opt()
        bad_opt.iterations = 29_999
        cases.append((self._dataset(), bad_opt, self._args("OFF")))
        bad_opt = self._opt()
        bad_opt.position_lr_max_steps = 20_000
        cases.append((self._dataset(), bad_opt, self._args("OFF")))
        for field, value in (
            ("binocular_consistency", True),
            ("disable_training_time_evaluation", False),
            ("start_checkpoint", "old.pth"),
            ("stop_after_checkpoint_iter", 21_000),
            ("i1_b1_weak_continuation", True),
            ("paired_sampling_tape", "legacy.json"),
        ):
            bad_args = self._args("OFF")
            setattr(bad_args, field, value)
            cases.append((self._dataset(), self._opt(), bad_args))
        for field, value in (
            ("eval", False),
            ("n_views", 4),
            ("view_manifest", ""),
            ("suffix", "wrong"),
            ("initial_opacity_arm", "confidence"),
            ("initial_opacity_scores", "scores.npy"),
        ):
            bad_dataset = self._dataset()
            setattr(bad_dataset, field, value)
            cases.append((bad_dataset, self._opt(), self._args("OFF")))
        for field, value in (
            ("enable_oe_structural_loss", False),
            ("oe_depth_cache_dir", ""),
            ("oe_structural_start_iter", 0),
            ("oe_structural_end_iter", 8001),
            ("oe_structural_weight", 0.09),
            ("oe_structure_gate_mode", "none"),
            ("oe_stable_quantile", 0.7),
        ):
            bad_args = self._args("ON")
            setattr(bad_args, field, value)
            cases.append((self._dataset(), self._opt(), bad_args))
        off_with_i1 = self._args("OFF")
        off_with_i1.enable_oe_structural_loss = True
        cases.append((self._dataset(), self._opt(), off_with_i1))

        with _working_directory(self.root):
            for index, (dataset, opt, args) in enumerate(cases):
                with self.subTest(case=index):
                    with self.assertRaises(NS30KContractError):
                        validate_ns30k_training_configuration(
                            dataset, opt, args, self._table()
                        )

    def test_every_frozen_common_parameter_drift_is_rejected(self):
        opt_drifts = (
            ("position_lr_init", 0.00017),
            ("position_lr_final", 0.0000017),
            ("lambda_dssim", 0.21),
            ("densify_from_iter", 999),
            ("densify_until_iter", 15_001),
            ("densification_interval", 101),
            ("densify_grad_threshold", 0.0003),
            ("random_background", True),
        )
        dataset_drifts = (("resolution", 4), ("sh_degree", 2))
        arg_drifts = (
            ("dropout_factor", 0.19),
            ("sigma_noise", 0.01),
            ("opacity_decay", False),
            ("opacity_decay_factor", 0.994),
        )
        cases = []
        for field, value in opt_drifts:
            opt = self._opt()
            setattr(opt, field, value)
            cases.append((self._dataset(), opt, self._args("OFF"), field))
        for field, value in dataset_drifts:
            dataset = self._dataset()
            setattr(dataset, field, value)
            cases.append((dataset, self._opt(), self._args("OFF"), field))
        for field, value in arg_drifts:
            args = self._args("OFF")
            setattr(args, field, value)
            cases.append((self._dataset(), self._opt(), args, field))

        with _working_directory(self.root):
            for dataset, opt, args, field in cases:
                with self.subTest(field=field):
                    with self.assertRaises(NS30KContractError):
                        validate_ns30k_training_configuration(
                            dataset, opt, args, self._table()
                        )

    def test_i1_densification_stat_preservation_is_arm_specific(self):
        with _working_directory(self.root):
            for arm, invalid in (("OFF", True), ("ON", False)):
                args = self._args(arm)
                args.oe_preserve_baseline_densification_stats = invalid
                with self.subTest(arm=arm):
                    with self.assertRaisesRegex(
                        NS30KContractError, "densification statistics|densification-stat"
                    ):
                        validate_ns30k_training_configuration(
                            self._dataset(), self._opt(), args, self._table()
                        )

    def test_save_and_checkpoint_nodes_are_exact_and_only_final_save_may_duplicate(self):
        accepted_saves = (
            [5_000, 10_000, 20_000, 30_000],
            [5_000, 10_000, 20_000, 30_000, 30_000],
        )
        with _working_directory(self.root):
            for saves in accepted_saves:
                args = self._args("OFF")
                args.save_iterations = list(saves)
                validate_ns30k_training_configuration(
                    self._dataset(), self._opt(), args, self._table()
                )

            invalid_pairs = (
                ([5_000, 10_000, 30_000], [5_000, 10_000, 20_000, 30_000]),
                ([5_000, 10_000, 20_000, 25_000, 30_000], [5_000, 10_000, 20_000, 30_000]),
                ([5_000, 10_000, 10_000, 20_000, 30_000], [5_000, 10_000, 20_000, 30_000]),
                ([5_000, 10_000, 20_000, 30_000], [5_000, 10_000, 20_000]),
                ([5_000, 10_000, 20_000, 30_000], [5_000, 10_000, 20_000, 25_000, 30_000]),
                ([5_000, 10_000, 20_000, 30_000], [5_000, 10_000, 20_000, 30_000, 30_000]),
            )
            for saves, checkpoints in invalid_pairs:
                args = self._args("OFF")
                args.save_iterations = saves
                args.checkpoint_iterations = checkpoints
                with self.subTest(saves=saves, checkpoints=checkpoints):
                    with self.assertRaisesRegex(NS30KContractError, "save/checkpoint nodes"):
                        validate_ns30k_training_configuration(
                            self._dataset(), self._opt(), args, self._table()
                        )

    def test_output_directory_allows_only_empty_or_train_log_only(self):
        nonexistent = self.root / "safe_nonexistent"
        validate_ns30k_output_directory(nonexistent)

        empty = self.root / "safe_empty"
        empty.mkdir()
        validate_ns30k_output_directory(empty)

        log_only = self.root / "safe_log_only"
        log_only.mkdir()
        (log_only / "train.log").write_text("launcher started\n", encoding="utf-8")
        validate_ns30k_output_directory(log_only)

        artifact_names = (
            "cfg_args",
            "chkpnt5000.pth",
            "point_cloud",
            "ns30k_sampling_trace.csv",
            "unrelated_previous_output.txt",
        )
        for index, name in enumerate(artifact_names):
            output = self.root / f"unsafe_output_{index}"
            output.mkdir()
            (output / "train.log").write_text("keep\n", encoding="utf-8")
            artifact = output / name
            if name == "point_cloud":
                artifact.mkdir()
            else:
                artifact.write_text("old", encoding="utf-8")
            with self.subTest(name=name):
                with self.assertRaisesRegex(NS30KContractError, "historical experiment artifacts"):
                    validate_ns30k_output_directory(output)

    def test_sequential_consumption_and_scalar_only_trace_are_auditable(self):
        table = self._table()
        first = table.consume(1)
        self.assertEqual(first["iteration"], 1)
        with self.assertRaises(NS30KContractError):
            table.consume(3)
        with self.assertRaises(NS30KContractError):
            table.assert_complete()

        trace_dir = self.root / "trace"
        trace_dir.mkdir()
        logger = NS30KTraceLogger(trace_dir)
        logger.record(first, first["camera_image_name"], table)
        header = (trace_dir / "ns30k_sampling_trace.csv").read_text(
            encoding="utf-8"
        ).splitlines()[0]
        self.assertNotIn("mask", header.lower())
        self.assertNotIn("tensor", header.lower())
        self.assertIn("sampling_file_sha256", header)
        self.assertIn("actual_camera_image_name", header)

    def test_dropout_pairing_never_claims_elementwise_identity_for_shape_changes(self):
        self.assertEqual(dropout_pairing_level((100,), (100,)), DROPOUT_PAIRING_LEVEL)
        self.assertEqual(dropout_pairing_level((100,), (120,)), DROPOUT_PAIRING_LEVEL)
        self.assertNotIn("ELEMENTWISE_IDENTICAL", DROPOUT_PAIRING_LEVEL)

    def test_legacy_p7_module_and_arguments_remain_unchanged(self):
        self.assertEqual(sha256_file(LEGACY_P7_PATH), LEGACY_P7_SHA256)
        source = TRAIN_PATH.read_text(encoding="utf-8")
        for original in (
            'parser.add_argument("--paired_sampling_tape", type=str, default="")',
            'parser.add_argument("--paired_sampling_tape_sha256", type=str, default="")',
            'parser.add_argument("--paired_trace_dir", type=str, default="")',
            'parser.add_argument("--paired_arm", choices=("OFF", "ON"), default=None)',
            "args.stop_after_checkpoint_iter != PAIRED_END_ITERATION",
            "if paired_sampling_enabled:",
        ):
            self.assertIn(original, source)

    def test_new_protocol_defaults_are_inert_and_existing_cli_switches_survive(self):
        parser = _train_cli_parser()
        defaults = parser.parse_args([])
        self.assertEqual(defaults.ns30k_sampling_tape, "")
        self.assertEqual(defaults.ns30k_sampling_tape_sha256, "")
        self.assertIsNone(defaults.ns30k_arm)
        self.assertIs(defaults.binocular_consistency, True)
        self.assertIs(defaults.disable_training_time_evaluation, False)

        explicit = parser.parse_args(
            [
                "--ns30k_sampling_tape", "table.json",
                "--ns30k_sampling_tape_sha256", "abc",
                "--ns30k_arm", "OFF",
                "--no-binocular-consistency",
                "--disable_training_time_evaluation",
            ]
        )
        self.assertIs(explicit.binocular_consistency, False)
        self.assertIs(explicit.disable_training_time_evaluation, True)
        self.assertEqual(explicit.ns30k_arm, "OFF")

        training = next(
            node
            for node in _train_tree().body
            if isinstance(node, ast.FunctionDef) and node.name == "training"
        )
        training_source = ast.unparse(training)
        self.assertIn("else:\n            viewpoint_cam = random.choice(viewpoint_stack)", training_source)
        self.assertLess(
            training_source.index("validate_ns30k_training_configuration"),
            training_source.index("prepare_output_and_logger"),
        )

    def test_train_wiring_uses_name_selection_and_existing_generator_interfaces(self):
        source = TRAIN_PATH.read_text(encoding="utf-8")
        self.assertIn('camera_name = ns30k_entry["camera_image_name"]', source)
        self.assertIn("viewpoint_cam = scheduled_camera_by_name[camera_name]", source)
        for purpose in SEED_PURPOSES:
            self.assertIn(
                f'seeded_ns30k_cuda_generator("{purpose}")',
                source,
            )
        self.assertIn("dropout_generator=dropout_generator", source)
        self.assertIn("opacity_noise_generator=opacity_noise_generator", source)
        self.assertIn("random_generator=densification_generator", source)


if __name__ == "__main__":
    unittest.main()
