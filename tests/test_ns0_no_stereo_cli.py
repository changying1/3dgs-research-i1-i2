import argparse
import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "train.py"
OE_ARGUMENTS_PATH = ROOT / "utils" / "evidence_structural_loss.py"


def _oe_argument_registrar():
    tree = ast.parse(
        OE_ARGUMENTS_PATH.read_text(encoding="utf-8"),
        filename=str(OE_ARGUMENTS_PATH),
    )
    selected = [
        node
        for node in tree.body
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "OE_STRUCTURE_GATE_MODES"
                for target in node.targets
            )
        )
        or (isinstance(node, ast.FunctionDef) and node.name == "add_oe_structural_arguments")
    ]
    assert len(selected) == 2, "I1 argument registrar could not be extracted"
    namespace = {}
    module = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
    exec(compile(module, str(OE_ARGUMENTS_PATH), "exec"), namespace)
    return namespace["add_oe_structural_arguments"]


def _main_body():
    tree = ast.parse(TRAIN_PATH.read_text(encoding="utf-8"), filename=str(TRAIN_PATH))
    for node in tree.body:
        if isinstance(node, ast.If) and ast.unparse(node.test) == "__name__ == '__main__'":
            return node.body
    raise AssertionError("train.py has no __main__ argument-registration block")


def _is_registration_statement(node):
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
    if isinstance(function, ast.Attribute) and isinstance(function.value, ast.Name):
        return function.attr == "add_argument" and function.value.id in {
            "parser",
            "binocular_group",
        }
    return isinstance(function, ast.Name) and function.id == "add_oe_structural_arguments"


def _train_cli_parser():
    """Execute train.py's real registration AST without importing its training stack."""
    parser = argparse.ArgumentParser(prog="train.py")
    namespace = {
        "parser": parser,
        "add_oe_structural_arguments": _oe_argument_registrar(),
    }
    statements = [node for node in _main_body() if _is_registration_statement(node)]
    assert statements, "no train.py argument registrations were extracted"
    module = ast.fix_missing_locations(ast.Module(body=statements, type_ignores=[]))
    exec(compile(module, str(TRAIN_PATH), "exec"), namespace)
    return parser


def _parse(arguments=()):
    return _train_cli_parser().parse_args(list(arguments))


class TestNoStereoCLI(unittest.TestCase):
    def test_default_keeps_binocular_consistency_enabled(self):
        self.assertIs(_parse().binocular_consistency, True)

    def test_legacy_explicit_enable_remains_compatible(self):
        self.assertIs(_parse(["--binocular_consistency"]).binocular_consistency, True)

    def test_explicit_no_stereo_switch_disables_binocular_consistency(self):
        self.assertIs(_parse(["--no-binocular-consistency"]).binocular_consistency, False)

    def test_enable_and_disable_switches_are_mutually_exclusive(self):
        with self.assertRaises(SystemExit) as error:
            _parse(["--binocular_consistency", "--no-binocular-consistency"])
        self.assertEqual(error.exception.code, 2)

    def test_shift_camera_start_default_is_unchanged(self):
        self.assertEqual(_parse().shift_cam_start, 20_000)

    def test_training_branch_is_still_gated_by_binocular_flag_and_shift_start(self):
        tree = ast.parse(TRAIN_PATH.read_text(encoding="utf-8"), filename=str(TRAIN_PATH))
        expected = ast.dump(
            ast.parse(
                "args.binocular_consistency and iteration > args.shift_cam_start",
                mode="eval",
            ).body,
            include_attributes=False,
        )
        matching_conditions = [
            node.test
            for node in ast.walk(tree)
            if isinstance(node, ast.If)
            and ast.dump(node.test, include_attributes=False) == expected
        ]
        self.assertEqual(len(matching_conditions), 1)

    def test_dropout_noise_opacity_and_i1_defaults_are_unchanged(self):
        args = _parse()
        self.assertEqual(args.dropout_factor, 0.2)
        self.assertEqual(args.sigma_noise, 0.0)
        self.assertIs(args.opacity_decay, True)
        self.assertEqual(args.opacity_decay_factor, 0.995)
        self.assertIs(args.enable_oe_structural_loss, False)
        self.assertIs(args.oe_preserve_baseline_densification_stats, False)
        self.assertEqual(args.oe_structural_weight, 0.0)
        self.assertEqual(args.oe_structural_start_iter, 0)
        self.assertEqual(args.oe_structural_end_iter, 30_000)
        self.assertEqual(args.oe_stable_quantile, 0.80)
        self.assertEqual(args.oe_structure_gate_mode, "stable_evidence")
        self.assertEqual(args.oe_conflict_threshold, 0.75)
        self.assertIs(args.i1_b1_weak_continuation, False)
        self.assertEqual(args.i1_b1_weak_weight, 0.02)
        self.assertEqual(args.i1_b1_weak_start_iter, 8_001)
        self.assertEqual(args.i1_b1_weak_end_iter, 30_000)
        self.assertEqual(args.i1_b1_weak_weight_schedule, "constant")
        self.assertEqual(args.i1_b1_weak_evidence_gamma, 1.0)
        self.assertEqual(args.oe_depth_cache_dir, "")

    def test_old_and_new_switch_forms_leave_all_other_parsed_values_identical(self):
        common = [
            "--shift_cam_start", "1234",
            "--dropout_factor", "0.15",
            "--sigma_noise", "0.03",
            "--opacity_decay_factor", "0.99",
            "--cam_trans_dist", "0.25",
            "--dataset_name", "CUSTOM",
            "--n_views", "5",
            "--suffix", "ns0-audit",
            "--oe_structural_weight", "0.07",
        ]
        legacy = vars(_parse(["--binocular_consistency", *common]))
        no_stereo = vars(_parse(["--no-binocular-consistency", *common]))
        self.assertIs(legacy.pop("binocular_consistency"), True)
        self.assertIs(no_stereo.pop("binocular_consistency"), False)
        self.assertEqual(legacy, no_stereo)


if __name__ == "__main__":
    unittest.main()
