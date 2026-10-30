import argparse
import ast
import hashlib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "train.py"


def _tree():
    return ast.parse(TRAIN_PATH.read_text(encoding="utf-8"), filename=str(TRAIN_PATH))


def _main_body():
    for node in _tree().body:
        if isinstance(node, ast.If) and ast.unparse(node.test) == "__name__ == '__main__'":
            return node.body
    raise AssertionError("train.py has no __main__ argument-registration block")


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


class _Scalar:
    def __init__(self, value):
        self.value = float(value)

    def item(self):
        return self.value

    def mean(self):
        return self

    def double(self):
        return self

    def __add__(self, other):
        return _Scalar(self.value + float(other))

    __radd__ = __add__

    def __truediv__(self, other):
        return _Scalar(self.value / float(other))

    def __float__(self):
        return self.value

    def __str__(self):
        return str(self.value)


class _Image:
    def to(self, _device):
        return self


class _Camera:
    original_image = _Image()
    image_name = "cpu-camera"


class _Cuda:
    empty_cache_calls = 0

    @classmethod
    def empty_cache(cls):
        cls.empty_cache_calls += 1


class _Torch:
    cuda = _Cuda

    @staticmethod
    def clamp(value, _minimum, _maximum):
        return value


class _Writer:
    def __init__(self):
        self.scalars = []

    def add_scalar(self, name, value, iteration):
        self.scalars.append((name, value, iteration))


class _Scene:
    def __init__(self):
        self.test_calls = 0
        self.train_calls = 0
        self.gaussians = object()
        self.cameras = [_Camera()]

    def getTestCameras(self):
        self.test_calls += 1
        return self.cameras

    def getTrainCameras(self):
        self.train_calls += 1
        return self.cameras


class _PoisonScene:
    gaussians = object()

    def getTestCameras(self):
        raise AssertionError("disabled evaluation accessed test cameras")

    def getTrainCameras(self):
        raise AssertionError("disabled evaluation accessed training cameras")


def _training_report():
    function = next(
        node
        for node in _tree().body
        if isinstance(node, ast.FunctionDef) and node.name == "training_report"
    )
    namespace = {
        "Scene": object,
        "torch": _Torch,
        "psnr": lambda *_args: _Scalar(1.0),
        "ssim": lambda *_args: _Scalar(1.0),
        "lpips": lambda *_args, **_kwargs: _Scalar(1.0),
    }
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    exec(compile(module, str(TRAIN_PATH), "exec"), namespace)
    return namespace["training_report"]


def _invoke_report(iteration, testing_iterations, scene, writer=None, enable_evaluation=None):
    render_calls = []

    def render(*_args):
        render_calls.append(iteration)
        return {"render": _Image()}

    arguments = (
        writer, iteration, _Scalar(0.1), _Scalar(0.2),
        lambda *_args: _Scalar(0.3), 1.5, testing_iterations,
        scene, render, (),
    )
    report = _training_report()
    if enable_evaluation is None:
        report(*arguments)
    else:
        report(*arguments, enable_evaluation=enable_evaluation)
    return render_calls


def _digest(nodes):
    normalized = "\n".join(ast.dump(node, include_attributes=False) for node in nodes)
    return hashlib.sha256(normalized.encode()).hexdigest()


class TestTrainingTimeEvaluationIsolation(unittest.TestCase):
    def test_cli_default_is_compatible_and_explicit_switch_enables_isolation(self):
        parser = _train_cli_parser()
        self.assertIs(parser.parse_args([]).disable_training_time_evaluation, False)
        self.assertIs(
            parser.parse_args(["--disable_training_time_evaluation"])
            .disable_training_time_evaluation,
            True,
        )

    def test_old_and_new_cli_forms_preserve_every_other_direct_argument(self):
        common = [
            "--no-binocular-consistency",
            "--dropout_factor", "0.15",
            "--sigma_noise", "0.03",
            "--shift_cam_start", "1234",
            "--test_iterations", "1000", "5000",
        ]
        old = vars(_train_cli_parser().parse_args(common))
        isolated = vars(
            _train_cli_parser().parse_args(
                ["--disable_training_time_evaluation", *common]
            )
        )
        self.assertIs(old.pop("disable_training_time_evaluation"), False)
        self.assertIs(isolated.pop("disable_training_time_evaluation"), True)
        self.assertEqual(old, isolated)

    def test_default_runtime_keeps_periodic_and_configured_evaluation_conditions(self):
        quiet_scene = _Scene()
        self.assertEqual(_invoke_report(999, [], quiet_scene), [])
        self.assertEqual(quiet_scene.test_calls, 0)

        periodic_scene = _Scene()
        self.assertGreater(len(_invoke_report(1000, [], periodic_scene)), 0)
        self.assertEqual(periodic_scene.test_calls, 1)

        configured_scene = _Scene()
        self.assertGreater(len(_invoke_report(123, [123], configured_scene)), 0)
        self.assertEqual(configured_scene.test_calls, 1)

    def test_isolation_blocks_all_formal_boundary_iterations(self):
        for iteration in (1000, 5000, 10_000, 20_000, 30_000):
            with self.subTest(iteration=iteration):
                renders = _invoke_report(
                    iteration,
                    [iteration],
                    _PoisonScene(),
                    enable_evaluation=False,
                )
                self.assertEqual(renders, [])

    def test_empty_test_iterations_is_not_treated_as_isolation(self):
        scene = _Scene()
        renders = _invoke_report(1000, [], scene)
        self.assertGreater(len(renders), 0)
        self.assertEqual(scene.test_calls, 1)

    def test_training_scalars_are_logged_while_evaluation_is_disabled(self):
        writer = _Writer()
        renders = _invoke_report(
            1000, [1000], _PoisonScene(), writer=writer, enable_evaluation=False
        )
        self.assertEqual(renders, [])
        self.assertEqual(
            [name for name, _value, _iteration in writer.scalars],
            [
                "train_loss_patches/l1_loss",
                "train_loss_patches/total_loss",
                "iter_time",
            ],
        )

    def test_training_loop_passes_only_the_new_evaluation_control_keyword(self):
        calls = [
            node
            for node in ast.walk(_tree())
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "training_report"
        ]
        self.assertEqual(len(calls), 1)
        keywords = {keyword.arg: keyword.value for keyword in calls[0].keywords}
        self.assertEqual(set(keywords), {"enable_evaluation"})
        self.assertEqual(
            ast.unparse(keywords["enable_evaluation"]),
            "not args.disable_training_time_evaluation",
        )

    def test_binocular_dropout_i1_and_densification_asts_are_unchanged(self):
        nodes = list(ast.walk(_tree()))
        binocular = [
            node
            for node in nodes
            if isinstance(node, ast.If)
            and ast.unparse(node.test)
            == "args.binocular_consistency and iteration > args.shift_cam_start"
        ]
        i1_calls = [
            node
            for node in nodes
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "apply_oe_structural_training_loss"
        ]
        densification_calls = [
            node
            for node in nodes
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "densify_and_prune"
        ]
        training_renders = [
            node
            for node in nodes
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "render"
            and any(keyword.arg == "dropout_factor" for keyword in node.keywords)
        ]
        self.assertEqual(
            _digest(binocular),
            "90d4614479bb6ac4934f66ae4abae3b34d4956025960c80fec70c693906beea6",
        )
        self.assertEqual(
            _digest(i1_calls),
            "78f2579632b62aff77898b9fc4f308d11003a34a7b15687c52027ea3706a1810",
        )
        self.assertEqual(
            _digest(densification_calls),
            "545c29ec8015650871b719a9b0f887b8adad3eb14a634863e87e827773dfc652",
        )
        self.assertEqual(
            _digest(training_renders),
            "ee1c5c386f7e4f231e5187ee6e01deff7f2f898ce317ba31f0d064ebb308df8d",
        )


if __name__ == "__main__":
    unittest.main()
