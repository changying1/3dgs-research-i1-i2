#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import random
import json
import copy
from pathlib import Path
import numpy as np

import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, SmoothLoss
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from lpipsPyTorch import lpips
from utils.image_utils import psnr
from utils.graphics_utils import inverse_warp_images
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from llff_view_protocol import canonical_image_name, load_view_manifest
from utils.densification_diagnostics import (
    DensificationEventLogger,
    add_densification_diagnostics_argument,
)
from utils.evidence_structural_loss import (
    add_oe_structural_arguments,
)
from utils.oe_depth_cache import create_oe_depth_cache_if_enabled
from utils.oe_densification import (
    add_densification_stats_with_baseline_gradient,
    effective_densify_until_iter,
    map_dropout_visibility_to_full_mask,
    maybe_extract_baseline_viewspace_gradient,
)
from utils.oe_training import (
    I1Diagnostics,
    apply_oe_structural_training_loss,
    validate_oe_training_configuration,
)
from utils.resume_replay_diagnostics import ResumeReplayDiagnostics
from utils.ns30k_sampling import (
    NS30KContractError,
    NS30KSamplingTable,
    NS30KTraceLogger,
    validate_ns30k_training_configuration,
)
from utils.paired_sampling import (
    END_ITERATION as PAIRED_END_ITERATION,
    START_ITERATION as PAIRED_START_ITERATION,
    PairedRunLogger,
    SamplingContractError,
    SamplingTape,
    gaussian_model_state_sha256,
    save_phase,
    sha256_file,
    tensor_summary,
)
from tools.probe_i1_gradient_conflict import (
    align_gradient_groups,
    compare_gradient_parameter_spaces,
    EXPECTED_CHECKPOINT_SHA256,
    pair_aligned_gradient_groups,
    gradient_conflict_stats,
    gradient_pair_group_stats,
    readonly_gradient_components,
    validate_gradient_additivity,
    weighted_i1_term_from_contract,
    write_probe_artifacts_to_directory,
)
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
TENSORBOARD_FOUND = False

NS30K_SMOKE_ITERATIONS = 10
NS30K_SMOKE_AUTH_ENV = "NS30K_SMOKE_GPU_AUTHORIZED"


def _write_json_exclusive(path, payload):
    path = Path(path)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _audit_state_equal(left, right):
    if torch.is_tensor(left) and torch.is_tensor(right):
        return left.dtype == right.dtype and left.shape == right.shape and torch.equal(left, right)
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return left.dtype == right.dtype and left.shape == right.shape and np.array_equal(left, right)
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _audit_state_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (tuple, list)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(
            _audit_state_equal(a, b) for a, b in zip(left, right)
        )
    return left == right


def write_ns30k_initial_state_audit(scene, gaussians, args, ns30k_table):
    """Record the read-only initialized Gaussian state before iteration one."""
    python_rng_before = random.getstate()
    numpy_rng_before = copy.deepcopy(np.random.get_state())
    torch_rng_before = torch.random.get_rng_state().clone()
    cuda_rng_before = [state.clone() for state in torch.cuda.get_rng_state_all()]
    optimizer_before = copy.deepcopy(gaussians.optimizer.state_dict())
    before_count = int(gaussians._xyz.shape[0])
    initial_sha256 = gaussian_model_state_sha256(gaussians)
    after_count = int(gaussians._xyz.shape[0])
    after_sha256 = gaussian_model_state_sha256(gaussians)
    rng_unchanged = (
        python_rng_before == random.getstate()
        and _audit_state_equal(numpy_rng_before, np.random.get_state())
        and torch.equal(torch_rng_before, torch.random.get_rng_state())
        and _audit_state_equal(cuda_rng_before, torch.cuda.get_rng_state_all())
    )
    optimizer_unchanged = _audit_state_equal(
        optimizer_before, gaussians.optimizer.state_dict()
    )
    if (
        before_count != after_count
        or initial_sha256 != after_sha256
        or not rng_unchanged
        or not optimizer_unchanged
    ):
        raise NS30KContractError(
            "initial Gaussian, optimizer, or RNG state changed during read-only hashing"
        )
    payload = {
        "audit_version": "ns30k-initial-state-audit-v1",
        "scene": ns30k_table.document["scene"],
        "arm": args.ns30k_arm,
        "experiment_arm": "B0+I1" if args.ns30k_arm == "ON" else "B0",
        "initialization_seed": 0,
        "manifest_sha256": ns30k_table.document["manifest_sha256"],
        "pdc_sha256": ns30k_table.document["pdc_sha256"],
        "sampling_table_file_sha256": ns30k_table.file_sha256,
        "initial_gaussian_count": before_count,
        "initial_gaussian_model_sha256": initial_sha256,
        "train_py_sha256": sha256_file(Path(__file__).resolve()),
        "python_version": sys.version,
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "read_only_verification": {
            "gaussian_state_unchanged": initial_sha256 == after_sha256,
            "gaussian_count_unchanged": before_count == after_count,
            "optimizer_state_unchanged": optimizer_unchanged,
            "python_numpy_torch_cuda_rng_state_unchanged": rng_unchanged,
        },
        "SMOKE_ONLY": True,
        "NOT_FORMAL_RESULT": True,
    }
    _write_json_exclusive(
        Path(scene.model_path) / "ns30k_initial_state_audit.json", payload
    )
    return payload


def _validate_resume_device_consistency(gaussians, device):
    """Fail early if any restored Gaussian/optimizer tensor stayed on CPU."""
    trainable_parameters = {
        "xyz": gaussians._xyz,
        "features_dc": gaussians._features_dc,
        "features_rest": gaussians._features_rest,
        "opacity": gaussians._opacity,
        "scaling": gaussians._scaling,
        "rotation": gaussians._rotation,
        "max_radii2D": gaussians.max_radii2D,
        "xyz_gradient_accum": gaussians.xyz_gradient_accum,
        "denom": gaussians.denom,
    }
    mismatches = [
        f"{name}={parameter.device}"
        for name, parameter in trainable_parameters.items()
        if parameter.device != device
    ]
    for parameter, state in gaussians.optimizer.state.items():
        for state_name, value in state.items():
            if torch.is_tensor(value) and value.device != device:
                mismatches.append(
                    f"optimizer[{id(parameter)}].{state_name}={value.device}"
                )
    if mismatches:
        raise RuntimeError(
            "Checkpoint restore left tensors on the wrong device: "
            + ", ".join(mismatches)
        )


def _checkpoint_safe_globals():
    """Allow only the NumPy scalar/dtype classes used by legacy checkpoints."""
    safe = [np._core.multiarray.scalar, np.dtype]
    for name in ("Float32DType", "Float64DType"):
        dtype = getattr(getattr(np, "dtypes", None), name, None)
        if dtype is not None:
            safe.append(dtype)
    return safe


def _probe_parameter_groups(gaussians):
    """Return the actual trainable Gaussian tensors, without cloning them."""
    return {
        "xyz": gaussians._xyz,
        "features_dc": gaussians._features_dc,
        "features_rest": gaussians._features_rest,
        "opacity": gaussians._opacity,
        "scaling": gaussians._scaling,
        "rotation": gaussians._rotation,
    }


def _probe_group_gradients(group_names, parameters, gradients):
    return {
        name: gradient
        for name, gradient in zip(group_names, gradients)
    }


def _probe_loss_tensor(value, reference):
    return value if torch.is_tensor(value) else reference * 0.0 + float(value)


def _probe_snapshot(gaussians):
    return {
        name: tensor.detach().clone()
        for name, tensor in _probe_parameter_groups(gaussians).items()
    }


def _assert_probe_snapshot(gaussians, snapshot):
    for name, before in snapshot.items():
        after = _probe_parameter_groups(gaussians)[name].detach()
        if not torch.equal(before, after):
            raise RuntimeError(f"read-only I1 probe changed Gaussian parameter: {name}")


def _probe_optimizer_snapshot(gaussians):
    """Deep-copy optimizer metadata/state for the read-only probe contract."""
    return copy.deepcopy(gaussians.optimizer.state_dict())


def _assert_probe_optimizer_snapshot(gaussians, snapshot):
    if gaussians.optimizer.state_dict().keys() != snapshot.keys():
        raise RuntimeError("read-only I1 probe changed optimizer state structure")
    current = gaussians.optimizer.state_dict()
    if current["param_groups"] != snapshot["param_groups"]:
        raise RuntimeError("read-only I1 probe changed optimizer param groups")
    for parameter_id, before_state in snapshot["state"].items():
        after_state = current["state"].get(parameter_id)
        if after_state is None or after_state.keys() != before_state.keys():
            raise RuntimeError("read-only I1 probe changed optimizer state entries")
        for name, before in before_state.items():
            after = after_state[name]
            if torch.is_tensor(before):
                if not torch.equal(before, after):
                    raise RuntimeError(f"read-only I1 probe changed optimizer tensor: {name}")
            elif before != after:
                raise RuntimeError(f"read-only I1 probe changed optimizer value: {name}")


def _validate_probe_camera_split(dataset, scene):
    """Require loaded probe cameras to match the frozen manifest in order."""
    manifest = load_view_manifest(
        dataset.view_manifest,
        expected_dataset="LLFF",
        expected_scene=os.path.basename(os.path.normpath(dataset.source_path)),
        expected_view_count=3,
    )
    expected_names = [canonical_image_name(name) for name in manifest["selected_train_images"]]
    expected_stems = [Path(name).stem for name in expected_names]
    actual_stems = [str(camera.image_name) for camera in scene.getTrainCameras()]
    if actual_stems != expected_stems:
        raise RuntimeError(
            "I1 gradient probe training cameras do not match frozen manifest name/count/order: "
            f"expected={expected_stems}, actual={actual_stems}"
        )
    return expected_names


def _collect_i1_gradient_probe(
    *,
    losses,
    total_loss,
    gaussians,
    camera,
    iteration,
    checkpoint_sha256,
    random_contract,
    oe_struct_loss,
    oe_stats,
):
    """Collect gradients from one shared forward graph, never calling backward."""
    if oe_struct_loss is None or oe_stats is None:
        raise RuntimeError("I1 gradient probe requires an active I1 structural loss")
    direct_weighted_i1_loss, applied_weight = weighted_i1_term_from_contract(
        total_loss,
        losses["baseline_total"],
        oe_struct_loss,
        oe_stats,
    )

    groups = _probe_parameter_groups(gaussians)
    group_names = list(groups)
    parameters = tuple(groups.values())
    all_losses = dict(losses)
    all_losses["baseline_repeat"] = losses["baseline_total"]
    all_losses["weighted_i1"] = direct_weighted_i1_loss
    component_grads = readonly_gradient_components(all_losses, parameters)
    baseline_grads = component_grads["baseline_total"]
    i1_grads = component_grads["weighted_i1"]
    additivity = validate_gradient_additivity(
        {name: component_grads[name] for name in ("rgb", "binocular", "alpha")},
        baseline_grads,
        parameter_group_names=group_names,
        baseline_repeat=component_grads["baseline_repeat"],
        atol=1e-6,
        rtol=1e-5,
    )

    def group_map(grads):
        return _probe_group_gradients(group_names, parameters, grads)

    baseline_alignment = align_gradient_groups(groups, group_map(baseline_grads))
    baseline_flat = baseline_alignment["valid"]
    result = {
        "event": "gradient_probe",
        "iteration": int(iteration),
        "camera_id": int(getattr(camera, "uid", -1)),
        "camera_image_name": str(getattr(camera, "image_name", "")),
        "checkpoint_sha256": checkpoint_sha256,
        "random_contract": dict(random_contract),
        "i1_applied_weight": applied_weight,
        "raw_struct_loss": float(oe_struct_loss.detach().item()),
        "weighted_i1_loss": float(direct_weighted_i1_loss.detach().item()),
        "loss_additivity": additivity,
        "global": {},
        "by_parameter_group": {},
    }
    for name in ("rgb", "binocular", "alpha", "baseline_total", "baseline_repeat", "weighted_i1"):
        current = component_grads[name] if name != "weighted_i1" else i1_grads
        current_alignment = align_gradient_groups(groups, group_map(current))
        current_flat = current_alignment["valid"]
        result["global"][name] = {
            "finite_gradient_elements": int(torch.isfinite(current_flat).sum().item()) if current_flat is not None else 0,
            "nonfinite_gradient_elements": int((~torch.isfinite(current_flat)).sum().item()) if current_flat is not None else 0,
            "gradient_norm": float(torch.linalg.vector_norm(current_flat).item()) if current_flat is not None else None,
            "missing_parameter_groups": current_alignment["missing"],
        }
        if current_flat is None:
            result["global"][name]["status"] = "MISSING_GRADIENT"
        elif result["global"][name]["nonfinite_gradient_elements"]:
            result["global"][name]["status"] = "UNVERIFIED"
        else:
            result["global"][name]["status"] = "PASS"
    if baseline_flat is None:
        raise RuntimeError("baseline gradient is entirely missing")
    baseline_group_map = group_map(baseline_grads)
    for name, target_grads in (
        [("rgb", component_grads["rgb"]),
         ("binocular", component_grads["binocular"]),
         ("alpha", component_grads["alpha"]),
         ("weighted_i1", i1_grads)]
    ):
        result["global"][f"{name}_vs_baseline"] = compare_gradient_parameter_spaces(
            groups, baseline_group_map, group_map(target_grads)
        )
    for name in ("rgb", "binocular", "alpha"):
        result["by_parameter_group"][f"{name}_vs_baseline"] = gradient_pair_group_stats(
            _probe_group_gradients(group_names, parameters, baseline_grads),
            _probe_group_gradients(group_names, parameters, component_grads[name]),
        )
    result["by_parameter_group"]["weighted_i1_vs_baseline"] = gradient_pair_group_stats(
        baseline_group_map, group_map(i1_grads),
    )
    result["by_parameter_group"]["loss_additivity"] = additivity["by_parameter_group"]
    i1_comparison = result["global"]["weighted_i1_vs_baseline"]
    i1_scope_statuses = {
        scope: i1_comparison[scope].get("status", "UNVERIFIED")
        for scope in ("full_parameter_space", "shared_active_groups")
    }
    nonfinite_components = {
        name: details["nonfinite_gradient_elements"]
        for name, details in result["global"].items()
        if isinstance(details, dict) and "nonfinite_gradient_elements" in details
        and details["nonfinite_gradient_elements"] > 0
    }
    result["i1_gradient_status"] = {
        "status": "PASS" if all(value == "PASS" for value in i1_scope_statuses.values()) else "UNVERIFIED",
        "scope_statuses": i1_scope_statuses,
        "nonfinite_components": nonfinite_components,
    }
    if not additivity.get("pass"):
        result["status"] = additivity.get("status", "UNVERIFIED")
    elif nonfinite_components or result["i1_gradient_status"]["status"] != "PASS":
        result["status"] = "UNVERIFIED"
    else:
        result["status"] = "PASS"
    return result

def training(dataset, opt, pipe, args):
    first_iter = 0
    initial_opacity_arm = getattr(args, "initial_opacity_arm", "control")
    if initial_opacity_arm not in ("control", "confidence", "shuffle"):
        raise ValueError("unsupported initial_opacity_arm: {}".format(initial_opacity_arm))
    p2a_protocol_enabled = bool(getattr(args, "initial_opacity_score_manifest", ""))
    if (initial_opacity_arm != "control" or p2a_protocol_enabled) and args.start_checkpoint:
        raise RuntimeError(
            "I1-NewA-P2A arms must start from initialization; "
            "--start_checkpoint is forbidden"
        )
    probe_enabled = bool(getattr(args, "i1_gradient_probe", False))
    paired_sampling_enabled = bool(getattr(args, "paired_sampling_tape", ""))
    ns30k_enabled = bool(getattr(args, "ns30k_sampling_tape", ""))
    ns30k_smoke_enabled = bool(getattr(args, "ns30k_smoke", False))
    ns30k_table = None
    ns30k_logger = None
    ns30k_initial_audit = None
    ns30k_smoke_observations = []
    paired_tape = None
    paired_logger = None
    paired_terminal_artifacts = {}
    if ns30k_enabled and paired_sampling_enabled:
        raise NS30KContractError(
            "NS-30k and legacy --paired_sampling_tape are mutually exclusive"
        )
    if ns30k_smoke_enabled:
        if not ns30k_enabled:
            raise NS30KContractError("--ns30k_smoke requires the formal NS-30k sampling table")
        if paired_sampling_enabled:
            raise NS30KContractError("NS-30k Smoke cannot use the legacy P7 sampling path")
        if os.environ.get(NS30K_SMOKE_AUTH_ENV) != "1":
            raise NS30KContractError(
                "NS-30k Smoke GPU execution requires independent launcher authorization"
            )
    if bool(getattr(args, "ns30k_sampling_tape_sha256", "")) != ns30k_enabled:
        raise NS30KContractError(
            "--ns30k_sampling_tape and --ns30k_sampling_tape_sha256 must be provided together"
        )
    if bool(getattr(args, "ns30k_arm", None)) != ns30k_enabled:
        raise NS30KContractError(
            "--ns30k_arm is required exactly when an NS-30k sampling table is used"
        )
    if bool(getattr(args, "paired_sampling_tape_sha256", "")) != paired_sampling_enabled:
        raise SamplingContractError(
            "--paired_sampling_tape and --paired_sampling_tape_sha256 must be provided together"
        )
    if paired_sampling_enabled:
        p7c_scenes = {"horns", "orchids"}
        p7b_scenes = {"fortress", "room"}
        if args.paired_scene not in p7b_scenes | p7c_scenes:
            raise SamplingContractError("paired sampling requires an explicitly supported scene")
        if (args.paired_scene in p7c_scenes
                and os.environ.get("P7C_GPU_AUTHORIZED") != "1"):
            raise SamplingContractError(
                "P7-C paired GPU training requires launcher authorization "
                "P7C_GPU_AUTHORIZED=1"
            )
        if (args.paired_scene in p7b_scenes
                and os.environ.get("P7B_GPU_AUTHORIZED") != "1"):
            raise SamplingContractError(
                "paired GPU training requires launcher authorization P7B_GPU_AUTHORIZED=1"
            )
        if args.paired_arm not in ("OFF", "ON"):
            raise SamplingContractError("paired sampling requires --paired_arm OFF or ON")
        if opt.iterations != 30_000 or args.stop_after_checkpoint_iter != PAIRED_END_ITERATION:
            raise SamplingContractError("paired run requires iterations=30000 and stop=21000")
        if list(args.checkpoint_iterations) != [PAIRED_END_ITERATION]:
            raise SamplingContractError("paired run requires checkpoint_iterations exactly [21000]")
        if list(args.save_iterations) != [PAIRED_END_ITERATION]:
            raise SamplingContractError("paired run requires save_iterations exactly [21000]")
        if args.paired_scene in p7c_scenes:
            if int(opt.position_lr_max_steps) != 20_000:
                raise SamplingContractError(
                    "P7-C G2 paired run requires position_lr_max_steps=20000"
                )
            if int(args.shift_cam_start) != 10_000:
                raise SamplingContractError(
                    "P7-C G2 paired run requires shift_cam_start=10000"
                )
        if args.paired_arm == "ON":
            valid_i1 = (
                args.enable_oe_structural_loss
                and int(args.oe_structural_end_iter) < PAIRED_START_ITERATION
                and args.i1_b1_weak_continuation
                and float(args.i1_b1_weak_weight) == 0.02
                and int(args.i1_b1_weak_start_iter) == PAIRED_START_ITERATION
                and int(args.i1_b1_weak_end_iter) == PAIRED_END_ITERATION
            )
            if not valid_i1:
                raise SamplingContractError("ON arm must apply weak I1 weight 0.02 at 20001--21000")
        elif args.enable_oe_structural_loss or args.i1_b1_weak_continuation:
            raise SamplingContractError("OFF arm must not enable or apply I1")
    if ns30k_enabled:
        ns30k_scene = os.path.basename(os.path.normpath(dataset.source_path)).lower()
        ns30k_table = NS30KSamplingTable(
            args.ns30k_sampling_tape,
            args.ns30k_sampling_tape_sha256,
            expected_scene=ns30k_scene,
        )
        validate_ns30k_training_configuration(dataset, opt, args, ns30k_table)
    probe_checkpoint_sha256 = None
    probe_results = []
    probe_group_results = []
    if probe_enabled:
        if os.environ.get("I1_D4_PROBE_AUTHORIZED") != "1":
            raise RuntimeError(
                "I1 gradient probe requires the preflight authorization environment marker"
            )
        if not args.start_checkpoint:
            raise RuntimeError("I1 gradient probe requires --start_checkpoint")
        if not args.enable_oe_structural_loss:
            raise RuntimeError("I1 gradient probe requires the existing I1 loss to be enabled")
        probe_output = os.path.abspath(args.i1_gradient_probe_output)
        probe_artifact_output = os.path.abspath(args.i1_gradient_probe_artifact_output)
        normalized_probe_output = probe_output.replace("\\", "/").lower()
        if "/reports/i1_d4_gradient_conflict" not in normalized_probe_output:
            raise RuntimeError("I1 gradient probe output must be reports/i1_d4_gradient_conflict")
        if os.path.exists(probe_output) and os.listdir(probe_output):
            raise RuntimeError("I1 gradient probe output directory must be absent or empty")
        if os.path.exists(probe_artifact_output):
            raise RuntimeError("I1 gradient probe artifact directory must not already exist")
        dataset.model_path = probe_output
    # Fail invalid I1-labelled experiments before creating an output directory.
    configured_active_interval = validate_oe_training_configuration(
        args.enable_oe_structural_loss,
        args.oe_structural_weight,
        args.oe_structural_start_iter,
        args.oe_structural_end_iter,
        opt.iterations,
        weak_continuation=args.i1_b1_weak_continuation,
        weak_start_iter=args.i1_b1_weak_start_iter,
        weak_end_iter=args.i1_b1_weak_end_iter,
        weak_weight=args.i1_b1_weak_weight,
        weak_evidence_gamma=args.i1_b1_weak_evidence_gamma,
        weak_weight_schedule=args.i1_b1_weak_weight_schedule,
    )
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, shuffle=not probe_enabled)
    probe_camera_names = _validate_probe_camera_split(dataset, scene) if probe_enabled else None
    gaussians.training_setup(opt)
    if ns30k_smoke_enabled:
        ns30k_initial_audit = write_ns30k_initial_state_audit(
            scene, gaussians, args, ns30k_table
        )
    densification_event_logger = None
    if getattr(args, "densification_diagnostics", False):
        densification_event_logger = DensificationEventLogger(scene.model_path)
    if args.start_checkpoint:
        with torch.serialization.safe_globals(_checkpoint_safe_globals()):
            (model_params, first_iter) = torch.load(
                args.start_checkpoint, weights_only=True, map_location="cuda:0"
            )
        gaussians.restore(model_params, opt)
        _validate_resume_device_consistency(gaussians, torch.device("cuda:0"))
        print(
            "[RESUME] checkpoint={} first_iter={} position_lr_max_steps={} "
            "scheduler_iteration_range=[{}, {}]".format(
                args.start_checkpoint, first_iter, opt.position_lr_max_steps,
                first_iter + 1, opt.position_lr_max_steps,
            )
        )
        if probe_enabled:
            if int(first_iter) != 20000:
                raise RuntimeError("I1 gradient probe requires an iteration=20000 checkpoint")
            import hashlib
            with open(args.start_checkpoint, "rb") as checkpoint_handle:
                probe_checkpoint_sha256 = hashlib.sha256(checkpoint_handle.read()).hexdigest()
            scene_name = "fortress" if "fortress" in str(args.source_path).lower() else "room"
            expected = EXPECTED_CHECKPOINT_SHA256.get(scene_name)
            if expected is None or probe_checkpoint_sha256 != expected:
                raise RuntimeError("I1 gradient probe checkpoint SHA256 mismatch")
            if not isinstance(model_params, (tuple, list)) or len(model_params) < 11:
                raise RuntimeError("I1 gradient probe checkpoint lacks optimizer state")
            optimizer_state = model_params[10]
            if not isinstance(optimizer_state, dict) or "state" not in optimizer_state or "param_groups" not in optimizer_state:
                raise RuntimeError("I1 gradient probe checkpoint optimizer state is malformed")
    if paired_sampling_enabled:
        if not args.start_checkpoint or int(first_iter) != 20_000:
            raise SamplingContractError("paired run requires an iteration=20000 checkpoint")
        paired_tape = SamplingTape(
            args.paired_sampling_tape,
            args.paired_sampling_tape_sha256,
            expected_scene=args.paired_scene,
        )
        if Path(args.paired_trace_dir).resolve() != Path(scene.model_path).resolve():
            raise SamplingContractError("paired trace directory must equal the model output directory")
        paired_logger = PairedRunLogger(args.paired_trace_dir)
    if ns30k_enabled:
        ns30k_logger = NS30KTraceLogger(scene.model_path)
    if args.stop_after_checkpoint_iter is not None:
        if args.stop_after_checkpoint_iter not in args.checkpoint_iterations:
            raise ValueError(
                "stop_after_checkpoint_iter must be present in checkpoint_iterations"
            )
        if args.stop_after_checkpoint_iter <= first_iter:
            raise ValueError(
                "stop_after_checkpoint_iter must be after the resumed checkpoint iteration"
            )
        if args.stop_after_checkpoint_iter > opt.iterations:
            raise ValueError(
                "stop_after_checkpoint_iter cannot exceed iterations"
            )
    if args.enable_oe_structural_loss:
        configured_active_interval = validate_oe_training_configuration(
            True,
            args.oe_structural_weight,
            args.oe_structural_start_iter,
            args.oe_structural_end_iter,
            opt.iterations,
            first_iteration=first_iter + 1,
            weak_continuation=args.i1_b1_weak_continuation,
            weak_start_iter=args.i1_b1_weak_start_iter,
            weak_end_iter=args.i1_b1_weak_end_iter,
            weak_weight=args.i1_b1_weak_weight,
            weak_evidence_gamma=args.i1_b1_weak_evidence_gamma,
            weak_weight_schedule=args.i1_b1_weak_weight_schedule,
        )
    oe_depth_cache = None
    oe_diagnostics = None
    replay_diagnostics = None
    if getattr(args, "resume_replay_diagnostics", False):
        replay_diagnostics = ResumeReplayDiagnostics(
            os.path.join(scene.model_path, "resume_replay_diagnostics.jsonl")
        )
    if args.enable_oe_structural_loss:
        images_dir = os.path.join(dataset.source_path, dataset.images or "images")
        oe_depth_cache = create_oe_depth_cache_if_enabled(
            True,
            args.oe_depth_cache_dir,
            dataset.view_manifest,
            scene.getTrainCameras(),
            images_dir,
        )
        oe_diagnostics = I1Diagnostics(
            {
                "enabled": True,
                "weight": args.oe_structural_weight,
                "configured_start_iter": args.oe_structural_start_iter,
                "configured_end_iter": args.oe_structural_end_iter,
                "gate_mode": args.oe_structure_gate_mode,
                "conflict_threshold": args.oe_conflict_threshold,
                "stable_quantile": args.oe_stable_quantile,
                "b1_weak_continuation": bool(args.i1_b1_weak_continuation),
                "b1_weak_weight": args.i1_b1_weak_weight,
                "b1_weak_start_iter": args.i1_b1_weak_start_iter,
                "b1_weak_end_iter": args.i1_b1_weak_end_iter,
                "b1_weak_evidence_gamma": args.i1_b1_weak_evidence_gamma,
                "b1_weak_weight_schedule": args.i1_b1_weak_weight_schedule,
                "b1_weak_decay_endpoint": (
                    args.i1_b1_weak_end_iter
                    if args.i1_b1_weak_weight_schedule == "linear_decay"
                    else None
                ),
                "depth_cache_dir": os.path.abspath(args.oe_depth_cache_dir),
                "preserve_baseline_densification_stats": bool(
                    args.oe_preserve_baseline_densification_stats
                ),
            },
            configured_active_interval,
            writer=tb_writer,
        )

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = scene.getTrainCameras().copy()
    scheduled_camera_by_name = None
    if paired_sampling_enabled or ns30k_enabled:
        scheduled_camera_by_name = {}
        for camera in viewpoint_stack:
            name = str(camera.image_name)
            if name in scheduled_camera_by_name:
                raise SamplingContractError(f"duplicate training camera image name: {name}")
            scheduled_camera_by_name[name] = camera
        if ns30k_enabled and set(scheduled_camera_by_name) != set(
            ns30k_table.document["training_camera_image_names"]
        ):
            raise NS30KContractError(
                "loaded training cameras do not exactly match the frozen NS-30k Manifest"
            )
    probe_view_count = int(getattr(args, "i1_gradient_probe_views", 3))
    if probe_enabled and probe_view_count not in (1, 3):
        raise RuntimeError("I1 gradient probe view count must be exactly 1 (smoke) or 3 (formal)")
    probe_viewpoints = viewpoint_stack[:probe_view_count]
    if probe_enabled and len(probe_viewpoints) != probe_view_count:
        raise RuntimeError("I1 gradient probe does not have the requested number of training cameras")
    if probe_enabled and len({int(cam.uid) for cam in probe_viewpoints}) != probe_view_count:
        raise RuntimeError("I1 gradient probe training cameras are not unique")
    probe_snapshot = _probe_snapshot(gaussians) if probe_enabled else None
    probe_optimizer_snapshot = _probe_optimizer_snapshot(gaussians) if probe_enabled else None

    # Create tensors for binocular consistency
    image_height = viewpoint_stack[0].image_height
    image_width = viewpoint_stack[0].image_width
    row_indices = torch.arange(0, image_height).view(-1, 1).repeat(1, image_width).cuda()
    column_indices = torch.arange(0, image_width).repeat(image_height, 1).cuda()
    mask = torch.ones((1, image_height, image_width), dtype=torch.float32).cuda()

    smooth_loss = SmoothLoss()

    ema_loss_for_log = 0.0
    first_iter += 1
    loop_end = first_iter + probe_view_count - 1 if probe_enabled else opt.iterations
    if ns30k_smoke_enabled:
        if first_iter != 1 or int(opt.iterations) != 30_000:
            raise NS30KContractError(
                "NS-30k Smoke must start at iteration 1 with iterations=30000"
            )
        loop_end = NS30K_SMOKE_ITERATIONS
    progress_bar = tqdm(range(first_iter, loop_end + 1), desc="Training progress")
    for iteration in range(first_iter, loop_end + 1):
        tape_entry = paired_tape.consume(iteration) if paired_tape is not None else None
        ns30k_entry = ns30k_table.consume(iteration) if ns30k_table is not None else None
        dropout_generator = None
        opacity_noise_generator = None
        background_generator = None
        densification_generator = None
        binocular_render_generator = None
        if tape_entry is not None:
            def seeded_cuda_generator(purpose):
                generator = torch.Generator(device="cuda")
                generator.manual_seed(int(tape_entry["seeds"][purpose]))
                return generator
            dropout_generator = seeded_cuda_generator("dropout")
            binocular_render_generator = seeded_cuda_generator("binocular_render")
            opacity_noise_generator = seeded_cuda_generator("opacity_noise")
            background_generator = seeded_cuda_generator("background")
            densification_generator = seeded_cuda_generator("densification")
        elif ns30k_entry is not None:
            def seeded_ns30k_cuda_generator(purpose):
                generator = torch.Generator(device="cuda")
                generator.manual_seed(int(ns30k_entry["seeds"][purpose]))
                return generator
            dropout_generator = seeded_ns30k_cuda_generator("dropout")
            opacity_noise_generator = seeded_ns30k_cuda_generator("opacity_noise")
            background_generator = seeded_ns30k_cuda_generator("background")
            densification_generator = seeded_ns30k_cuda_generator("densification")
        # if network_gui.conn == None:
        #     network_gui.try_connect()
        # while network_gui.conn != None:
        #     try:
        #         net_image_bytes = None
        #         custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
        #         if custom_cam != None:
        #             net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
        #             net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
        #         network_gui.send(net_image_bytes, dataset.source_path)
        #         if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
        #             break
        #     except Exception as e:
        #         network_gui.conn = None

        iter_start.record()

        position_lr = None if probe_enabled else gaussians.update_learning_rate(iteration)
        if iteration == first_iter:
            print(
                "[RESUME_LR] iteration={} xyz_lr={} configured_max_steps={}".format(
                    iteration, position_lr, opt.position_lr_max_steps
                )
            )

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if not probe_enabled and iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        # random_idx = randint(0, len(viewpoint_stack)-1)
        # viewpoint_cam = viewpoint_stack[random_idx]
        if probe_enabled:
            random.seed(int(args.i1_gradient_probe_seed) + iteration)
            torch.manual_seed(int(args.i1_gradient_probe_seed) + iteration)
            torch.cuda.manual_seed_all(int(args.i1_gradient_probe_seed) + iteration)
            viewpoint_cam = probe_viewpoints[(iteration - first_iter) % 3]
        elif tape_entry is not None:
            camera_name = tape_entry["camera_image_name"]
            if camera_name not in scheduled_camera_by_name:
                raise SamplingContractError(f"tape camera is absent from training split: {camera_name}")
            viewpoint_cam = scheduled_camera_by_name[camera_name]
        elif ns30k_entry is not None:
            camera_name = ns30k_entry["camera_image_name"]
            if camera_name not in scheduled_camera_by_name:
                raise NS30KContractError(
                    f"NS-30k camera is absent from training split: {camera_name}"
                )
            viewpoint_cam = scheduled_camera_by_name[camera_name]
            ns30k_logger.record(ns30k_entry, viewpoint_cam.image_name, ns30k_table)
            if ns30k_smoke_enabled:
                ns30k_smoke_observations.append({
                    "iteration": iteration,
                    "planned_camera_image_name": camera_name,
                    "actual_camera_image_name": str(viewpoint_cam.image_name),
                    "camera_matches_sampling_table": str(viewpoint_cam.image_name) == camera_name,
                    "dropout_seed": int(ns30k_entry["seeds"]["dropout"]),
                    "densification_seed": int(ns30k_entry["seeds"]["densification"]),
                    "dropout_generator_seed_wired": True,
                    "densification_generator_seed_wired": True,
                })
        else:
            viewpoint_cam = random.choice(viewpoint_stack)

        # Render
        if (iteration - 1) == args.debug_from:
            pipe.debug = True

        bg = (
            torch.rand((3), device="cuda", generator=background_generator)
            if opt.random_background else background
        )

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, 
                            dropout_factor=args.dropout_factor, 
                            sigma_noise=args.sigma_noise, train=True,
                            dropout_generator=dropout_generator,
                            opacity_noise_generator=opacity_noise_generator)
        
        image = render_pkg["render"]
        viewspace_point_tensor = render_pkg["viewspace_points"]
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]
        depth = render_pkg["rendered_depth"]
        alpha = render_pkg["rendered_alpha"]
        dropout_mask = render_pkg["dropout_mask"]
        dropout_random_values = render_pkg["dropout_random_values"]

        if replay_diagnostics is not None:
            replay_diagnostics.record_render(iteration, viewpoint_cam, dropout_mask, gaussians)

        gt_image = viewpoint_cam.original_image.cuda()
        
        # print(gt_image)

        ######## from DNGaussian ###########
        bg_mask = None
        if args.dataset_name == 'DTU':
            if 'scan110' not in args.source_path:
                bg_mask = (gt_image.max(0, keepdim=True).values < 30 / 255)
            else:
                bg_mask = (gt_image.max(0, keepdim=True).values < 15 / 255)
            bg_mask_clone = bg_mask.clone()
            for i in range(1, 50):
                bg_mask[:, i:] *= bg_mask_clone[:, :-i]
            bg_mask = bg_mask.float()
        #####################################

        disparity_loss=0.0
        if args.binocular_consistency and iteration > args.shift_cam_start:
            if tape_entry is not None:
                trans_dist = (
                    float(tape_entry["binocular_unit"])
                    * int(tape_entry["binocular_sign"])
                    * float(args.cam_trans_dist)
                )
            else:
                trans_dist = torch.rand(1) * args.cam_trans_dist
                trans_dist = (trans_dist * random.choice([-1.0, 1.0])).item()
            shifted_cam = scene.getShiftedCamera(viewpoint_cam, trans_dist)
            render_pkg = render(
                shifted_cam, gaussians, pipe, bg,
                dropout_generator=binocular_render_generator,
            )
            shifted_image = render_pkg["render"]
            focal_x, focal_y = viewpoint_cam.get_focal()
            disparity = focal_x * (-trans_dist) / (depth + 1e-5)
            warped_image = inverse_warp_images(shifted_image.unsqueeze(0), disparity.unsqueeze(0), row_indices, column_indices)
            shift_mask = inverse_warp_images(mask.unsqueeze(0), disparity.unsqueeze(0), row_indices, column_indices)

            disparity_loss = (l1_loss(warped_image, gt_image.unsqueeze(0), mask=shift_mask) +
                              0.05 * smooth_loss.forward(disparity=disparity*shift_mask, image=gt_image.unsqueeze(0)))

        if paired_logger is not None:
            dropout_summary = tensor_summary(dropout_mask)
            dropout_input_summary = tensor_summary(dropout_random_values)
            paired_logger.record_trace({
                "iteration": iteration,
                "camera_id": tape_entry["camera_id"],
                "camera_uid_observed": int(viewpoint_cam.uid),
                "camera_image_name": str(viewpoint_cam.image_name),
                "tape_id": paired_tape.tape_id,
                "tape_file_sha256": paired_tape.file_sha256,
                "tape_cursor": tape_entry["cursor"],
                "tape_entry_sha256": tape_entry["entry_sha256"],
                "dropout_seed": tape_entry["seeds"]["dropout"],
                "dropout_input_shape": json.dumps(dropout_input_summary["shape"]),
                "dropout_input_sha256": dropout_input_summary["sha256"],
                "dropout_input_pairing_level": "RULE_AND_SEED; ELEMENTWISE_ONLY_WHEN_SHAPES_MATCH",
                "dropout_mask_shape": json.dumps(dropout_summary["shape"]),
                "dropout_mask_sha256": dropout_summary["sha256"],
                "dropout_mask_true_count": dropout_summary.get("true_count"),
                "binocular_unit": tape_entry["binocular_unit"],
                "binocular_sign": tape_entry["binocular_sign"],
                "binocular_translation": trans_dist if args.binocular_consistency and iteration > args.shift_cam_start else "",
                "planned_fields": "camera_id,camera_image_name,tape,seeds,binocular_unit,binocular_sign",
                "observed_fields": "camera_uid_observed,dropout_shape,dropout_mask,binocular_translation",
            })


        alpha_loss = 0.0
        if viewpoint_cam.gt_alpha_mask is not None:
            alpha_loss = torch.mean(torch.abs(alpha) * (1-viewpoint_cam.gt_alpha_mask))
        elif bg_mask is not None:
            alpha_loss = torch.mean(torch.abs(alpha) * bg_mask)

        # Loss
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        baseline_total_loss = loss + disparity_loss + alpha_loss
        # ``depth`` was captured from the primary training-view render;
        # ``render_pkg`` may have been rebound by binocular consistency.
        total_loss, oe_struct_loss, oe_stats = apply_oe_structural_training_loss(
            baseline_total_loss,
            enabled=args.enable_oe_structural_loss,
            iteration=iteration,
            start_iter=args.oe_structural_start_iter,
            end_iter=args.oe_structural_end_iter,
            weight=args.oe_structural_weight,
            depth_cache=oe_depth_cache,
            camera=viewpoint_cam,
            rendered_depth=depth,
            gt_rgb=gt_image,
            gate_mode=args.oe_structure_gate_mode,
            stable_quantile=args.oe_stable_quantile,
            conflict_threshold=args.oe_conflict_threshold,
            weak_continuation=args.i1_b1_weak_continuation,
            weak_start_iter=args.i1_b1_weak_start_iter,
            weak_end_iter=args.i1_b1_weak_end_iter,
            weak_weight=args.i1_b1_weak_weight,
            weak_evidence_gamma=args.i1_b1_weak_evidence_gamma,
            weak_weight_schedule=args.i1_b1_weak_weight_schedule,
            diagnostics=oe_diagnostics,
        )
        if ns30k_smoke_enabled:
            smoke_observation = ns30k_smoke_observations[-1]
            smoke_observation.update({
                "i1_structural_loss_computed": oe_struct_loss is not None,
                "i1_raw_loss": (
                    float(oe_struct_loss.detach().item())
                    if oe_struct_loss is not None else None
                ),
                "i1_applied_weight": (
                    float(oe_stats.get("i1_applied_weight", 0.0)) if oe_stats else 0.0
                ),
            })
        finite_loss = True
        if paired_sampling_enabled:
            finite_loss = bool(torch.isfinite(total_loss).all().item())
        if ns30k_smoke_enabled:
            finite_loss = bool(torch.isfinite(total_loss).all().item())
            smoke_observation["finite_total_loss"] = finite_loss
            if not finite_loss:
                raise FloatingPointError(
                    f"non-finite total loss at NS-30k Smoke iteration {iteration}"
                )
        if paired_sampling_enabled and not finite_loss:
            paired_logger.record_metrics({
                "iteration": iteration, "rgb_loss": float(loss.detach().item()),
                "binocular_loss": float(torch.as_tensor(disparity_loss).detach().item()),
                "i1_raw_loss": float(oe_struct_loss.detach().item()) if oe_struct_loss is not None else 0.0,
                "i1_applied_weight": float(oe_stats.get("i1_applied_weight", 0.0)) if oe_stats else 0.0,
                "i1_weighted_loss": "NONFINITE", "total_loss": "NONFINITE",
                "gaussian_count": int(gaussians.get_xyz.shape[0]),
                "optimizer_step_executed": False, "finite_status": "NONFINITE_LOSS",
                "field_provenance": "observed_runtime",
            })
            raise FloatingPointError(f"non-finite total loss at paired iteration {iteration}")
        if probe_enabled:
            probe_losses = {
                "rgb": loss,
                "binocular": _probe_loss_tensor(disparity_loss, baseline_total_loss),
                "alpha": _probe_loss_tensor(alpha_loss, baseline_total_loss),
                "baseline_total": baseline_total_loss,
            }
            probe_result = _collect_i1_gradient_probe(
                losses=probe_losses,
                total_loss=total_loss,
                gaussians=gaussians,
                camera=viewpoint_cam,
                iteration=iteration,
                checkpoint_sha256=probe_checkpoint_sha256,
                random_contract={
                    "seed": int(args.i1_gradient_probe_seed) + iteration,
                    "dropout_factor": float(args.dropout_factor),
                    "sigma_noise": float(args.sigma_noise),
                    "shift_cam_start": int(args.shift_cam_start),
                },
                oe_struct_loss=oe_struct_loss,
                oe_stats=oe_stats,
            )
            probe_results.append(probe_result)
            for comparison, rows in probe_result.get("by_parameter_group", {}).items():
                for row in rows:
                    probe_group_results.append({
                        "camera_id": probe_result.get("camera_id"),
                        "comparison": comparison,
                        **row,
                    })
            print("[I1_GRADIENT_PROBE] " + json.dumps(probe_result, ensure_ascii=False))
            _assert_probe_snapshot(gaussians, probe_snapshot)
            _assert_probe_optimizer_snapshot(gaussians, probe_optimizer_snapshot)
            continue
        current_densify_until_iter = effective_densify_until_iter(
            iteration,
            opt.densify_from_iter,
            opt.densify_until_iter,
            opt.iterations,
        )
        baseline_viewspace_grad = maybe_extract_baseline_viewspace_gradient(
            baseline_total_loss,
            viewspace_point_tensor,
            preserve_flag=args.oe_preserve_baseline_densification_stats,
            oe_loss_active=oe_struct_loss is not None,
            iteration=iteration,
            densify_until_iter=current_densify_until_iter,
        )
        preserve_oe_densification_stats = baseline_viewspace_grad is not None
        total_loss.backward()
        finite_gradients = True
        if paired_sampling_enabled:
            finite_gradients = all(
                parameter.grad is None or bool(torch.isfinite(parameter.grad).all().item())
                for group in gaussians.optimizer.param_groups
                for parameter in group["params"]
            )
        if paired_sampling_enabled and not finite_gradients:
            paired_logger.record_metrics({
                "iteration": iteration, "rgb_loss": float(loss.detach().item()),
                "binocular_loss": float(torch.as_tensor(disparity_loss).detach().item()),
                "i1_raw_loss": float(oe_struct_loss.detach().item()) if oe_struct_loss is not None else 0.0,
                "i1_applied_weight": float(oe_stats.get("i1_applied_weight", 0.0)) if oe_stats else 0.0,
                "i1_weighted_loss": "UNAPPLIED", "total_loss": float(total_loss.detach().item()),
                "gaussian_count": int(gaussians.get_xyz.shape[0]),
                "optimizer_step_executed": False, "finite_status": "NONFINITE_GRADIENT",
                "field_provenance": "observed_runtime",
            })
            raise FloatingPointError(f"non-finite gradient at paired iteration {iteration}")

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.6f}",
                                          "disp_loss": f"{disparity_loss:.6f}",
                                          "points": f"{len(gaussians.get_xyz)}",
                                          })
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            # Formal paired continuations never inspect test metrics while consuming
            # the training tape. Evaluation is a later, separately frozen stage.
            if not paired_sampling_enabled:
                training_report(
                    tb_writer, iteration, Ll1, loss, l1_loss,
                    iter_start.elapsed_time(iter_end), args.test_iterations,
                    scene, render, (pipe, background),
                    enable_evaluation=not args.disable_training_time_evaluation,
                )
            if save_phase(
                paired_mode=paired_sampling_enabled,
                iteration=iteration,
                save_iterations=args.save_iterations,
            ) == "pre_optimizer_step":
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            if iteration > opt.densify_from_iter:
                opt.densify_until_iter = opt.iterations

            if args.opacity_decay:
                gaussians.opacity_decay(factor=args.opacity_decay_factor)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                combined_mask = map_dropout_visibility_to_full_mask(
                    dropout_mask,
                    visibility_filter,
                    viewspace_point_tensor,
                )
                gaussians.max_radii2D[combined_mask] = torch.max(gaussians.max_radii2D[combined_mask], radii[visibility_filter])
                if preserve_oe_densification_stats:
                    add_densification_stats_with_baseline_gradient(
                        gaussians,
                        viewspace_point_tensor,
                        combined_mask,
                        baseline_viewspace_grad,
                    )
                    if oe_diagnostics is not None:
                        oe_diagnostics.record_densification_gradients(
                            iteration,
                            baseline_viewspace_grad,
                            viewspace_point_tensor.grad,
                            combined_mask,
                            protection_effective=True,
                        )
                else:
                    gaussians.add_densification_stats(viewspace_point_tensor, combined_mask)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    if args.opacity_decay:
                        size_threshold = None
                    size_threshold = None
                    collect_event = densification_event_logger is not None or paired_logger is not None
                    if not collect_event:
                        gaussians.densify_and_prune(
                            opt.densify_grad_threshold, 0.005, scene.cameras_extent,
                            size_threshold, random_generator=densification_generator,
                        )
                    else:
                        event_counts = gaussians.densify_and_prune(
                            opt.densify_grad_threshold,
                            0.005,
                            scene.cameras_extent,
                            size_threshold,
                            collect_diagnostics=True,
                            random_generator=densification_generator,
                        )
                        if densification_event_logger is not None:
                            densification_event_logger.record(iteration, event_counts, gaussians.optimizer)
                        if paired_logger is not None:
                            paired_logger.record_event({"iteration": iteration, **event_counts})
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    pass
                    # if args.opacity_decay:
                    #     pass
                    # else:
                    #     gaussians.reset_opacity()

            # Optimizer step
            optimizer_step_executed = False
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                optimizer_step_executed = True

            if paired_logger is not None:
                i1_weight = float(oe_stats.get("i1_applied_weight", 0.0)) if oe_stats else 0.0
                i1_raw = float(oe_struct_loss.detach().item()) if oe_struct_loss is not None else 0.0
                finite_parameters = all(
                    bool(torch.isfinite(parameter).all().item())
                    for group in gaussians.optimizer.param_groups
                    for parameter in group["params"]
                )
                paired_logger.record_metrics({
                    "iteration": iteration,
                    "rgb_loss": float(loss.detach().item()),
                    "binocular_loss": float(torch.as_tensor(disparity_loss).detach().item()),
                    "i1_raw_loss": i1_raw,
                    "i1_applied_weight": i1_weight,
                    "i1_weighted_loss": i1_raw * i1_weight,
                    "total_loss": float(total_loss.detach().item()),
                    "gaussian_count": int(gaussians.get_xyz.shape[0]),
                    "optimizer_step_executed": optimizer_step_executed,
                    "finite_status": "FINITE" if finite_loss and finite_gradients and finite_parameters else "NONFINITE_PARAMETER",
                    "field_provenance": "observed_runtime",
                })
                if not finite_parameters:
                    raise FloatingPointError(f"non-finite parameter after optimizer step at paired iteration {iteration}")

            # Paired terminal PLY is intentionally written only after the normal
            # opacity/densification/optimizer update.  The ordinary training path
            # above retains its historical pre-step save ordering.
            if save_phase(
                paired_mode=paired_sampling_enabled,
                iteration=iteration,
                save_iterations=args.save_iterations,
            ) == "post_optimizer_step":
                if not optimizer_step_executed:
                    raise SamplingContractError("paired terminal PLY requires a completed optimizer step")
                model_sha_before_ply = gaussian_model_state_sha256(gaussians)
                print("\n[ITER {}] Saving post-update Gaussians".format(iteration))
                scene.save(iteration)
                model_sha_after_ply = gaussian_model_state_sha256(gaussians)
                ply_path = (
                    Path(scene.model_path) / "point_cloud" /
                    f"iteration_{iteration}" / "point_cloud.ply"
                )
                if not ply_path.is_file():
                    raise SamplingContractError("paired terminal PLY was not created")
                paired_terminal_artifacts.update({
                    "save_order": [
                        "opacity_decay", "densification_or_pruning", "optimizer_step",
                        "ply_save", "checkpoint_save",
                    ],
                    "model_state_sha256_before_ply": model_sha_before_ply,
                    "model_state_sha256_after_ply": model_sha_after_ply,
                    "ply_path": str(ply_path),
                    "ply_sha256": sha256_file(ply_path),
                })

            if replay_diagnostics is not None:
                replay_diagnostics.record_optimizer(iteration, gaussians)

            if (iteration in args.checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                checkpoint_path = Path(scene.model_path) / f"chkpnt{iteration}.pth"
                model_sha_before_checkpoint = (
                    gaussian_model_state_sha256(gaussians) if paired_sampling_enabled else None
                )
                torch.save((gaussians.capture(), iteration), checkpoint_path)
                if paired_sampling_enabled:
                    model_sha_after_checkpoint = gaussian_model_state_sha256(gaussians)
                    paired_terminal_artifacts.update({
                        "model_state_sha256_before_checkpoint": model_sha_before_checkpoint,
                        "model_state_sha256_after_checkpoint": model_sha_after_checkpoint,
                        "checkpoint_path": str(checkpoint_path),
                        "checkpoint_sha256": sha256_file(checkpoint_path),
                    })
                if (args.stop_after_checkpoint_iter is not None
                        and iteration == args.stop_after_checkpoint_iter):
                    print("\n[STOP_AFTER_CHECKPOINT] iteration={} checkpoint_saved=true optimizer_step_completed=true".format(iteration))
                    break
    if ns30k_table is not None:
        if ns30k_smoke_enabled:
            consumed_exactly_ten = (
                ns30k_table.consumed_count == NS30K_SMOKE_ITERATIONS
                and len(ns30k_smoke_observations) == NS30K_SMOKE_ITERATIONS
            )
            cameras_match = all(
                item["camera_matches_sampling_table"]
                for item in ns30k_smoke_observations
            )
            seeds_wired = all(
                item["dropout_generator_seed_wired"]
                and item["densification_generator_seed_wired"]
                for item in ns30k_smoke_observations
            )
            nonfinite_loss_seen = any(
                not item.get("finite_total_loss", False)
                for item in ns30k_smoke_observations
            )
            i1_computation_count = sum(
                bool(item["i1_structural_loss_computed"])
                for item in ns30k_smoke_observations
            )
            i1_expected = args.ns30k_arm == "ON"
            i1_matches_arm = (
                i1_computation_count == NS30K_SMOKE_ITERATIONS
                if i1_expected else i1_computation_count == 0
            )
            model_path = Path(scene.model_path)
            formal_artifacts = [
                str(path) for path in (
                    list(model_path.glob("chkpnt*.pth"))
                    + list((model_path / "point_cloud").glob("iteration_*"))
                )
            ]
            smoke_ok = (
                consumed_exactly_ten
                and cameras_match
                and seeds_wired
                and not nonfinite_loss_seen
                and i1_matches_arm
                and not formal_artifacts
            )
            smoke_result = {
                "audit_version": "ns30k-smoke-result-v1",
                "status": "SMOKE_PASS" if smoke_ok else "SMOKE_FAIL",
                "SMOKE_ONLY": True,
                "NOT_FORMAL_RESULT": True,
                "scene": ns30k_table.document["scene"],
                "arm": args.ns30k_arm,
                "experiment_arm": "B0+I1" if args.ns30k_arm == "ON" else "B0",
                "configured_iterations": int(opt.iterations),
                "actual_completed_training_iterations": len(ns30k_smoke_observations),
                "sampling_records_consumed": ns30k_table.consumed_count,
                "sampling_consumption_exactly_10": consumed_exactly_ten,
                "camera_sequence_matches_sampling_table": cameras_match,
                "seed_wiring_records": ns30k_smoke_observations,
                "initial_gaussian_model_sha256": ns30k_initial_audit[
                    "initial_gaussian_model_sha256"
                ],
                "i1_structural_loss_expected": i1_expected,
                "i1_structural_loss_computation_count": i1_computation_count,
                "i1_structural_loss_observation_matches_arm": i1_matches_arm,
                "nonfinite_loss_seen": nonfinite_loss_seen,
                "formal_checkpoint_or_ply_artifacts": formal_artifacts,
                "densification_validation": "NOT_REACHED_ITERATION_1000",
                "limitation": (
                    "The 10-iteration Smoke validates only the initial training path; "
                    "it is not a complete 30k result and does not validate densification."
                ),
            }
            _write_json_exclusive(
                model_path / "ns30k_smoke_result.json", smoke_result
            )
            if not smoke_ok:
                raise NS30KContractError("NS-30k Smoke result audit failed")
        else:
            ns30k_table.assert_complete()
    if paired_tape is not None:
        paired_tape.assert_complete()
        checkpoint_path = Path(scene.model_path) / f"chkpnt{PAIRED_END_ITERATION}.pth"
        runtime_metrics = paired_logger.metrics_summary()
        expected_weights = [0.02] if args.paired_arm == "ON" else [0.0]
        audit_ok = (
            checkpoint_path.is_file()
            and paired_terminal_artifacts.get("ply_path") is not None
            and paired_terminal_artifacts.get("model_state_sha256_before_ply")
                == paired_terminal_artifacts.get("model_state_sha256_after_ply")
                == paired_terminal_artifacts.get("model_state_sha256_before_checkpoint")
                == paired_terminal_artifacts.get("model_state_sha256_after_checkpoint")
            and runtime_metrics["optimizer_step_count"] == 1000
            and runtime_metrics["observed_i1_weights"] == expected_weights
            and runtime_metrics["metric_row_count"] == 1000
            and runtime_metrics["nonfinite_count"] == 0
        )
        capture_audit = {
            "status": "PASS" if audit_ok else "FAIL",
            "observed_runtime": True,
            "arm": args.paired_arm,
            "scene": args.paired_scene,
            "start_iteration": 20_000,
            "last_completed_iteration": PAIRED_END_ITERATION,
            "normal_update_count": paired_tape.consumed_count,
            "optimizer_step_completed_at_21000": runtime_metrics["optimizer_step_count"] == 1000,
            "checkpoint_21000_exists": checkpoint_path.is_file(),
            "terminal_artifacts": paired_terminal_artifacts,
            "in_memory_model_state_stable_across_saves": bool(
                paired_terminal_artifacts.get("model_state_sha256_before_ply")
                and paired_terminal_artifacts.get("model_state_sha256_before_ply")
                == paired_terminal_artifacts.get("model_state_sha256_after_ply")
                == paired_terminal_artifacts.get("model_state_sha256_before_checkpoint")
                == paired_terminal_artifacts.get("model_state_sha256_after_checkpoint")
            ),
            "ply_checkpoint_file_content_equivalence": "UNVERIFIED",
            "sampling_tape_id": paired_tape.tape_id,
            "sampling_tape_file_sha256": paired_tape.file_sha256,
            "test_metrics_read_for_sampling": False,
            "runtime_metrics": runtime_metrics,
            "expected_i1_weights": expected_weights,
            "i1_weight_observation_matches_arm": runtime_metrics["observed_i1_weights"] == expected_weights,
        }
        (Path(scene.model_path) / "capture_audit.json").write_text(
            json.dumps(capture_audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if not audit_ok:
            raise SamplingContractError("paired run capture/weight/finite audit failed")
    if probe_enabled:
        camera_statuses = [item.get("status", "UNVERIFIED") for item in probe_results]
        if any(status == "FAIL" for status in camera_statuses):
            summary_status = "FAIL"
        elif all(status == "PASS" for status in camera_statuses):
            summary_status = "PASS"
        else:
            summary_status = "UNVERIFIED"
        summary = {
            "status": summary_status,
            "camera_statuses": camera_statuses,
            "i1_gradient_statuses": [item.get("i1_gradient_status") for item in probe_results],
            "probe_view_count": probe_view_count,
            "camera_result_count": len(probe_results),
            "frozen_training_camera_names": probe_camera_names,
            "camera_name_count_order_exact": True,
            "checkpoint_sha256": probe_checkpoint_sha256,
            "optimizer_step_called": False,
            "model_update_allowed": False,
            "results": probe_results,
        }
        write_probe_artifacts_to_directory(
            Path(probe_artifact_output), per_camera=probe_results,
            by_parameter_group=probe_group_results, summary=summary,
        )
        if summary["status"] != "PASS":
            raise RuntimeError("I1 gradient probe produced FAIL/UNVERIFIED results")
    if oe_diagnostics is not None:
        oe_diagnostics.finish()
    if replay_diagnostics is not None:
        replay_diagnostics.finish()

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, enable_evaluation=True):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if enable_evaluation and (iteration in testing_iterations or iteration % 1000 == 0):
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test, psnr_test, ssim_test, lpips_test = 0.0, 0.0, 0.0, 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    # if tb_writer and (idx < 5):
                    #     tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                    #     if iteration == testing_iterations[0]:
                    #         tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    # _mask = None
                    _psnr = psnr(image, gt_image).mean().double()
                    _ssim = ssim(image, gt_image).mean().double()
                    _lpips = lpips(image, gt_image, net_type='vgg')
                    psnr_test += _psnr
                    ssim_test += _ssim
                    lpips_test += _lpips
                psnr_test /= len(config['cameras'])
                ssim_test /= len(config['cameras'])
                lpips_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} SSIM {} LPIPS {} ".format(iteration, config['name'], l1_test, psnr_test, ssim_test, lpips_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        # if tb_writer:
        #     tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
        #     tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[3_000, 10_000, 20_000, 30_000])
    parser.add_argument("--disable_training_time_evaluation", action="store_true", default=False)
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument(
        "--stop_after_checkpoint_iter", type=int, default=None,
        help="Safely stop after the selected iteration's normal optimizer step and checkpoint save.",
    )
    parser.add_argument(
        "--resume_replay_diagnostics", action="store_true", default=False,
        help="write read-only RNG/mask/camera/parameter diagnostics for the R2 replay audit",
    )
    parser.add_argument(
        "--i1_gradient_probe", action="store_true", default=False,
        help="run the dedicated read-only I1 gradient probe branch; default is disabled",
    )
    parser.add_argument(
        "--i1_gradient_probe_output", type=str,
        default="./reports/i1_d4_gradient_conflict",
    )
    parser.add_argument(
        "--i1_gradient_probe_artifact_output", type=str,
        default="./reports/i1_d4_gradient_conflict",
    )
    parser.add_argument(
        "--i1_gradient_probe_views", type=int, choices=(1, 3), default=3,
        help="read-only probe cameras: 1 for smoke, 3 for formal",
    )
    parser.add_argument("--i1_gradient_probe_seed", type=int, default=20261022)
    parser.add_argument("--paired_sampling_tape", type=str, default="")
    parser.add_argument("--paired_sampling_tape_sha256", type=str, default="")
    parser.add_argument("--paired_trace_dir", type=str, default="")
    parser.add_argument(
        "--paired_scene", choices=("fortress", "room", "horns", "orchids"), default=None
    )
    parser.add_argument("--paired_arm", choices=("OFF", "ON"), default=None)
    parser.add_argument("--ns30k_sampling_tape", type=str, default="")
    parser.add_argument("--ns30k_sampling_tape_sha256", type=str, default="")
    parser.add_argument("--ns30k_arm", choices=("OFF", "ON"), default=None)
    parser.add_argument(
        "--ns30k_smoke", action="store_true", default=False,
        help="authorized 10-iteration NS30K integration Smoke; never a formal result",
    )
    parser.add_argument("--opacity_decay", action="store_true", default=True)
    parser.add_argument("--opacity_decay_factor", type=float, default=0.995)
    parser.add_argument("--dropout_factor", type=float, default=0.2) # 0.2
    parser.add_argument("--sigma_noise", type=float, default=0.0) # 0.8 0.05
    parser.add_argument("--cam_trans_dist", type=float, default=0.4)
    binocular_group = parser.add_mutually_exclusive_group()
    binocular_group.add_argument(
        "--binocular_consistency",
        dest="binocular_consistency",
        action="store_true",
        default=True,
    )
    binocular_group.add_argument(
        "--no-binocular-consistency",
        dest="binocular_consistency",
        action="store_false",
    )
    parser.add_argument("--shift_cam_start", type=int, default=20000)
    parser.add_argument("--dataset_name", type=str, default="LLFF")
    parser.add_argument("--n_views", type=int, default=3)
    parser.add_argument("--suffix", type=str, default=None)
    add_oe_structural_arguments(parser)
    add_densification_diagnostics_argument(parser)
    args = parser.parse_args(sys.argv[1:])
    # Preserve the historical final-iteration auto-save for ordinary training.
    # Paired runs freeze exactly one post-update PLY node at iteration 21000.
    if not args.paired_sampling_tape:
        args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args)

    # All done
    print("\nTraining complete." + args.model_path)
