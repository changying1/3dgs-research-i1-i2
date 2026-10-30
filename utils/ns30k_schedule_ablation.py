"""Strict contract for the NS30K-I1-Schedule-v1 ablation.

This module deliberately replaces only the legacy NS30K configuration
validator.  Sampling, tracing, rendering, loss computation, optimization,
densification, pruning, and artifact creation continue to come from the
frozen training path.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

from utils.ns30k_sampling import (
    END_ITERATION,
    FROZEN_SAVE_NODES,
    NS30KContractError,
    NS30KSamplingTable,
)
PROTOCOL_VERSION = "NS30K-I1-Schedule-v1"
SUPPORTED_SCENES = ("fortress", "horns", "room")
SUPPORTED_VARIANTS = ("Strong30k", "Weak30k")
VARIANT_ENV = "NS30K_I1_SCHEDULE_VARIANT"
AUTH_ENV = "NS30K_I1_SCHEDULE_AUTHORIZED"
BOUNDARY_ITERATIONS = (1, 8_000, 8_001, 20_000, 30_000)

EXPECTED_WEIGHTS = {
    "Strong30k": {1: 0.1, 8_000: 0.1, 8_001: 0.1, 20_000: 0.1, 30_000: 0.1},
    "Weak30k": {1: 0.1, 8_000: 0.1, 8_001: 0.02, 20_000: 0.02, 30_000: 0.02},
}


def _equal_float(value, expected: float) -> bool:
    try:
        return math.isclose(float(value), expected, rel_tol=0.0, abs_tol=1e-15)
    except (TypeError, ValueError):
        return False


def _require(failures: list[str], condition: bool, message: str) -> None:
    if not condition:
        failures.append(message)


def effective_weight_from_contract(args, iteration: int) -> float:
    """Dependency-free equivalent of the frozen activation selection."""
    iteration = int(iteration)
    if (
        args.enable_oe_structural_loss
        and int(args.oe_structural_start_iter) <= iteration <= int(args.oe_structural_end_iter)
        and float(args.oe_structural_weight) > 0.0
    ):
        return float(args.oe_structural_weight)
    if (
        args.enable_oe_structural_loss
        and args.i1_b1_weak_continuation
        and int(args.i1_b1_weak_start_iter) <= iteration <= int(args.i1_b1_weak_end_iter)
    ):
        weight = float(args.i1_b1_weak_weight)
        if args.i1_b1_weak_weight_schedule == "linear_decay":
            denominator = int(args.i1_b1_weak_end_iter) - int(args.i1_b1_weak_start_iter)
            if denominator <= 0:
                raise ValueError("linear_decay requires weak_start_iter < weak_end_iter")
            weight *= max(0.0, min(1.0, (int(args.i1_b1_weak_end_iter) - iteration) / denominator))
        return weight if weight > 0.0 else 0.0
    return 0.0


def effective_weight_from_training_path(args, iteration: int) -> float:
    """Resolve weight with the exact activation helper used by GPU training."""
    from utils.oe_training import oe_activation_details

    activation = oe_activation_details(
        args.enable_oe_structural_loss,
        int(iteration),
        args.oe_structural_start_iter,
        args.oe_structural_end_iter,
        args.oe_structural_weight,
        weak_continuation=args.i1_b1_weak_continuation,
        weak_start_iter=args.i1_b1_weak_start_iter,
        weak_end_iter=args.i1_b1_weak_end_iter,
        weak_weight=args.i1_b1_weak_weight,
        weak_weight_schedule=args.i1_b1_weak_weight_schedule,
    )
    return float(activation["weight"]) if activation["active"] else 0.0


def schedule_snapshot(args) -> dict:
    return {
        str(iteration): effective_weight_from_contract(args, iteration)
        for iteration in BOUNDARY_ITERATIONS
    }


def _save_nodes_are_frozen(save_iterations, checkpoint_iterations) -> bool:
    try:
        saves = [int(value) for value in save_iterations]
        checkpoints = [int(value) for value in checkpoint_iterations]
    except (TypeError, ValueError):
        return False
    return saves in (list(FROZEN_SAVE_NODES), [*FROZEN_SAVE_NODES, END_ITERATION]) and (
        checkpoints == list(FROZEN_SAVE_NODES)
    )


def validate_ns30k_schedule_ablation_training_configuration(
    dataset, opt, args, table: NS30KSamplingTable
) -> Path:
    """Fail before output creation unless the complete schedule contract holds."""
    from utils.oe_training import validate_oe_training_configuration

    variant = os.environ.get(VARIANT_ENV, "")
    failures: list[str] = []
    scene = os.path.basename(os.path.normpath(str(dataset.source_path))).lower()

    _require(failures, os.environ.get(AUTH_ENV) == "1", f"missing {AUTH_ENV}=1")
    _require(failures, variant in SUPPORTED_VARIANTS, f"invalid or missing {VARIANT_ENV}")
    _require(
        failures,
        scene in SUPPORTED_SCENES,
        "scene must be fortress, horns, or room",
    )
    _require(failures, scene == table.document.get("scene"), "sampling-table scene mismatch")
    _require(failures, int(opt.iterations) == END_ITERATION, "iterations must be exactly 30000")
    _require(failures, int(opt.position_lr_max_steps) == END_ITERATION, "position_lr_max_steps must be 30000")
    _require(failures, _equal_float(opt.position_lr_init, 0.00016), "position_lr_init must be 0.00016")
    _require(failures, _equal_float(opt.position_lr_final, 0.0000016), "position_lr_final must be 0.0000016")
    _require(failures, _equal_float(opt.lambda_dssim, 0.2), "lambda_dssim must be 0.2")
    _require(failures, int(opt.densify_from_iter) == 1000, "densify_from_iter must be 1000")
    _require(failures, int(opt.densify_until_iter) == 15_000, "densify_until_iter must be 15000")
    _require(failures, int(opt.densification_interval) == 100, "densification_interval must be 100")
    _require(failures, _equal_float(opt.densify_grad_threshold, 0.0002), "densify_grad_threshold must be 0.0002")
    _require(failures, opt.random_background is False, "random_background must be False")
    _require(failures, dataset.dataset_name == "LLFF", "dataset_name must be LLFF")
    _require(failures, dataset.eval is True, "--eval is required")
    _require(failures, int(dataset.n_views) == 3, "n_views must be 3")
    _require(failures, int(dataset.resolution) == 8, "resolution must be 8")
    _require(failures, int(dataset.sh_degree) == 1, "sh_degree must be 1")
    _require(failures, bool(dataset.view_manifest), "frozen view_manifest is required")
    _require(failures, bool(dataset.suffix), "frozen PDC suffix is required")
    _require(failures, dataset.suffix == table.document.get("pdc_suffix"), "PDC suffix mismatch")
    _require(failures, args.binocular_consistency is False, "--no-binocular-consistency is required")
    _require(failures, args.disable_training_time_evaluation is True, "training-time evaluation must be disabled")
    _require(failures, not args.start_checkpoint, "checkpoint resume is forbidden")
    _require(failures, args.stop_after_checkpoint_iter is None, "early stop is forbidden")
    _require(failures, not args.paired_sampling_tape, "legacy paired sampling is forbidden")
    _require(failures, args.ns30k_arm == "ON", "ns30k_arm must be ON")
    _require(failures, _equal_float(args.dropout_factor, 0.2), "dropout_factor must be 0.2")
    _require(failures, _equal_float(args.sigma_noise, 0.0), "sigma_noise must be 0")
    _require(failures, args.opacity_decay is True, "opacity_decay must be enabled")
    _require(failures, _equal_float(args.opacity_decay_factor, 0.995), "opacity_decay_factor must be 0.995")
    _require(failures, _save_nodes_are_frozen(args.save_iterations, args.checkpoint_iterations), "save/checkpoint nodes must be 5000,10000,20000,30000")
    _require(failures, getattr(dataset, "initial_opacity_arm", "control") == "control", "initial opacity arm must be control")
    _require(failures, not getattr(dataset, "initial_opacity_scores", ""), "opacity scores are forbidden")
    _require(failures, not getattr(dataset, "initial_opacity_score_manifest", ""), "opacity score manifest is forbidden")

    _require(failures, args.enable_oe_structural_loss is True, "I1 structural loss must be enabled")
    _require(failures, bool(args.oe_depth_cache_dir), "explicit MiDaS cache is required")
    _require(failures, args.oe_preserve_baseline_densification_stats is True, "baseline densification statistics must be preserved")
    _require(failures, int(args.oe_structural_start_iter) == 1, "strong interval must start at 1")
    _require(failures, _equal_float(args.oe_structural_weight, 0.1), "strong weight must be 0.1")
    _require(failures, args.oe_structure_gate_mode == "stable_evidence", "gate mode must be stable_evidence")
    _require(failures, _equal_float(args.oe_stable_quantile, 0.8), "stable quantile must be 0.8")

    if variant == "Strong30k":
        _require(failures, int(args.oe_structural_end_iter) == 30_000, "Strong30k strong interval must end at 30000")
        _require(failures, args.i1_b1_weak_continuation is False, "Strong30k must not enable weak continuation")
    elif variant == "Weak30k":
        _require(failures, int(args.oe_structural_end_iter) == 8_000, "Weak30k strong interval must end at 8000")
        _require(failures, args.i1_b1_weak_continuation is True, "Weak30k must enable weak continuation")
        _require(failures, int(args.i1_b1_weak_start_iter) == 8_001, "weak interval must start at 8001")
        _require(failures, int(args.i1_b1_weak_end_iter) == 30_000, "weak interval must end at 30000")
        _require(failures, _equal_float(args.i1_b1_weak_weight, 0.02), "weak weight must be 0.02")
        _require(failures, args.i1_b1_weak_weight_schedule == "constant", "weak schedule must be constant")
        _require(failures, _equal_float(args.i1_b1_weak_evidence_gamma, 1.0), "weak_evidence_gamma must be 1.0")

    try:
        active_interval = validate_oe_training_configuration(
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
        _require(failures, tuple(active_interval) == (1, 30_000), "I1 must be active continuously at iterations 1--30000")
    except (TypeError, ValueError) as exc:
        failures.append(f"invalid or overlapping I1 intervals: {exc}")

    if variant in EXPECTED_WEIGHTS:
        for iteration, expected in EXPECTED_WEIGHTS[variant].items():
            try:
                actual = effective_weight_from_training_path(args, iteration)
            except (TypeError, ValueError) as exc:
                failures.append(f"training-path weight failed at {iteration}: {exc}")
                continue
            _require(failures, _equal_float(actual, expected), f"effective training-path weight at {iteration} must be {expected}, got {actual}")

    expected_output = f"NS30K_I1SCHED_v1_{scene}_{variant}_seed0"
    _require(failures, Path(dataset.model_path).name == expected_output, f"output directory name must be {expected_output}")
    if failures:
        raise NS30KContractError(PROTOCOL_VERSION + " contract failure: " + "; ".join(failures))
    return Path(args.oe_depth_cache_dir)


# The wrapper patches the legacy symbol to this name before executing train.py.
validate_ns30k_training_configuration = validate_ns30k_schedule_ablation_training_configuration
