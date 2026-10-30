"""CPU-testable Innovation 1 training-path integration and diagnostics."""

from __future__ import annotations

import json
import math

import torch

from utils.evidence_structural_loss import compute_oe_structural_loss, should_apply_oe_structural_loss
from utils.oe_depth_cache import validate_rendered_depth


I1_DIAGNOSTIC_INTERVAL = 1000


def validate_oe_training_configuration(
    enabled, weight, start_iter, end_iter, total_iterations, *, first_iteration=1,
    weak_continuation=False, weak_start_iter=8001, weak_end_iter=30000,
    weak_weight=0.02, weak_evidence_gamma=1.0, weak_weight_schedule="constant",
):
    """Reject enabled I1 runs whose primary and weak phases are both unreachable."""
    if not bool(enabled):
        return None
    weight = float(weight)
    start_iter = int(start_iter)
    end_iter = int(end_iter)
    total_iterations = int(total_iterations)
    first_iteration = int(first_iteration)
    if not math.isfinite(weight) or weight <= 0.0:
        raise ValueError("I1 is enabled but --oe_structural_weight must be finite and > 0.")
    if start_iter > end_iter:
        raise ValueError("I1 activation interval is invalid: start_iter is greater than end_iter.")
    intervals = [(start_iter, end_iter)]
    if bool(weak_continuation):
        weak_weight = float(weak_weight)
        weak_start_iter = int(weak_start_iter)
        weak_end_iter = int(weak_end_iter)
        if not math.isfinite(weak_weight) or weak_weight <= 0.0:
            raise ValueError("I1 weak continuation weight must be finite and > 0.")
        weak_evidence_gamma = float(weak_evidence_gamma)
        if not math.isfinite(weak_evidence_gamma) or weak_evidence_gamma <= 0.0:
            raise ValueError("I1 weak continuation evidence gamma must be finite and > 0.")
        if weak_start_iter > weak_end_iter:
            raise ValueError("I1 weak continuation interval is invalid: start_iter is greater than end_iter.")
        if weak_weight_schedule not in ("constant", "linear_decay"):
            raise ValueError(f"Unsupported I1 weak continuation weight schedule: {weak_weight_schedule!r}.")
        if weak_weight_schedule == "linear_decay" and weak_start_iter >= weak_end_iter:
            raise ValueError("I1 linear-decay weak continuation requires start_iter < end_iter.")
        weak_last_active_iter = (
            weak_end_iter - 1 if weak_weight_schedule == "linear_decay" else weak_end_iter
        )
        intervals.append((weak_start_iter, weak_last_active_iter))
    reachable = [
        (max(first_iteration, phase_start), min(total_iterations, phase_end))
        for phase_start, phase_end in intervals
    ]
    reachable = [(start, end) for start, end in reachable if start <= end]
    if not reachable:
        raise ValueError(
            "I1 is enabled but cannot activate: no structural phase can activate during this training run: "
            f"run=[{first_iteration}, {total_iterations}]."
        )
    return min(start for start, _ in reachable), max(end for _, end in reachable)


def oe_activation_details(
    enabled,
    iteration,
    start_iter,
    end_iter,
    weight,
    *,
    weak_continuation=False,
    weak_start_iter=8001,
    weak_end_iter=30000,
    weak_weight=0.02,
    weak_weight_schedule="constant",
):
    """Return the effective I1 phase and weight without touching tensors."""
    primary = should_apply_oe_structural_loss(enabled, iteration, start_iter, end_iter, weight)
    if weak_weight_schedule not in ("constant", "linear_decay"):
        raise ValueError(f"Unsupported I1 weak continuation weight schedule: {weak_weight_schedule!r}.")
    effective_weak_weight = float(weak_weight)
    if weak_weight_schedule == "linear_decay":
        if int(weak_start_iter) >= int(weak_end_iter):
            raise ValueError("I1 linear-decay weak continuation requires start_iter < end_iter.")
        fraction = (int(weak_end_iter) - int(iteration)) / (
            int(weak_end_iter) - int(weak_start_iter)
        )
        effective_weak_weight *= min(max(fraction, 0.0), 1.0)
    if not math.isfinite(effective_weak_weight) or effective_weak_weight < 0.0:
        raise ValueError("I1 weak continuation applied weight must be finite and >= 0.")
    weak = bool(weak_continuation) and should_apply_oe_structural_loss(
        enabled, iteration, weak_start_iter, weak_end_iter, effective_weak_weight
    )
    if primary:
        return {"active": True, "phase": "primary", "weight": float(weight)}
    if weak:
        return {"active": True, "phase": "b1_weak_continuation", "weight": effective_weak_weight}
    return {"active": False, "phase": "inactive", "weight": 0.0}


def apply_oe_structural_training_loss(
    baseline_total_loss,
    *,
    enabled,
    iteration,
    start_iter,
    end_iter,
    weight,
    depth_cache=None,
    camera=None,
    rendered_depth=None,
    gt_rgb=None,
    gate_mode="stable_evidence",
    stable_quantile=0.80,
    conflict_threshold=0.75,
    weak_continuation=False,
    weak_start_iter=8001,
    weak_end_iter=30000,
    weak_weight=0.02,
    weak_evidence_gamma=1.0,
    weak_weight_schedule="constant",
    diagnostics=None,
    compute_fn=compute_oe_structural_loss,
    validate_fn=validate_rendered_depth,
):
    """Combine losses without touching cache or depth when the I1 gate is closed."""
    activation = oe_activation_details(
        enabled, iteration, start_iter, end_iter, weight,
        weak_continuation=weak_continuation, weak_start_iter=weak_start_iter,
        weak_end_iter=weak_end_iter, weak_weight=weak_weight,
        weak_weight_schedule=weak_weight_schedule,
    )
    if not activation["active"]:
        if (
            diagnostics is not None
            and weak_continuation
            and weak_weight_schedule == "linear_decay"
            and int(iteration) == int(weak_end_iter)
        ):
            diagnostics.record_phase_transition(
                iteration,
                phase="inactive",
                applied_weight=0.0,
                reason="linear_decay_endpoint_reached",
            )
        return baseline_total_loss, None, None
    if depth_cache is None:
        raise RuntimeError("I1 is active but its validated depth cache is unavailable.")
    depth = validate_fn(rendered_depth, camera.image_height, camera.image_width)
    reference = depth_cache.load_for_camera(camera, device=depth.device, dtype=depth.dtype)
    applied_gamma = (
        float(weak_evidence_gamma)
        if activation["phase"] == "b1_weak_continuation"
        else 1.0
    )
    if not math.isfinite(applied_gamma) or applied_gamma <= 0.0:
        raise ValueError("I1 weak continuation evidence gamma must be finite and > 0.")
    structural_loss, stats, _ = compute_fn(
        depth,
        reference,
        gt_rgb=gt_rgb,
        gate_mode=gate_mode,
        stable_quantile=stable_quantile,
        conflict_threshold=conflict_threshold,
        evidence_gamma=applied_gamma,
    )
    applied_weight = activation["weight"]
    stats["i1_phase"] = activation["phase"]
    stats["i1_applied_weight"] = applied_weight
    stats["i1_evidence_gamma"] = applied_gamma
    weighted_loss = applied_weight * structural_loss
    if not torch.isfinite(structural_loss).all() or not torch.isfinite(weighted_loss).all():
        if diagnostics is not None:
            diagnostics.record_nonfinite(iteration, structural_loss, weighted_loss, stats)
        raise FloatingPointError(f"I1 produced a non-finite structural loss at iteration {iteration}.")
    if diagnostics is not None:
        diagnostics.record(
            iteration, structural_loss, weighted_loss, stats,
            force=int(iteration) in (8001, int(weak_start_iter)),
        )
    return baseline_total_loss + weighted_loss, structural_loss, stats


class I1Diagnostics:
    """Emit auditable JSON lines and retain an end-of-run activation summary."""

    def __init__(self, parameters, active_interval, *, interval=I1_DIAGNOSTIC_INTERVAL, emit=print, writer=None):
        self.parameters = dict(parameters)
        self.active_interval = tuple(active_interval)
        self.interval = int(interval)
        if self.interval <= 0:
            raise ValueError("I1 diagnostic interval must be positive.")
        self.emit = emit
        self.writer = writer
        self.activation_count = 0
        self.valid_pair_count = 0
        self.active_pair_count = 0
        self.nonfinite_seen = False
        self._emit({
            "event": "configuration",
            "parameters": self.parameters,
            "actual_active_start": self.active_interval[0],
            "actual_active_end": self.active_interval[1],
            "diagnostic_interval": self.interval,
        })

    def record(self, iteration, structural_loss, weighted_loss, stats, *, force=False):
        self.activation_count += 1
        valid_pairs = int(stats.get("valid_pair_count", 0))
        active_pairs = int(stats.get("active_pair_count", 0))
        self.valid_pair_count += valid_pairs
        self.active_pair_count += active_pairs
        payload = {
            "event": "activation",
            "iteration": int(iteration),
            "oe_struct_loss": float(structural_loss.detach().item()),
            "weighted_oe_loss": float(weighted_loss.detach().item()),
            "valid_pair_count": valid_pairs,
            "active_pair_count": active_pairs,
            "valid_structure_pair_count": valid_pairs,
            "mean_pair_evidence": stats.get("mean_pair_evidence"),
            "reference_transform": stats.get("reference_transform"),
            "i1_phase": stats.get("i1_phase"),
            "i1_applied_weight": stats.get("i1_applied_weight"),
            "i1_evidence_gamma": stats.get("i1_evidence_gamma"),
        }
        if force or self.activation_count == 1 or int(iteration) % self.interval == 0:
            self._emit(payload)
        if self.writer is not None:
            self.writer.add_scalar("i1/oe_struct_loss", payload["oe_struct_loss"], iteration)
            self.writer.add_scalar("i1/weighted_oe_loss", payload["weighted_oe_loss"], iteration)
            self.writer.add_scalar("i1/valid_pair_count", valid_pairs, iteration)
            self.writer.add_scalar("i1/active_pair_count", active_pairs, iteration)
            if payload["mean_pair_evidence"] is not None:
                self.writer.add_scalar("i1/mean_pair_evidence", payload["mean_pair_evidence"], iteration)
            self.writer.add_scalar("i1/evidence_gamma", payload["i1_evidence_gamma"], iteration)
            self.writer.add_text("i1/reference_transform", str(payload["reference_transform"]), iteration)

    def record_nonfinite(self, iteration, structural_loss, weighted_loss, stats):
        self.nonfinite_seen = True
        self.activation_count += 1
        valid_pairs = int(stats.get("valid_pair_count", 0))
        active_pairs = int(stats.get("active_pair_count", 0))
        self.valid_pair_count += valid_pairs
        self.active_pair_count += active_pairs
        self._emit({
            "event": "nonfinite_loss",
            "iteration": int(iteration),
            "oe_struct_loss": _safe_tensor_value(structural_loss),
            "weighted_oe_loss": _safe_tensor_value(weighted_loss),
            "valid_pair_count": valid_pairs,
            "active_pair_count": active_pairs,
            "mean_pair_evidence": stats.get("mean_pair_evidence"),
            "reference_transform": stats.get("reference_transform"),
            "i1_phase": stats.get("i1_phase"),
            "i1_applied_weight": stats.get("i1_applied_weight"),
            "i1_evidence_gamma": stats.get("i1_evidence_gamma"),
            "actual_activation_count": self.activation_count,
            "total_valid_pair_count": self.valid_pair_count,
            "total_active_pair_count": self.active_pair_count,
            "nonfinite_loss_seen": True,
        })

    def record_phase_transition(self, iteration, *, phase, applied_weight, reason):
        """Log an inactive boundary explicitly without counting it as an activation."""
        payload = {
            "event": "phase_transition",
            "iteration": int(iteration),
            "i1_active": False,
            "i1_phase": str(phase),
            "i1_applied_weight": float(applied_weight),
            "reason": str(reason),
        }
        self._emit(payload)
        if self.writer is not None:
            self.writer.add_scalar("i1/applied_weight", payload["i1_applied_weight"], iteration)
            self.writer.add_text("i1/phase", payload["i1_phase"], iteration)
        return payload

    def record_densification_gradients(
        self,
        iteration,
        baseline_gradient,
        joint_gradient,
        visible_mask,
        *,
        protection_effective,
    ):
        """Emit gradient comparisons only at the configured low-frequency interval."""
        if int(iteration) % self.interval != 0:
            return None
        baseline_norm = torch.norm(baseline_gradient[:, :2], dim=-1)
        joint_norm = torch.norm(joint_gradient[:, :2], dim=-1)
        visible_baseline = baseline_norm[visible_mask]
        visible_joint = joint_norm[visible_mask]
        payload = {
            "event": "densification_gradient",
            "iteration": int(iteration),
            "protection_effective": bool(protection_effective),
            "baseline_gradient_mean": float(baseline_norm.mean().item()),
            "joint_gradient_mean": float(joint_norm.mean().item()),
            "visible_baseline_gradient_mean": (
                float(visible_baseline.mean().item()) if visible_baseline.numel() else 0.0
            ),
            "visible_joint_gradient_mean": (
                float(visible_joint.mean().item()) if visible_joint.numel() else 0.0
            ),
            "visible_gaussian_count": int(visible_baseline.numel()),
        }
        self._emit(payload)
        if self.writer is not None:
            self.writer.add_scalar(
                "i1/densification_baseline_gradient_mean",
                payload["visible_baseline_gradient_mean"],
                iteration,
            )
            self.writer.add_scalar(
                "i1/densification_joint_gradient_mean",
                payload["visible_joint_gradient_mean"],
                iteration,
            )
        return payload

    def finish(self):
        payload = {
            "event": "summary",
            "actual_activation_count": self.activation_count,
            "total_valid_pair_count": self.valid_pair_count,
            "total_active_pair_count": self.active_pair_count,
            "nonfinite_loss_seen": self.nonfinite_seen,
        }
        self._emit(payload)
        return payload

    def _emit(self, payload):
        self.emit("[I1] " + json.dumps(payload, sort_keys=True, allow_nan=False))


def _safe_tensor_value(value):
    result = float(torch.as_tensor(value).detach().item())
    return result if math.isfinite(result) else str(result)
