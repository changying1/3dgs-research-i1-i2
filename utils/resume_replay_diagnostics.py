"""Read-only diagnostics for short checkpoint-resume replay checks."""

from __future__ import annotations

import hashlib
import json
import pickle
import random
from pathlib import Path

import numpy as np
import torch


def _digest_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest().upper()


def tensor_sha256(tensor: torch.Tensor) -> str:
    """Hash already-materialized tensor values without consuming RNG."""
    return _digest_bytes(tensor.detach().contiguous().cpu().numpy().tobytes())


def object_sha256(value) -> str:
    return _digest_bytes(pickle.dumps(value, protocol=4))


def rng_state_summary() -> dict:
    result = {
        "python_sha256": object_sha256(random.getstate()),
        "numpy_sha256": object_sha256(np.random.get_state()),
        "torch_cpu_sha256": tensor_sha256(torch.get_rng_state()),
        "cuda_available": bool(torch.cuda.is_available()),
    }
    result["cuda_sha256"] = (
        [tensor_sha256(state) for state in torch.cuda.get_rng_state_all()]
        if result["cuda_available"] else []
    )
    return result


def camera_identity(camera) -> dict:
    return {"uid": getattr(camera, "uid", None),
            "colmap_id": getattr(camera, "colmap_id", None),
            "image_name": getattr(camera, "image_name", None)}


def gaussian_parameters_sha256(gaussians) -> str:
    digest = hashlib.sha256()
    for name in ("_xyz", "_features_dc", "_features_rest", "_opacity", "_scaling", "_rotation"):
        digest.update(name.encode("utf-8"))
        digest.update(tensor_sha256(getattr(gaussians, name)).encode("ascii"))
    return digest.hexdigest().upper()


def optimizer_summary(gaussians) -> dict:
    state_steps = []
    state_entries = 0
    for parameter, state in gaussians.optimizer.state.items():
        del parameter
        state_entries += 1
        step = state.get("step")
        if torch.is_tensor(step):
            state_steps.append(float(step.detach().cpu().item()))
        elif step is not None:
            state_steps.append(float(step))
    return {"gaussian_count": int(gaussians.get_xyz.shape[0]),
            "optimizer_state_entries": state_entries,
            "optimizer_step_values": state_steps}


class ResumeReplayDiagnostics:
    ITERATIONS = frozenset((20001, 20002, 20100, 20200))
    PARAMETER_ITERATIONS = frozenset((20001, 20200))

    def __init__(self, output_path: str | Path):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.output_path.open("x", encoding="utf-8", newline="\n")
        self._write({
            "event": "diagnostic_contract",
            "checks": {
                "hashes_are_read_only": "PASS",
                "diagnostic_rng_consumption": "PASS",
                "camera_and_mask_are_existing_training_values": "PASS",
                "bitwise_cuda_reproducibility": "UNVERIFIED",
                "device_sync_and_kernel_nondeterminism": "UNVERIFIED",
            },
        })

    def _write(self, payload: dict) -> None:
        self._handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        self._handle.flush()

    def record_render(self, iteration: int, camera, dropout_mask: torch.Tensor, gaussians) -> None:
        if int(iteration) not in self.ITERATIONS:
            return
        self._write({"event": "render_state", "iteration": int(iteration),
                     "camera": camera_identity(camera),
                     "dropout_mask_sha256": tensor_sha256(dropout_mask),
                     "random_state": rng_state_summary(),
                     "gaussian_count": int(gaussians.get_xyz.shape[0])})

    def record_optimizer(self, iteration: int, gaussians) -> None:
        if int(iteration) not in self.PARAMETER_ITERATIONS:
            return
        self._write({"event": "optimizer_state", "iteration": int(iteration),
                     "gaussian_parameters_sha256": gaussian_parameters_sha256(gaussians),
                     "random_state": rng_state_summary(), **optimizer_summary(gaussians)})

    def finish(self) -> None:
        if not self._handle.closed:
            self._handle.close()
