"""Pure-Python contracts for observation-guided initial Gaussian opacity."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Sequence


VALID_ARMS = ("control", "confidence", "shuffle")
BASE_OPACITY = 0.1
DEFAULT_BETA = 0.25
DEFAULT_SHUFFLE_SEED = 20261023


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_arm(arm: str, beta: float) -> None:
    if arm not in VALID_ARMS:
        raise ValueError(f"initial opacity arm must be one of {VALID_ARMS}, got {arm!r}")
    if not math.isfinite(beta) or not 0 <= beta < 1:
        raise ValueError("initial opacity beta must be finite and in [0, 1)")


def rank_confidences(valid_errors: Sequence[tuple[int, float]], point_count: int) -> tuple[list, list]:
    """Return 1-based ranks and confidences; invalid points remain neutral."""
    if point_count < 1:
        raise ValueError("point_count must be positive")
    ordered = sorted(valid_errors, key=lambda item: (item[1], item[0]))
    if len({index for index, _ in ordered}) != len(ordered):
        raise ValueError("duplicate point index in valid errors")
    ranks = [None] * point_count
    confidence = [0.5] * point_count
    valid_count = len(ordered)
    for rank, (index, error) in enumerate(ordered, 1):
        if not 0 <= index < point_count or not math.isfinite(error):
            raise ValueError("invalid point index or photometric error")
        ranks[index] = rank
        confidence[index] = 1.0 - (rank - 0.5) / valid_count
    return ranks, confidence


def normalized_opacities(confidence: Sequence[float], beta: float = DEFAULT_BETA,
                         base_opacity: float = BASE_OPACITY) -> list[float]:
    if not confidence:
        raise ValueError("confidence array is empty")
    if not 0 < base_opacity < 1:
        raise ValueError("base opacity must be in (0, 1)")
    validate_arm("confidence", beta)
    raw = []
    for value in confidence:
        if not math.isfinite(value) or not 0 < value < 1:
            raise ValueError("confidence must be finite and strictly inside (0, 1)")
        raw.append(base_opacity * (1.0 + beta * (2.0 * value - 1.0)))
    target_sum = base_opacity * len(raw)
    normalization = target_sum / math.fsum(raw)
    result = [value * normalization for value in raw]
    # Correct the final floating-point residue without changing rank direction.
    result[-1] += target_sum - math.fsum(result)
    if any(not math.isfinite(value) or not 0 < value < 1 for value in result):
        raise ValueError("modulated opacity is not finite and legal")
    if not math.isclose(math.fsum(result), target_sum, rel_tol=0, abs_tol=1e-12):
        raise ValueError("opacity budget normalization failed")
    return result


def build_arm_opacities(confidence: Sequence[float], beta: float = DEFAULT_BETA,
                        shuffle_seed: int = DEFAULT_SHUFFLE_SEED) -> dict[str, list[float]]:
    control = [BASE_OPACITY] * len(confidence)
    guided = normalized_opacities(confidence, beta)
    shuffled = guided.copy()
    random.Random(shuffle_seed).shuffle(shuffled)
    return {"control": control, "confidence": guided, "shuffle": shuffled}


def probability_to_logit(value: float) -> float:
    if not 0 < value < 1:
        raise ValueError("opacity probability must be in (0, 1)")
    return math.log(value / (1.0 - value))


def load_verified_arm_opacities(score_csv, score_manifest, source_ply, points,
                                arm: str, beta: float, shuffle_seed: int) -> list[float]:
    """Load scoring artifact and hard-check its hash and PLY/Gaussian row identity."""
    validate_arm(arm, beta)
    score_csv = Path(score_csv).resolve()
    score_manifest = Path(score_manifest).resolve()
    source_ply = Path(source_ply).resolve()
    metadata = json.loads(score_manifest.read_text(encoding="utf-8"))
    if metadata.get("source_ply_sha256") != file_sha256(source_ply):
        raise ValueError("initial opacity scoring PLY SHA256 mismatch")
    if metadata.get("score_csv_sha256") != file_sha256(score_csv):
        raise ValueError("initial opacity score CSV SHA256 mismatch")
    if metadata.get("point_count") != len(points):
        raise ValueError("initial opacity scoring/Gaussian point-count mismatch")
    confidence = []
    with score_csv.open(encoding="utf-8", newline="") as handle:
        rows = csv.DictReader(handle)
        for expected_index, (row, point) in enumerate(zip(rows, points)):
            if int(row["point_index"]) != expected_index:
                raise ValueError("initial opacity score row index is not one-to-one/in order")
            scored_xyz = tuple(float(row[key]) for key in ("x", "y", "z"))
            if scored_xyz != tuple(float(value) for value in point):
                raise ValueError(f"initial opacity score XYZ row mismatch at {expected_index}")
            confidence.append(float(row["confidence"]))
        extra = next(rows, None)
    if extra is not None or len(confidence) != len(points):
        raise ValueError("initial opacity score CSV row-count mismatch")
    return build_arm_opacities(confidence, beta, shuffle_seed)[arm]
