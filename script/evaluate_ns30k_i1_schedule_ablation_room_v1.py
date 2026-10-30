"""Isolated Room evaluation for NS30K-I1-Schedule-v1.

This entry point reuses the frozen rendering, metric, staging, and audit logic
from ``evaluate_ns30k_i1_schedule_ablation_v1.py`` while selecting only Room
and a separate report directory.  The original Fortress/Horns report is never
read as a staging target or modified.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from script import evaluate_ns30k_i1_schedule_ablation_v1 as shared  # noqa: E402


REPORT = ROOT / "reports" / "ns30k_i1_schedule_ablation_room_v1"
SCENES = ("room",)
FROZEN_REFERENCE_MODEL_DIRS = {
    ("room", "B0"): "NS30K_v1_room_B0_seed0",
    ("room", "End8k"): "NS30K_v1_room_B0+I1_seed0",
}


@contextmanager
def room_profile():
    with shared.evaluation_profile(
        report=REPORT,
        scenes=SCENES,
        references=FROZEN_REFERENCE_MODEL_DIRS,
    ):
        yield


def preflight() -> dict:
    with room_profile():
        return shared.preflight()


def model_path(group: str) -> Path:
    with room_profile():
        return shared.model_path("room", group)


def expected_report_rows() -> tuple[int, int]:
    with room_profile():
        return shared.expected_report_rows()


def main(argv=None) -> int:
    with room_profile():
        return shared.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
