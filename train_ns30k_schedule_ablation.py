"""Training entry point for NS30K-I1-Schedule-v1 only.

It executes the frozen train.py and changes only which strict NS30K protocol
validator is imported.  Direct invocation without launcher authorization is
rejected by that validator before output creation.
"""

from __future__ import annotations

import runpy
from pathlib import Path

from utils import ns30k_sampling
from utils.ns30k_schedule_ablation import (
    validate_ns30k_schedule_ablation_training_configuration,
)


ns30k_sampling.validate_ns30k_training_configuration = (
    validate_ns30k_schedule_ablation_training_configuration
)
runpy.run_path(str(Path(__file__).resolve().with_name("train.py")), run_name="__main__")
