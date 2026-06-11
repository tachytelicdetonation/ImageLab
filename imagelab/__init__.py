"""
imagelab — a small lab for single-GPU model experimentation.

The framework owns the run lifecycle: every run is a self-describing folder + a ledger
row, with tiered execution (smoke/overfit/fast/full) and a CLI for comparison. You own
everything scientific: model, data, loss, metrics — declared on ONE class:

    from imagelab import Task, StepOutput

See imagelab/task.py for the contract and examples/ for complete model families
(DDPM, rectified flow) built on it.
"""

from imagelab.config import ConfigError, CoreConfig  # noqa: F401
from imagelab.loop import StepOutput                 # noqa: F401
from imagelab.registry import available, build, get, meta, register  # noqa: F401
from imagelab.task import Task                       # noqa: F401

__version__ = "0.4.0"
