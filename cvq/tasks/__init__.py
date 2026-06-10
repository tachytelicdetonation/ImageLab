"""Importing cvq.tasks registers the built-in tasks. Add new task modules here."""

from . import car        # noqa: F401  registers task/car
from . import e2e        # noqa: F401  registers task/e2e
from . import tokenizer  # noqa: F401  registers task/tokenizer

from .base import StepOutput, Task  # noqa: F401
