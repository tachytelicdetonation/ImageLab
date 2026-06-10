"""Back-compat shim — metrics live in cvq.eval.metrics, grad_norm in cvq.utils."""

from cvq.eval.metrics import validate  # noqa: F401
from cvq.utils import grad_norm  # noqa: F401
