"""The lab layer: runs as data.

The training stack (cvq/trainer.py + cvq/tasks/) executes ONE experiment; this package
remembers and compares ALL of them. The organizing rule: the filesystem is the database —
every run is a self-describing folder under runs/, the ledger is a JSONL index over them,
and the `lab` CLI is just queries over those files. No server, no external service.
"""

from cvq.lab.rundir import RunDir, ledger_append, ledger_rows  # noqa: F401
