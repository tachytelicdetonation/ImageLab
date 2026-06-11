"""The lab layer: runs as data.

The trainer (imagelab/trainer.py) executes ONE experiment; this package remembers and
compares ALL of them. The organizing rule: the filesystem is the database — every run is
a self-describing folder under runs/, the ledger is a JSONL index over them, and the
`lab` CLI is just queries over those files. No server, no external service.
"""

from imagelab.lab.rundir import RunDir, ledger_append, ledger_rows  # noqa: F401
