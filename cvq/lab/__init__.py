"""cvq's lab extensions — contract checkers and scaffolds for cvq's component kinds
(quantizer, ar_head, ...), registered into imagelab's `lab check` / `lab new` seams.
The lab layer itself (run dirs, ledger, tiers, CLI) lives in imagelab/lab/."""

from imagelab.lab.rundir import RunDir, ledger_append, ledger_rows  # noqa: F401
