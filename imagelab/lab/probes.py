"""
`lab check <kind> <name>` — component contracts as instant probes, not just CI.

The harness is generic; the probes are a project's own. A project defines what its
component kinds must satisfy and registers a checker per kind (e.g. a quantizer
checker asserting shapes, encode/decode round-trips, straight-through gradients):

    from imagelab.lab.probes import Probe, register_checker

    @register_checker("quantizer")
    def check_quantizer(name: str, kwargs: dict) -> int:
        p = Probe()
        p.check("constructs", lambda: ...)
        ...
        return p.done("quantizer", name)

Checker modules get imported via pyproject.toml [tool.imagelab] imports. The framework
ships exactly one checker — "task" — because the Task seam is the only contract it owns.
"""

from __future__ import annotations

from typing import Callable

CHECKERS: dict[str, Callable[[str, dict], int]] = {}


def register_checker(kind: str):
    def deco(fn):
        CHECKERS[kind] = fn
        return fn
    return deco


class Probe:
    """✓/✗ probe runner: each check is one bug class caught in seconds, not GPU-hours."""

    def __init__(self):
        self.failed = 0

    def check(self, desc: str, fn):
        try:
            fn()
            print(f"  \033[32m✓\033[0m {desc}")
        except Exception as e:
            print(f"  \033[31m✗\033[0m {desc}\n      {type(e).__name__}: {e}")
            self.failed += 1

    def done(self, kind: str, name: str) -> int:
        if self.failed:
            print(f"\n{kind}/{name}: {self.failed} contract violation(s) — fix before "
                  f"training (a run would fail later and less legibly)")
            return 1
        print(f"\n{kind}/{name}: contract OK — ready for `lab run ... --tier overfit`")
        return 0


def run_checks(kind: str, name: str, kwargs: dict) -> int:
    import torch
    torch.manual_seed(0)
    fn = CHECKERS.get(kind)
    if fn is None:
        print(f"no checker registered for kind {kind!r} (have: "
              f"{sorted(CHECKERS) or '(none)'})\n"
              f"register one with @imagelab.lab.probes.register_checker({kind!r}) in a "
              f"module listed under [tool.imagelab] imports")
        return 1
    print(f"checking {kind}/{name} {kwargs or ''}")
    return fn(name, kwargs)


def assert_(cond, msg: str):
    if not cond:
        raise AssertionError(msg)


# --------------------------------------------------------------------------- #
# The one framework-owned contract: the Task seam itself
# --------------------------------------------------------------------------- #
@register_checker("task")
def _check_task(name: str, _kwargs: dict) -> int:
    from imagelab.task import Task
    if ":" in name:                              # file.py:Class / module:Class specs work too
        from imagelab.trainer import resolve_task
        cls = resolve_task(name)
    else:
        from imagelab.registry import get
        try:
            cls = get("task", name)
        except KeyError as e:
            print(e.args[0])
            print("(is the defining module listed under [tool.imagelab] imports?)")
            return 1
    p = Probe()
    p.check("subclasses imagelab.Task", lambda: assert_(
        isinstance(cls, type) and issubclass(cls, Task), f"{cls} is not a Task"))
    p.check("implements setup()", lambda: assert_(
        cls.setup is not Task.setup, "setup() not overridden"))
    step_impl = (cls.step_fn is not Task.step_fn) or \
                (cls.generator_fn is not Task.generator_fn)
    p.check("implements step_fn (or generator_fn for gan=True)", lambda: assert_(
        step_impl, "no per-step work implemented"))
    p.check("overfit_gate is well-formed (or None)", lambda: assert_(
        cls.overfit_gate is None or (len(cls.overfit_gate) == 3
                                     and cls.overfit_gate[1] in ("<=", ">=")),
        f"got {cls.overfit_gate!r}, want (metric, '<='|'>=', threshold)"))
    if cls.overfit_gate is None:
        print("  i no overfit_gate declared -> `--tier overfit` will report 'unknown'")
    wiring = "GANStep (generator_fn/discriminator_fn)" if cls.gan else "NoGANStep (step_fn)"
    print(f"  i gan={cls.gan} -> trainer wires {wiring}; ckpt prefix {cls.ckpt_prefix!r}")
    return p.done("task", name)
