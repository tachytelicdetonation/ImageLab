"""Legacy entry point — now a shim. The tokenizer recipe lives in cvq/tasks/tokenizer.py;
the shared training spine in cvq/trainer.py.

    python -m cvq.train --config configs/cvq_pokemon.yaml
"""

from cvq.trainer import run

if __name__ == "__main__":
    run(default_task="tokenizer", default_config="configs/cvq_pokemon.yaml")
