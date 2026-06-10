"""Legacy entry point — now a shim. The joint EOSTok recipe lives in cvq/tasks/e2e.py;
the shared training spine in cvq/trainer.py.

    python -m cvq.train_e2e --config configs/car_e2e_pokemon.yaml
"""

from cvq.trainer import run

if __name__ == "__main__":
    run(default_task="e2e", default_config="configs/car_e2e_pokemon.yaml")
