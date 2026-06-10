"""Legacy entry point — now a shim. The frozen-tokenizer CAR recipe lives in
cvq/tasks/car.py; the shared training spine in cvq/trainer.py.

    python -m cvq.train_car --config configs/car_pokemon_qwen.yaml --tokenizer_ckpt checkpoints/best.pt
"""

from cvq.trainer import run

if __name__ == "__main__":
    run(default_task="car", default_config="configs/car_pokemon_qwen.yaml")
