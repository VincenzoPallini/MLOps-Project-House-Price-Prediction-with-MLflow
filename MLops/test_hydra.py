print("--- Starting test_hydra.py ---")
import hydra
from omegaconf import DictConfig
import os
print(f"Hydra imported. Current working directory: {os.getcwd()}")


@hydra.main(config_path="conf", config_name="config", version_base=None)
def my_app(cfg: DictConfig):
    print("--- INSIDE my_app function ---")
    try:
        print(f"Config object type: {type(cfg)}")
        print(f"Value from config: {cfg.test_param}")
        print("--- Hydra minimal test SUCCESS ---")
    except Exception as e:
        print(f"Error inside my_app: {e}")

print("--- Decorators processed (my_app defined) ---")

if __name__ == "__main__":
    print("--- Calling my_app via entry point ---")
    my_app()
    print("--- my_app call finished ---")