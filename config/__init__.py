from .config import ExperimentConfig, load_config
import argparse

def get_args():
    parser = argparse.ArgumentParser("Dual-Constraint Human Gaussian Splatting")
    parser.add_argument("--config", required=False, default="config/testdual512.yaml", help="path to config file")
    args, extras = parser.parse_known_args()
    return args,extras
