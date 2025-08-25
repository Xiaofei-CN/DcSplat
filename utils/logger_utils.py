import shutil
import os
from pathlib import Path
import logging
from omegaconf import OmegaConf
logger = logging.getLogger()

def file_backup(cfg, args, train_script):
    exp_path = cfg.record.file_path
    if cfg.model.debug:
        logger.info(f"No need to save files while debugging ")
        return
    logger.info("Saving training script")
    shutil.copy(train_script, exp_path)
    logger.info("Saving models")
    shutil.copytree('models', os.path.join(exp_path, 'models'), dirs_exist_ok=True)
    logger.info("Saving config")
    shutil.copytree('config', os.path.join(exp_path, 'config'), dirs_exist_ok=True)
    logger.info("Saving data")
    shutil.copytree('data', os.path.join(exp_path, 'data'), dirs_exist_ok=True)

    a = OmegaConf.to_yaml(cfg)
    with open(os.path.join(exp_path, os.path.basename(args.config)), "w", encoding="utf-8") as f:
        f.write(a)

class AverageMeter:
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count