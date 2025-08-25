import open3d as o3d
import time
import datetime
import sys
import warnings
from accelerate.tracking import TensorBoardTracker

import torch
from models.lr_scheduler import LinearWarmupMultiStepDecayLRScheduler
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs,set_seed
from utils.utils import *
from utils.logger_utils import *
from models.lossFunction import l1_loss, ssim

warnings.filterwarnings("ignore", category=UserWarning)

from config import ExperimentConfig, load_config, get_args
from models.GS3D import *
from models.baseFunction import UnetExtractor,DINOv2
from torch.cuda.amp import autocast as autocast
from BaseTrainer import BaseTrainer
from models.perceptual import PerceptualLoss
from models.depth_refiner import Refiner


import logging
logger = logging.getLogger()

class Trainer(BaseTrainer):
    def __init__(self, cfg,accelerator):
        super().__init__(cfg,accelerator)
        logger.info("Set up image encoder...")
        self.img_encoder = UnetExtractor(in_channel=3, encoder_dim=self.cfg.model.raft.encoder_dims).cuda()
        self.depth_refiner = Refiner(in_channel=1,rgb_dim=3,encoder_dim=self.cfg.model.gsnet.encoder_dims,
                                           decoder_dim=self.cfg.model.gsnet.decoder_dims, predict_depth=True)

        self.model_registry += ["img_encoder", "depth_refiner"]

        trainable_params,params = self.caculate_totoal_params(self.model_registry)
        logger.info(f"number of trainable parameters: {trainable_params}")

        logger.info("preparing optimizer...")
        self.optimizer = torch.optim.AdamW(params, lr=self.cfg.model.lr, weight_decay=self.cfg.model.wdecay, eps=1e-8)


        self.img_encoder, self.depth_refiner,self.train_dataloader, self.val_dataloader,self.optimizer = \
            self.accelerator.prepare(
                self.img_encoder, self.depth_refiner, self.train_dataloader,
                self.val_dataloader, self.optimizer)
        if cfg.model.resume:
            self.resume()

        self.lr_scheduler = LinearWarmupMultiStepDecayLRScheduler(
        self.optimizer, 1000, 0.1, 0.1,
        cfg.model.num_epochs, [cfg.model.num_epochs//2], len(self.train_dataloader),
        last_epoch=len(self.train_dataloader)*self.start_epoch-1, override_lr=0.)

    def step(self, data, is_train=True):
        bs, v = data["source_view"]["img"].shape[:2]
        img = rearrange(data["source_view"]["img"].cuda(), "b v c h w -> (b v) c h w")
        depth_gt = rearrange(data["source_view"]["depth"].cuda(), "b v h w -> (b v) h w").unsqueeze(1)
        depth_smpl = rearrange(data["source_view"]["depth_smpl"].cuda(), "b v h w -> (b v) h w").unsqueeze(1)
        mask = rearrange(data["source_view"]["mask"].cuda(), "b v c h w -> (b v) c h w").to(torch.bool)[:,:1,...]
        with autocast(enabled=self.cfg.model.raft.mixed_precision):
            img_feat = self.img_encoder(img)
        out = self.depth_refiner(depth_smpl,img_feat[-1])
        out_refined_depth = out.clone()
        if is_train:
            depth_mask = depth_gt != 0
            valid_mask = depth_mask & mask
            refine_loss = (depth_gt[valid_mask].detach() - out[valid_mask]).abs().mean()
            return refine_loss
        else:
            return out_refined_depth


    def train(self):
        logger.info("start training...")
        start_time = time.time()
        end_time = time.time()
        self.set_all_model_train()
        for epoch in range(self.start_epoch,self.end_epoch):
            epoch_time = time.time()
            logger.info(f"epoch {epoch + 1} start")
            batch_time = AverageMeter()
            total_loss = AverageMeter()
            for index, data in enumerate(self.train_dataloader):
                with self.accelerator.accumulate(
                        self.img_encoder,self.depth_refiner):
                    self.optimizer.zero_grad()

                    loss = self.step(data)

                    if torch.isnan(loss).any():
                        self.accelerator.set_trigger()
                    if self.accelerator.check_trigger():
                        logger.info("loss is nan, stop training")
                        self.accelerator.end_training()

                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        self.total_steps += 1
                    self.optimizer.step()
                    self.lr_scheduler.step()


                batch_time.update(time.time() - end_time)
                total_loss.update(loss.item())
                end_time = time.time()

                if (self.total_steps) % self.cfg.record.loss_freq == 0 or self.cfg.model.debug:
                    self.accelerator.log(
                        {
                            "loss_avg": total_loss.avg,
                            "lr": self.optimizer.param_groups[-1]["lr"]
                        },step=self.total_steps
                    )
                    etas = batch_time.avg * (len(self.train_dataloader) - 1 - index)
                    logger.info(
                        f"Train EPOCH[{epoch+1}|{index + 1}/{len(self.train_dataloader)}] "
                        f"Time {batch_time.val:.4f}({batch_time.avg:.4f})  "
                        f"Loss {total_loss.val:.4f}({total_loss.avg:.4f})  "
                        f"Lr {self.optimizer.param_groups[-1]['lr']:.8f}  "
                        f"Eta {datetime.timedelta(seconds=int(etas))}")

                if (self.total_steps) % self.cfg.record.save_freq == 0 or self.cfg.model.debug:
                    logger.info(f"Saving model {self.total_steps}...")
                    path = os.path.join(self.cfg.record.ckpt_path, f"iteration_{(epoch + 1):03d}")
                    self.accelerator.save_state(path)
            logger.info(
                f"epoch {epoch + 1} finished, running time {datetime.timedelta(seconds=int(time.time() - epoch_time))}")
        logger.info(f"Saving model {self.total_steps}...")
        path = os.path.join(self.cfg.record.ckpt_path, f"iteration_latest")
        self.accelerator.save_state(path)
        logger.info(f'training completed, running time {datetime.timedelta(seconds=int(time.time() - start_time))}')
        self.accelerator.end_training()


def main():
    fmt = "[%(asctime)s %(filename)s:%(lineno)s] %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    args, extras = get_args()
    cfg: ExperimentConfig = load_config(args.config, cli_args=extras)

    exp_name = f"{cfg.dataset['name']}_{cfg.model['name']}"
    cfg.record.ckpt_path = "experiments/exp/%s/ckpt" % (exp_name)
    cfg.record.show_path = "experiments/exp/%s/show" % (exp_name)
    cfg.record.logs_path = "experiments/exp/%s/logs" % (exp_name)
    cfg.record.file_path = "experiments/exp/%s/file" % (exp_name)

    makeDirsFromList([cfg.record.ckpt_path, cfg.record.show_path,
                      cfg.record.logs_path, cfg.record.file_path])
    accelerator = Accelerator(
        log_with=["tensorboard"],
        project_dir=cfg.record.ckpt_path,
        mixed_precision="no",
        gradient_accumulation_steps=1,
        kwargs_handlers=[DistributedDataParallelKwargs(bucket_cap_mb=200, gradient_as_bucket_view=True)]
    )

    if accelerator.is_main_process:
        accelerator.trackers = []
        accelerator.trackers.append(TensorBoardTracker(exp_name, cfg.record.logs_path))

    accelerator.wait_for_everyone()

    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        datefmt=datefmt,
        filename=f"{cfg.record.logs_path}/log_{exp_name}.txt",
        filemode="a"
    )
    if accelerator.is_main_process:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter(fmt, datefmt))
        logger.addHandler(console_handler)

    file_backup(cfg, args, train_script=os.path.basename(__file__))
    logger.info("Setup seed")
    set_seed(12138)

    start_time = time.time()
    trainer = Trainer(cfg,accelerator)
    trainer.train()
    # trainer.test()
    # test(cfg)
    train_time = time.time() - start_time
    logger.info(f'training completed, running time {datetime.timedelta(seconds=int(train_time))}')


if __name__ == '__main__':
    main()