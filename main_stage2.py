import numpy as np
import open3d as o3d
import time
import datetime
import sys
import warnings
from accelerate.tracking import TensorBoardTracker

import torch
from models.lr_scheduler import LinearWarmupMultiStepDecayLRScheduler
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, set_seed
from utils.utils import *
from utils.logger_utils import *
from models.lossFunction import l1_loss, ssim
from data import build_dataloader

warnings.filterwarnings("ignore", category=UserWarning)

from config import ExperimentConfig, load_config, get_args
from models.GS3D import GSRegresser
from models.baseFunction import UnetExtractor
from torch.cuda.amp import autocast as autocast
from BaseTrainer import BaseTrainer
from models.perceptual import PerceptualLoss
from models.depth_refiner import Refiner
from models.transformer import SideDecoder

import logging

logger = logging.getLogger()


class Trainer(BaseTrainer):
    def __init__(self, cfg, accelerator):
        super().__init__(cfg, accelerator)
        logger.info("Set up 3DGs...")
        self.gs_parm_regresser = GSRegresser(self.cfg, rgb_dim=3, depth_dim=1).cuda()
        logger.info("Set up image encoder...")
        self.img_encoder = UnetExtractor(in_channel=3, encoder_dim=self.cfg.model.raft.encoder_dims).cuda()
        self.ip_encoder = UnetExtractor(in_channel=3, encoder_dim=self.cfg.model.raft.encoder_dims).cuda()
        self.depth_encoder = UnetExtractor(in_channel=1, encoder_dim=cfg.model.gsnet.encoder_dims).cuda()
        self.transformer = SideDecoder(n_ctx=1024, ctx_dim=self.cfg.model.raft.encoder_dims[-1], heads=8, depth=4).cuda()
        self.model_registry += ["img_encoder", "ip_encoder", "gs_parm_regresser", "transformer", "depth_encoder"]

        trainable_params, params = self.caculate_totoal_params(self.model_registry)
        logger.info(f"number of trainable parameters: {trainable_params}")

        logger.info("preparing optimizer...")
        self.vgg_loss = PerceptualLoss().cuda()
        self.optimizer = torch.optim.AdamW(params, lr=self.cfg.model.lr, weight_decay=self.cfg.model.wdecay, eps=1e-8)
        self.depth_refiner = None
        if cfg.model.stage1:
            self.depth_refiner = Refiner(in_channel=1, rgb_dim=3, encoder_dim=self.cfg.model.gsnet.encoder_dims,
                                         decoder_dim=self.cfg.model.gsnet.decoder_dims, predict_depth=True).cuda()
            self.resume_stage1(cfg.model.stage1)

        self.gs_parm_regresser, self.img_encoder, self.ip_encoder, self.depth_encoder, self.transformer, self.train_dataloader, self.val_dataloader, self.optimizer = \
            self.accelerator.prepare(
                self.gs_parm_regresser, self.img_encoder, self.ip_encoder, self.depth_encoder,self.transformer, self.train_dataloader,
                self.val_dataloader, self.optimizer)
        if cfg.model.resume:
            self.resume()

        self.lr_scheduler = LinearWarmupMultiStepDecayLRScheduler(
            self.optimizer, 1000, 0.1, 0.1,
            cfg.model.num_epochs, [cfg.model.num_epochs // 2], len(self.train_dataloader),
            last_epoch=len(self.train_dataloader) * self.start_epoch - 1, override_lr=0.)

    def gs_multi_view(self, bs, depth, data, rgb_maps, rot_maps, scale_maps, opacity_maps):
        rgb_maps = rearrange(rgb_maps, "(b v) c h w -> b v c h w", v=self.cfg.dataset.pvs)
        rot_maps = rearrange(rot_maps, "(b v) c h w -> b v c h w", v=self.cfg.dataset.pvs)
        scale_maps = rearrange(scale_maps, "(b v) c h w -> b v c h w", v=self.cfg.dataset.pvs)
        opacity_maps = rearrange(opacity_maps, "(b v) c h w -> b v c h w", v=self.cfg.dataset.pvs)

        pts_valid = depth != 0.0
        pts_valid = pts_valid.view(pts_valid.shape[0], -1)
        extr = data["source_view"]['extr'].cuda()
        intr = data["source_view"]['intr'].cuda()
        pc = depth2pc(depth, extr, intr)
        pc = rearrange(pc, "(b v) h w -> b v h w", v=self.cfg.dataset.pvs)
        pts_valid = rearrange(pts_valid, "(b v) c -> b v c", v=self.cfg.dataset.pvs)
        render_novel_list = []
        for i in range(bs):
            xyz_i_valid = []
            rgb_i_valid = []
            rot_i_valid = []
            scale_i_valid = []
            opacity_i_valid = []
            for vs in range(pts_valid.shape[1]):
                valid_i = pts_valid[i][vs]
                xyz_i = pc[i][vs]
                rgb_i = rgb_maps[i][vs]#source_img[i] * 0.5 + 0.5 if vs == 0 else rgb_maps[i][vs]
                rgb_i = rgb_i.permute(1, 2, 0).view(-1, 3)
                rot_i = rot_maps[i][vs].permute(1, 2, 0).view(-1, 4)
                scale_i = scale_maps[i][vs].permute(1, 2, 0).view(-1, 3)
                opacity_i = opacity_maps[i][vs].permute(1, 2, 0).view(-1, 1)

                xyz_i_valid.append(xyz_i[valid_i].view(-1, 3))
                rgb_i_valid.append(rgb_i[valid_i].view(-1, 3))
                rot_i_valid.append(rot_i[valid_i].view(-1, 4))
                scale_i_valid.append(scale_i[valid_i].view(-1, 3))
                opacity_i_valid.append(opacity_i[valid_i].view(-1, 1))

            xyz_i_valid = torch.cat(xyz_i_valid, dim=0)

            rgb_i_valid = torch.cat(rgb_i_valid, dim=0)  # * 0.5 + 0.5
            rot_i_valid = torch.cat(rot_i_valid, dim=0)
            scale_i_valid = torch.cat(scale_i_valid, dim=0)
            opacity_i_valid = torch.cat(opacity_i_valid, dim=0)

            render_novel_i, render_novel_pc, _ = render(data["target_view_novel"], i, xyz_i_valid, rgb_i_valid,
                                                        rot_i_valid,
                                                        scale_i_valid, opacity_i_valid,
                                                        bg_color=self.cfg.dataset.bg_color)
            render_novel_list.append(render_novel_i.unsqueeze(0))
        return torch.cat(render_novel_list, dim=0)

    def step(self, data, is_train=False):
        bs, v = data["source_view"]["img"].shape[:2]
        img = data["source_view"]["img"][:, 0, ...].cuda()
        img_pose = rearrange(data["source_view"]["img"], "b v c h w -> (b v) c h w").cuda()
        with autocast(enabled=self.cfg.model.raft.mixed_precision):
            img_feat = self.img_encoder(img)
            img_pose_feat = self.ip_encoder(img_pose)

        if self.cfg.model.stage1:
            depth_smpl = rearrange(data["source_view"]["depth"].cuda(), "b v h w -> (b v) h w").unsqueeze(1) # smpl depth
        else:
            depth_smpl = rearrange(data["source_view"]["depth_smpl"].cuda(), "b v h w -> (b v) h w").unsqueeze(1)
        depth_feat = self.depth_encoder(depth_smpl)
        all_pose_feat = self.transformer(img_feat[-1], img_pose_feat[-1], depth_feat[-1],v)

        if self.depth_refiner is not None:
            refined_depth = self.depth_refiner(depth_smpl,all_pose_feat)
        else:
            refined_depth = depth_smpl

        rot_maps, scale_maps, opacity_maps, rgb_maps = self.gs_parm_regresser(
            side_feat=all_pose_feat, depth_feat=depth_feat, depth=refined_depth, pose_feat=img_pose_feat,
            img_feat=img_feat)
        if is_train:
            return self.gs_multi_view(bs, refined_depth, data, rgb_maps, rot_maps, scale_maps, opacity_maps),refined_depth
        else:
            return self.gs_multi_view(bs, refined_depth, data, rgb_maps, rot_maps, scale_maps, opacity_maps)
    def cal_depth_loss(self,batch,refined_depth):
        depth_gt = rearrange(batch["source_view"]["depth"].cuda(), "b v h w -> (b v) h w").unsqueeze(1)
        mask = rearrange(batch["source_view"]["mask"].cuda(), "b v c h w -> (b v) c h w").to(torch.bool)[:,:1,...]
        depth_mask = depth_gt != 0
        valid_mask = depth_mask & mask
        return (depth_gt[valid_mask].detach() - refined_depth[valid_mask]).abs().mean()

    def train(self):
        logger.info("start training...")
        start_time = time.time()
        end_time = time.time()
        score_text = ""
        best_score = {
            "lpips": 10000, "psnr": 0, "ssim": 0, "ssim256": 0, "l1": 10000,
        }
        self.set_all_model_train()
        for epoch in range(self.start_epoch, self.end_epoch):
            epoch_time = time.time()
            logger.info(f"epoch {epoch + 1} start")
            batch_time = AverageMeter()
            total_loss = AverageMeter()
            for index, data in enumerate(self.train_dataloader):
                with self.accelerator.accumulate(
                        self.gs_parm_regresser, self.img_encoder, self.ip_encoder, self.depth_encoder,self.transformer):
                    self.optimizer.zero_grad()

                    render_img,refined_depth = self.step(data,True)

                    gt = data['target_view_novel']['img'].cuda()

                    Ll1 = l1_loss(render_img, gt)
                    Lvgg = self.vgg_loss(render_img, gt)
                    Lssim = 1.0 - ssim(render_img, gt)
                    L_d = self.cal_depth_loss(data, refined_depth)
                    loss = 0.8 * Ll1 + 0.2 * Lssim + 0.01 * Lvgg + 0.1 * L_d

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
                            "loss_l1": Ll1.item(),
                            "loss_vgg": Lvgg.item(),
                            "loss_vssim": Lssim.item(),
                            "lr": self.optimizer.param_groups[-1]["lr"]
                        }, step=self.total_steps
                    )
                    etas = batch_time.avg * (len(self.train_dataloader) - 1 - index)
                    logger.info(
                        f"Train EPOCH[{epoch + 1}|{index + 1}/{len(self.train_dataloader)}] "
                        f"Time {batch_time.val:.4f}({batch_time.avg:.4f})  "
                        f"Loss {total_loss.val:.4f}({total_loss.avg:.4f})  "
                        f"Lr {self.optimizer.param_groups[-1]['lr']:.8f}  "
                        f"Eta {datetime.timedelta(seconds=int(etas))}")

                if (self.total_steps) % self.cfg.record.save_freq == 0 or self.cfg.model.debug:
                    logger.info(f"Saving model {self.total_steps}...")
                    path = os.path.join(self.cfg.record.ckpt_path, f"iteration_{(epoch + 1):03d}")
                    self.accelerator.save_state(path)
                    path = os.path.join(self.cfg.record.ckpt_path, f"iteration_latest")
                    self.accelerator.save_state(path)
                if (self.total_steps) % self.cfg.record.eval_freq == 0 or self.cfg.model.debug:
                    torch.cuda.empty_cache()
                    self.set_all_model_eval()
                    if self.accelerator.is_main_process:
                        current_score_txt, best_score_txt, best_score = self.test(self.val_dataloader, self.metric,
                                                                                  best_score=best_score, phase="eval",
                                                                                  epoch=epoch + 1)
                        score_text += f"{self.total_steps}:" + current_score_txt + '\n'
                        if len(best_score_txt) != 0:
                            score_text += f"New Best score \n {self.total_steps}:" + best_score_txt + '\n'
                            path = os.path.join(self.cfg.record.ckpt_path, f"iteration_best")
                            self.accelerator.save_state(path)
                            with open(os.path.join(self.cfg.record.file_path, "score.txt"), 'a') as f:
                                f.write(score_text)
                    else:
                        self.test(self.val_dataloader, self.metric,
                                  best_score=best_score, phase="eval",
                                  epoch=epoch + 1)
                    self.accelerator.wait_for_everyone()
                    self.set_all_model_train()
            logger.info(
                f"epoch {epoch + 1} finished, running time {datetime.timedelta(seconds=int(time.time() - epoch_time))}")
        logger.info(f"Saving model {self.total_steps}...")
        path = os.path.join(self.cfg.record.ckpt_path, f"iteration_latest")
        self.accelerator.save_state(path)
        self.set_all_model_eval()
        if self.accelerator.is_main_process:
            current_score_txt, best_score_txt, best_score = self.test(self.val_dataloader, self.metric,
                                                                      best_score=best_score, phase="eval",
                                                                      epoch=epoch + 1)
            score_text += f"{self.total_steps}:" + current_score_txt + '\n'
            if len(best_score_txt) != 0:
                score_text += f"New Best score \n {self.total_steps}:" + best_score_txt + '\n'
                path = os.path.join(self.cfg.record.ckpt_path, f"iteration_best")
                self.accelerator.save_state(path)
                with open(os.path.join(self.cfg.record.file_path, "score.txt"), 'a') as f:
                    f.write(score_text)
        else:
            self.test(self.val_dataloader, self.metric,
                      best_score=best_score, phase="eval",
                      epoch=epoch + 1)
        self.accelerator.wait_for_everyone()
        logger.info(f'training completed, running time {datetime.timedelta(seconds=int(time.time() - start_time))}')
        self.accelerator.end_training()

    def test(self, val_dataloader, metric, best_score, phase="test", epoch=None):
        logging.info(f"Doing validation ...")
        new_score = {
            "lpips": [], "ssim256": [],
            "psnr": [], "ssim": [], "l1": [],
        }
        with torch.no_grad():
            val_time = AverageMeter()
            start_time = time.time()
            end_time = time.time()
            for index, data in enumerate(val_dataloader):
                pred_img = self.step(data, False)

                gt_images = data['target_view_novel']['img'].cuda()
                lpips, psnr, ssim, ssim_256, l1 = metric(gt_images, pred_img)
                new_score["lpips"].append(self.accelerator.gather_for_metrics(lpips).cpu().numpy())
                new_score["psnr"].append(self.accelerator.gather_for_metrics(psnr).cpu().numpy())
                new_score["ssim"].append(self.accelerator.gather_for_metrics(ssim).cpu().numpy())
                new_score["ssim256"].append(self.accelerator.gather_for_metrics(ssim_256).cpu().numpy())
                new_score["l1"].append(self.accelerator.gather_for_metrics(l1).cpu().numpy())

                val_time.update(time.time() - end_time)

                if (index + 1) % 10 == 0 or self.cfg.model.debug:
                    etas = val_time.avg * (len(val_dataloader) - 1 - index)
                    logger.info(
                        f"Val [{index + 1}/{len(val_dataloader)}] "
                        f"Eta {datetime.timedelta(seconds=int(etas))}")

                if (index + 1) <= 100 or self.cfg.model.debug or phase == "test":
                    Path(
                        os.path.join(self.cfg.record.show_path, f"epoch_{epoch}_{self.total_steps}/{phase}/fig")).mkdir(
                        exist_ok=True,
                        parents=True)
                    pred_img = torch.clamp(pred_img.permute(0, 2, 3, 1)[0] * 255, min=0., max=255.)
                    pred_img = pred_img.detach().cpu().numpy()
                    pred_img = pred_img[:, :, ::-1].astype(np.uint8)
                    cv2.imwrite(
                        os.path.join(self.cfg.record.show_path,
                                     f"epoch_{epoch}_{self.total_steps}/{phase}/fig/{data['pair'][0]}_{self.accelerator.process_index}.png"),
                        pred_img)
                if (index + 1) > 100 and phase == "eval":
                    break
        logger.info("Evaluation Results:")
        if self.accelerator.is_main_process:
            for key, value in new_score.items():
                if key == "l1":
                    total_value = np.mean(value)
                else:
                    total_value = np.mean(np.concatenate(value, axis=0))
                new_score[key] = total_value
                logger.info(f"{key.upper()}: {total_value:.4f}")

            logger.info(
                f'validation completed, running time {datetime.timedelta(seconds=int(time.time() - start_time))}')

        torch.cuda.empty_cache()
        if self.accelerator.is_main_process:
            return get_best(best_score, new_score)
        else:
            return None

    def val(self,save=None):
        val_dataloader, _ = build_dataloader(self.cfg, "test")
        logging.info(f"Doing validation ...")
        new_score = {
            "lpips": [], "ssim256": [],
            "psnr": [], "ssim": [], "l1": [],
        }
        epoch =self.cfg.model.last_epoch if self.cfg.model.last_epoch != 0 else "latest"
        path = os.path.join(self.cfg.record.show_path, f"epoch_{epoch}_{self.total_steps}/val/fig") if save is None else (
            os.path.join(self.cfg.record.show_path, f"{save}"))
        with torch.no_grad():
            val_time = AverageMeter()
            start_time = time.time()
            end_time = time.time()
            times = []
            for index, data in enumerate(val_dataloader):
                start_time = time.time()
                pred_img = self.step(data, False)
                times.append(time.time() - start_time)

                gt_images = data['target_view_novel']['img'].cuda()
                lpips, psnr, ssim, ssim_256, l1 = self.metric(gt_images, pred_img)
                new_score["lpips"].append(self.accelerator.gather_for_metrics(lpips).cpu().numpy())
                new_score["psnr"].append(self.accelerator.gather_for_metrics(psnr).cpu().numpy())
                new_score["ssim"].append(self.accelerator.gather_for_metrics(ssim).cpu().numpy())
                new_score["ssim256"].append(self.accelerator.gather_for_metrics(ssim_256).cpu().numpy())
                new_score["l1"].append(self.accelerator.gather_for_metrics(l1).cpu().numpy())

                val_time.update(time.time() - end_time)

                if (index + 1) % 10 == 0 or self.cfg.model.debug:
                    etas = val_time.avg * (len(val_dataloader) - 1 - index)
                    logger.info(
                        f"Val [{index + 1}/{len(val_dataloader)}] "
                        f"Eta {datetime.timedelta(seconds=int(etas))}")

                Path(path).mkdir(exist_ok=True,parents=True)
                pred_img = torch.clamp(pred_img.permute(0, 2, 3, 1)[0] * 255, min=0., max=255.)
                pred_img = pred_img.detach().cpu().numpy()
                pred_img = pred_img[:, :, ::-1].astype(np.uint8)
                cv2.imwrite(
                    os.path.join(path,
                                 f"{data['pair'][0]}_{self.accelerator.process_index}.png"),
                    pred_img)

        logger.info("Evaluation Results:")
        if self.accelerator.is_main_process:
            for key, value in new_score.items():
                if key == "l1":
                    total_value = np.mean(value)
                else:
                    total_value = np.mean(np.concatenate(value, axis=0))
                new_score[key] = total_value
                logger.info(f"{key.upper()}: {total_value:.4f}")
            logger.info(f"use time {np.mean(times)}")
            logger.info(
                f'validation completed, running time {datetime.timedelta(seconds=int(time.time() - start_time))}')

        torch.cuda.empty_cache()


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

    trainer = Trainer(cfg, accelerator)
    trainer.train()



if __name__ == '__main__':
    main()