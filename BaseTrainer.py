import os.path
import time
import datetime
import warnings
from utils.utils import *
from utils.logger_utils import *
from utils.metrics_utils import build_metric
from data import build_dataloader

from models.GS3D import *
from models.lossFunction import l1_loss, ssim
import torchvision.transforms.functional as TF

# from test import test, eval
from torch.cuda.amp import autocast as autocast
from torch.utils.tensorboard import SummaryWriter
warnings.filterwarnings("ignore", category=UserWarning)
import logging
logger = logging.getLogger()

class BaseTrainer(nn.Module):
    def __init__(self,cfg,accelerator):
        super().__init__()
        self.cfg = cfg
        logger.info("Get train dataloder....")
        self.train_dataloader, self.train_dataset = build_dataloader(cfg, "train")
        self.val_dataloader, self.val_dataset = build_dataloader(cfg, "val")
        self.metric = build_metric().cuda()
        self.total_steps = 0
        self.start_epoch = cfg.model.last_epoch
        self.end_epoch = cfg.model.num_epochs

        self.model_registry = []

        self.accelerator = accelerator



    def gs_forward_pc_from_depth(self, bs, rot_maps, scale_maps, opacity_maps, img, depth, data, require_pc=False):

        pts_valid = depth != 0.0
        pts_valid = pts_valid.view(bs, -1)
        pc = depth2pc(depth, data["source_view"]['extr'][:, :1, ...].cuda(),
                      data["source_view"]['intr'][:, :1, ...].cuda())

        render_novel_list = []
        if require_pc:
            render_3dgs = []
        for i in range(bs):
            valid_i = pts_valid[i]
            xyz_i = pc[i, :, :]
            rgb_i = img[i, :, :, :].permute(1, 2, 0).view(-1, 3)
            rot_i = rot_maps[i, :, :, :].permute(1, 2, 0).view(-1, 4)
            scale_i = scale_maps[i, :, :, :].permute(1, 2, 0).view(-1, 3)
            opacity_i = opacity_maps[i, :, :, :].permute(1, 2, 0).view(-1, 1)

            assert valid_i.shape[0] == rgb_i.shape[0] == xyz_i.shape[0] == rot_i.shape[0] == scale_i.shape[0] == \
                   opacity_i.shape[0]
            xyz_i_valid = xyz_i[valid_i].view(-1, 3)
            rgb_i_valid = rgb_i[valid_i].view(-1, 3) * 0.5 + 0.5
            rot_i_valid = rot_i[valid_i].view(-1, 4)
            scale_i_valid = scale_i[valid_i].view(-1, 3)
            opacity_i_valid = opacity_i[valid_i].view(-1, 1)

            render_novel_i, render_pc, _ = render(data["target_view"], i, xyz_i_valid, rgb_i_valid, rot_i_valid,
                                                  scale_i_valid, opacity_i_valid,
                                                  bg_color=self.cfg.dataset.bg_color)
            # torch_max = torch.max(render_novel_i)
            # torch_min = torch.min(render_novel_i)
            render_novel_list.append(render_novel_i.unsqueeze(0))
            if require_pc:
                render_3dgs.append(render_pc)
        if require_pc:
            return torch.cat(render_novel_list, dim=0), torch.cat(render_3dgs, dim=0)
        else:
            return torch.cat(render_novel_list, dim=0)


    def gs_forward_pc_from_smplx(self, bs, rot_maps, scale_maps, opacity_maps, img, data, require_pc=False):

        pc = data["source_view"]['pc'].cuda()

        render_novel_list = []
        if require_pc:
            render_3dgs = []
        for i in range(bs):
            xyz_i = pc[i, :, :]
            rgb_i = img[i, :, :, :].permute(1, 2, 0).view(-1, 3)
            rot_i = rot_maps[i, :, :, :].permute(1, 2, 0).view(-1, 4)
            scale_i = scale_maps[i, :, :, :].permute(1, 2, 0).view(-1, 3)
            opacity_i = opacity_maps[i, :, :, :].permute(1, 2, 0).view(-1, 1)

            assert rgb_i.shape[0] == xyz_i.shape[0] == rot_i.shape[0] == scale_i.shape[0] == \
                   opacity_i.shape[0]
            xyz_i_valid = xyz_i.view(-1, 3)
            rgb_i_valid = rgb_i.view(-1, 3) * 0.5 + 0.5
            rot_i_valid = rot_i.view(-1, 4)
            scale_i_valid = scale_i.view(-1, 3)
            opacity_i_valid = opacity_i.view(-1, 1)

            render_novel_i, render_pc, _ = render(data["target_view"], i, xyz_i_valid, rgb_i_valid, rot_i_valid,
                                                  scale_i_valid, opacity_i_valid,
                                                  bg_color=self.cfg.dataset.bg_color)
            render_novel_list.append(render_novel_i.unsqueeze(0))
            if require_pc:
                render_3dgs.append(render_pc)
        if require_pc:
            return torch.cat(render_novel_list, dim=0), torch.cat(render_3dgs, dim=0)
        else:
            return torch.cat(render_novel_list, dim=0)

    def step(self, data, require_pc=False):
        bs, v = data["source_view"]["img"].shape[:2]
        img = rearrange(data["source_view"]["img"].cuda(), "b v c h w -> (b v) c h w")

        with autocast(enabled=self.cfg.model.raft.mixed_precision):
            img_feat = self.img_encoder(TF.resize(img, (224, 224)),
                                        modulation_cond=None)  # TF.resize(img,(252,252))
            img_feat = rearrange(img_feat, "(b v) c d -> b v d c", v=v) # bs 257 768
            img = rearrange(img, "(b v) c h w -> b v c h w", b=bs, v=v)[:, 0, ...]

        # get gs params -----------------------------
        depth = data["source_view"]["depth"][:,:1,...].cuda()
        rot_maps, scale_maps, opacity_maps = self.gs_parm_regresser(func_name=f"forward_{self.cfg.model.gsnet.net_type}"
                                                                    , img=img, depth=depth, img_feat=img_feat)
        # render img
        return self.gs_forward_pc_from_depth(bs, rot_maps, scale_maps, opacity_maps, img, depth, data, require_pc)
        # return self.gs_forward_pc_from_smplx(bs, rot_maps, scale_maps, opacity_maps, img, data, require_pc)

    def train(self):
        logger.info("start training...")
        start_time = time.time()
        end_time = time.time()
        score_text = ""
        best_score = {
            "lpips": 10000, "psnr": 0, "ssim": 0, "ssim256": 0, "l1": 10000,
        }
        for epoch in range(self.start_epoch,self.end_epoch):

            epoch_time = time.time()
            logger.info(f"epoch {epoch + 1} start")
            batch_time = AverageMeter()
            total_loss = AverageMeter()
            for index, data in enumerate(self.train_dataloader):
                self.optimizer.zero_grad()

                render_img = self.step(data)

                gt_novel = data['target_view']['img'].cuda()

                Ll1 = l1_loss(render_img, gt_novel)
                Lssim = 1.0 - ssim(render_img, gt_novel)
                loss = 0.8 * Ll1 + 0.2 * Lssim  # + 0.5 * L_per
                self.writer.add_scalar("Loss/loss_total", loss.item(), self.total_steps)
                self.writer.add_scalar("Loss/loss_l1", Ll1.item(), self.total_steps)
                self.writer.add_scalar("Loss/loss_ssim", Lssim.item(), self.total_steps)

                batch_time.update(time.time() - end_time)
                total_loss.update(loss.item())
                end_time = time.time()

                if (self.total_steps + 1) % self.cfg.record.loss_freq == 0 or self.cfg.model.debug:
                    etas = batch_time.avg * (len(self.train_dataloader) - 1 - index)
                    logger.info(
                        f"Train [{self.total_steps + 1}/{len(self.train_dataloader) * self.cfg.model.num_epochs}] "
                        f"Time {batch_time.val:.4f}({batch_time.avg:.4f})  "
                        f"Loss {total_loss.val:.4f}({total_loss.avg:.4f})  "
                        f"Lr {self.optimizer.param_groups[-1]['lr']:.8f}  "
                        f"Eta {datetime.timedelta(seconds=int(etas))}")

                if (self.total_steps + 1) % self.cfg.record.save_freq == 0 or self.cfg.model.debug:
                    logger.info(f"Saving model {self.total_steps + 1}...")
                    path = os.path.join(self.cfg.record.ckpt_path, f"iteration_{(epoch + 1):03d}.pth")
                    self.save_ckpt(epoch, path)

                if (self.total_steps + 1) % self.cfg.record.eval_freq == 0 or self.cfg.model.debug:
                    torch.cuda.empty_cache()
                    self.img_encoder.eval()
                    self.gs_parm_regresser.eval()
                    current_score_txt,best_score_txt,best_score = self.test(self.val_dataloader, self.metric,
                                                                            best_score=best_score, phase="eval")
                    score_text += f"{self.total_steps}:" + current_score_txt + '\n'
                    if len(best_score_txt) != 0:
                        score_text += f"New Best score \n {self.total_steps}:" + best_score_txt + '\n'
                        path = os.path.join(self.cfg.record.ckpt_path, f"iteration_best.pth")
                        self.save_ckpt(epoch,path)
                        with open(os.path.join(self.cfg.record.file_path, "score.txt"), 'a') as f:
                            f.write(score_text)
                    self.img_encoder.train()
                    self.gs_parm_regresser.train()


                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                self.total_steps += 1

            logger.info(
                f"epoch {epoch + 1} finished, running time {datetime.timedelta(seconds=int(time.time() - epoch_time))}")

        logger.info(f"Saving model {self.total_steps + 1}...")
        path = os.path.join(self.cfg.record.ckpt_path, f"iteration_latest.pth")
        self.save_ckpt(epoch,path)
        self.img_encoder.eval()
        self.gs_parm_regresser.eval()
        current_score_txt, best_score_txt, best_score = self.test(self.val_dataloader, self.metric,
                                                                  best_score=best_score, phase="test")
        score_text += f"{self.total_steps}:" + current_score_txt + '\n'
        if len(best_score_txt) != 0:
            score_text += f"New Best score \n {self.total_steps}:" + best_score_txt + '\n'
            path = os.path.join(self.cfg.record.ckpt_path, f"iteration_best.pth")
            self.save_checkpoint(epoch,path)
        with open(os.path.join(self.cfg.record.file_path, "score.txt"), 'a') as f:
            f.write(score_text)
        logger.info(f'training completed, running time {datetime.timedelta(seconds=int(time.time() - start_time))}')

    def test(self, val_dataloader, metric, best_score, phase="test",epoch=None):
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
                pred_img, pred_3dgs = self.step(data, True)

                gt_images = data['target_view']['img'].cuda()
                lpips, psnr, ssim, ssim_256, l1 = metric(gt_images, pred_img)
                new_score["lpips"].append(lpips.cpu().numpy())
                new_score["psnr"].append(psnr.cpu().numpy())
                new_score["ssim"].append(ssim.cpu().numpy())
                new_score["ssim256"].append(ssim_256.cpu().numpy())
                new_score["l1"].append(l1.cpu().numpy())

                val_time.update(time.time() - end_time)

                if (index + 1) % 10 == 0 or self.cfg.model.debug:
                    etas = val_time.avg * (len(val_dataloader) - 1 - index)
                    logger.info(
                        f"Val [{index + 1}/{len(val_dataloader)}] "
                        f"Eta {datetime.timedelta(seconds=int(etas))}")

                if (index + 1) <= 100 or self.cfg.model.debug or phase == "test":
                    Path(os.path.join(self.cfg.record.show_path, f"epoch_{epoch}_{self.total_steps}/{phase}/fig")).mkdir(exist_ok=True,
                                                                                                      parents=True)
                    pred_img = torch.clamp(pred_img.permute(0, 2, 3, 1)[0] * 255, min=0., max=255.)
                    pred_img = pred_img.detach().cpu().numpy()
                    pred_img = pred_img[:, :, ::-1].astype(np.uint8)
                    cv2.imwrite(
                        os.path.join(self.cfg.record.show_path, f"epoch_{epoch}_{self.total_steps}/{phase}/fig/{data['pair'][0]}.png"),
                        pred_img)
                if (index+1) > 500 and phase == "eval":
                    break
        logger.info("Evaluation Results:")
        for key, value in new_score.items():
            if key == "l1":
                total_value = np.mean(value)
            else:
                total_value = np.mean(np.concatenate(value, axis=0))
            new_score[key] = total_value
            logger.info(f"{key.upper()}: {total_value:.4f}")

        logger.info(f'validation completed, running time {datetime.timedelta(seconds=int(time.time() - start_time))}')
        torch.cuda.empty_cache()
        return get_best(best_score, new_score)

    def val(self):
        self.test(self.img_encoder, self.gs_parm_regresser, self.val_dataloader, self.metric, phase="eval")


    def save_ckpt(self,epoch,path):
        save_dcit = {'epoch': epoch + 1,'optimizer': self.optimizer.state_dict(),'scheduler': self.scheduler.state_dict()}
        for i in self.model_registry:
            save_dcit[i] = getattr(self,i).state_dict()
        torch.save(save_dcit, path)
    def load_ckpt(self,epoch=None):
        if epoch is None:
            path = os.path.join(self.cfg.record.ckpt_path, f"iteration_best")
        else:
            path = os.path.join(self.cfg.record.ckpt_path, f"iteration_{epoch}")
        ckpts = sorted(os.listdir(path))
        assert len(ckpts) == len(self.model_registry)
        for m,c in zip(self.model_registry,ckpts):
            logger.info(getattr(self,m).load_state_dict(torch.load(
                os.path.join(path,c), map_location="cpu"
            ), strict=False))

    def resume(self,epoch=None):
        if epoch is not None:
            path = os.path.join(self.cfg.record.ckpt_path, f"iteration_{epoch}")
        else:
            path = os.path.join(self.cfg.record.ckpt_path, f"iteration_latest")
        logger.info(f"loading states from {path}")
        self.accelerator.load_state(path)
        self.start_epoch = self.cfg.model.last_epoch
        self.total_steps = self.start_epoch * len(self.train_dataloader)
    def resume_stage1(self,path):
        self.img_encoder.load_state_dict(torch.load(os.path.join(path,"")))
        self.depth_refiner.load_state_dict(torch.load(os.path.join(path,"")))

    def caculate_totoal_params(self,model_registry):
        trainable_params_nums = 0
        trainable_params = []
        for i in model_registry:
            trainable_params_nums += sum([p.numel() for p in getattr(self,i).parameters() if p.requires_grad])
            trainable_params += [p for p in getattr(self,i).parameters() if p.requires_grad]
        return trainable_params_nums,trainable_params

    def set_all_model_train(self):
        for i in self.model_registry:
            getattr(self,i).train()

    def set_all_model_eval(self):
        for i in self.model_registry:
            getattr(self,i).eval()
