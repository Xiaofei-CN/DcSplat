import os
import torch
import numpy as np
from pathlib import Path
from utils.typing_utils import *
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd
import torch.nn.functional as F
from omegaconf import OmegaConf
from einops import rearrange
from torchvision.utils import make_grid
from PIL import Image
import cv2
from copy import deepcopy


def normalize_intr(x, new_resolution, original_resolution):
    scale = new_resolution / original_resolution
    x_new = deepcopy(x)
    x_new[:, 0, 0] = x[:, 0, 0] * scale  # f_x
    x_new[:, 1, 1] = x[:, 1, 1] * scale  # f_y
    x_new[:, 0, 2] = x[:, 0, 2] * scale  # c_x
    x_new[:, 1, 2] = x[:, 1, 2] * scale  # c_y
    return x_new
def load_ckpt(model, gs,ckpt_path, iteration=None):
    ckpt_path = os.path.join(ckpt_path, "iteration_latest.pth") if iteration \
            is None else os.path.join(ckpt_path,f"iteration_{iteration}.pth")

    state_dict = torch.load(ckpt_path)
    model.load_state_dict(state_dict["network"])
    gs.load_state_dict(state_dict["gs"])
    return model,gs


def convert_data(data):
    if data is None:
        return None
    elif isinstance(data, np.ndarray):
        return data
    elif isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    elif isinstance(data, list):
        return [convert_data(d) for d in data]
    elif isinstance(data, dict):
        return {k: convert_data(v) for k, v in data.items()}
    else:
        raise TypeError(
            "Data must be in type numpy.ndarray, torch.Tensor, list or dict, getting",
            type(data),
        )


def get_rgb_image(img, data_range=[0, 1], rgba=False):
    img = convert_data(img)

    def _get_rgb_img(img):
        if img.dtype != np.uint8:
            img = img.clip(min=data_range[0], max=data_range[1])
            img = (
                    (img - data_range[0]) / (data_range[1] - data_range[0]) * 255.0
            ).astype(np.uint8)
        nc = 4 if rgba else 3
        imgs = [img[..., start: start + nc] for start in range(0, img.shape[-1], nc)]
        imgs = [
            img_
            if img_.shape[-1] == nc
            else np.concatenate(
                [
                    img_,
                    np.zeros(
                        (img_.shape[0], img_.shape[1], nc - img_.shape[2]),
                        dtype=img_.dtype,
                    ),
                ],
                axis=-1,
            )
            for img_ in imgs
        ]
        img = np.concatenate(imgs, axis=1)
        if rgba:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

    res = []
    for i in img:
        res.append(_get_rgb_img(i))
    return np.stack(res)


def postprocess_image_list(tensor, seq_len):
    tensor = tensor * 255.
    tensor = torch.clamp(tensor, min=0., max=255.)
    tensor = rearrange(tensor, 'b n c h w -> (b n) c h w')
    tensor = make_grid(tensor, nrow=seq_len, padding=0)
    img = tensor.cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
    return Image.fromarray(img)


def save_render(path,gt,pred,source,name):
    mpath = f"{path}/mutil_fig"
    spath = f"{path}/single_fig"
    Path(mpath).mkdir(exist_ok=True, parents=True)
    Path(spath).mkdir(exist_ok=True, parents=True)

    if isinstance(gt, torch.Tensor): gt = gt.detach().cpu().numpy()
    if isinstance(pred, torch.Tensor): pred = pred.detach().cpu().numpy()
    if isinstance(source, torch.Tensor): source = source.detach().cpu().numpy()

    B, V, C, H, W = source.shape

    # 转为 uint8 图像格式（B, C, H, W）→ (B, H, W, C)
    def to_img(arr):
        arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
        return arr.transpose(0, 2, 3, 1)  # (B, H, W, 3)

    gt = to_img(gt)  # (B, H, W, 3)
    pred = to_img(pred)  # (B, H, W, 3)
    source = source.reshape(B * V, C, H, W)
    source = to_img(source)  # (B*V, H, W, 3)
    source = source.reshape(B, V, H, W, 3)

    for b in range(B):
        # 保存单张 pred 图
        pred_img = pred[b]
        cv2.imwrite(f"{spath}/{name}_pred.png", cv2.cvtColor(pred_img, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_PNG_COMPRESSION, 9])

        # 拼接 source[b], gt[b], pred[b]
        images = list(source[b]) + [gt[b], pred[b]]  # V + 2 张图
        concat_img = np.concatenate(images, axis=1)  # 横向拼接
        cv2.imwrite(f"{mpath}/{name}_full.png", cv2.cvtColor(concat_img, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_PNG_COMPRESSION, 9])


def makeDirsFromList(path: List[str]) -> None:
    for p in path:
        Path(p).mkdir(exist_ok=True, parents=True)



def get_trhee_views(views):
    def check(angle):
        if angle == 360:
            return 0
        elif angle > 360:
            return angle - 360
        else:
            return angle

    v0, v1, v2 = check(views + 90), check(views + 180), check(views + 270)
    return str(v0)+".png", str(v1)+".png", str(v2)+".png"


class _TruncExp(Function):  # pylint: disable=abstract-method
    # Implementation from torch-ngp:
    # https://github.com/ashawkey/torch-ngp/blob/93b08a0d4ec1cc6e69d85df7f0acdfb99603b628/activation.py
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, x):  # pylint: disable=arguments-differ
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @custom_bwd
    def backward(ctx, g):  # pylint: disable=arguments-differ
        x = ctx.saved_tensors[0]
        return g * torch.exp(torch.clamp(x, max=15))


trunc_exp = _TruncExp.apply


def get_activation(name) -> Callable:
    if name is None:
        return lambda x: x
    name = name.lower()
    if name == "none":
        return lambda x: x
    elif name == "lin2srgb":
        return lambda x: torch.where(
            x > 0.0031308,
            torch.pow(torch.clamp(x, min=0.0031308), 1.0 / 2.4) * 1.055 - 0.055,
            12.92 * x,
        ).clamp(0.0, 1.0)
    elif name == "exp":
        return lambda x: torch.exp(x)
    elif name == "shifted_exp":
        return lambda x: torch.exp(x - 1.0)
    elif name == "trunc_exp":
        return trunc_exp
    elif name == "shifted_trunc_exp":
        return lambda x: trunc_exp(x - 1.0)
    elif name == "sigmoid":
        return lambda x: torch.sigmoid(x)
    elif name == "tanh":
        return lambda x: torch.tanh(x)
    elif name == "shifted_softplus":
        return lambda x: F.softplus(x - 1.0)
    elif name == "scale_-11_01":
        return lambda x: x * 0.5 + 0.5
    else:
        try:
            return getattr(F, name)
        except AttributeError:
            raise ValueError(f"Unknown activation function: {name}")


def parse_structured(fields: Any, cfg: Optional[Union[dict, DictConfig]] = None) -> Any:
    scfg = OmegaConf.structured(fields(**cfg))
    return scfg


def get_rank():
    # SLURM_PROCID can be set even if SLURM is not managing the multiprocessing,
    # therefore LOCAL_RANK needs to be checked first
    rank_keys = ("RANK", "LOCAL_RANK", "SLURM_PROCID", "JSM_NAMESPACE_RANK")
    for key in rank_keys:
        rank = os.environ.get(key)
        if rank is not None:
            return int(rank)
    return 0


def get_device():
    return torch.device(f"cuda:{get_rank()}")

def depth2pc(depth, extrinsic, intrinsic):
    extrinsic = rearrange(extrinsic, "b c h w -> (b c) h w")
    intrinsic = rearrange(intrinsic, "b c h w -> (b c) h w")
    B, C, S, S = depth.shape
    depth = depth[:, 0, :, :]
    rot = extrinsic[:, :3, :3]
    trans = extrinsic[:, :3, 3:]

    y, x = torch.meshgrid(torch.linspace(0.5, S-0.5, S, device=depth.device), torch.linspace(0.5, S-0.5, S, device=depth.device))
    pts_2d = torch.stack([x, y, torch.ones_like(x)], dim=-1).unsqueeze(0).repeat(B, 1, 1, 1)  # B S S 3

    pts_2d[..., 2] = 1.0 / (depth + 1e-8)
    pts_2d[:, :, :, 0] -= intrinsic[:, None, None, 0, 2]
    pts_2d[:, :, :, 1] -= intrinsic[:, None, None, 1, 2]
    pts_2d_xy = pts_2d[:, :, :, :2] * pts_2d[:, :, :, 2:]
    pts_2d = torch.cat([pts_2d_xy, pts_2d[..., 2:]], dim=-1)

    pts_2d[..., 0] /= intrinsic[:, 0, 0][:, None, None]
    pts_2d[..., 1] /= intrinsic[:, 1, 1][:, None, None]

    pts_2d = pts_2d.view(B, -1, 3).permute(0, 2, 1)
    rot_t = rot.permute(0, 2, 1)
    pts = torch.bmm(rot_t, pts_2d) - torch.bmm(rot_t, trans)

    return pts.permute(0, 2, 1)

def get_best(best,new):
    count = 0

    for key in best.keys():
        if key in ["lpips","l1"]:
            if best[key] > new[key]:
                count += 1
        elif key in ["ssim","psnr"]:
            if best[key] < new[key]:
                count += 1
    if count >= 3:
        return get_txt_from_dict(new),get_txt_from_dict(new),new
    else:
        return get_txt_from_dict(new),"",best

def get_txt_from_dict(data,step='\t'):
    assert isinstance(data, dict)
    return f"{step}".join(f"{k.upper()}={v:.4f}" for k, v in data.items())
