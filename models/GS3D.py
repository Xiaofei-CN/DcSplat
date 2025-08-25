import torch
from torch import nn
from models.baseFunction import ResidualBlock
import math
import numpy as np
from einops import rearrange
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer


class GSRegresser(nn.Module):
    def  __init__(self, cfg, rgb_dim=3, depth_dim=1, norm_fn='group'):
        super().__init__()
        self.cfg = cfg
        self.rgb_dims = cfg.model.raft.encoder_dims
        # self.depth_dims = cfg.model.gsnet.encoder_dims
        self.decoder_dims = cfg.model.gsnet.decoder_dims
        self.head_dim = cfg.model.gsnet.parm_head_dim
        self.decoder = [
            nn.Sequential(
                ResidualBlock(int(256*2.5), 128, norm_fn=norm_fn),
                ResidualBlock(128, 128, norm_fn=norm_fn),
            ),
            nn.Sequential(
                ResidualBlock(int(128*2.5), 64, norm_fn=norm_fn),
                ResidualBlock(64, 64, norm_fn=norm_fn),
            ),
            nn.Sequential(
                ResidualBlock(int(64*2.5), 32, norm_fn=norm_fn),
                ResidualBlock(32, 32, norm_fn=norm_fn),
            ),
            nn.Sequential(
                ResidualBlock(int(32*2.5), 32, norm_fn=norm_fn),
                ResidualBlock(32, 32, norm_fn=norm_fn),
            ),
        ]

        self.decoder = nn.ModuleList(self.decoder)
        self.color_decoder = [
            nn.Sequential(
                ResidualBlock(512, 128, norm_fn=norm_fn),
                ResidualBlock(128, 128, norm_fn=norm_fn),
            ),
            nn.Sequential(
                ResidualBlock(384, 64, norm_fn=norm_fn),
                ResidualBlock(64, 64, norm_fn=norm_fn),
            ),
            nn.Sequential(
                ResidualBlock(192, 32, norm_fn=norm_fn),
                ResidualBlock(32, 32, norm_fn=norm_fn),
            ),
            nn.Sequential(
                ResidualBlock(96, 16, norm_fn=norm_fn),
                ResidualBlock(16, 16, norm_fn=norm_fn),
            ),
        ]

        self.color_decoder = nn.ModuleList(self.color_decoder)

        self.up = nn.Upsample(scale_factor=2, mode="bilinear")
        self.out_conv = nn.Conv2d(49, self.head_dim, kernel_size=3, padding=1)
        self.out_relu = nn.ReLU(inplace=True)
        self.rot_head = nn.Sequential(
            nn.Conv2d(self.head_dim, self.head_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.head_dim, 4, kernel_size=1),
        )
        self.scale_head = nn.Sequential(
            nn.Conv2d(self.head_dim, self.head_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.head_dim, 3, kernel_size=1),
            nn.Softplus(beta=100)
        )
        self.opacity_head = nn.Sequential(
            nn.Conv2d(self.head_dim, self.head_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.head_dim, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.rgb_head = nn.Sequential(
            nn.Conv2d(self.head_dim, self.head_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.head_dim, 3, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, depth, depth_feat, side_feat, pose_feat, img_feat):
        dout,cout = None,None
        for dnet, cnet in zip(self.decoder, self.color_decoder):
            if cout is None:
                inp = torch.cat([rearrange(img_feat.pop().unsqueeze(1).expand(-1, self.cfg.dataset.pvs, -1, -1, -1),
                                "b v c h w -> (b v) c h w"), pose_feat.pop()],dim=1)
            else:
                inp = torch.cat([rearrange(img_feat.pop().unsqueeze(1).expand(-1, self.cfg.dataset.pvs, -1, -1, -1),
                                           "b v c h w -> (b v) c h w"), pose_feat.pop(),cout],dim=1)
            cout = cnet(inp)
            if dout is None:
                inp = torch.cat([cout,side_feat,depth_feat.pop()], dim=1)
            else:
                inp = torch.cat([cout,dout,depth_feat.pop()], dim=1)
            dout = self.up(dnet(inp))
            cout = self.up(cout)


        out = torch.cat([cout, dout, depth], dim=1)
        out = self.out_conv(out)
        out = self.out_relu(out)

        # rot head
        rot_out = self.rot_head(out)
        rot_out = torch.nn.functional.normalize(rot_out, dim=1)

        # scale head
        scale_out = torch.clamp_max(self.scale_head(out), 0.01)

        # opacity head
        opacity_out = self.opacity_head(out)
        rgb_out = self.rgb_head(out)
        return rot_out, scale_out, opacity_out, rgb_out

inverse_sigmoid = lambda x: np.log(x / (1 - x))

def render(target_view, idx, pts_xyz, pts_rgb, rotations, scales, opacity, bg_color):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """
    bg_color = torch.tensor(bg_color, dtype=torch.float32).cuda()

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pts_xyz, dtype=torch.float32, requires_grad=True).cuda() + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = torch.tensor(math.tan(target_view['FovX'][idx] * 0.5)).cuda()
    tanfovy = torch.tensor(math.tan(target_view['FovY'][idx] * 0.5)).cuda()

    raster_settings = GaussianRasterizationSettings(
        image_height=int(target_view['height'][idx]),
        image_width=int(target_view['width'][idx]),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=1.0,
        viewmatrix=target_view['world_view_transform'][idx],
        projmatrix=target_view['full_proj_transform'][idx],
        sh_degree=3,
        campos=target_view['camera_center'][idx],
        prefiltered=False,
        debug=False, antialiasing=False
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    return rasterizer(
        means3D=pts_xyz,
        means2D=screenspace_points,
        shs=None,
        colors_precomp=pts_rgb,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=None)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
