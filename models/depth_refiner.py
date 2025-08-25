import torch.nn as nn
import torch
from models.baseFunction import ResidualBlock
class Refiner(nn.Module):
    def __init__(self, in_channel=3, rgb_dim=3, head_dim=32, encoder_dim=[32, 48, 96], decoder_dim=[48, 64, 96],
                 norm_fn='group', predict_depth=False):
        super().__init__()
        self.predict_depth = predict_depth
        self.head_dim = head_dim
        self.d_out = head_dim
        self.in_ds = nn.Sequential(
            nn.Conv2d(in_channel, 32, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(num_groups=8, num_channels=32),
            nn.ReLU(inplace=True)
        )

        self.res1 = nn.Sequential(
            ResidualBlock(32, encoder_dim[0], norm_fn=norm_fn),
            ResidualBlock(encoder_dim[0], encoder_dim[0], norm_fn=norm_fn)
        )
        self.res2 = nn.Sequential(
            ResidualBlock(encoder_dim[0], encoder_dim[1], stride=2, norm_fn=norm_fn),
            ResidualBlock(encoder_dim[1], encoder_dim[1], norm_fn=norm_fn)
        )

        self.res3 = nn.Sequential(
            ResidualBlock(encoder_dim[1], encoder_dim[2], stride=2, norm_fn=norm_fn),
            ResidualBlock(encoder_dim[2], encoder_dim[2], norm_fn=norm_fn),
        )
        self.res4 = nn.Sequential(
            ResidualBlock(encoder_dim[2], encoder_dim[3], stride=2, norm_fn=norm_fn),
            ResidualBlock(encoder_dim[3], encoder_dim[3], norm_fn=norm_fn),
        )

        self.decoder4 = nn.Sequential(
            ResidualBlock(encoder_dim[3]+256, decoder_dim[3], norm_fn=norm_fn),
            ResidualBlock(decoder_dim[3], decoder_dim[3], norm_fn=norm_fn)
        )
        self.decoder3 = nn.Sequential(
            ResidualBlock(decoder_dim[3]+128, decoder_dim[2], norm_fn=norm_fn),
            ResidualBlock(decoder_dim[2], decoder_dim[2], norm_fn=norm_fn)
        )

        self.decoder2 = nn.Sequential(
            ResidualBlock(decoder_dim[2]+64, decoder_dim[1],
                          norm_fn=norm_fn),
            ResidualBlock(decoder_dim[1], decoder_dim[1], norm_fn=norm_fn)
        )

        self.decoder1 = nn.Sequential(
            ResidualBlock(decoder_dim[1]+32, decoder_dim[0],
                          norm_fn=norm_fn),
            ResidualBlock(decoder_dim[0], decoder_dim[0], norm_fn=norm_fn)
        )
        self.up = nn.Upsample(scale_factor=2, mode="bilinear")
        self.out_conv1 = nn.Conv2d(decoder_dim[2], head_dim, kernel_size=3, padding=1)
        # self.out_conv2 = nn.Conv2d(decoder_dim[0] + rgb_dim, head_dim, kernel_size=3, padding=1)
        self.out_conv2 = nn.Conv2d(decoder_dim[0]+1, head_dim, kernel_size=3, padding=1)

        self.out_relu = nn.ReLU(inplace=True)
        if predict_depth:
            self.depth_head = nn.Sequential(
                nn.Conv2d(self.head_dim, self.head_dim, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.head_dim, 1, kernel_size=1),
                # nn.Tanh()
            )

    def forward(self, depth,img_feat):
    # def forward(self, img, depth,img_feat):
    #     if depth is not None:
    #         x = torch.cat([img, depth], dim=1)
        x = depth
        x = self.in_ds(x)  # (32,512,512)
        x1 = self.res1(x)  # (32,256)
        x2 = self.res2(x1)  # (64,128)
        x3 = self.res3(x2)  # (128,64)
        x4 = self.res4(x3)  # (256,32)

        x4 = torch.cat([x4, img_feat], dim=1) # (512,32)
        y4 = self.decoder4(x4)  # (256，32)
        y4 = self.up(y4)  # (256,64)

        y4 = torch.cat([y4, x3], dim=1)
        y3 = self.decoder3(y4)  # (128,64)
        y3 = self.up(y3)  # (128,128)

        y3 = torch.cat([y3, x2], dim=1)
        y2 = self.decoder2(y3)  # (64,256,256)
        y2 = self.up(y2)  # (64,512,512)

        y2 = torch.cat([y2, x1], dim=1)
        y1 = self.decoder1(y2)  # (48,512,512)
        up1 = self.up(y1)  # (48,1024,1024)

        x = torch.cat([up1, depth], dim=1)
        x = self.out_conv2(x)  # (32,1024,1024)
        x = self.out_relu(x)

        depth_out = None
        if self.predict_depth:
            depth_out = self.depth_head(x)

        return depth_out
        # return depth_out, out1, x