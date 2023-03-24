import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class Block(nn.Module):
    def __init__(self, in_ch, out_ch, padding=0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=padding, padding_mode="zeros")
        self.relu  = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3)
    
    def forward(self, x):
        return self.relu(self.conv2(self.relu(self.conv1(x))))


class Encoder(nn.Module):
    def __init__(self, chs=(3, 16, 32, 64)): # (3, 64, 128, 256, 512, 1024)):
        super().__init__()
        self.enc_blocks = nn.ModuleList([Block(chs[i], chs[i + 1]) for i in range(len(chs) - 1)])
        self.pool       = nn.MaxPool2d(2)
    
    def forward(self, x):
        ftrs = []
        for block in self.enc_blocks:
            x = block(x)
            ftrs.append(x)
            x = self.pool(x)
        return ftrs


class Decoder(nn.Module):
    def __init__(self, chs=(64, 32, 16)): # (1024, 512, 256, 128, 64)):
        super().__init__()
        self.chs        = chs
        self.upconvs    = nn.ModuleList([nn.ConvTranspose2d(chs[i], chs[i + 1], 2, 2) for i in range(len(chs) - 1)])
        self.dec_blocks = nn.ModuleList([Block(chs[i], chs[i + 1]) for i in range(len(chs) - 1)])
        
    def forward(self, x, encoder_features):
        for i in range(len(self.chs)-1):
            x        = self.upconvs[i](x)
            enc_ftrs = self.crop(encoder_features[i], x)
            x        = torch.cat([x, enc_ftrs], dim=1)
            x        = self.dec_blocks[i](x)
        return x
    
    def crop(self, enc_ftrs, x):
        _, _, H, W = x.shape
        enc_ftrs   = torchvision.transforms.CenterCrop([H, W])(enc_ftrs)
        return enc_ftrs


class UNet(nn.Module):
    def __init__(self, enc_chs=(3, 16, 32, 64), dec_chs=(64, 32, 16), num_class=11, retain_dim=True, out_sz=(64, 64)):
        super().__init__()
        self.encoder     = Encoder(enc_chs)
        self.decoder     = Decoder(dec_chs)
        self.head        = nn.Sequential(
            nn.ConvTranspose2d(dec_chs[-1], dec_chs[-1], 2, 2),
            nn.Conv2d(dec_chs[-1], num_class, 1)
        )
        self.retain_dim  = retain_dim
        self.out_sz = out_sz

    def forward(self, x):
        enc_ftrs = self.encoder(x)
        out      = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        out      = self.head(out)
        if self.retain_dim:
            out = F.interpolate(out, self.out_sz)
        return out


class UNetG(nn.Module):
    def __init__(self, enc_chs=(3, 16, 32, 64), pretrained=False):
        super().__init__()
        self.encoder = Encoder(enc_chs)

    def forward(self, x):
        out = self.encoder(x)
        return out


class UNetC(nn.Module):
    def __init__(self, dec_chs=(64, 32, 16), num_classes=11, retain_dim=True, input_shape=(64, 64)):
        super().__init__()
        self.decoder     = Decoder(dec_chs)
        self.head1       = nn.ConvTranspose2d(dec_chs[-1], dec_chs[-1], 2, 2)
        self.head2       = nn.Conv2d(dec_chs[-1], num_classes, 1)
        self.relu        = nn.ReLU()
        self.retain_dim  = retain_dim
        self.input_shape = input_shape

    def forward(self, enc_ftrs, return_all_features=False):
        fl3      = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        fl6      = self.relu(self.head1(fl3))
        out      = self.head2(fl6)
        if self.retain_dim:
            out = F.interpolate(out, self.input_shape)
        if return_all_features:
            return out, fl6.flatten(start_dim=1), fl3.flatten(start_dim=1)
        else:
            return out


if __name__ == "__main__":
    model = UNet().cuda()
    from torchsummary import summary
    summary(model, (3, 64, 64))
    out = model(torch.randn(1, 3, 64, 64).cuda())
    print(out.shape)