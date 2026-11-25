"""Minimal exact-shape port for scrollprize/ink_9um seed42/43 checkpoints.

Architecture sources and normalization are pinned under hybrid_reference/.
2D body follows villa residual encoder and ConvBlock decoder. Duplicate
all_modules aliases are verified then omitted on load; all unique keys strict.
"""
import torch
from torch import nn
from hybrid_reference.hybrid_3d2d import Local3DStem2DUNet


class ConvNorm(nn.Module):
    def __init__(self, cin, cout, kernel=3, stride=1, bias=True, activate=True):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, kernel, stride, (kernel-1)//2, bias=bias)
        self.norm = nn.InstanceNorm2d(cout, affine=True, eps=1e-5)
        self.nonlin = nn.LeakyReLU(0.01, inplace=True) if activate else nn.Identity()

    def forward(self, x):
        return self.nonlin(self.norm(self.conv(x)))


class BasicBlockD(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.conv1 = ConvNorm(cin, cout, stride=stride)
        self.conv2 = ConvNorm(cout, cout, activate=False)
        self.nonlin2 = nn.LeakyReLU(0.01, inplace=True)
        skip = []
        if stride != 1:
            skip.append(nn.AvgPool2d(stride, stride))
        if cin != cout:
            skip.append(ConvNorm(cin, cout, kernel=1, bias=False, activate=False))
        self.skip = nn.Sequential(*skip) if skip else nn.Identity()

    def forward(self, x):
        return self.nonlin2(self.conv2(self.conv1(x)) + self.skip(x))


class ResidualStage(nn.Module):
    def __init__(self, cin, cout, stride, count):
        super().__init__()
        self.blocks = nn.Sequential(BasicBlockD(cin, cout, stride),
                                    *[BasicBlockD(cout, cout, 1) for _ in range(count-1)])

    def forward(self, x):
        return self.blocks(x)


class ConvStage(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.convs = nn.Sequential(ConvNorm(cin, cout))

    def forward(self, x):
        return self.convs(x)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = ConvStage(32, 32)
        features = [32, 32, 64, 128, 256, 320, 320]
        counts = [1, 3, 4, 6, 6, 6]
        self.stages = nn.Sequential(*[
            ResidualStage(features[i], features[i+1], 1 if i==0 else 2, counts[i])
            for i in range(6)])

    def forward(self, x):
        x = self.stem(x)
        skips = []
        for stage in self.stages:
            x = stage(x)
            skips.append(x)
        return skips


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        sizes = [320, 320, 256, 128, 64, 32]
        self.stages = nn.ModuleList([ConvStage(2*c, c) for c in sizes[1:]])
        self.transpconvs = nn.ModuleList([
            nn.ConvTranspose2d(sizes[i], sizes[i+1], 2, 2, bias=True) for i in range(5)])

    def forward(self, skips):
        x = skips[-1]
        for i, (up, stage) in enumerate(zip(self.transpconvs, self.stages)):
            x = stage(torch.cat((up(x), skips[-(i+2)]), dim=1))
        return x


class Body(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared_encoder = Encoder()
        self.task_heads = nn.ModuleDict({'ink': nn.Conv2d(32, 1, 1)})
        self.shared_decoder = Decoder()

    def forward(self, x):
        return self.task_heads['ink'](self.shared_decoder(self.shared_encoder(x)))


def make_hybrid(checkpoint, *, spatial_patch_size=128):
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    config = payload['config']
    assert config['model_type'] == 'vesuvius_unet_3d_stem_2d'
    assert config['patch_size'] == [17, spatial_patch_size, spatial_patch_size]
    assert config['image_normalization']['mode'] == 'robust_mad'
    state = payload['model']
    unique = {}
    aliases = 0
    for k, v in state.items():
        if '.all_modules.' in k:
            base = k.replace('.all_modules.0.', '.conv.').replace('.all_modules.1.', '.norm.')
            if base not in state or not torch.equal(v, state[base]):
                raise ValueError(f'Unverified duplicate module alias: {k}')
            aliases += 1
        else:
            unique[k] = v
    model = Local3DStem2DUNet(Body(), input_depth=17, stem_channels=16)
    model.load_state_dict(unique, strict=True)
    return model.eval(), {'config': config, 'step': payload['step'], 'aliases_checked': aliases,
                          'parameter_count': sum(p.numel() for p in model.parameters())}
