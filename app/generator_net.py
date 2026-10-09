import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.vision_transformer import Encoder


# -------------------------------
# CNN Encoder
# -------------------------------
class CNNEncoder(nn.Module):
    def __init__(self, in_channels=3, base=64):
        super().__init__()
        self.base = base

        self.enc1 = nn.Sequential(nn.Conv2d(in_channels, base, 3, padding=1),
                                  nn.InstanceNorm2d(base),
                                  nn.SiLU(inplace=True))
        self.down1 = nn.Conv2d(base, base * 2, 4, stride=2, padding=1)

        self.enc2 = nn.Sequential(nn.Conv2d(base * 2, base * 2, 3, padding=1),
                                  nn.InstanceNorm2d(base * 2),
                                  nn.SiLU(inplace=True))
        self.down2 = nn.Conv2d(base * 2, base * 4, 4, stride=2, padding=1)

        self.enc3 = nn.Sequential(nn.Conv2d(base * 4, base * 4, 3, padding=1),
                                  nn.InstanceNorm2d(base * 4),
                                  nn.SiLU(inplace=True))
        self.down3 = nn.Conv2d(base * 4, base * 8, 4, stride=2, padding=1)

        self.enc4 = nn.Sequential(nn.Conv2d(base * 8, base * 8, 3, padding=1),
                                  nn.InstanceNorm2d(base * 8),
                                  nn.SiLU(inplace=True))
        self.down4 = nn.Conv2d(base * 8, base * 8, 4, stride=2, padding=1)

        self.enc5 = nn.Sequential(nn.Conv2d(base * 8, base * 8, 3, padding=1),
                                  nn.InstanceNorm2d(base * 8),
                                  nn.SiLU(inplace=True))

    def forward(self, x):
        f1 = self.enc1(x)               # [B,64,224,224]
        f2 = self.enc2(self.down1(f1))  # [B,128,112,112]
        f3 = self.enc3(self.down2(f2))  # [B,256,56,56]
        f4 = self.enc4(self.down3(f3))  # [B,512,28,28]
        f5 = self.enc5(self.down4(f4))  # [B,512,14,14]
        return [f1, f2, f3, f4, f5]


# -------------------------------
# ViT Extractor
# -------------------------------
class ViTExtractor(nn.Module):
    def __init__(self, img_size=224, patch_size=16, embed_dim=192, depth=2, num_heads=4):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (img_size // patch_size) ** 2
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, embed_dim))
        self.encoder = Encoder(seq_length=num_patches,
                               num_layers=depth,
                               num_heads=num_heads,
                               hidden_dim=embed_dim,
                               mlp_dim=embed_dim * 4,
                               dropout=0.0,
                               attention_dropout=0.0)
        self.norm = nn.LayerNorm(embed_dim)
        self.embed_dim = embed_dim

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.patch_embed(x)                 # [B,192,14,14]
        x = x.flatten(2).transpose(1, 2)        # [B,196,192]
        x = x + self.pos_embedding
        x = self.encoder(x)
        x = self.norm(x)
        h = w = int(x.shape[1] ** 0.5)
        x = x.transpose(1, 2).reshape(B, self.embed_dim, h, w)
        return x


# -------------------------------
# ViT Decoder
# -------------------------------
class ViTDecoder(nn.Module):
    def __init__(self, vit_dim=192, base=64):
        super().__init__()
        self.stages = nn.ModuleList([
            nn.ConvTranspose2d(vit_dim, base * 8, 2, stride=2),
            nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2),
            nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2),
            nn.ConvTranspose2d(base * 2, base, 2, stride=2),
        ])
        self.proj = nn.ModuleList([
            nn.Conv2d(base * 8, base * 8, 1),
            nn.Conv2d(base * 4, base * 4, 1),
            nn.Conv2d(base * 2, base * 2, 1),
            nn.Conv2d(base, base, 1),
        ])
        self.out_conv = nn.Conv2d(base, 3, 3, padding=1)

    def forward(self, vit_feat, target_sizes):
        features = []
        v = vit_feat
        for i, stage in enumerate(self.stages):
            v = stage(v)
            v = F.interpolate(self.proj[i](v), size=target_sizes[i],
                              mode="bilinear", align_corners=False)
            features.append(v)
        vit_out = self.out_conv(v)
        return features, vit_out


# -------------------------------
# CNN Decoder
# -------------------------------
class CNNDecoder(nn.Module):
    def __init__(self, base=64):
        super().__init__()
        self.base = base

        self.up = nn.ModuleList([
            nn.ConvTranspose2d(base * 8, base * 8, 2, stride=2),
            nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2),
            nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2),
            nn.ConvTranspose2d(base * 2, base, 2, stride=2),
        ])

        self.stage_proj = nn.ModuleList([
            nn.Conv2d(1536, base * 8, kernel_size=1),
            nn.Conv2d(768, base * 4, kernel_size=1),
            nn.Conv2d(384, base * 2, kernel_size=1),
            nn.Conv2d(192, base, kernel_size=1),
        ])

        self.out_conv = nn.Conv2d(base, 3, 3, padding=1)

    def forward(self, cnn_feats, vit_feats):
        f1, f2, f3, f4, bottleneck = cnn_feats
        x = bottleneck
        cnn_skips = [f4, f3, f2, f1]

        for i in range(4):
            x = self.up[i](x)

            s_cnn = cnn_skips[i]
            s_vit = vit_feats[i]

            if s_cnn.shape[2:] != x.shape[2:]:
                s_cnn = F.interpolate(s_cnn, size=x.shape[2:], mode="bilinear", align_corners=False)
            if s_vit.shape[2:] != x.shape[2:]:
                s_vit = F.interpolate(s_vit, size=x.shape[2:], mode="bilinear", align_corners=False)

            x = torch.cat([x, s_cnn, s_vit], dim=1)
            x = self.stage_proj[i](x)

        return self.out_conv(x)


# -------------------------------
# Hybrid Generator (Guided Fusion)
# -------------------------------
class Generator(nn.Module):
    def __init__(self, base=64, vit_dim=192):
        super().__init__()
        self.encoder = CNNEncoder(base=base)
        self.vit_extractor = ViTExtractor(embed_dim=vit_dim)
        self.vit_decoder = ViTDecoder(vit_dim=vit_dim, base=base)
        self.cnn_decoder = CNNDecoder(base=base)
        self.fusion_weight = nn.Parameter(torch.tensor([0.0]))

    def forward(self, x):
        cnn_feats = self.encoder(x)
        vit_feat = self.vit_extractor(x)

        B, C, H, W = x.shape
        target_sizes = [(H // 8, W // 8), (H // 4, W // 4), (H // 2, W // 2), (H, W)]

        vit_stage_feats, vit_out = self.vit_decoder(vit_feat, target_sizes)
        cnn_out = self.cnn_decoder(cnn_feats, vit_stage_feats)

        vit_out_resized = F.interpolate(vit_out, size=cnn_out.shape[2:],
                                        mode="bilinear", align_corners=False)

        w = torch.sigmoid(self.fusion_weight)
        out = (w * cnn_out) + ((1 - w) * vit_out_resized)

        return out, cnn_out, vit_out_resized


def build_generator() -> nn.Module:
    """Same constructor arguments as in training (base=64, vit_dim=192 are the defaults)."""
    return Generator()