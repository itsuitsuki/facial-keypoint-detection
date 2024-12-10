import torch
from torch import nn


class Conv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # Kernel size: 3
        # stride: 1
        # padding: 1
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activ = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Conv2d(3,1,1)+BN+GELU
        return self.activ(self.bn(self.conv(x)))


class DownConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              3, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activ = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activ(self.bn(self.conv(x)))


class UpConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.tconv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activ = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activ(self.bn(self.tconv(x)))


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = Conv(in_channels, out_channels)
        self.conv2 = Conv(out_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.conv1(x))


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.dconv = DownConv(in_channels, out_channels)
        self.conv = ConvBlock(out_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.dconv(x))


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.uconv = UpConv(in_channels, out_channels)
        self.conv = ConvBlock(out_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.uconv(x))


class FCBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.linear1 = nn.Linear(in_channels, out_channels)
        self.activ = nn.GELU()
        self.linear2 = nn.Linear(out_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.activ(self.linear1(x)))


class Flatten(nn.Module):
    def __init__(self):
        super().__init__()
        self.avgpool = nn.AvgPool2d(kernel_size=7)
        self.activ = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activ(self.avgpool(x))


class Unflatten(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.tconv = nn.ConvTranspose2d(
            in_channels, in_channels, kernel_size=7, stride=7, padding=0)
        self.bn = nn.BatchNorm2d(in_channels)
        self.activ = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activ(self.bn(self.tconv(x)))

def heatmap_to_landmarks_mean(heatmaps):
    B, C, H, W = heatmaps.shape
    
    row_indices = torch.arange(H, device=heatmaps.device).view(1, 1, H, 1)  # (1, 1, H, 1)
    col_indices = torch.arange(W, device=heatmaps.device).view(1, 1, 1, W)  # (1, 1, 1, W)
    
    weighted_mean_row = (heatmaps * row_indices).sum(dim=(2, 3)) / heatmaps.sum(dim=(2, 3))  # (B, C)
    weighted_mean_col = (heatmaps * col_indices).sum(dim=(2, 3)) / heatmaps.sum(dim=(2, 3))  # (B, C)
    
    weighted_mean_indices = torch.stack([weighted_mean_row, weighted_mean_col], dim=-1)  # (B, C, 2)
    # print(weighted_mean_indices)
    return weighted_mean_indices


def heatmap_to_landmarks_max(heatmaps):
    B, C, H, W = heatmaps.shape
    
    # max_indices = heatmaps.view(B, C, -1).argmax(dim=-1)  # (B, C)
    
    # row_indices = max_indices // W  # (B, C)
    # col_indices = max_indices % W  # (B, C)
    
    # max_indices = torch.stack([row_indices, col_indices], dim=-1)  # (B, C, 2)
    
    # find 4 largest indices for each heatmap in (B,C)
    max_indices = torch.topk(heatmaps.view(B, C, -1), 4, dim=-1).indices # (B, C, 4)
    top_values = torch.topk(heatmaps.view(B, C, -1), 4, dim=-1).values # (B, C, 4)
    # back to 2D: (B, C, 4) -> (B, C, 4, 2)
    row_indices = max_indices // W  # (B, C, 4)
    col_indices = max_indices % W  # (B, C, 4)
    # max_indices = torch.stack([row_indices, col_indices], dim=-1)  # (B, C, 4, 2)
    top_prob = top_values / top_values.sum(dim=-1, keepdim=True)
    
    # should aggregate the indices by normalized means to get the final 2 indices
    # (B, C, 4, 2) -> (B, C, 2)
    # weighted mean by the heatmap values
    # p1 * (r1, c1) + p2 * (r2, c2) + p3 * (r3, c3) + p4 * (r4, c4)
    # (p1*r1 + p2*r2 + p3*r3 + p4*r4, p1*c1 + p2*c2 + p3*c3 + p4*c4)
    weighted_mean_row = (row_indices * top_prob).sum(dim=-1)  # (B, C)
    weighted_mean_col = (col_indices * top_prob).sum(dim=-1)  # (B, C)
    
    soft_argmax_indices = torch.stack([weighted_mean_row, weighted_mean_col], dim=-1)  # (B, C, 2)
    
    
    return soft_argmax_indices

def heatmap_to_landmarks(heatmaps, method='max'):
    if method == 'mean':
        return heatmap_to_landmarks_mean(heatmaps)
    elif method == 'max':
        return heatmap_to_landmarks_max(heatmaps)
    else:
        raise ValueError(f"Invalid method: {method}")
class PixelwiseClassificationUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 68,  # dimension of t
        num_hiddens: int = 128,
        height: int = 224,
        width: int = 224,
    ):
        # in: (B, 1, 224, 224)
        # out: (B, 68, 224, 224)

        super().__init__()
        self.height = height
        self.width = width
        self.in_channels = in_channels
        self.num_hiddens = num_hiddens
        self.conv0 = Conv(in_channels, num_hiddens)  # (B, 128, 224, 224)
        self.down1 = DownBlock(num_hiddens, num_hiddens)  # (B, 128, 112, 112)
        self.down2 = DownBlock(
            num_hiddens, 2 * num_hiddens)  # (B, 256, 56, 56)
        self.flatten = Flatten()  # (B, 256, 7, 7)
        self.unflatten = Unflatten(2 * num_hiddens)  # (B, 256, 56, 56)
        self.up1 = UpBlock(4 * num_hiddens, num_hiddens)  # (B, 128, 112, 112)
        self.up2 = UpBlock(2 * num_hiddens, num_hiddens)  # (B, 128, 224, 224)
        self.conv1 = Conv(2 * num_hiddens, num_hiddens)  # (B, 128, 224, 224)
        self.conv2 = nn.Conv2d(num_hiddens, num_classes, 1)  # (B, 68, 224, 224)

    def forward(self, x: torch.Tensor, return_heatmap=True) -> torch.Tensor:
        # x: (B, 1, 224, 224)
        assert x.shape[2] == self.height and x.shape[3] == self.width
        # skip connections
        x = self.conv0(x)  # (B, D, 224, 224)
        xdown1 = self.down1(x)  # (B, D, 112, 112)
        xdown2 = self.down2(xdown1)  # (B, 2D, 56, 56)
        flat = self.flatten(xdown2)  # (B, 2D, 7, 7)
        unflat = self.unflatten(flat)  # (B, 2D, 56, 56)
        xup1 = self.up1(torch.cat([unflat, xdown2], dim=1)) # (B, 4D, 56, 56) -> (B, D, 112, 112)
        xup2 = self.up2(torch.cat([xup1, xdown1], dim=1)) # (B, 2D, 112, 112) -> (B, D, 224, 224)
        x = self.conv1(torch.cat([xup2, x], dim=1)) # (B, 2D, 224, 224) -> (B, D, 224, 224)
        logits = self.conv2(x) # (B, D, 224, 224) -> (B, 68, 224, 224)
        # softmax for every i in 68, over (224, 224) for each batch
        
        # SOFTMAX OVER 224x224
        logits = logits.view(-1, 68, 224*224) # (B, 68, 224*224)
        if return_heatmap:
            return logits # to be sigmoided
        probs = torch.softmax(logits, dim=2)
        probs = probs.view(-1, 68, 224, 224)
        with torch.no_grad():
            # ensure sum of probs over 224x224 is 1
            sum_over = torch.sum(probs, dim=(2, 3), keepdim=True) # (B, 68, 1, 1)
            assert torch.allclose(sum_over, torch.ones_like(sum_over))
        # probs: (B, 68, 224, 224)
        # should return (B, 68, 2) for each batch
        return heatmap_to_landmarks_mean(probs) / 224.0
