import torch
import torch.nn as nn


def positional_encode(images):
    """Copied from https://github.com/Lyusungwon/film_pytorch/blob/master/utils.py"""

    try:
        device = images.get_device()
    except RuntimeError:
        device = torch.device("cpu")
    n, c, h, w = images.size()
    x_coordinate = (
        torch.linspace(-1, 1, w).view(1, 1, 1, w).expand(n, 1, h, w).to(device)
    )
    y_coordinate = (
        torch.linspace(-1, 1, h).view(1, 1, h, 1).expand(n, 1, h, w).to(device)
    )
    images = torch.cat([images, x_coordinate, y_coordinate], 1)
    return images


class FilmResModule(nn.Module):
    def __init__(self, filter_size, kernel_size):
        super().__init__()

        self.conv1 = nn.Conv2d(filter_size + 2, filter_size, 1, 1, 0)
        self.conv2 = nn.Conv2d(
            filter_size, filter_size, kernel_size, 1, (kernel_size - 1) // 2
        )
        self.batch_norm = nn.BatchNorm2d(filter_size)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, betagamma):
        x = positional_encode(x)
        x = self.relu(self.conv1(x))
        residual = x
        beta = betagamma[:, 0].unsqueeze(2).unsqueeze(3).expand_as(x)
        gamma = betagamma[:, 1].unsqueeze(2).unsqueeze(3).expand_as(x)
        x = self.batch_norm(self.conv2(x))
        x = self.relu(x * beta + gamma)
        x = x + residual
        return x


class FilmModule(nn.Module):
    def __init__(
        self,
        in_channels,
        res_channels,
        res_kernel,
        bg_hidden_size,
        out_channels,
        out_kernel,
        n_blocks=3,
    ):
        super().__init__()

        self.relu = nn.ReLU()
        self.film_fc = nn.Linear(bg_hidden_size, res_channels * n_blocks * 2,)
        self.first_conv = nn.Sequential(
            nn.Conv2d(in_channels, res_channels, 1, 1, 0),
            nn.BatchNorm2d(res_channels),
            self.relu,
        )
        self.res_blocks = nn.ModuleList(
            [FilmResModule(res_channels, res_kernel) for _ in range(n_blocks)]
        )
        self.last_conv = nn.Conv2d(res_channels + 2, out_channels, out_kernel, 1, 0)

    def forward(self, x, h):
        betagamma = self.film_fc(h)
        x = self.first_conv(x)
        for n, block in enumerate(self.res_blocks):
            x = block(x, betagamma[:n])
        x = self.last_conv(x)
        return x
