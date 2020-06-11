import torch
from torch import nn

class addCoords(nn.Module):
    def __init__(self):
        super(addCoords, self).__init__()

    def forward(self, x):
        in_batch, in_ch, in_h, in_w = x.shape
        width_coords = torch.linspace(-1, 1, steps=in_w).to(x.device)
        wc = width_coords.repeat(in_batch, 1, in_h, 1)
        if in_h != 1:
            height_coords = torch.linspace(-1, 1, steps=in_h).view(in_h, -1).to(x.device)
            hc = height_coords.repeat(in_batch, 1, 1, in_w)
            coord_x = torch.cat((x, wc, hc), 1)
        else:
            coord_x = torch.cat((x, wc), 1)
        return coord_x


class addCoords_1D(nn.Module):
    def __init__(self):
        super(addCoords_1D, self).__init__()

    def forward(self, x):
        in_batch, in_ch, in_w = x.shape
        width_coords = torch.linspace(-1, 1, steps=in_w).to(x.device)
        wc = width_coords.repeat(in_batch, 1, 1)
        coord_x = torch.cat((x, wc), 1)
        return coord_x



