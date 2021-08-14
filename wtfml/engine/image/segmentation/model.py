
import torch.nn as nn
import numpy as np

class FacetypeLocationGenerator(nn.Module):
    """
    Based on DCGAN
    Args:
        nn ([type]): [description]
    """

    def __init__(self, input_size = 512,mid_layer_num=None):
        super(FacetypeLocationGenerator, self).__init__()

        self.init_size = 160 // 4
        if mid_layer_num is None:
            self.l1 = nn.Sequential(nn.Linear(input_size, 128 * self.init_size ** 2))
        else:
            self.l1 = nn.Sequential(
                nn.Linear(input_size, mid_layer_num),
                nn.ReLU(),
                nn.Linear(mid_layer_num, mid_layer_num),
                nn.ReLU(),
                nn.Linear(mid_layer_num, 128 * self.init_size ** 2),
            )
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 1, 3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


#%%
class FacetypeLocationLightGenerator(nn.Module):
    def __init__(self, img_shape=(1, 160, 160)):
        super(FacetypeLocationLightGenerator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(512, 5, normalize=False),
            *block(5, 512),
            # *block(512, 1024),
            nn.Linear(512, int(np.prod(img_shape))),
            nn.Sigmoid()
        )
        self.img_shape = img_shape

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img