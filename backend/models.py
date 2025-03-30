import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(channels)
        )

    def forward(self, x):
        return x + self.block(x)

# Define Generator
class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super(Generator, self).__init__()
        # Initial Convolution Block
        model = [nn.Conv2d(input_nc, 64, kernel_size=7, stride=1, padding=3),
                 nn.InstanceNorm2d(64),
                 nn.ReLU(inplace=True)]

        # Downsampling
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model += [nn.Conv2d(in_features, out_features, kernel_size=3, stride=2, padding=1),
                      nn.InstanceNorm2d(out_features),
                      nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features * 2

        # Residual Blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        # Upsampling
        out_features = in_features // 2
        for _ in range(2):
            model += [nn.ConvTranspose2d(in_features, out_features, kernel_size=3, stride=2, padding=1, output_padding=1),
                      nn.InstanceNorm2d(out_features),
                      nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features // 2

        # Output Layer
        model += [nn.Conv2d(64, output_nc, kernel_size=7, stride=1, padding=3),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

# Define Discriminator
class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()
        model = [
            nn.Conv2d(input_nc, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        in_features = 64
        for _ in range(3):
            out_features = in_features * 2
            model += [
                nn.Conv2d(in_features, out_features, kernel_size=4, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.LeakyReLU(0.2, inplace=True)
            ]
            in_features = out_features

        model += [nn.Conv2d(in_features, 1, kernel_size=4, stride=1, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

# Load VGG model for perceptual loss
vgg = models.vgg19(pretrained=True).features.to(device).eval()

# Define feature loss (perceptual loss) using VGG features
class FeatureLoss(nn.Module):
    def __init__(self, vgg_model, layer_ids):
        super(FeatureLoss, self).__init__()
        self.vgg_model = vgg_model
        self.layer_ids = layer_ids
        self.mse_loss = nn.MSELoss()

    def forward(self, gen_img, target_img):
        gen_features = []
        target_features = []

        x_gen = gen_img
        x_target = target_img

        for i, layer in enumerate(self.vgg_model):
            x_gen = layer(x_gen)
            x_target = layer(x_target)

            if i in self.layer_ids:
                gen_features.append(x_gen)
                target_features.append(x_target)

        loss = 0
        for gen_feat, target_feat in zip(gen_features, target_features):
            loss += self.mse_loss(gen_feat, target_feat)
        return loss

# Set layer indices for feature loss
feature_loss = FeatureLoss(vgg, layer_ids=[1, 6, 11, 20, 29]).to(device)  # Using VGG19 layers