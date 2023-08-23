import torchvision.models as models
import torch.nn as nn

from utils import *


def save_debug_image(style_images, content_images, transformed_images, filename):
    width = 256
    style_images = [recover_image(x) for x in style_images]
    content_images = [recover_image(x) for x in content_images]
    transformed_images = [recover_image(x) for x in transformed_images]

    new_im = Image.new('RGB', ((width + 5) * 8, width * 3 + 5))
    for i, (a, b, c) in enumerate(zip(style_images, content_images, transformed_images)):
        new_im.paste(Image.fromarray(a), ((width + 5) * i, 0))
        new_im.paste(Image.fromarray(b), ((width + 5) * i, width + 5))
        new_im.paste(Image.fromarray(c), ((width + 5) * i, width * 2 + 10))

    new_im.save(filename)


def get_IN_mean_std(features):
    """input: b * c * h * w
        return: features_mean, features_std: b * c * 1 * 1"""
    batch_size, c = features.size()[0:2]
    features_mean = features.reshape(batch_size, c, -1).mean(dim=2).reshape(batch_size, c, 1, 1)
    features_std = features.reshape(batch_size, c, -1).std(dim=2).reshape(batch_size, c, 1, 1) + 1e-6
    return features_mean, features_std


def adain(content_features, style_features):
    """Adaptive Instance Normalization"""
    content_mean, content_std = get_IN_mean_std(content_features)
    style_mean, style_std = get_IN_mean_std(style_features)
    normalized_features = style_std * (content_features - content_mean) / content_std + style_mean
    return normalized_features


class VGGEncoder(nn.Module):
    """ input channels number: 3
    output channels number: 512"""

    def __init__(self):
        super(VGGEncoder, self).__init__()
        vgg = models.vgg19(pretrained=True).features
        self.slice1 = vgg[: 2]
        self.slice2 = vgg[2: 7]
        self.slice3 = vgg[7: 12]
        self.slice4 = vgg[12: 21]
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        h1 = self.slice1(x)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        h4 = self.slice4(h3)
        return h1, h2, h3, h4


def ConvLayer(in_channels, out_channels, kernel_size=3, stride=1,
              upsampling=False, relu=True):
    layers = []
    if upsampling:
        layers.append(nn.UpsamplingNearest2d(scale_factor=2))
    layers.append(nn.ReflectionPad2d(kernel_size // 2))
    layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                            stride=stride))
    if relu:
        layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class Decoder(nn.Module):
    """ input channels number: 512,
    output channels number: 3"""

    def __init__(self):
        super(Decoder, self).__init__()
        self.sequential = nn.Sequential(
            ConvLayer(512, 256),
            ConvLayer(256, 256, upsampling=True),
            ConvLayer(256, 256),
            ConvLayer(256, 256),
            ConvLayer(256, 128),
            ConvLayer(128, 128, upsampling=True),
            ConvLayer(128, 64),
            ConvLayer(64, 64, upsampling=True),
            ConvLayer(64, 3, relu=False)
        )

    def forward(self, x):
        return self.sequential(x)


def content_loss(y_hat, y):
    return F.mse_loss(y_hat, y.detach())


def style_loss(y_hat, y):
    loss = 0
    for i in range(len(y)):
        mean_hat, std_hat = get_IN_mean_std(y_hat[i])
        mean_y, std_y = get_IN_mean_std(y[i])
        loss += F.mse_loss(mean_y, mean_hat) + F.mse_loss(std_y, std_hat)
    return loss


class TransformNet(nn.Module):
    """风格转换网络"""

    def __init__(self):
        super(TransformNet, self).__init__()
        self.encoder = VGGEncoder()
        self.decoder = Decoder()

    def generate(self, content_images, style_images, alpha=1.0):
        content_features = self.encoder(content_images)[-1]
        style_features = self.encoder(style_images)[-1]
        t = adain(content_features, style_features)
        t = alpha * t + (1 - alpha) * content_features
        out = self.decoder(t)
        return out

    def forward(self, content_images, style_images, content_weight, style_weight, show=0):
        content_features = self.encoder(content_images)
        style_features = self.encoder(style_images)
        target_features = adain(content_features[-1], style_features[-1])
        mixed_images = self.decoder(target_features)
        mixed_features = self.encoder(mixed_images)
        if show != 0:
            save_debug_image(style_images, content_images, mixed_images,
                             f"debug_{show}.jpg")

        cl = content_loss(mixed_features[-1], target_features) * content_weight
        sl = style_loss(mixed_features, style_features) * style_weight
        loss = cl + sl
        return loss
