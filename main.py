import torch.utils.data

from utils import *
from models import *

import argparse

from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)
lr = 5e-5
batch_size = 8
epochs = 5

style_path = "E:/ArtisticStyle/style/Vangogh.jpg"
style_image = read_image(style_path).to(device)
plt.ion()
plt.figure()
imshow(style_image, title='Style Image')

data_transform = transforms.Compose([
    transforms.RandomResizedCrop(256, scale=(256 / 480, 1), ratio=(1, 1)),
    transforms.ToTensor(),
    tensor_normalizer
])

content_dataset = LoadData('E:/ArtisticStyle/archive/coco2017', transform=data_transform)
style_dataset = torchvision.datasets.ImageFolder('E:/ArtisticStyle/archive/wikiart/images', transform=data_transform)


def train():
    net = TransformNet().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    smooth_loss = Smooth()
    for epoch in range(1, epochs + 1):
        print(f'Start {epoch} epoch')
        style_loader = torch.utils.data.DataLoader(style_dataset, batch_size=batch_size,
                                           shuffle=True, generator=torch.Generator(device='cuda'))
        content_loader = torch.utils.data.DataLoader(content_dataset, batch_size=batch_size,
                                             shuffle=True, generator=torch.Generator(device='cuda'))
        style_iters = len(style_dataset)
        content_iters = len(content_dataset)
        print(f'Length of style image pairs: {style_iters}')
        print(f'Length of content image pairs: {content_iters}')

        iter = zip(content_loader, style_loader)
        n_batch = min(len(content_loader), len(style_loader))
        with tqdm(iter, total=n_batch) as pbar:
            i = 0
            for (content_image, style_image) in pbar:
                is_show = 0
                style_image = style_image[0].to(device)
                content_image = content_image.to(device)
                if not content_image.size()[0] == style_image.size()[0]:
                    break
                if i % 500 == 0:
                    is_show = i
                    print(f'[{epoch}/total {epochs} epoch], /'
                          f'loss: {smooth_loss:3f}')
                loss = net(content_image, style_image, 1, 10, is_show)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                i += 1
                smooth_loss += loss.item()
    torch.save(net.state_dict(), f'{epoch}_epoch.pth')


def test():
    transform_net = TransformNet()
    transform_net.load_state_dict(torch.load('C:/Users/vcc/PycharmProjects/ArbitraryStyleTransfer/output/3_epoch.pth'))
    style_path = "E:/ArtisticStyle/style/Picasso.jpg"
    style_img = read_image(style_path).to(device)
    content_img = read_image('E:/ArtisticStyle/content.jpg').to(device)

    output_img = transform_net.generate(content_img, style_img)

    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    imshow(style_img, title='Style Image')

    plt.subplot(1, 3, 2)
    imshow(content_img, title='Content Image')

    plt.subplot(1, 3, 3)
    imshow(output_img.detach(), title='Output Image')

test()
plt.ioff()
plt.show()
