import random
import argparse

import matplotlib.pyplot as plt
import torch
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader

from saliency import *

plt.ion()

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--type',
        help = 'Type of saliency to generate',
        choices = ['guided', 'vanilla', 'deconv'],
        default = 'vanilla',
        required = False
    )

    choices_map = {
        'guided': SaliencyMethod.GUIDED,
        'vanilla': SaliencyMethod.VANILLA,
        'deconv': SaliencyMethod.DECONV
    }

    args = parser.parse_args()

    saliency_type = choices_map[args.type]

    model = models.vgg16(pretrained=True)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    data_transform = transforms.Compose([
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize
            ])

    cats_vs_dogs = datasets.ImageFolder(root = './images/', transform = data_transform)
    dataset_loader = torch.utils.data.DataLoader(cats_vs_dogs,
                                                 batch_size=4,
                                                 shuffle=True,
                                                 num_workers=4)

    random_image = random.choice(range(len(dataset_loader.dataset)))
    input = dataset_loader.dataset[random_image][0].unsqueeze(0)

    prediction = model(input)

    _, prediction = prediction.max(1)

    saliency = generate_saliency(model, input, prediction, type = saliency_type)

    figure = plt.figure(figsize = (8, 8), facecolor='w')

    plt.subplot(2, 2, 1)
    plt.title("Original Image")
    plt.imshow(input.squeeze(0).mean(0), cmap="gray")

    plt.subplot(2, 2, 2)
    plt.title("Positive Saliency")
    plt.imshow(saliency[MapType.POSITIVE].mean(0), cmap='gray')

    plt.subplot(2, 2, 3)
    plt.title("Negative Saliency")
    plt.imshow(saliency[MapType.NEGATIVE].mean(0), cmap='gray')

    plt.subplot(2, 2, 4)
    plt.title("Absolute Saliency")
    plt.imshow(saliency[MapType.ABSOLUTE].mean(0), cmap='gray')

    plt.show(block = True)


if __name__ == '__main__':
    main()
