Pytorch Extension for Saliency Maps
====================================

Pytorch implementation of different saliency methods.


# Dependencies

* Pytorch 4.0

# Installing

## From source

```

git clone https://github.com/Ema93sh/pytorch-saliency

cd pytorch-saliency

python setup.py install

```


# Example Usage
For more detailed example checkout main.py

```

from torchvision import models

from saliency import SaliencyMethod, MapType, generate_saliency


# Load your image
input_image =  ...


# Load your model
model = models.vgg16(pretrained=True)

# Choose your saliency method
saliency_method = SaliencyMethod.GUIDED

# Choose your targets for saliency
targets = [2]

saliency_maps = generate_saliency(model, input_image, targets, saliency_method)

# Select the type that you want to display
saliency = saliency_maps[MapType.POSITIVE]

```

# Implemented Saliency Methods

*  Vanilla backprop
*  Guided backprop
*  DeConv


# Map types

* Positive (MapType.POSITIVE)
* Negative (MapType.NEGATIVE)
* Absolute (MapType.ABSOLUTE)
* Original (MapType.ORIGINAL)


# Testing
To test your saliency method run main.py. It will pick a random image from images folder and display the saliency.

```

python main.py --type guided

```

# References / Credits

* https://github.com/Lasagne/Recipes/blob/master/examples/Saliency%20Maps%20and%20Guided%20Backpropagation.ipynb
