import torch
import torch.nn as nn

# vgg family configurations
# Note 1: While the paper provides the A-E and A-Local Response Normalisation
# 6 vgg structure. We will only implement A (vgg11), B(vgg13), D(vgg16)
# and E (vgg19) as per torchvision implementation.
# These are also the more popular structures.

# Note 2: Compared to AlexNet, vgg's convolutional layers have more regular
# and consistent structures. It popularised the use of 3x3 conv filters.
# For this reason, we can write a config dictionary to easily specify their
# structures. 

# m in the config indicates a 2x2 max pooling layer with a stride of 2
# number indicates the num of 3x3 kernels (channels of the layer)


# TODO: Write a summary for initialisation for the readers
# read the paper for xavier and kaiming init

# the initialisation method in the pytorch seems to be confusing
# for the fan out... also the linear layers are initialised using that proposed in 
# vgg paper. It is a mix and match

config = {
    "vgg11": [64, "m", 128, "m", 256, 256, "m", 512, 512, "m", 512, 512, "m"],
    "vgg13": [64, 64, "m", 128, 128, "m", 256, 256, "m", 512, 512, "m", 512, 512, "m"],
    "vgg16": [64, 64, "m", 128, 128, "m", 256, 256, 256], "m",  512, 512, 512, "m", 512, 512, 512, "m"],
    "vgg19": [64, 64, "m", 128, 128, "m", 256, 256, 256, 256, "m", 512, 512, 512, 512, "m", 512, 512, 512, 512, "m"]
}


class VGG(nn.Module):
    def __init__(self, name, num_classes = 1000, head = "fc", bn = False, init_weights = True):
        assert head in ["fc", "conv"], "classification head must be fc or conv"
        self.features = self._get_conv_layers(name)
        if head == "fc":
            self.classifier = self._get_fc_classifier(num_classes)
        else:
            self.classifier = self._get_conv_classifier(num_classes)

    
    def _get_conv_layers(self, name):
        cfg = config[name]
        # As the input image has RGB three channels
        num_prev_channels = 3
        layers = []
        for layer in cfg:
            if layer == "m":
                layers.append(nn.MaxPool2d(kernel_size = 2, stride = 2))
            else:
                layers.append(nn.Conv2d(num_prev_channels, layer, kernel_size = 3, stride = 1, padding = 1))
                # store the current number of channels
                num_prev_channels = layer
                if self.bn:
                    # batch normalisation is usually added after convolution
                    # before activations, though some researchers may argue for the
                    # case of putting it after activation
                    layers.append(nn.BatchNorm2D(layer))
                # append ReLU activation
                # again, inplace is set to True to save memory
                layers.append(nn.ReLU(inplace = True))

        # pack all layers into a module
        return nn.Sequential(*conv_layers)

    def _get_fc_classifier(self, num_classes):
        return nn.Sequential(
            # in case the input is not 224 and hence the final
            # feature map has spatial dimensions different from (7, 7)
            nn.AdaptiveAvgPool2d((7, 7)),
            # flatten the feature map for FC layers
            nn.Flatten(),
            # FC 1
            nn.Linear(in_features=7 * 7 * 512, out_features=4096, bias=True),
            nn.ReLu(inplace = True),
            # FC 2
            # Dropout is usually added after activation and before convolution
            # again, some researchers may argue for other case
            nn.Dropout(),
            nn.Linear(in_features = 4096, out_features = 4096, bias = True),
            nn.ReLU(inplace = True),
            # FC 3
            nn.Dropout(),
            nn.Linear(in_features = 4096, num_classes)
        )


    def _get_conv_classifier(self, num_classes):
        return nn.Sequential(
            # Conv for FC 1
            nn.Conv2d(in_channels = 512, out_channels = 4096, kernel_size = 7),
            # Conv for FC 2
            nn.Dropout(),
            nn.Conv2d(in_channels = 4096, out_channels = 4096, kernel_size = 1),
            # Conv for FC3
            nn.Dropout(),
            nn.Conv2d(in_channels = 4096, out_channels = num_classes, kernel_size = 1)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        return x


    def _init_weights(self):
        # use a mix of Kaiming init and Xavier init
