""" Each encoder should have following attributes and methods and be inherited from `_base.EncoderMixin`

Attributes:

    _out_channels (list of int): specify number of channels for each encoder feature tensor
    _depth (int): specify number of stages in decoder (in other words number of downsampling operations)
    _in_channels (int): default number of input channels in first Conv2d layer for encoder (usually 3)

Methods:

    forward(self, x: torch.Tensor)
        produce list of features of different spatial resolutions, each feature is a 4D torch.tensor of
        shape NCHW (features should be sorted in descending order according to spatial resolution, starting
        with resolution same as input `x` tensor).

        Input: `x` with shape (1, 3, 64, 64)
        Output: [f0, f1, f2, f3, f4, f5] - features with corresponding shapes
                [(1, 3, 64, 64), (1, 64, 32, 32), (1, 128, 16, 16), (1, 256, 8, 8),
                (1, 512, 4, 4), (1, 1024, 2, 2)] (C - dim may differ)

        also should support number of features according to specified depth, e.g. if depth = 5,
        number of feature tensors = 6 (one with same resolution as input and 5 downsampled),
        depth = 3 -> number of feature tensors = 4 (one with same resolution as input and 3 downsampled).
"""

import torch.nn as nn

from torchvision.models.resnet import ResNet
from torchvision.models.resnet import BasicBlock
from torchvision.models.resnet import Bottleneck
from .pretrainedmodels.models.torchvision_models import pretrained_settings

from ._base import EncoderMixin

#pretrained_settings = {"alexnet": {"imagenet": {"url": "https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth", "input_space": "RGB", "input_size": [3, 224, 224], "input_range": [0, 1], "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225], "num_classes": 1000}}, "densenet121": {"imagenet": {"url": "http://data.lip6.fr/cadene/pretrainedmodels/densenet121-fbdb23505.pth", "input_space": "RGB", "input_size": [3, 224, 224], "input_range": [0, 1], "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225], "num_classes": 1000}}, "densenet169": {"imagenet": {"url": "http://data.lip6.fr/cadene/pretrainedmodels/densenet169-f470b90a4.pth", "input_space": "RGB", "input_size": [3, 224, 224], "input_range": [0, 1], "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225], "num_classes": 1000}}, "densenet201": {"imagenet": {"url": "http://data.lip6.fr/cadene/pretrainedmodels/densenet201-5750cbb1e.pth", "input_space": "RGB", "input_size": [3, 224, 224], "input_range": [0, 1], "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225], "num_classes": 1000}}, "densenet161": {"imagenet": {"url": "http://data.lip6.fr/cadene/pretrainedmodels/densenet161-347e6b360.pth", "input_space": "RGB", "input_size": [3, 224, 224], "input_range": [0, 1], "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225], "num_classes": 1000}}, "resnet18": {"imagenet": {"url": "https://download.pytorch.org/models/resnet18-5c106cde.pth", "input_space": "RGB", "input_size": [3, 224, 224], "input_range": [0, 1], "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225], "num_classes": 1000}}, "resnet34": {"imagenet": {"url": "https://download.pytorch.org/models/resnet34-333f7ec4.pth", "input_space": "RGB", "input_size": [3, 224, 224], "input_range": [0, 1], "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225], "num_classes": 1000}}, "resnet50": {"imagenet": {"url": "https://download.pytorch.org/models/resnet50-19c8e357.pth", "input_space": "RGB", "input_size": [3, 224, 224], "input_range": [0, 1], "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225], "num_classes": 1000}}, "resnet101": {"imagenet": {"url": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth", "input_space": "RGB", "input_size": [3, 224, 224], "input_range": [0, 1], "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225], "num_classes": 1000}}, "resnet152": {"imagenet": {"url": "https://download.pytorch.org/models/resnet152-b121ed2d.pth", "input_space": "RGB", "input_size": [3, 224, 224], "input_range": [0, 1], "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225], "num_classes": 1000}}, "inceptionv3": {"imagenet": {"url": "https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth", "input_space": "RGB", "input_size": [3, 299, 299], "input_range": [0, 1], "mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5], "num_classes": 1000}}, "squeezenet1_0": {"imagenet": {"url": "https://download.pytorch.org/models/squeezenet1_0-a815701f.pth", "input_space": "RGB", "input_size": [3, 224, 224], "input_range": [0, 1], "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225], "num_classes": 1000}}, "squeezenet1_1": {"imagenet": {"url": "https://download.pytorch.org/models/squeezenet1_1-f364aa15.pth", "input_space": "RGB", "input_size": [3, 224, 224], "input_range": [0, 1], "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225], "num_classes": 1000}}, "vgg11": {"imagenet": {"url": "https://download.pytorch.org/models/vgg11-bbd30ac9.pth", "input_space": "RGB", "input_size": [3, 224, 224], "input_range": [0, 1], "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225], "num_classes": 1000}}, "vgg11_bn": {"imagenet": {"url": "https://download.pytorch.org/models/vgg11_bn-6002323d.pth", "input_space": "RGB", "input_size": [3, 224, 224], "input_range": [0, 1], "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225], "num_classes": 1000}}, "vgg13": {"imagenet": {"url": "https://download.pytorch.org/models/vgg13-c768596a.pth", "input_space": "RGB", "input_size": [3, 224, 224], "input_range": [0, 1], "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225], "num_classes": 1000}}, "vgg13_bn": {"imagenet": {"url": "https://download.pytorch.org/models/vgg13_bn-abd245e5.pth", "input_space": "RGB", "input_size": [3, 224, 224], "input_range": [0, 1], "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225], "num_classes": 1000}}, "vgg16": {"imagenet": {"url": "https://download.pytorch.org/models/vgg16-397923af.pth", "input_space": "RGB", "input_size": [3, 224, 224], "input_range": [0, 1], "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225], "num_classes": 1000}}, "vgg16_bn": {"imagenet": {"url": "https://download.pytorch.org/models/vgg16_bn-6c64b313.pth", "input_space": "RGB", "input_size": [3, 224, 224], "input_range": [0, 1], "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225], "num_classes": 1000}}, "vgg19_bn": {"imagenet": {"url": "https://download.pytorch.org/models/vgg19_bn-c79401a0.pth", "input_space": "RGB", "input_size": [3, 224, 224], "input_range": [0, 1], "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225], "num_classes": 1000}}, "vgg19": {"imagenet": {"url": "https://download.pytorch.org/models/vgg19-dcbb9e9d.pth", "input_space": "RGB", "input_size": [3, 224, 224], "input_range": [0, 1], "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225], "num_classes": 1000}}}



class ResNetEncoder(ResNet, EncoderMixin):
    def __init__(self, out_channels, depth=5, **kwargs):
        super().__init__(**kwargs)
        self._depth = depth
        self._out_channels = out_channels
        self._in_channels = 3

        del self.fc
        del self.avgpool

    def get_stages(self):
        return [
            nn.Identity(),
            nn.Sequential(self.conv1, self.bn1, self.relu),
            nn.Sequential(self.maxpool, self.layer1),
            self.layer2,
            self.layer3,
            self.layer4,
        ]

    def forward(self, x):
        stages = self.get_stages()

        features = []
        for i in range(self._depth + 1):
            x = stages[i](x)
            features.append(x)

        return features

    def load_state_dict(self, state_dict, **kwargs):
        state_dict.pop("fc.bias")
        state_dict.pop("fc.weight")
        super().load_state_dict(state_dict, **kwargs)


resnet_encoders = {
    "resnet18": {
        "encoder": ResNetEncoder,
        "pretrained_settings": pretrained_settings["resnet18"],
        "params": {
            "out_channels": (3, 64, 64, 128, 256, 512),
            "block": BasicBlock,
            "layers": [2, 2, 2, 2],
        },
    },
    "resnet34": {
        "encoder": ResNetEncoder,
        "pretrained_settings": pretrained_settings["resnet34"],
        "params": {
            "out_channels": (3, 64, 64, 128, 256, 512),
            "block": BasicBlock,
            "layers": [3, 4, 6, 3],
        },
    },
    "resnet50": {
        "encoder": ResNetEncoder,
        "pretrained_settings": pretrained_settings["resnet50"],
        "params": {
            "out_channels": (3, 64, 256, 512, 1024, 2048),
            "block": Bottleneck,
            "layers": [3, 4, 6, 3],
        },
    },
    "resnet101": {
        "encoder": ResNetEncoder,
        "pretrained_settings": pretrained_settings["resnet101"],
        "params": {
            "out_channels": (3, 64, 256, 512, 1024, 2048),
            "block": Bottleneck,
            "layers": [3, 4, 23, 3],
        },
    },
    "resnet152": {
        "encoder": ResNetEncoder,
        "pretrained_settings": pretrained_settings["resnet152"],
        "params": {
            "out_channels": (3, 64, 256, 512, 1024, 2048),
            "block": Bottleneck,
            "layers": [3, 8, 36, 3],
        },
    },
    "resnext50_32x4d": {
        "encoder": ResNetEncoder,
        "pretrained_settings": {
            "imagenet": {
                "url": "https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth",
                "input_space": "RGB",
                "input_size": [3, 224, 224],
                "input_range": [0, 1],
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
                "num_classes": 1000,
            }
        },
        "params": {
            "out_channels": (3, 64, 256, 512, 1024, 2048),
            "block": Bottleneck,
            "layers": [3, 4, 6, 3],
            "groups": 32,
            "width_per_group": 4,
        },
    },
    "resnext101_32x8d": {
        "encoder": ResNetEncoder,
        "pretrained_settings": {
            "imagenet": {
                "url": "https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth",
                "input_space": "RGB",
                "input_size": [3, 224, 224],
                "input_range": [0, 1],
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
                "num_classes": 1000,
            },
            "instagram": {
                "url": "https://download.pytorch.org/models/ig_resnext101_32x8-c38310e5.pth",
                "input_space": "RGB",
                "input_size": [3, 224, 224],
                "input_range": [0, 1],
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
                "num_classes": 1000,
            },
        },
        "params": {
            "out_channels": (3, 64, 256, 512, 1024, 2048),
            "block": Bottleneck,
            "layers": [3, 4, 23, 3],
            "groups": 32,
            "width_per_group": 8,
        },
    },
    "resnext101_32x16d": {
        "encoder": ResNetEncoder,
        "pretrained_settings": {
            "instagram": {
                "url": "https://download.pytorch.org/models/ig_resnext101_32x16-c6f796b0.pth",
                "input_space": "RGB",
                "input_size": [3, 224, 224],
                "input_range": [0, 1],
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
                "num_classes": 1000,
            }
        },
        "params": {
            "out_channels": (3, 64, 256, 512, 1024, 2048),
            "block": Bottleneck,
            "layers": [3, 4, 23, 3],
            "groups": 32,
            "width_per_group": 16,
        },
    },
    "resnext101_32x32d": {
        "encoder": ResNetEncoder,
        "pretrained_settings": {
            "instagram": {
                "url": "https://download.pytorch.org/models/ig_resnext101_32x32-e4b90b00.pth",
                "input_space": "RGB",
                "input_size": [3, 224, 224],
                "input_range": [0, 1],
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
                "num_classes": 1000,
            }
        },
        "params": {
            "out_channels": (3, 64, 256, 512, 1024, 2048),
            "block": Bottleneck,
            "layers": [3, 4, 23, 3],
            "groups": 32,
            "width_per_group": 32,
        },
    },
    "resnext101_32x48d": {
        "encoder": ResNetEncoder,
        "pretrained_settings": {
            "instagram": {
                "url": "https://download.pytorch.org/models/ig_resnext101_32x48-3e41cc8a.pth",
                "input_space": "RGB",
                "input_size": [3, 224, 224],
                "input_range": [0, 1],
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
                "num_classes": 1000,
            }
        },
        "params": {
            "out_channels": (3, 64, 256, 512, 1024, 2048),
            "block": Bottleneck,
            "layers": [3, 4, 23, 3],
            "groups": 32,
            "width_per_group": 48,
        },
    },
}
