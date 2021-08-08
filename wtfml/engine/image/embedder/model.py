from numpy import extract
import torch
import torch.nn as nn
from torch.nn import (
    Linear,
    Conv2d,
    BatchNorm1d,
    BatchNorm2d,
    PReLU,
    ReLU,
    Sigmoid,
    Dropout,
    MaxPool2d,
    AdaptiveAvgPool2d,
    Sequential,
    Module,
)
from collections import namedtuple


# Support: ['IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101', 'IR_SE_152']


class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)

    return output


class SEModule(Module):
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = AdaptiveAvgPool2d(1)
        self.fc1 = Conv2d(
            channels, channels // reduction, kernel_size=1, padding=0, bias=False
        )

        nn.init.xavier_uniform_(self.fc1.weight.data)

        self.relu = ReLU(inplace=True)
        self.fc2 = Conv2d(
            channels // reduction, channels, kernel_size=1, padding=0, bias=False
        )

        self.sigmoid = Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)

        return module_input * x


class bottleneck_IR(Module):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                BatchNorm2d(depth),
            )
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            PReLU(depth),
            Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            BatchNorm2d(depth),
        )

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)

        return res + shortcut


class bottleneck_IR_SE(Module):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR_SE, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                BatchNorm2d(depth),
            )
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            PReLU(depth),
            Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            BatchNorm2d(depth),
            SEModule(depth, 16),
        )

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)

        return res + shortcut


class Bottleneck(namedtuple("Block", ["in_channel", "depth", "stride"])):
    """A named tuple describing a ResNet block."""


def get_block(in_channel, depth, num_units, stride=2):

    return [Bottleneck(in_channel, depth, stride)] + [
        Bottleneck(depth, depth, 1) for i in range(num_units - 1)
    ]


def get_blocks(num_layers):
    blocks = None
    if num_layers == 50:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=4),
            get_block(in_channel=128, depth=256, num_units=14),
            get_block(in_channel=256, depth=512, num_units=3),
        ]
    elif num_layers == 100:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=13),
            get_block(in_channel=128, depth=256, num_units=30),
            get_block(in_channel=256, depth=512, num_units=3),
        ]
    elif num_layers == 152:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=8),
            get_block(in_channel=128, depth=256, num_units=36),
            get_block(in_channel=256, depth=512, num_units=3),
        ]

    return blocks


class Backbone(Module):
    def __init__(self, input_size, num_layers, mode="ir"):
        super(Backbone, self).__init__()
        assert input_size[0] in [
            112,
            224,
        ], "input_size should be [112, 112] or [224, 224]"
        assert num_layers in [50, 100, 152], "num_layers should be 50, 100 or 152"
        assert mode in ["ir", "ir_se"], "mode should be ir or ir_se"
        blocks = get_blocks(num_layers)
        if mode == "ir":
            unit_module = bottleneck_IR
        elif mode == "ir_se":
            unit_module = bottleneck_IR_SE
        else:
            raise NotImplementedError()
        self.input_layer = Sequential(
            Conv2d(3, 64, (3, 3), 1, 1, bias=False), BatchNorm2d(64), PReLU(64)
        )
        if input_size[0] == 112:
            self.output_layer = Sequential(
                BatchNorm2d(512),
                Dropout(),
                Flatten(),
                Linear(512 * 7 * 7, 512),
                BatchNorm1d(512),
            )
        else:
            self.output_layer = Sequential(
                BatchNorm2d(512),
                Dropout(),
                Flatten(),
                Linear(512 * 14 * 14, 512),
                BatchNorm1d(512),
            )

        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(
                    unit_module(
                        bottleneck.in_channel, bottleneck.depth, bottleneck.stride
                    )
                )
        self.body = Sequential(*modules)

        self._initialize_weights()

    def forward(self, x):
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_layer(x)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()


def IR_50(input_size):
    """Constructs a ir-50 model."""
    model = Backbone(input_size, 50, "ir")

    return model


def IR_101(input_size):
    """Constructs a ir-101 model."""
    model = Backbone(input_size, 100, "ir")

    return model


def IR_152(input_size):
    """Constructs a ir-152 model."""
    model = Backbone(input_size, 152, "ir")

    return model


def IR_SE_50(input_size):
    """Constructs a ir_se-50 model."""
    model = Backbone(input_size, 50, "ir_se")

    return model


def IR_SE_101(input_size):
    """Constructs a ir_se-101 model."""
    model = Backbone(input_size, 100, "ir_se")

    return model


def IR_SE_152(input_size):
    """Constructs a ir_se-152 model."""
    model = Backbone(input_size, 152, "ir_se")

    return model


class IR_50_classificator(nn.Module):
    def __init__(
        self, feature_extractor=IR_50, num_classes=23, backbone_model_path=None
    ):
        super(IR_50_classificator, self).__init__()
        self.feature_extractor = feature_extractor(input_size=[112, 112])
        if backbone_model_path:
            self.feature_extractor.load_state_dict(
                torch.load(backbone_model_path, map_location=torch.device("cpu"))
            )
        self.fc = nn.Linear(512, num_classes)

    def forward(self, input_image):
        x = self.feature_extractor(input_image)
        x = self.fc(x)
        return x

    def extract_feature(self, input_image):
        x = self.feature_extractor(input_image)
        return x



def classification_model(
    save_model_path=None,
    transfer_learning=False,
    num_classes=None,
    classify=False,
    pretrained=True,
    model_name="ir50",
):

    if model_name == "ir50":
        if classify and num_classes:
            if save_model_path:
                base_model = IR_50_classificator(
                    num_classes=num_classes, backbone_model_path=save_model_path
                )
            else:
                base_model = IR_50_classificator(num_classes=num_classes)
        else:
            base_model = IR_50(input_size=[112, 112])
            if save_model_path:
                base_model.load_state_dict(
                torch.load(save_model_path, map_location=torch.device("cpu"))
                )

        if transfer_learning:
            for _, module in base_model._modules.items():
                module.requires_grad = False  # 全ての層を凍結
                base_model.fc = nn.Linear(512, num_classes)

    return base_model