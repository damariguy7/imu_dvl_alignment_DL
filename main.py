import math
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torch
import random
from torch import nn
from torch import Tensor
from typing import Type, Any, Callable, Union, List, Optional
from sklearn.model_selection import train_test_split
import dask.dataframe as dd


# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


__all__ = [
    "ResNet",
    "resnet18_1d",
    "resnet34_1d",
    "resnet50_1d",
    "resnet101_1d",
    "resnet152_1d",
    "resnext50_32x4d_1d",
    "resnext101_32x8d_1d",
    "wide_resnet50_2_1d",
    "wide_resnet101_2_1d",
]


# model_urls = {
#     "resnet18": "https://download.pytorch.org/models/resnet18-f37072fd.pth",
#     "resnet34": "https://download.pytorch.org/models/resnet34-b627a593.pth",
#     "resnet50": "https://download.pytorch.org/models/resnet50-0676ba61.pth",
#     "resnet101": "https://download.pytorch.org/models/resnet101-63fe2227.pth",
#     "resnet152": "https://download.pytorch.org/models/resnet152-394f9c45.pth",
#     "resnext50_32x4d": "https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth",
#     "resnext101_32x8d": "https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth",
#     "wide_resnet50_2": "https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth",
#     "wide_resnet101_2": "https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth",
# }


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv1d:
    """3x3 convolution with padding"""
    return nn.Conv1d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv1d:
    """1x1 convolution"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv1d(6, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _resnet(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any,
) -> ResNet:
    model = ResNet(block, layers, **kwargs)
    # if pretrained:
    #     state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
    #     model.load_state_dict(state_dict)
    return model


def resnet18_1d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet18", BasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)


def resnet34_1d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet34", BasicBlock, [3, 4, 6, 3], pretrained, progress, **kwargs)


def resnet50_1d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet50", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)


def resnet101_1d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet101", Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs)


def resnet152_1d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet152", Bottleneck, [3, 8, 36, 3], pretrained, progress, **kwargs)


def resnext50_32x4d_1d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 4
    return _resnet("resnext50_32x4d", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)


def resnext101_32x8d_1d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 8
    return _resnet("resnext101_32x8d", Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs)


def wide_resnet50_2_1d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["width_per_group"] = 64 * 2
    return _resnet("wide_resnet50_2", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)


def wide_resnet101_2_1d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["width_per_group"] = 64 * 2
    return _resnet("wide_resnet101_2", Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs)






class Resnet1chDnet(nn.Module):
    def __init__(self, in_channels=6, output_features=3):
        self.in_channels = in_channels
        super(Resnet1chDnet, self).__init__()

        self.model = resnet18_1d()

        # "ResNet",
        # "resnet18_1d",
        # "resnet34_1d",
        # "resnet50_1d",
        # "resnet101_1d",
        # "resnet152_1d",
        # "resnext50_32x4d_1d",
        # "resnext101_32x8d_1d",
        # "wide_resnet50_2_1d",
        # "wide_resnet101_2_1d",

        # Changed Conv2d to Conv1d since we're working with 1D data
        self.model.conv1 = nn.Conv1d(self.in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Update the final fully connected layer to match the expected dimensions
        num_features = self.model.fc.in_features  # Get the number of input features
        self.model.fc = nn.Linear(num_features, output_features)

    def forward(self, x):
        # Permute the dimensions from [batch_size, sequence_length, channels] to [batch_size, channels, sequence_length]
        x = x.permute(0, 2, 1)
        return self.model(x)


#
# # Define the CNN model
# class IMUDVLCNN(nn.Module):
#     def __init__(self, dropout_rate=0.2):  # Reduced dropout rate
#         super(IMUDVLCNN, self).__init__()
#         # Increase network capacity slightly
#         self.bn1 = nn.BatchNorm1d(128)
#         self.bn2 = nn.BatchNorm1d(256)
#         self.bn3 = nn.BatchNorm1d(512)
#
#         self.conv1 = nn.Conv1d(6, 128, kernel_size=5, padding=1)
#         self.conv2 = nn.Conv1d(128, 256, kernel_size=5, padding=1)
#         self.conv3 = nn.Conv1d(256, 512, kernel_size=5, padding=1)
#
#         self.pool = nn.AdaptiveAvgPool1d(1)
#         self.dropout = nn.Dropout(dropout_rate)
#
#         self.fc1 = nn.Linear(512, 1024)
#         self.fc2 = nn.Linear(1024, 3)
#         self.relu = nn.ReLU()
#
#     def forward(self, x):
#         x = x.permute(0, 2, 1)
#
#         # Residual connection for first block
#         identity = self.conv1(x)
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = x + identity  # Residual connection
#         x = self.dropout(x)
#
#         # Second block
#         x = self.conv2(x)
#         x = self.bn2(x)
#         x = self.relu(x)
#         x = self.dropout(x)
#
#         # Third block
#         x = self.conv3(x)
#         x = self.bn3(x)
#         x = self.relu(x)
#         x = self.dropout(x)
#
#         x = self.pool(x)
#         x = x.view(x.size(0), -1)
#
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.dropout(x)
#         x = self.fc2(x)
#
#         return x
#
#
# class ResBlock1D(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=3):
#         super(ResBlock1D, self).__init__()
#         self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
#         self.bn1 = nn.BatchNorm1d(out_channels)
#         self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size // 2)
#         self.bn2 = nn.BatchNorm1d(out_channels)
#         self.relu = nn.ReLU(inplace=True)
#
#         # Projection shortcut if dimensions change
#         self.shortcut = nn.Sequential()
#         if in_channels != out_channels:
#             self.shortcut = nn.Sequential(
#                 nn.Conv1d(in_channels, out_channels, 1),
#                 nn.BatchNorm1d(out_channels)
#             )
#
#     def forward(self, x):
#         identity = self.shortcut(x)
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#
#         out += identity
#         out = self.relu(out)
#
#         return out
#
#
# class SimplerIMUResNet(nn.Module):
#     def __init__(self, dropout_rate=0.2):
#         super(SimplerIMUResNet, self).__init__()
#
#         # Initial convolution
#         self.conv1 = nn.Conv1d(6, 64, kernel_size=7, padding=3)
#         self.bn1 = nn.BatchNorm1d(64)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
#
#         # ResNet blocks with increasing channels
#         self.res1_1 = ResBlock1D(64, 128)
#         # self.res1_2 = ResBlock1D(64, 64)
#
#         self.res2_1 = ResBlock1D(128, 256)
#         # self.res2_2 = ResBlock1D(128, 128)
#
#         self.res3_1 = ResBlock1D(256, 512)
#         # self.res3_2 = ResBlock1D(256, 256)
#
#         self.res4_1 = ResBlock1D(512, 1024)
#         # self.res4_2 = ResBlock1D(512, 512)
#
#         # Global pooling and final layers
#         self.avgpool = nn.AdaptiveAvgPool1d(1)
#         self.dropout = nn.Dropout(dropout_rate)
#         self.fc = nn.Linear(1024, 3)
#
#     def forward(self, x):
#         # Input shape: (batch, time, features)
#         x = x.permute(0, 2, 1)  # to (batch, features, time)
#
#         # Initial convolution block
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
#
#         # ResNet blocks
#         # Stage 1
#         x = self.res1_1(x)
#         # x = self.res1_2(x)
#
#         # Stage 2
#         x = self.res2_1(x)
#         # x = self.res2_2(x)
#
#         # Stage 3
#         x = self.res3_1(x)
#         # x = self.res3_2(x)
#
#         # Stage 4
#         x = self.res4_1(x)
#         # x = self.res4_2(x)
#
#         # Global pooling and prediction
#         x = self.avgpool(x)
#         x = torch.flatten(x, start_dim=1)
#         # x = self.dropout(x)
#         x = self.fc(x)
#
#         return x


def train_model(model, train_loader, val_loader, optimizer, num_epochs, device,
                scheduler=None):
    model.to(device)
    best_val_loss = float('inf')
    patience = 1
    patience_counter = 0
    best_model_state = None
    criterion = EulerAnglesLoss()  # Using the specialized loss for Euler angles

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            # Calculate loss considering periodic nature of angles
            loss = criterion(outputs, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            running_loss += loss.item()

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        print(f'Epoch {epoch + 1}/{num_epochs}, '
              f'Training Loss: {avg_train_loss:.4f}, '
              f'Validation Loss: {avg_val_loss:.4f}')

        if scheduler is not None:
            scheduler.step(avg_val_loss)

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after epoch {epoch + 1}")
                model.load_state_dict(best_model_state)  # Restore best model
                break

    return model, best_val_loss


class EulerAnglesLoss(nn.Module):
    def __init__(self):
        """
        Custom loss function for Euler angles that handles periodicity and works with angles in degrees
        """
        super(EulerAnglesLoss, self).__init__()

    def normalize_angle(self, angle):
        """
        Normalize angle to [-180, 180] range
        Args:
            angle (torch.Tensor): Input angle in degrees
        Returns:
            torch.Tensor: Normalized angle in [-180, 180] range
        """
        return ((angle + 180) % 360) - 180

    def forward(self, pred, target):
        """
        Calculate the loss between predicted and target Euler angles
        Args:
            pred (torch.Tensor): Predicted Euler angles [batch_size, 3] in degrees
            target (torch.Tensor): Target Euler angles [batch_size, 3] in degrees
        Returns:
            torch.Tensor: Scalar loss value
        """
        # Normalize both predicted and target angles to [-180, 180] range
        pred_normalized = torch.stack([self.normalize_angle(a) for a in pred.t()]).t()
        target_normalized = torch.stack([self.normalize_angle(a) for a in target.t()]).t()

        # Calculate the angular difference considering periodicity
        diff = pred_normalized - target_normalized
        diff_normalized = torch.stack([self.normalize_angle(d) for d in diff.t()]).t()

        # Calculate MSE loss on the normalized differences
        loss = torch.mean(diff_normalized ** 2)

        return loss



class QuaternionNormLoss(nn.Module):
    def __init__(self, norm_weight=0.1):
        super(QuaternionNormLoss, self).__init__()
        self.norm_weight = norm_weight

    def forward(self, pred, target):
        # Orientation loss
        inner_product = torch.sum(pred * target, dim=-1)
        orientation_loss = 1 - torch.abs(inner_product)

        # Unit norm constraint
        norm_loss = torch.abs(torch.sum(pred * pred, dim=-1) - 1.0)

        return torch.mean(orientation_loss + self.norm_weight * norm_loss)


class AngularMSELoss(nn.Module):
    def __init__(self, period=360.0):
        """
        Custom loss function for angular values that handles wrapping.

        Args:
            period (float): The period of the angle in degrees (360 for full circle, 180 for half circle)
        """
        super(AngularMSELoss, self).__init__()
        self.period = period

    def forward(self, pred, target):
        """
        Calculate the MSE loss accounting for angular wrapping.

        Args:
            pred (torch.Tensor): Predicted angles in degrees (batch_size x 3 for roll, pitch, yaw)
            target (torch.Tensor): Target angles in degrees (batch_size x 3 for roll, pitch, yaw)

        Returns:
            torch.Tensor: Mean squared angular difference
        """
        # Calculate the absolute difference
        diff = pred - target

        # Handle wrapping for each angle separately
        wrapped_diff = torch.remainder(diff + self.period / 2, self.period) - self.period / 2

        # Calculate MSE on the wrapped differences
        return torch.mean(wrapped_diff ** 2)


def quaternion_to_euler(q):
    """Convert quaternions to Euler angles (roll, pitch, yaw).

    Args:
        q (torch.Tensor): Quaternions with shape (..., 4) where q = [w, x, y, z]

    Returns:
        torch.Tensor: Euler angles [roll, pitch, yaw] in degrees
    """
    # Extract quaternion components
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    pitch = torch.where(
        torch.abs(sinp) >= 1,
        torch.sign(sinp) * torch.tensor(math.pi / 2, device=q.device),
        torch.asin(sinp)
    )

    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    # Convert from radians to degrees
    roll_deg = roll * 180.0 / math.pi
    pitch_deg = pitch * 180.0 / math.pi
    yaw_deg = yaw * 180.0 / math.pi

    return torch.stack([roll_deg, pitch_deg, yaw_deg], dim=-1)


def single_quaternion_to_euler(q):
    """Convert a single quaternion to Euler angles (roll, pitch, yaw).

    Args:
        q (list or array): Single quaternion [w, x, y, z]

    Returns:
        tuple: Euler angles (roll, pitch, yaw) in degrees
    """
    # Extract quaternion components
    w, x, y, z = q[0], q[1], q[2], q[3]

    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    # Check for gimbal lock, pitch = ±90°
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)
    else:
        pitch = math.asin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    # Convert from radians to degrees
    roll_deg = roll * 180.0 / math.pi
    pitch_deg = pitch * 180.0 / math.pi
    yaw_deg = yaw * 180.0 / math.pi

    return roll_deg, pitch_deg, yaw_deg

def save_baseline_results_numpy(mean_rmse_svd_degrees_per_num_samples_list, svd_time_list,rmse_test_list, current_time_test_list, config):
    """
    Save results as NumPy files.

    Args:
        mean_rmse_svd_degrees_per_num_samples_list: List of RMSE values
        svd_time_list: List of time values
        config: Includes directory to save the results
    """
    data_path = config['data_path']
    saved_baseline_results_file_name = config['saved_baseline_results_file_name']
    # rmse_saved_baseline_results_file_name = f"rmse_{saved_baseline_results_file_name}.npy"



    np.save(os.path.join(data_path, 'baseline_rmse_results', f"rmse_{saved_baseline_results_file_name}.npy"), np.array(mean_rmse_svd_degrees_per_num_samples_list))
    np.save(os.path.join(data_path, 'baseline_rmse_results', f"timeline_{saved_baseline_results_file_name}.npy"), np.array(svd_time_list))
    np.save(os.path.join(data_path, 'baseline_rmse_results', f"rmse.npy"), np.array(rmse_test_list))
    np.save(os.path.join(data_path, 'baseline_rmse_results', f"timeline.npy"), np.array(current_time_test_list))




    print(f"Results saved to {saved_baseline_results_file_name}")


def load_baseline_results_numpy(config):
    """
    Load results from NumPy files.

    Args:
        config: Includes directory containing the results

    Returns:
        Loaded arrays
    """
    data_path = config['data_path']
    loaded_baseline_results_file_names_list = config['loaded_baseline_results_file_names_list']
    loaded_rmse_baseline_results_file_names_list = []
    loaded_timeline_baseline_results_file_names_list = []


    for file_name in loaded_baseline_results_file_names_list:
        loaded_rmse_baseline_results_file_names_list.append(np.load(os.path.join(data_path,'baseline_rmse_results',f"rmse_{file_name}.npy")).tolist())
        loaded_timeline_baseline_results_file_names_list.append(np.load(os.path.join(data_path,'baseline_rmse_results',f"timeline_{file_name}.npy")).tolist())


    # mean_rmse_svd = np.load(f"baseline_rmse_results/mean_rmse_svd.npy").tolist()
    # svd_time = np.load(f"baseline_rmse_results/svd_time.npy").tolist()

    return loaded_rmse_baseline_results_file_names_list, loaded_timeline_baseline_results_file_names_list



    #
    # for test_idx, test_sequence in enumerate(v_imu_dvl_test_series_list):
    #     v_imu_seq = test_sequence[0]
    #     v_dvl_seq = test_sequence[1]
    #     eul_body_dvl_gt_seq = test_sequence[2]
    #     dataset_len = len(v_imu_seq)
    #
    #     rmse_svd_degrees_per_num_samples_list = []
    #
    #     for num_samples in range(start_num_of_samples, dataset_len, num_of_samples_slot):  # should Start from 2
    #         v_imu_sampled = v_imu_seq[0:num_samples, :]
    #         v_dvl_sampled = v_dvl_seq[0:num_samples, :]
    #         euler_body_dvl_gt = eul_body_dvl_gt_seq[0, :]
    #
    #         euler_angles_svd_rads = run_svd_solution_for_wahba_problem(v_imu_sampled.T, v_dvl_sampled.T)
    #         euler_angles_svd_degrees = np.degrees(euler_angles_svd_rads)
    #
    #         squared_error_svd_baseline = rmse_angle(np.array(euler_angles_svd_degrees),
    #                                                                 euler_body_dvl_gt)
    #
    #         rmse_angles_svd = np.sqrt(squared_error_svd_baseline)
    #
    #         rmse_svd_degrees_per_num_samples_list.append(rmse_angles_svd)
    #
    #         rmse_svd_across_all_samples = np.sqrt(np.mean(rmse_svd_degrees_per_num_samples_list))
    #
    #         # Store in dictionary
    #         if num_samples not in rmse_svd_all_tests_by_num_of_samples_dict:
    #             rmse_svd_all_tests_by_num_of_samples_dict[num_samples] = []
    #
    #         rmse_svd_all_tests_by_num_of_samples_dict[num_samples].append(rmse_svd_across_all_samples)
    #
    # mean_rmse_svd_degrees_per_num_samples_list = []
    #
    # for num_samples in range(start_num_of_samples, dataset_len, num_of_samples_slot):
    #     mean_euler_angles_svd_per_num_samples = np.mean(rmse_svd_all_tests_by_num_of_samples_dict[num_samples])
    #     mean_rmse_svd_degrees_per_num_samples_list.append(mean_euler_angles_svd_per_num_samples)
    #
    # svd_time_list = list(range(0, 323))
    #
    # return mean_rmse_svd_degrees_per_num_samples_list, svd_time_list




def evaluate_model(model, test_loader, device):
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    mse_angle_list = []

    # Lists to store per-dataset metrics
    dataset_rmse_lists = []
    current_dataset_predictions = []
    current_dataset_targets = []
    samples_per_dataset = len(next(iter(test_loader))[0])  # Get batch size
    batch_count = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            # Store predictions and targets
            current_dataset_predictions.extend(outputs.cpu().numpy())
            current_dataset_targets.extend(targets.cpu().numpy())
            batch_count += 1

            # Calculate per-batch metrics
            for ii in range(len(targets)):
                mse_angle = calc_squared_err_angles(targets[ii].cpu().numpy(), outputs[ii].cpu().numpy())

                mse_angle_list.append(mse_angle)

                #         rmse_angles_svd = np.sqrt(squared_error_svd_baseline)
                #
                #         rmse_svd_degrees_per_num_samples_list.append(rmse_angles_svd)
                #
                #         rmse_svd_across_all_samples = np.sqrt(np.mean(rmse_svd_degrees_per_num_samples_list))




            # # Check if we've completed a dataset
            # if batch_count * samples_per_dataset >= len(test_loader.dataset):
            #     # Calculate dataset-level metrics
            #     dataset_predictions = np.array(current_dataset_predictions)
            #     dataset_targets = np.array(current_dataset_targets)
            #
            #     # Calculate RMSE components for this dataset
            #     rmse_components = np.sqrt(np.mean((dataset_predictions - dataset_targets) ** 2, axis=0))
            #     dataset_rmse_lists.append(rmse_components)
            #
            #     # Reset for next dataset
            #     current_dataset_predictions = []
            #     current_dataset_targets = []
            #     batch_count = 0

            # # Calculate loss
            # criterion = nn.CosineSimilarity()
            # loss = torch.mean(torch.abs(criterion(targets, outputs)))
            # loss = 1 - loss
            # total_loss += loss.item()

            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    # Calculate overall metrics
    # avg_loss = total_loss / len(test_loader)
    all_predictions = np.concatenate(all_predictions)
    all_targets = np.concatenate(all_targets)

    # Calculate overall RMSE components
    rmse_components = np.sqrt(np.mean((all_predictions - all_targets) ** 2, axis=0))

    # Calculate total RMSE
    total_rmse = np.sqrt(np.mean((all_predictions - all_targets) ** 2))

    # Calculate mean RMSE across all samples
    rmse = np.sqrt(np.mean(mse_angle_list))

    # # Calculate mean RMSE components across datasets
    # if dataset_rmse_lists:
    #     mean_dataset_rmse = np.mean(dataset_rmse_lists, axis=0)
    # else:
    #     mean_dataset_rmse = rmse_components

    return rmse


class IMUDVLWindowedDataset(Dataset):
    def __init__(self, series, window_size):
        self.imu_series = torch.FloatTensor(series[0])
        self.dvl_series = torch.FloatTensor(series[1])

        # Convert euler angles to degrees if they're in radians
        # Assuming the input is in degrees (based on your data),
        # but we'll normalize to the range [-180, 180]
        euler_angles = torch.FloatTensor(series[2])

        # Normalize angles to [-180, 180] range
        euler_angles = ((euler_angles + 180) % 360) - 180

        self.euler_body_dvl_series = euler_angles
        self.window_size = window_size

    def __len__(self):
        return len(self.imu_series) - self.window_size

    def __getitem__(self, idx):
        imu_window = self.imu_series[idx:idx + self.window_size]
        dvl_window = self.dvl_series[idx:idx + self.window_size]
        euler_body_dvl_window = self.euler_body_dvl_series[idx:idx + self.window_size]

        # Combine IMU and DVL data
        input_data = torch.cat((imu_window, dvl_window), dim=1)

        # Return features (IMU and DVL data for the window) and target (Euler angles in degrees)
        return input_data, euler_body_dvl_window[0]


def windowed_dataset(series_list, window_size, batch_size, shuffle):
    """Generates dataset windows from multiple time series

    Args:
      series_list (list of arrays of float) - list of time series
      window_size (int) - the number of time steps to include in the feature
      batch_size (int) - the batch size
      shuffle (bool) - whether to shuffle the dataset

    Returns:
      dataloader (torch.utils.data.DataLoader) - DataLoader containing time windows from all series
    """

    datasets = [IMUDVLWindowedDataset(series, window_size) for series in series_list]
    if len(datasets) > 1:
        combined_dataset = ConcatDataset(datasets)
    else:
        combined_dataset = datasets[0]

    dataloader = DataLoader(
        combined_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True
    )

    return dataloader


def save_model(model, test_type, window_size, base_path="models"):
    """
    Save the trained model to a file with window size in the filename.

    Args:
        model (torch.nn.Module): The trained model to save
        window_size (int): The window size used for training
        base_path (str): Base directory to save models
    """
    import os
    # # Create models directory if it doesn't exist
    # os.makedirs(base_path, exist_ok=True)

    # Create filename with window size
    filepath = os.path.join(base_path, f'imu_dvl_model_{test_type}_window_{window_size}.pth')

    # Save the model
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to {filepath}")
    return filepath


def vector_to_skew(vector):
    return np.array([
        [0, -vector[2], vector[1]],
        [vector[2], 0, -vector[0]],
        [-vector[1], vector[0], 0]
    ])


def skew_symetric(vectors):
    """
    Convert multiple 3D vectors to skew-symmetric matrices.

    Args:
        vectors: array-like, shape (3, N)

    Returns:
        Array of skew-symmetric matrices, shape (N, 3, 3)
    """
    if vectors.shape[0] != 3:
        raise ValueError("Input array must have shape (3, N)")

    N = vectors.shape[1]
    result = np.zeros((3, 3, N))

    for i in range(N):
        result[:, :, i] = vector_to_skew(vectors[:, i])

    return result


def run_acc_gradient_descent(v_imu, v_dvl, omega_skew_imu, a_imu, max_iterations=1000, learning_rate=0.0001,
                             tolerance=1e-6):
    """
    Implement acceleration-based gradient descent optimization to find optimal rotation matrix.

    Args:
        v_imu: IMU velocity data (3xN array)
        v_dvl: DVL velocity data (3xN array)
        omega_skew_imu: Skew-symmetric matrices of angular velocities (3x3xN array)
        a_imu: IMU acceleration data (3xN array)
        max_iterations: Maximum number of iterations for gradient descent
        learning_rate: Learning rate for gradient descent
        tolerance: Convergence tolerance

    Returns:
        R: Optimal rotation matrix
        euler_angles_deg: Euler angles in degrees [roll, pitch, yaw]
    """

    def skew_to_vec(S):
        """Convert 3x3 skew symmetric matrix to 3x1 vector"""
        return np.array([S[2, 1], S[0, 2], S[1, 0]])

    def rotation_matrix_to_euler_xyz(R):
        """
        Convert rotation matrix to XYZ (roll, pitch, yaw) Euler angles

        Args:
            R: 3x3 rotation matrix

        Returns:
            np.array: [roll, pitch, yaw] in radians
        """
        pitch = np.arcsin(-R[0, 2])

        if np.cos(pitch) > 1e-6:
            roll = np.arctan2(R[1, 2], R[2, 2])
            yaw = np.arctan2(R[0, 1], R[0, 0])
        else:
            # Gimbal lock case
            roll = 0
            yaw = np.arctan2(-R[1, 0], R[1, 1])

        return np.array([roll, pitch, yaw])

    def euler_xyz_to_rotation_matrix(angles):
        """
        Convert XYZ (roll, pitch, yaw) Euler angles to rotation matrix

        Args:
            angles: [roll, pitch, yaw] in radians

        Returns:
            R: 3x3 rotation matrix
        """
        roll, pitch, yaw = angles

        Rx = np.array([[1, 0, 0],
                       [0, np.cos(roll), -np.sin(roll)],
                       [0, np.sin(roll), np.cos(roll)]])

        Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                       [0, 1, 0],
                       [-np.sin(pitch), 0, np.cos(pitch)]])

        Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                       [np.sin(yaw), np.cos(yaw), 0],
                       [0, 0, 1]])

        return Rx @ Ry @ Rz

    def compute_error(R):
        """Compute sum of squared errors"""
        total_error = 0
        for i in range(v_imu.shape[1]):
            # Predicted acceleration using current rotation estimate
            a_pred = omega_skew_imu[:, :, i] @ R @ v_dvl[:, i] + R @ np.gradient(v_dvl[:, i])

            # Error between predicted and measured acceleration
            error = a_imu[:, i] - a_pred
            total_error += np.sum(error ** 2)

        return total_error

    def compute_gradient(R):
        """Compute gradient of error with respect to rotation matrix"""
        gradient = np.zeros((3, 3))

        for i in range(v_imu.shape[1]):
            omega_skew = omega_skew_imu[:, :, i]
            v_dvl_i = v_dvl[:, i]
            v_dvl_dot = np.gradient(v_dvl_i)

            # Predicted acceleration
            a_pred = omega_skew @ R @ v_dvl_i + R @ v_dvl_dot

            # Error
            error = a_imu[:, i] - a_pred

            # Gradient contribution from this sample
            gradient -= 2 * (np.outer(error, v_dvl_i) @ omega_skew.T +
                             np.outer(error, v_dvl_dot))

        return gradient

    # Initialize with identity rotation
    euler_angles_rads = np.zeros(3)
    R = euler_xyz_to_rotation_matrix(euler_angles_rads)

    prev_error = float('inf')

    for iteration in range(max_iterations):
        # Compute current error
        current_error = compute_error(R)

        # Check convergence
        if abs(current_error - prev_error) < tolerance:
            break

        # Compute gradient
        gradient = compute_gradient(R)

        # Update rotation matrix using gradient descent on SO(3)
        # Project gradient onto tangent space of SO(3)
        A = gradient @ R.T - R @ gradient.T
        omega = skew_to_vec(A)

        # Update Euler angles
        euler_angles_rads -= learning_rate * omega

        # Convert back to rotation matrix ensuring SO(3) constraint
        R = euler_xyz_to_rotation_matrix(euler_angles_rads)

        prev_error = current_error

    # Convert final result to degrees
    euler_angles_deg = np.degrees(euler_angles_rads)

    return euler_angles_deg


def calc_squared_err_angles(a, b):
    squared_diff = np.zeros(3)

    for i in range(3):
        # Normalize both angles to [-180, 180]
        a_norm = ((a[i] + 180) % 360) - 180
        b_norm = ((b[i] + 180) % 360) - 180

        # Calculate the absolute difference
        diff = abs(a_norm - b_norm)

        # Handle wrap-around case
        if diff > 180:
            diff = 360 - diff

        squared_diff[i] = diff ** 2

    # mean_squared_err = np.mean(squared_diff)
    squared_err = squared_diff

    # rmse = np.sqrt(mean_squared_err)

    return squared_err


def squared_angle_difference(a, b):
    return (min((a - b) % 360, (b - a) % 360)) ** 2


def euler_angles_to_rotation_matrix(roll_rads, pitch_rads, yaw_rads):
    # Convert Euler angles (roll, pitch, yaw) to a rotation matrix.
    # ZYX convention: Rotate about z-axis, then y-axis, then x-axis.

    # Compute sines and cosines
    c_roll = np.cos(roll_rads)
    s_roll = np.sin(roll_rads)
    c_pitch = np.cos(pitch_rads)
    s_pitch = np.sin(pitch_rads)
    c_yaw = np.cos(yaw_rads)
    s_yaw = np.sin(yaw_rads)

    # Compute rotation matrix
    Rz = np.array([[c_yaw, s_yaw, 0],
                   [-s_yaw, c_yaw, 0],
                   [0, 0, 1]])

    Ry = np.array([[c_pitch, 0, -s_pitch],
                   [0, 1, 0],
                   [s_pitch, 0, c_pitch]])

    Rx = np.array([[1, 0, 0],
                   [0, c_roll, s_roll],
                   [0, -s_roll, c_roll]])

    # Combine rotation matrices
    R = np.dot(Rx, np.dot(Ry, Rz))

    return R

def convert_deg_to_rads(roll_deg, pitch_deg, yaw_deg):
    roll_rads = np.radians(roll_deg)
    pitch_rads = np.radians(pitch_deg)
    yaw_rads = np.radians(yaw_deg)
    return roll_rads, pitch_rads, yaw_rads


def run_svd_solution_for_wahba_problem(velocity_ins_sampled, velocity_dvl_sampled):
    """
    Solve the Wahba problem using Singular Value Decomposition (SVD).

    Parameters:
        v_imu_sampled: Sampled INS velocity vector.
        v_dvl_sampled: Sampled DVL velocity vector.

    Returns:
        rotation_matrix_wahba: Optimal rotation matrix.
    """

    # Compute the cross-correlation matrix
    w = np.dot(velocity_ins_sampled, velocity_dvl_sampled.T)

    # Perform Singular Value Decomposition (SVD)
    U, S, Vt = np.linalg.svd(w)

    # Compute the optimal rotation matrix
    rotation_matrix_wahba = np.dot(Vt.T, U.T)

    curr_euler_angles_rads = rotation_matrix_to_euler_zyx(rotation_matrix_wahba)

    # Convert angles to degrees
    return curr_euler_angles_rads


def rotation_matrix_to_euler_zyx(R):
    """
    Extract Euler angles from a rotation matrix using ZYX convention.

    Args:
        R (numpy.ndarray): 3x3 rotation matrix.

    Returns:
        numpy.ndarray: Euler angles [yaw, pitch, roll] in radians.
    """

    # R = R.T

    # Extracting individual elements from the rotation matrix
    r11, r12, r13 = R[0, 0], R[0, 1], R[0, 2]
    r23, r33 = R[1, 2], R[2, 2]

    # Computing roll (around X axis)
    roll = np.arctan2(r23, r33)
    roll = np.arctan2(r23, r33)

    # Computing pitch (around Y axis)
    pitch = np.arcsin(-r13)

    # Computing yaw (around Z axis)
    yaw = np.arctan2(r12, r11)

    return np.array([roll, pitch, yaw])


# # Acceleration-based Method: Run gradient descent solution
# euler_angles_gd_degrees = run_acc_gradient_descent(v_imu_sampled, v_dvl_sampled, omega_skew_imu_sampled,
#                                                    a_imu_sampled)
# euler_angles_gd_degrees_list.append(euler_angles_gd_degrees)

# squared_error_gd_baseline = squared_angular_difference(np.array(euler_angles_gd_degrees),euler_body_dvl_gt)

# squared_error_gd_baseline_list.append(squared_error_gd_baseline)


def calc_mean_rmse_svd_degrees_per_num_samples(v_imu_dvl_test_series_list, sample_freq, config):

    dataset_len = len(v_imu_dvl_test_series_list[0][0])
    window_sizes_sec = config['window_sizes_sec']

    # fix graph parameters for transformed_real_data
    # start_num_of_samples = sample_freq*10
    # end_num_of_samples = sample_freq*200
    # #end_num_of_samples = dataset_len
    # num_of_samples_slot = sample_freq*5


    start_num_of_samples = sample_freq*1
    #end_num_of_samples = sample_freq*230
    end_num_of_samples = dataset_len
    # end_num_of_samples = dataset_len
    num_of_samples_slot = sample_freq*1
    rmse_svd_all_tests_by_num_of_samples_dict = {}
    rmse_acc_gd_all_tests_by_num_of_samples_dict = {}

    for test_idx, test_sequence in enumerate(v_imu_dvl_test_series_list):
        # print(f'Calc baseline for test sequence {test_idx} ')
        v_imu_seq = test_sequence[0]
        v_dvl_seq = test_sequence[1]
        eul_body_dvl_gt_seq = test_sequence[2]
        omega_ned_to_body_rad = test_sequence[3]
        est_acc_eb_b = test_sequence[4]

        for num_samples in range(start_num_of_samples, dataset_len, num_of_samples_slot):
            v_imu_sampled = v_imu_seq[0:num_samples, :]
            v_dvl_sampled = v_dvl_seq[0:num_samples, :]
            euler_body_dvl_gt = eul_body_dvl_gt_seq[0, :]
            #omega_ned_to_body_rad_sampled = omega_ned_to_body_rad[0:num_samples, :]
            #omega_skew_ned_to_body_rad_sampled = skew_symetric(omega_ned_to_body_rad_sampled.T)
            #est_acc_eb_b_sampled = est_acc_eb_b[0:num_samples, :]

            euler_angles_svd_rads = run_svd_solution_for_wahba_problem(v_imu_sampled.T, v_dvl_sampled.T)
            euler_angles_svd_degrees = np.degrees(euler_angles_svd_rads)
            svd_squared_err_angles = calc_squared_err_angles(np.array(euler_angles_svd_degrees), euler_body_dvl_gt)

            # euler_angles_acc_gd_rads = run_acc_gradient_descent(v_imu_sampled.T, v_dvl_sampled.T, omega_skew_ned_to_body_rad_sampled, est_acc_eb_b_sampled.T)
            # euler_angles_acc_gd_degrees = np.degrees(euler_angles_acc_gd_rads)
            # acc_gd_squared_err_angles = calc_squared_err_angles(np.array(euler_angles_acc_gd_degrees), euler_body_dvl_gt)

            # Store in dictionary
            if num_samples not in rmse_svd_all_tests_by_num_of_samples_dict:
                rmse_svd_all_tests_by_num_of_samples_dict[num_samples] = []

            # # Store in dictionary
            # if num_samples not in rmse_acc_gd_all_tests_by_num_of_samples_dict:
            #     rmse_acc_gd_all_tests_by_num_of_samples_dict[num_samples] = []



            rmse_svd_all_tests_by_num_of_samples_dict[num_samples].append(svd_squared_err_angles)
            # rmse_acc_gd_all_tests_by_num_of_samples_dict[num_samples].append(acc_gd_squared_err_angles)

    mean_rmse_svd_degrees_per_num_samples_list = []
    mean_rmse_acc_gd_degrees_per_num_samples_list = []

    for num_samples in range(start_num_of_samples, end_num_of_samples, num_of_samples_slot):
        mean_euler_angles_svd_per_num_samples = np.mean(rmse_svd_all_tests_by_num_of_samples_dict[num_samples])
        mean_rmse_svd_degrees_per_num_samples_list.append(np.sqrt(mean_euler_angles_svd_per_num_samples))

        # mean_euler_angles_acc_gd_per_num_samples = np.mean(rmse_acc_gd_all_tests_by_num_of_samples_dict[num_samples])
        # mean_rmse_acc_gd_degrees_per_num_samples_list.append(np.sqrt(mean_euler_angles_acc_gd_per_num_samples))

    #svd_time_list = list(range(0, dataset_len-start_num_of_samples))
    svd_time_list = list(range(start_num_of_samples//sample_freq, (end_num_of_samples)//sample_freq, num_of_samples_slot//sample_freq))

    for window in window_sizes_sec:
        print(f'window {window} is:{mean_rmse_svd_degrees_per_num_samples_list[window // (num_of_samples_slot//sample_freq)]}')

    return mean_rmse_svd_degrees_per_num_samples_list, svd_time_list


def plot_results_graph_rmse_net_and_rmse_svd(svd_time_list, mean_rmse_svd_degrees_per_num_samples_list,
                                             current_time_test_list, rmse_test_list):
    # Create a single figure for total RMSE plot
    plt.figure(figsize=(12, 8))

    # Increase font size globally
    plt.rcParams.update({'font.size': 14})

    # Set colors
    baseline_color = 'blue'
    test_color = 'red'

    # Plot total RMSE for baseline and test
    plt.plot(svd_time_list, mean_rmse_svd_degrees_per_num_samples_list, color=baseline_color, linestyle='-',
             label='Baseline', linewidth=2)
    plt.scatter(current_time_test_list, rmse_test_list, color=test_color, marker='o', label='AlignNet', s=100)

    # Create tick marks that align with both datasets
    # First, ensure we have the key points for small intervals
    small_intervals = np.array([5, 25, 50, 75])

    # Then create regularly spaced large intervals
    max_time = max(max(svd_time_list), max(current_time_test_list))
    large_intervals = np.arange(0, max_time, 50)  # 0 to max_time in steps of 50

    # Combine and ensure all ticks are unique and sorted
    all_ticks = np.unique(np.concatenate([small_intervals, large_intervals]))

    # Set x-axis ticks with larger font
    plt.xticks(all_ticks, fontsize=22)

    # Increase y-axis tick font size
    plt.yticks(fontsize=22)

    # Add labels and title with larger font
    plt.xlabel('Time [sec]', fontsize=30)
    plt.ylabel('Alignment RMSE [deg]', fontsize=30)

    # Add primary grid - make sure grid aligns with actual ticks
    plt.grid(True, which='major', linestyle='-', alpha=0.5)

    # Add special grid lines for important intervals
    for x in small_intervals:
        plt.axvline(x=x, color='gray', linestyle=':', alpha=0.5)

    # Add legend with larger font
    plt.legend(fontsize=24, loc='upper right')

    # Adjust layout
    plt.tight_layout()

    # Show plot
    plt.show(block=False)


def plot_all_baseline_results(timeline_lists, rmse_lists, labels=None, current_time_test_list=None,
                              rmse_test_list=None):
    """
    Plot multiple baseline results on the same plot with the same style as plot_results_graph_rmse_net_and_rmse_svd.

    Args:
        timeline_lists: List of lists containing timeline data for each baseline
        rmse_lists: List of lists containing RMSE data for each baseline
        labels: Optional list of labels for each baseline
        current_time_test_list: Optional list of window sizes for test data
        rmse_test_list: Optional list of RMSE values for test data
    """
    # Create a single figure
    plt.figure(figsize=(12, 8))

    # Increase font size globally
    plt.rcParams.update({'font.size': 14})

    # Colors for different baselines - using the same color scheme as the original function
    colors = ['blue', 'green', 'red', 'brown', 'orange', 'pink', 'gray', 'cyan']

    # Plot each baseline - first one as continuous line, others as scatter points
    for i, (timeline, rmse) in enumerate(zip(timeline_lists, rmse_lists)):
        color = colors[i % len(colors)]
        label = labels[i] if labels is not None and i < len(labels) else f'Baseline {i + 1}'

        if i == 0:
            # First baseline as continuous line (like original baseline)
            plt.plot(timeline, rmse, color=color, linestyle='-', label=label, linewidth=2)
        else:
            # Other baselines as scatter points (like AlignNet style)
            plt.scatter(timeline, rmse, color=color, marker='o', label=label, s=100)

    # Plot test data if provided (scatter points like AlignNet in original function)
    if current_time_test_list is not None and rmse_test_list is not None:
        plt.scatter(current_time_test_list, rmse_test_list, color='red', marker='o', label='AlignNet', s=100)

    # Create tick marks for x-axis - same logic as original function
    max_time = max(max(t) for t in timeline_lists)
    if current_time_test_list:
        max_time = max(max_time, max(current_time_test_list))

    # Small intervals for important points
    small_intervals = np.array([5, 25, 50, 75])
    # Large intervals for regular spacing
    large_intervals = np.arange(0, max_time + 50, 50)  # 0 to max_time in steps of 50

    # Combine and ensure all ticks are unique and sorted
    all_ticks = np.unique(np.concatenate([small_intervals, large_intervals]))
    all_ticks = all_ticks[all_ticks <= max_time]

    # Set x-axis ticks with larger font
    plt.xticks(all_ticks, fontsize=22)

    # Increase y-axis tick font size
    plt.yticks(fontsize=22)

    # Add labels and title with larger font
    plt.xlabel('Time [sec]', fontsize=30)
    plt.ylabel('Alignment RMSE [deg]', fontsize=30)

    # Add primary grid - make sure grid aligns with actual ticks
    plt.grid(True, which='major', linestyle='-', alpha=0.5)

    # Add special grid lines for important intervals
    for x in small_intervals:
        plt.axvline(x=x, color='gray', linestyle=':', alpha=0.5)

    # Add legend with larger font
    plt.legend(fontsize=24, loc='upper right')

    # Adjust layout
    plt.tight_layout()

    # Show plot
    plt.show(block=False)

def split_data_properly(data_pd, num_sequences, sequence_length, train_size=0.6, val_size=0.2):
    # Create sequence indices
    sequence_indices = np.arange(num_sequences)

    # First split to separate test set
    train_val_indices, test_indices = train_test_split(
        sequence_indices,
        test_size=0.2,
        shuffle=True,
        random_state=42
    )

    # Then split remaining data into train and validation
    train_indices, val_indices = train_test_split(
        train_val_indices,
        test_size=val_size / (train_size + val_size),
        shuffle=True,
        random_state=42
    )

    # Extract data arrays
    v_imu_body_full = np.array(data_pd.iloc[:, 1:4].T)
    v_dvl_full = np.array(data_pd.iloc[:, 4:7].T)
    euler_body_dvl_full = np.array(data_pd.iloc[:, 7:10].T)
    est_eul_ned_to_body_rad_full = np.array(data_pd.iloc[:, 10:13].T)
    est_acc_eb_b_full = np.array(data_pd.iloc[:, 13:16].T)

    # Create sequence lists
    train_sequences = []
    val_sequences = []
    test_sequences = []

    # Fill training sequences
    for idx in train_indices:
        start_idx = idx * sequence_length
        end_idx = start_idx + sequence_length
        train_sequences.append([
            v_imu_body_full[:, start_idx:end_idx].T,
            v_dvl_full[:, start_idx:end_idx].T,
            euler_body_dvl_full[:, start_idx:end_idx].T,
            est_eul_ned_to_body_rad_full[:, start_idx:end_idx].T,
            est_acc_eb_b_full[:, start_idx:end_idx].T
        ])

    # Fill validation sequences
    for idx in val_indices:
        start_idx = idx * sequence_length
        end_idx = start_idx + sequence_length
        val_sequences.append([
            v_imu_body_full[:, start_idx:end_idx].T,
            v_dvl_full[:, start_idx:end_idx].T,
            euler_body_dvl_full[:, start_idx:end_idx].T,
            est_eul_ned_to_body_rad_full[:, start_idx:end_idx].T,
            est_acc_eb_b_full[:, start_idx:end_idx].T
        ])

    # Fill test sequences
    for idx in test_indices:
        start_idx = idx * sequence_length
        end_idx = start_idx + sequence_length
        test_sequences.append([
            v_imu_body_full[:, start_idx:end_idx].T,
            v_dvl_full[:, start_idx:end_idx].T,
            euler_body_dvl_full[:, start_idx:end_idx].T,
            est_eul_ned_to_body_rad_full[:, start_idx:end_idx].T,
            est_acc_eb_b_full[:, start_idx:end_idx].T
        ])

    return train_sequences, val_sequences, test_sequences, test_indices



def main(config):
    # Example usage

    # Main variables to run the simulation
    roll_gt_deg = config['roll_gt_deg']
    pitch_gt_deg = config['pitch_gt_deg']
    yaw_gt_deg = config['yaw_gt_deg']

    sample_freq = None
    single_dataset_duration_sec = None # should be same parameter value like in matlab simulation
    single_dataset_len = None # should be same parameter value like in matlab simulation

    if(config['test_type'] == 'simulated_data'):
        single_dataset_len = config['simulated_dataset_len']
        # single_dataset_duration_sec = config['simulated_dataset_duration_sec']
        sample_freq = 5
    elif((config['test_type'] == 'transformed_real_data') or (config['test_type'] == 'simulated_imu_from_real_gt_data')):
        single_dataset_len = config['real_dataset_len']
        # single_dataset_duration_sec = config['real_dataset_duration_sec']
        sample_freq = 1

    data_path = config['data_path']

    if(config['test_type'] == 'simulated_data'):
        file_name = config['simulated_data_file_name']
        data_pd = pd.read_csv(os.path.join(data_path, f'{file_name}'), header=None)

        # use for very large datasets
        #ddf = dd.read_csv(os.path.join(data_path, f'{file_name}'), header=None)


    elif(config['test_type'] == 'transformed_real_data'):
        file_name = config['transformed_real_data_file_name']
        data_pd = pd.read_csv(os.path.join(data_path, f'{file_name}'), header=None)
        simulated_imu_from_real_gt_data_file_name = config['simulated_imu_from_real_gt_data_file_name']
        simulated_imu_from_real_gt_data_pd = pd.read_csv(os.path.join(data_path, f'{simulated_imu_from_real_gt_data_file_name}'), header=None)
    elif (config['test_type'] == 'simulated_imu_from_real_gt_data'):
        file_name = config['simulated_imu_from_real_gt_data_file_name']
        data_pd = pd.read_csv(os.path.join(data_path, f'{file_name}'), header=None)


    real_data_file_name = config['real_data_file_name']
    trained_model_base_path = config['trained_model_path']
    window_sizes_sec = config['window_sizes_sec']
    batch_size = config['batch_size']


    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert degrees to radians
    roll_gt_rads, pitch_gt_rads, yaw_gt_rads = convert_deg_to_rads(roll_gt_deg, pitch_gt_deg, yaw_gt_deg)

    # Convert Euler angles to rotation matrix
    rotation_matrix_ins_to_dvl = euler_angles_to_rotation_matrix(roll_gt_rads, pitch_gt_rads, yaw_gt_rads)

    # Read data from the .csv file


    ##prepare real data dataset - for sim2real model
    real_data_pd = pd.read_csv(os.path.join(data_path, f'{real_data_file_name}'), header=None)
    v_imu_body_real_data_full = np.array(real_data_pd.iloc[:, 1:4].T)
    v_dvl_real_data_full = np.array(real_data_pd.iloc[:, 4:7].T)
    euler_body_dvl_real_data_full = np.array(real_data_pd.iloc[:, 7:10].T)
    real_dataset_len = len(v_imu_body_real_data_full[1])

    current_time_test_list = []
    rmse_test_list = []
    v_imu_dvl_test_real_data_list = []

    # Calculate number of sequences
    num_of_simulated_datasets = len(data_pd) // single_dataset_len

    train_sequences, val_sequences, test_sequences, test_indices = split_data_properly(
        data_pd=data_pd,
        num_sequences=num_of_simulated_datasets,
        sequence_length=single_dataset_len,
        train_size=0.6,
        val_size=0.2
    )

    # Replace your existing lists with the new split sequences
    v_imu_dvl_train_series_list = train_sequences
    v_imu_dvl_valid_series_list = val_sequences
    v_imu_dvl_test_series_list = test_sequences

    v_imu_dvl_test_real_data_list.append(
        [v_imu_body_real_data_full.T, v_dvl_real_data_full.T, euler_body_dvl_real_data_full.T])

    # train model ##################################################################
    if config['train_model']:
        for i in window_sizes_sec:
            print(f'train with window of {i} seconds')
            # Parameters
            num_of_samples_window_size = i * sample_freq

            train_loader = windowed_dataset(
                series_list=v_imu_dvl_train_series_list,
                window_size=num_of_samples_window_size,
                batch_size=batch_size,
                shuffle=True
            )

            val_loader = windowed_dataset(
                series_list=v_imu_dvl_valid_series_list,
                window_size=num_of_samples_window_size,
                batch_size=batch_size,
                shuffle=True
            )

            model = Resnet1chDnet()

            # # Loss function
            # criterion = EulerAnglesLoss()

            # Optimizer with weight decay (L2 regularization)

            optimizer = optim.Adam(model.parameters(), lr=0.0000001, weight_decay=0.01)

            #
            # Learning rate scheduler
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=0,
                verbose=True
            )

            # Train the model with the improved training function
            train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                num_epochs=25,
                device=device,
                scheduler=scheduler
                # l2_lambda=0.01
            )

            # Save the model and store its path
            model_path = save_model(model, config['test_type'], i, trained_model_base_path)

    # Test Model section #################
    if config['test_model']:
        for window_size in window_sizes_sec:
            print(f'\nEvaluating model with window size {window_size}')

            # Construct the specific model path for this window size
            test_type = config['test_type']
            #model_path = os.path.join(trained_model_base_path, f'imu_dvl_model_{test_type}_window_{window_size}.pth')
            #model_path = os.path.join(trained_model_base_path, f'imu_dvl_model_simulated_data_window_{window_size}.pth')
            # model_path = os.path.join(trained_model_base_path, f'imu_dvl_model_simulated_data_straight_descent_22_+2_ba_real_bg_0_1_window_{window_size}.pth')
            # model_path = os.path.join(trained_model_base_path, f'imu_dvl_model_simulated_data_straight_descent_22_+0_227_ba100_bg_0_1_window_{window_size}.pth')
            model_path = os.path.join(trained_model_base_path, f'imu_dvl_model_transformed_real_data_traj12_22_+2_window_{window_size}.pth')
            # Load model
            # model = SimplerIMUResNet(dropout_rate=0.3)
            model = Resnet1chDnet()

            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()

            if (config['test_type'] == 'simulated_data') or (config['test_type'] == 'transformed_real_data') or (config['test_type'] == 'simulated_imu_from_real_gt_data'):
                # Create test loader for this window size
                test_loader = windowed_dataset(
                    series_list=v_imu_dvl_test_series_list,
                    window_size=window_size,
                    batch_size=batch_size,
                    shuffle=True
                )

            elif config['test_type'] == 'real_data':
                # Create test loader for this window size
                test_loader = windowed_dataset(
                    series_list=v_imu_dvl_test_real_data_list,
                    window_size=window_size,
                    batch_size=batch_size,
                    shuffle=True
                )

            # Evaluate model
            rmse = evaluate_model(
                model, test_loader, device
            )

            rmse_test_list.append(rmse)
            current_time = window_size
            current_time_test_list.append(current_time)
            print(f"Test RMSE: {rmse:.4f}")


# Test Baseline Model section #################
    if config['test_baseline_model']:

        mean_rmse_svd_degrees_per_num_samples_list, svd_time_list = calc_mean_rmse_svd_degrees_per_num_samples(v_imu_dvl_test_series_list, sample_freq, config)
        # save_baseline_results_numpy(mean_rmse_svd_degrees_per_num_samples_list, svd_time_list,rmse_test_list, current_time_test_list, config)
        plot_results_graph_rmse_net_and_rmse_svd(svd_time_list, mean_rmse_svd_degrees_per_num_samples_list, current_time_test_list, rmse_test_list)

        # loaded_rmse_baseline_results_list, loaded_timeline_baseline_results_list = load_baseline_results_numpy(config)
        # print("Baseline RMSE")
        # # Create labels for the baselines (optional)
        # baseline_labels = [
        #     f"Baseline", f"AligNet - sim2real", f"AligNet"
        # ]
        #
        # # Plot all baselines on the same plot
        # plot_all_baseline_results(
        #     loaded_timeline_baseline_results_list,
        #     loaded_rmse_baseline_results_list,
        #     labels=baseline_labels,
        #     current_time_test_list=current_time_test_list if config['test_model'] else None,
        #     rmse_test_list=rmse_test_list if config['test_model'] else None
        # )


if __name__ == '__main__':
    # Default configuration
    default_config = {
        'roll_gt_deg': 45,
        'pitch_gt_deg': 10,
        'yaw_gt_deg': 120,
    }

    # User-defined configuration (can be read from a config file or command-line arguments)
    user_config = {
        'roll_gt_deg': -179.9,
        'pitch_gt_deg': 0.2,
        'yaw_gt_deg': -44.3,
        # 'window_sizes_sec': [5, 25, 50, 75, 100, 125, 150],
        # 'window_sizes_sec': [75],
        #'window_sizes_sec': [5,25,50,75,100,125,150,175,200],
        # 'window_sizes_sec': [5,25],
        'window_sizes_sec': [5,25,50,75,100],
        # 'window_sizes_sec': [175],
        'batch_size': 32,
        #'simulated_dataset_len': 1612, # important!! you have to update it, from the data output file, every time you change dataset
        #'simulated_dataset_len': 2311, # important!! you have to update it, from the data output file, every time you change dataset
        # 'simulated_dataset_len': 2076, # - straight descent - important!! you have to update it, from the data output file, every time you change dataset
        'simulated_dataset_len': 3234, # - lawn_mower_1 - important!! you have to update it, from the data output file, every time you change dataset - should be length of single trajectory
        'real_dataset_len': 400,
        'data_path': "C:\\Users\\damar\\MATLAB\\Projects\\modeling-and-simulation-of-an-AUV-in-Simulink-master\\Work",
        'test_type': 'transformed_real_data',  # Set to "real_data" or "simulated_data" or "transformed_real_data" or "simulated_imu_from_real_gt_data"
        'train_model': False,
        'test_model': True,
        'test_baseline_model': True,
        'trained_model_path': "C:\\Users\\damar\\MATLAB\\Projects\\modeling-and-simulation-of-an-AUV-in-Simulink-master\\Work\\trained_model",
        #'simulated_data_file_name': 'simulated_data_output.csv',
        # 'simulated_data_file_name': 'simulated_data_output_straight_descent_22_+2_ba_real_bg_0_01.csv',
        'simulated_data_file_name': 'simulated_data_output.csv',
        'real_data_file_name': 'real_data_output.csv',
        # 'transformed_real_data_file_name': 'transformed_real_data_output.csv',
        # 'transformed_real_data_file_name': 'transformed_real_data_output_traj12_22_+2.csv',
        'transformed_real_data_file_name': 'transformed_real_data_output.csv',
        'simulated_imu_from_real_gt_data_file_name': 'simulated_imu_from_real_gt_data_output.csv',
        'saved_baseline_results_file_name': 'baseline_results_transformed_real_data_output_traj12_22_+2',
        'loaded_baseline_results_file_names_list': ['baseline_results_transformed_real_data_output_traj12_22_+2','sim2real_traj12_22_+2', 'real_traj12_22_+2'],
        'real_imu_file_name': 'IMU_trajectory.csv',
        'real_dvl_file_name': 'DVL_trajectory.csv',
        'real_gt_file_name': 'GT_trajectory.csv'
    }

    # orzi_euler_config = {
    #     'roll_gt_deg': -179.9,
    #     'pitch_gt_deg': 0.2,
    #     'yaw_gt_deg': -44.3,
    # }

    # Merge default and user configurations
    config = {**default_config, **user_config}

    main(config)

    plt.show()


