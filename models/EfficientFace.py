"""
EfficientFace: An Efficient Deep Neural Network for Facial Emotion Recognition (FER)

Reference:
- EfficientFace: An Efficient Deep Neural Network for Facial Emotion Recognition
  https://github.com/kaiegg/EfficientFace

Key Architectural Components:
1. Inverted Residual Blocks (ShuffleNetV2-style):
   - Employs depthwise separable convolutions to drastically reduce parameter count and FLOPs.
   - Uses channel split and channel shuffle operations to encourage cross-channel information exchange.
2. Local Feature Extractor:
   - Partitions intermediate feature maps into 4 spatial quadrants (28x28 patches).
   - Extracts localized facial cues (e.g., eyes, eyebrows, nose, mouth/chin).
3. Modulator (Attention):
   - Computes channel and spatial attention weights to modulate global facial features.
   - Combines modulated stage2 features with local facial features.
"""

import torch
import torch.nn as nn
from models.modulator import Modulator


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """
    Standard 3x3 convolution with padding.
    Preserves spatial dimensions when stride=1 and dilation=1.
    """
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """
    Pointwise (1x1) convolution.
    Used for channel projection (expansion or reduction) without changing spatial resolution.
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def depthwise_conv(i, o, kernel_size, stride=1, padding=0, bias=False):
    """
    Depthwise Convolution: groups = in_channels.
    Applies a separate spatial filter to each input channel individually.
    Reduces computational cost by a factor of ~out_channels compared to standard conv.
    """
    return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)


def channel_shuffle(x, groups):
    """
    Channel Shuffle operation (from ShuffleNetV2).
    
    When using group convolutions or channel splits, information flow between channel
    groups is blocked. Channel shuffle interleaves channels from different groups:
      Shape transition:
      (N, C, H, W) -> (N, G, C//G, H, W) -> transpose(1, 2) -> (N, C//G, G, H, W) -> (N, C, H, W)
    """
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    # Reshape into (groups, channels_per_group)
    x = x.view(batchsize, groups, channels_per_group, height, width)
    # Transpose group and sub-channel dimensions
    x = torch.transpose(x, 1, 2).contiguous()
    # Flatten back to original channel dimension
    x = x.view(batchsize, -1, height, width)
    return x


class LocalFeatureExtractor(nn.Module):
    """
    Local Feature Extractor:
    Divides the intermediate 56x56 feature map into 4 quadrants (28x28 spatial patches):
      - patch_11: Top-Left (Left eye / eyebrow)
      - patch_12: Top-Right (Right eye / eyebrow)
      - patch_21: Bottom-Left (Left cheek / mouth corner)
      - patch_22: Bottom-Right (Right cheek / mouth corner)
      
    Each quadrant is processed independently through 2 depthwise convolution layers
    with stride 2 on the first layer, reducing each patch from 28x28 to 14x14.
    The resulting quadrants are stitched back together to form a 28x28 feature map.
    """

    def __init__(self, inplanes, planes, index):
        super(LocalFeatureExtractor, self).__init__()
        self.index = index

        norm_layer = nn.BatchNorm2d
        self.relu = nn.ReLU()

        # Branch 1: Top-Left quadrant
        self.conv1_1 = depthwise_conv(inplanes, planes, kernel_size=3, stride=2, padding=1)
        self.bn1_1 = norm_layer(planes)
        self.conv1_2 = depthwise_conv(planes, planes, kernel_size=3, stride=1, padding=1)
        self.bn1_2 = norm_layer(planes)

        # Branch 2: Bottom-Left quadrant
        self.conv2_1 = depthwise_conv(inplanes, planes, kernel_size=3, stride=2, padding=1)
        self.bn2_1 = norm_layer(planes)
        self.conv2_2 = depthwise_conv(planes, planes, kernel_size=3, stride=1, padding=1)
        self.bn2_2 = norm_layer(planes)

        # Branch 3: Top-Right quadrant
        self.conv3_1 = depthwise_conv(inplanes, planes, kernel_size=3, stride=2, padding=1)
        self.bn3_1 = norm_layer(planes)
        self.conv3_2 = depthwise_conv(planes, planes, kernel_size=3, stride=1, padding=1)
        self.bn3_2 = norm_layer(planes)

        # Branch 4: Bottom-Right quadrant
        self.conv4_1 = depthwise_conv(inplanes, planes, kernel_size=3, stride=2, padding=1)
        self.bn4_1 = norm_layer(planes)
        self.conv4_2 = depthwise_conv(planes, planes, kernel_size=3, stride=1, padding=1)
        self.bn4_2 = norm_layer(planes)

        ##################
        # TODO: Student Contribution - Local Feature Extraction
        # Ideas for experimentation:
        # 1. Modify patch partitioning (e.g., 3x3 grid instead of 2x2, or overlapping patches).
        # 2. Add spatial landmark attention to weight patches according to facial keypoints.
        # 3. Experiment with different activation functions (e.g., GELU, SiLU).
        ##################

    def forward(self, x):
        # Partition 56x56 feature map into 4 quadrants of size 28x28
        patch_11 = x[:, :, 0:28, 0:28]    # Top-Left
        patch_21 = x[:, :, 28:56, 0:28]   # Bottom-Left
        patch_12 = x[:, :, 0:28, 28:56]   # Top-Right
        patch_22 = x[:, :, 28:56, 28:56]  # Bottom-Right

        out_1 = self.conv1_1(patch_11)
        out_1 = self.bn1_1(out_1)
        out_1 = self.relu(out_1)
        out_1 = self.conv1_2(out_1)
        out_1 = self.bn1_2(out_1)
        out_1 = self.relu(out_1)

        out_2 = self.conv2_1(patch_21)
        out_2 = self.bn2_1(out_2)
        out_2 = self.relu(out_2)
        out_2 = self.conv2_2(out_2)
        out_2 = self.bn2_2(out_2)
        out_2 = self.relu(out_2)

        out_3 = self.conv3_1(patch_12)
        out_3 = self.bn3_1(out_3)
        out_3 = self.relu(out_3)
        out_3 = self.conv3_2(out_3)
        out_3 = self.bn3_2(out_3)
        out_3 = self.relu(out_3)

        out_4 = self.conv4_1(patch_22)
        out_4 = self.bn4_1(out_4)
        out_4 = self.relu(out_4)
        out_4 = self.conv4_2(out_4)
        out_4 = self.bn4_2(out_4)
        out_4 = self.relu(out_4)

        # Concatenate vertical pairs, then horizontal pairs to restore 28x28 grid
        out1 = torch.cat([out_1, out_2], dim=2)  # Height concatenation
        out2 = torch.cat([out_3, out_4], dim=2)  # Height concatenation
        out = torch.cat([out1, out2], dim=3)    # Width concatenation

        return out


class InvertedResidual(nn.Module):
    """
    ShuffleNetV2 Inverted Residual Block.
    
    Two operational modes depending on stride:
    1. stride == 1 (Feature Transformation & Identity Mapping):
       - Input channels split into two halves (c // 2 each).
       - Branch 1: Identity shortcut pass-through.
       - Branch 2: 1x1 conv -> 3x3 depthwise conv -> 1x1 conv.
       - Concat both branches + channel shuffle.
    2. stride > 1 (Downsampling & Channel Expansion):
       - Branch 1: 3x3 depthwise conv (stride 2) -> 1x1 conv.
       - Branch 2: 1x1 conv -> 3x3 depthwise conv (stride 2) -> 1x1 conv.
       - Concat both branches + channel shuffle (doubles output channels).
    """

    def __init__(self, inp, oup, stride):
        super(InvertedResidual, self).__init__()

        if not (1 <= stride <= 3):
            raise ValueError('illegal stride value')
        self.stride = stride

        branch_features = oup // 2
        assert (self.stride != 1) or (inp == branch_features << 1)

        if self.stride > 1:
            # Branch 1 downsampling path
            self.branch1 = nn.Sequential(
                depthwise_conv(inp, inp, kernel_size=3, stride=self.stride, padding=1),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True))

        # Branch 2 feature transformation path
        self.branch2 = nn.Sequential(
            nn.Conv2d(inp if (self.stride > 1) else branch_features,
                      branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            depthwise_conv(branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True))

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle(out, 2)
        return out


class EfficientFace(nn.Module):
    """
    EfficientFace Backbone Network for Facial Emotion Recognition.
    
    Data Flow:
      1. Input image: (B, 3, 224, 224)
      2. conv1 + maxpool: (B, 29, 56, 56)
      3. Stage 2 + Modulator + Local Feature Extractor:
         - stage2(x) produces global features (B, 116, 28, 28)
         - modulator(stage2(x)) applies channel + spatial attention
         - local(x) produces localized patch features (B, 116, 28, 28)
         - Fused: modulator(stage2(x)) + local(x)
      4. Stage 3: (B, 232, 14, 14)
      5. Stage 4: (B, 464, 7, 7)
      6. conv5: 1x1 projection to (B, 1024, 7, 7)
      7. Global Average Pooling: (B, 1024)
      8. Fully Connected (fc): (B, num_classes)
    """

    def __init__(self, stages_repeats, stages_out_channels, num_classes=7):
        super(EfficientFace, self).__init__()

        if len(stages_repeats) != 3:
            raise ValueError('expected stages_repeats as list of 3 positive ints')
        if len(stages_out_channels) != 5:
            raise ValueError('expected stages_out_channels as list of 5 positive ints')
        self._stage_out_channels = stages_out_channels

        input_channels = 3
        output_channels = self._stage_out_channels[0]
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )
        input_channels = output_channels

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Build Stage 2, 3, 4 inverted residual blocks
        stage_names = ['stage{}'.format(i) for i in [2, 3, 4]]
        for name, repeats, output_channels in zip(stage_names, stages_repeats, self._stage_out_channels[1:]):
            seq = [InvertedResidual(input_channels, output_channels, 2)]
            for i in range(repeats - 1):
                seq.append(InvertedResidual(output_channels, output_channels, 1))
            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels

        # Local Feature Extractor and Attention Modulator
        self.local = LocalFeatureExtractor(29, 116, 1)
        self.modulator = Modulator(116)

        output_channels = self._stage_out_channels[-1]

        # 1x1 Conv projection
        self.conv5 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )

        ##################
        # TODO: Student Contribution - Classifier Head & Regularization
        # You can add Dropout (e.g. nn.Dropout(p=0.2)) before self.fc to prevent
        # overfitting during fine-tuning on custom captured facial images.
        # Example:
        #   self.dropout = nn.Dropout(p=0.3)
        #   self.fc = nn.Linear(output_channels, num_classes)
        ##################
        self.fc = nn.Linear(output_channels, num_classes)

    def forward(self, x):
        # Stem
        x = self.conv1(x)
        x = self.maxpool(x)

        # Stage 2 + Attention Modulation + Local Feature Fusion
        x = self.modulator(self.stage2(x)) + self.local(x)

        # Stage 3 and 4
        x = self.stage3(x)
        x = self.stage4(x)

        # Conv5 + Global Average Pooling
        x = self.conv5(x)
        x = x.mean([2, 3])

        # Classification Head
        x = self.fc(x)
        return x


def efficient_face(num_classes=7):
    """
    Convenience factory function for creating the default EfficientFace model.
    Default repeats: [4, 8, 4] for stage2, stage3, stage4.
    Default channels: [29, 116, 232, 464, 1024].
    Default classes: 7 (Neutral, Happiness, Sadness, Surprise, Fear, Disgust, Anger).
    """
    model = EfficientFace([4, 8, 4], [29, 116, 232, 464, 1024], num_classes=num_classes)
    return model
