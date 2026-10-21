"""
Attention Modulator for EfficientFace

Components:
1. Channel Attention (Channel):
   - Squeeze-and-Excitation inspired module.
   - Computes channel-wise importance weights using Global Average Pooling + MLP.
2. Spatial Attention (Spatial):
   - Uses dilated convolutions (dilation=4) to expand the receptive field without downsampling.
   - Computes spatial importance weights across the feature map.
3. Modulator:
   - Fuses channel attention and spatial attention via Hadamard product and sigmoid activation.
   - Modulates input features: Output = Input * Sigmoid(Channel_Att * Spatial_Att).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Flatten(nn.Module):
    """Flattens input tensor into 2D (batch_size, -1)."""
    def forward(self, x):
        return x.view(x.size(0), -1)


class Channel(nn.Module):
    """
    Channel Attention Module:
    Captures inter-channel dependencies by aggregating spatial context via global pooling,
    followed by an MLP bottleneck to determine which feature channels are most informative
    for facial emotion recognition.
    """

    def __init__(self, gate_channel, reduction_ratio=16, num_layers=1):
        super(Channel, self).__init__()
        self.gate_c = nn.Sequential()
        self.gate_c.add_module('flatten', Flatten())

        # Construct MLP bottleneck: in -> (in // reduction_ratio) -> in
        gate_channels = [gate_channel]
        gate_channels += [gate_channel // reduction_ratio] * num_layers
        gate_channels += [gate_channel]

        for i in range(len(gate_channels) - 2):
            self.gate_c.add_module(f'gate_c_fc_{i}', nn.Linear(gate_channels[i], gate_channels[i + 1]))
            self.gate_c.add_module(f'gate_c_bn_{i + 1}', nn.BatchNorm1d(gate_channels[i + 1]))
            self.gate_c.add_module(f'gate_c_relu_{i + 1}', nn.ReLU())

        self.gate_c.add_module('gate_c_fc_final', nn.Linear(gate_channels[-2], gate_channels[-1]))

        ##################
        # TODO: Student Contribution - Channel Attention
        # Ideas for experimentation:
        # 1. Compare Average Pooling with Max Pooling (or combine both as in CBAM).
        # 2. Try Efficient Channel Attention (ECA-Net) using 1D convolutions without dimensionality reduction.
        ##################

    def forward(self, in_tensor):
        # Global Average Pooling: (B, C, H, W) -> (B, C, 1, 1)
        avg_pool = F.avg_pool2d(in_tensor, in_tensor.size(2), stride=in_tensor.size(2))
        # MLP: (B, C) -> (B, C), then unsqueeze back to (B, C, 1, 1) and expand
        return self.gate_c(avg_pool).unsqueeze(2).unsqueeze(3).expand_as(in_tensor)


class Spatial(nn.Module):
    """
    Spatial Attention Module:
    Captures spatial context using dilated convolutions to identify "where" salient
    facial emotion cues (e.g. squinting eyes, raised eyebrows, curved lips) are located.
    """

    def __init__(self, gate_channel, reduction_ratio=16, dilation_conv_num=2, dilation_val=4):
        super(Spatial, self).__init__()
        self.gate_s = nn.Sequential()

        # 1x1 conv to compress channels
        self.gate_s.add_module('gate_s_conv_reduce0', nn.Conv2d(gate_channel, gate_channel // reduction_ratio, kernel_size=1))
        self.gate_s.add_module('gate_s_bn_reduce0', nn.BatchNorm2d(gate_channel // reduction_ratio))
        self.gate_s.add_module('gate_s_relu_reduce0', nn.ReLU())

        # Dilated convolutions for wide receptive field without downsampling
        for i in range(dilation_conv_num):
            self.gate_s.add_module(
                f'gate_s_conv_di_{i}',
                nn.Conv2d(
                    gate_channel // reduction_ratio,
                    gate_channel // reduction_ratio,
                    kernel_size=3,
                    padding=dilation_val,
                    dilation=dilation_val,
                ),
            )
            self.gate_s.add_module(f'gate_s_bn_di_{i}', nn.BatchNorm2d(gate_channel // reduction_ratio))
            self.gate_s.add_module(f'gate_s_relu_di_{i}', nn.ReLU())

        # Final 1x1 conv to produce a 1-channel spatial attention map
        self.gate_s.add_module('gate_s_conv_final', nn.Conv2d(gate_channel // reduction_ratio, 1, kernel_size=1))

        ##################
        # TODO: Student Contribution - Spatial Attention
        # Ideas for experimentation:
        # 1. Experiment with different dilation values (e.g., dilation=2, 3, 6).
        # 2. Add deform conv or coordinate attention (CoordAttention) for better facial feature alignment.
        ##################

    def forward(self, in_tensor):
        return self.gate_s(in_tensor).expand_as(in_tensor)


class Modulator(nn.Module):
    """
    Unified Attention Modulator:
    Combines Channel and Spatial attention to gate feature representations.
    
    Calculation:
      att_map = sigmoid(Channel_Attention(x) * Spatial_Attention(x))
      output = att_map * x
    """

    def __init__(self, gate_channel):
        super(Modulator, self).__init__()
        self.channel_att = Channel(gate_channel)
        self.spatial_att = Spatial(gate_channel)

        ##################
        # TODO: Student Contribution - Attention Fusion
        # Ideas for experimentation:
        # 1. Try additive fusion: sigmoid(channel_att + spatial_att) instead of multiplicative.
        # 2. Try sequential application: spatial_att(channel_att(x)) as done in CBAM.
        ##################

    def forward(self, in_tensor):
        att = torch.sigmoid(self.channel_att(in_tensor) * self.spatial_att(in_tensor))
        return att * in_tensor
