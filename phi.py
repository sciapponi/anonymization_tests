from typing import List
from micromind.networks.phinet import PhiNetConvBlock
import torch
import torch.ao.nn.quantized as nnq
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class CausalConv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.causal_padding = self.dilation[0] * (self.kernel_size[0] - 1)

    def forward(self, x):
        return self._conv_forward(F.pad(x, [self.causal_padding, 0]), self.weight, self.bias)

class CausalConvTranspose1d(nn.ConvTranspose1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.causal_padding = self.dilation[0] * (self.kernel_size[0] - 1) + self.output_padding[0] + 1 - self.stride[0]
    
    def forward(self, x, output_size=None):
        if self.padding_mode != 'zeros':
            raise ValueError('Only `zeros` padding mode is supported for ConvTranspose1d')

        assert isinstance(self.padding, tuple)
        output_padding = self._output_padding(
            x, output_size, self.stride, self.padding, self.kernel_size, self.dilation)
        return F.conv_transpose1d(
            x, self.weight, self.bias, self.stride, self.padding,
            output_padding, self.groups, self.dilation)[...,:-self.causal_padding]    

def _make_divisible(v, divisor=8, min_value=None):
    """
    This function is taken from the original tf repo. It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py

    It ensures that all layers have a channel number that is divisible by divisor.

    Arguments
    ---------
    v : int
        The original number of channels.
    divisor : int, optional
        The divisor to ensure divisibility (default is 8).
    min_value : int or None, optional
        The minimum value for the divisible channels (default is None).

    Returns
    -------
    int
        The adjusted number of channels.
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def correct_pad(input_shape, kernel_size):
    """Returns a tuple for zero-padding for 2D convolution with downsampling.

    Arguments
    ---------
    input_shape : tuple or list
        Shape of the input tensor (height, width).
    kernel_size : int or tuple
        Size of the convolution kernel.

    Returns
    -------
    tuple
        A tuple representing the zero-padding in the format (left, right, top, bottom).
    """
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)

    if input_shape[0] is None:
        adjust = (1, 1)
    else:
        adjust = (1 - input_shape[0] % 2, 1 - input_shape[1] % 2)

    correct = (kernel_size[0] // 2, kernel_size[1] // 2)

    return (
        int(correct[1] - adjust[1]),
        int(correct[1]),
        int(correct[0] - adjust[0]),
        int(correct[0]),
    )


def preprocess_input(x, **kwargs):
    """Normalize input channels between [-1, 1].

    Arguments
    ---------
    x : torch.Tensor
        Input tensor to be preprocessed.

    Returns
    -------
    torch.Tensor
        Normalized tensor with values between [-1, 1].
    """

    return (x / 128.0) - 1


def get_xpansion_factor(t_zero, beta, block_id, num_blocks):
    """Compute the expansion factor based on the formula from the paper.

    Arguments
    ---------
    t_zero : float
        The base expansion factor.
    beta : float
        The shape factor.
    block_id : int
        The identifier of the current block.
    num_blocks : int
        The total number of blocks.

    Returns
    -------
    float
        The computed expansion factor.
    """
    return (t_zero * beta) * block_id / num_blocks + t_zero * (
        num_blocks - block_id
    ) / num_blocks


class ReLUMax(torch.nn.Module):
    """Implements ReLUMax.

    Arguments
    ---------
    max_value : float
        The maximum value for the clamp operation.

    """

    def __init__(self, max):
        super(ReLUMax, self).__init__()
        self.max = max

    def forward(self, x):
        """Forward pass of ReLUMax.

        Arguments
        ---------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor after applying ReLU with max value.
        """
        return torch.clamp(x, min=0, max=self.max)


class SEBlock(torch.nn.Module):
    """Implements squeeze-and-excitation block.

    Arguments
    ---------
    in_channels : int
        Input number of channels.
    out_channels : int
        Output number of channels.
    h_swish : bool, optional
        Whether to use the h_swish (default is True).

    """

    def __init__(self, in_channels, out_channels, h_swish=True):
        super(SEBlock, self).__init__()

        self.se_conv = CausalConv1d(
            in_channels,
            out_channels,
            kernel_size=1,
            padding=0,
            bias=False,
        )

        self.se_conv2 = CausalConv1d(
            out_channels, in_channels, kernel_size=1, bias=False, padding=0
        )

        if h_swish:
            self.activation = nn.Hardswish(inplace=True)
        else:
            self.activation = ReLUMax(6)

        # It serves for the quantization.
        # The behavior remains equivalent for the unquantized models.
        self.mult = nnq.FloatFunctional()

    def forward(self, x):
        """Executes the squeeze-and-excitation block.

        Arguments
        ---------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output of the squeeze-and-excitation block.
        """

        inp = x
        x = F.adaptive_avg_pool1d(x, 1)
        x = self.se_conv(x)
        x = self.activation(x)
        x = self.se_conv2(x)
        x = torch.sigmoid(x)

        return self.mult.mul(inp, x)  # Equivalent to ``torch.mul(a, b)``


class DepthwiseCausalConv(CausalConv1d):
    """Depthwise Causal 1D convolution layer.

    Arguments
    ---------
    in_channels : int
        Number of input channels.
    depth_multiplier : int, optional
        The channel multiplier for the output channels (default is 1).
    kernel_size : int or tuple, optional
        Size of the convolution kernel (default is 3).
    stride : int or tuple, optional
        Stride of the convolution (default is 1).
    padding : int or tuple, optional
        Zero-padding added to both sides of the input (default is 0).
    dilation : int or tuple, optional
        Spacing between kernel elements (default is 1).
    bias : bool, optional
        If True, adds a learnable bias to the output (default is False).
    padding_mode : str, optional
        'zeros' or 'circular'. Padding mode for convolution (default is 'zeros').

    """

    def __init__(
        self,
        in_channels,
        depth_multiplier=1,
        kernel_size=3,
        stride=1,
        padding=0,
        dilation=1,
        bias=False,
        padding_mode="zeros",
    ):
        out_channels = in_channels * depth_multiplier
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=bias,
            padding_mode=padding_mode,
        )


class SeparableCausalConv1d(torch.nn.Module):
    """Implements SeparableCausalConv1d.

    Arguments
    ---------
    in_channels : int
        Input number of channels.
    out_channels : int
        Output number of channels.
    activation : function, optional
        Activation function to apply (default is torch.nn.functional.relu).
    kernel_size : int, optional
        Kernel size (default is 3).
    stride : int, optional
        Stride for convolution (default is 1).
    padding : int, optional
        Padding for convolution (default is 0).
    dilation : int, optional
        Dilation factor for convolution (default is 1).
    bias : bool, optional
        If True, adds a learnable bias to the output (default is True).
    padding_mode : str, optional
        Padding mode for convolution (default is 'zeros').
    depth_multiplier : int, optional
        Depth multiplier (default is 1).

    """

    def __init__(
        self,
        in_channels,
        out_channels,
        activation=torch.nn.functional.relu,
        kernel_size=3,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
        padding_mode="zeros",
        depth_multiplier=1,
    ):
        super().__init__()

        self._layers = torch.nn.ModuleList()

        depthwise = CausalConv1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=1,
            groups=in_channels,
            bias=bias,
            padding_mode=padding_mode,
        )

        spatialConv = CausalConv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=dilation,
            # groups=in_channels,
            bias=bias,
            padding_mode=padding_mode,
        )

        bn = torch.nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.999)

        self._layers.append(depthwise)
        self._layers.append(spatialConv)
        self._layers.append(bn)
        self._layers.append(activation)

    def forward(self, x):
        """Executes the SeparableConv2d block.

        Arguments
        ---------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output of the convolution.
        """
        for layer in self._layers:
            x = layer(x)

        return x


class PhiNetCausalConvBlock(nn.Module):
    """Implements PhiNet's convolutional block.

    Arguments
    ---------
    in_shape : tuple
        Input shape of the conv block.
    expansion : float
        Expansion coefficient for this convolutional block.
    stride: int
        Stride for the conv block.
    filters : int
        Output channels of the convolutional block.
    block_id : int
        ID of the convolutional block.
    has_se : bool
        Whether to include use Squeeze and Excite or not.
    res : bool
        Whether to use the residual connection or not.
    h_swish : bool
        Whether to use HSwish or not.
    k_size : int
        Kernel size for the depthwise convolution.

    """

    def __init__(
        self,
        in_channels,
        expansion,
        stride,
        filters,
        has_se,
        block_id=None,
        res=True,
        h_swish=True,
        k_size=3,
        dp_rate=0.05,
        divisor=1,
        dilation=1,
    ):
        super(PhiNetCausalConvBlock, self).__init__()

        self.param_count = 0

        self.skip_conn = False
        
        self._layers = torch.nn.ModuleList()
        # in_channels = in_shape[0]
        
        # Define activation function
        if h_swish:
            activation = nn.Hardswish(inplace=True)
        else:
            activation = ReLUMax(6)

        if block_id:
            # Expand
            conv1 = CausalConv1d(
                in_channels,
                _make_divisible(int(expansion * in_channels), divisor=divisor),
                kernel_size=1,
                padding=0,
                bias=False,
            )

            bn1 = nn.BatchNorm1d(
                _make_divisible(int(expansion * in_channels), divisor=divisor),
                eps=1e-3,
                momentum=0.999,
            )

            self._layers.append(conv1)
            self._layers.append(bn1)
            self._layers.append(activation)

        if stride == 2:
            padding = correct_pad([res, res], 3)

        self._layers.append(nn.Dropout1d(dp_rate))

        d_mul = 1
        in_channels_dw = (
            _make_divisible(int(expansion * in_channels), divisor=divisor)
            if block_id
            else in_channels
        )
        out_channels_dw = in_channels_dw * d_mul
        dw1 = DepthwiseCausalConv(
            in_channels=in_channels_dw,
            depth_multiplier=d_mul,
            kernel_size=k_size,
            stride=stride,
            bias=False,
            padding= 0,#k_size // 2 if stride == 1 else (padding[1], padding[3]),
            dilation = dilation,
        )

        bn_dw1 = nn.BatchNorm1d(
            out_channels_dw,
            eps=1e-3,
            momentum=0.999,
        )

        # It is necessary to reinitialize the activation
        # for functions using Module.children() to work properly.
        # Module.children() does not return repeated layers.
        if h_swish:
            activation = nn.Hardswish(inplace=True)
        else:
            activation = ReLUMax(6)

        self._layers.append(dw1)
        self._layers.append(bn_dw1)
        self._layers.append(activation)

        if has_se:
            num_reduced_filters = _make_divisible(
                max(1, int(out_channels_dw / 6)), divisor=divisor
            )
            se_block = SEBlock(out_channels_dw, num_reduced_filters, h_swish=h_swish)
            self._layers.append(se_block)

        conv2 = CausalConv1d(
            in_channels=out_channels_dw,
            out_channels=filters,
            kernel_size=1,
            padding=0,
            bias=False,
        )

        bn2 = nn.BatchNorm1d(
            filters,
            eps=1e-3,
            momentum=0.999,
        )

        self._layers.append(conv2)
        self._layers.append(bn2)

        if res and in_channels == filters and stride == 1:
            self.skip_conn = True
            # It serves for the quantization.
            # The behavior remains equivalent for the unquantized models.
            self.op = nnq.FloatFunctional()

    def forward(self, x):
        """Executes the PhiNet convolutional block.

        Arguments
        ---------
        x : torch.Tensor
            Input to the convolutional block.

        Returns
        -------
        torch.Tensor
            Output of the convolutional block.
        """

        if self.skip_conn:
            inp = x

        for layer in self._layers:
            x = layer(x)

        if self.skip_conn:
            return self.op.add(x, inp)  # Equivalent to ``torch.add(a, b)``

        return x

class ResidualUnit(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 dilation,
                 pad_mode="reflect"):
        super(ResidualUnit, self).__init__()

        self.dilaton = dilation

        self.layers = nn.Sequential(
            PhiNetCausalConvBlock(in_channels=in_channels, 
                                  filters=out_channels, 
                                  k_size=7, 
                                  dilation=dilation,
                                  has_se=True,
                                  expansion=1,
                                  stride=1),
            PhiNetCausalConvBlock(in_channels=in_channels, 
                                  filters=out_channels, 
                                  k_size=1, 
                                  has_se=True,
                                  expansion=1,
                                  stride=1),
        )

    def forward(self, x):
        return x + self.layers(x)

class EncoderBlock(nn.Module):
    def __init__(self, out_channels, stride):
        super(EncoderBlock, self).__init__()

        self.layers = nn.Sequential(
            ResidualUnit(
                in_channels=out_channels // 2,
                out_channels=out_channels // 2,
                dilation=1
            ),
            ResidualUnit(
                in_channels=out_channels // 2,
                out_channels=out_channels // 2,
                dilation=3
            ),
            ResidualUnit(
                in_channels=out_channels // 2,
                out_channels=out_channels // 2,
                dilation=9
            ),
            PhiNetCausalConvBlock(
                in_channels=out_channels // 2,
                filters=out_channels,
                k_size=stride*2,
                stride=stride,
                has_se=True,
                expansion=1
            ),
        )

    def forward(self, x):
        return self.layers(x)
    
class Encoder(nn.Module):
    def __init__(self, C, D, strides=(4, 5, 16)):
        super(Encoder, self).__init__()

        self.layers = nn.Sequential(
            CausalConv1d(in_channels=1, out_channels=C, kernel_size=7),
            EncoderBlock(out_channels=2*C, stride=strides[0]),
            EncoderBlock(out_channels=4*C, stride=strides[1]),
            EncoderBlock(out_channels=8*C, stride=strides[2]),
            # EncoderBlock(out_channels=16*C, stride=strides[3]),
            CausalConv1d(in_channels=8*C, out_channels=D, kernel_size=3)
        )

    def forward(self, x):
        return self.layers(x)