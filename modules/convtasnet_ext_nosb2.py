'''
Copied from speechbrain's conv-tasnet implementation.
Reworked for extraction.
'''

""" Implementation of a popular speech separation model.
"""
import torch
import torch.nn as nn
#import speechbrain as sb
import torch.nn.functional as F
import inspect
import math
from .film import FiLM

EPS = 1e-8

def get_padding_elem(L_in: int, stride: int, kernel_size: int, dilation: int):
    """This function computes the number of elements to add for zero-padding.

    Arguments
    ---------
    L_in : int
    stride: int
    kernel_size : int
    dilation : int

    Returns
    -------
    padding : int
        The size of the padding to be added
    """
    if stride > 1:
        padding = [math.floor(kernel_size / 2), math.floor(kernel_size / 2)]

    else:
        L_out = (
            math.floor((L_in - dilation * (kernel_size - 1) - 1) / stride) + 1
        )
        padding = [
            math.floor((L_in - L_out) / 2),
            math.floor((L_in - L_out) / 2),
        ]
    return padding


class Conv1d(nn.Module):
    """This function implements 1d convolution.

    Arguments
    ---------
    out_channels : int
        It is the number of output channels.
    kernel_size : int
        Kernel size of the convolutional filters.
    input_shape : tuple
        The shape of the input. Alternatively use ``in_channels``.
    in_channels : int
        The number of input channels. Alternatively use ``input_shape``.
    stride : int
        Stride factor of the convolutional filters. When the stride factor > 1,
        a decimation in time is performed.
    dilation : int
        Dilation factor of the convolutional filters.
    padding : str
        (same, valid, causal). If "valid", no padding is performed.
        If "same" and stride is 1, output shape is the same as the input shape.
        "causal" results in causal (dilated) convolutions.
    groups : int
        Number of blocked connections from input channels to output channels.
    bias : bool
        Whether to add a bias term to convolution operation.
    padding_mode : str
        This flag specifies the type of padding. See torch.nn documentation
        for more information.
    skip_transpose : bool
        If False, uses batch x time x channel convention of speechbrain.
        If True, uses batch x channel x time convention.
    weight_norm : bool
        If True, use weight normalization,
        to be removed with self.remove_weight_norm() at inference
    conv_init : str
        Weight initialization for the convolution network
    default_padding: str or int
        This sets the default padding mode that will be used by the pytorch Conv1d backend.

    Example
    -------
    >>> inp_tensor = torch.rand([10, 40, 16])
    >>> cnn_1d = Conv1d(
    ...     input_shape=inp_tensor.shape, out_channels=8, kernel_size=5
    ... )
    >>> out_tensor = cnn_1d(inp_tensor)
    >>> out_tensor.shape
    torch.Size([10, 40, 8])
    """

    def __init__(
        self,
        out_channels,
        kernel_size,
        input_shape=None,
        in_channels=None,
        stride=1,
        dilation=1,
        padding="same",
        groups=1,
        bias=True,
        padding_mode="reflect",
        skip_transpose=False,
        weight_norm=False,
        conv_init=None,
        default_padding=0,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.padding_mode = padding_mode
        self.unsqueeze = False
        self.skip_transpose = skip_transpose

        if input_shape is None and in_channels is None:
            raise ValueError("Must provide one of input_shape or in_channels")

        if in_channels is None:
            in_channels = self._check_input_shape(input_shape)

        self.in_channels = in_channels

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            self.kernel_size,
            stride=self.stride,
            dilation=self.dilation,
            padding=default_padding,
            groups=groups,
            bias=bias,
        )

        if conv_init == "kaiming":
            nn.init.kaiming_normal_(self.conv.weight)
        elif conv_init == "zero":
            nn.init.zeros_(self.conv.weight)
        elif conv_init == "normal":
            nn.init.normal_(self.conv.weight, std=1e-6)

        if weight_norm:
            self.conv = nn.utils.weight_norm(self.conv)

    def forward(self, x):
        """Returns the output of the convolution.

        Arguments
        ---------
        x : torch.Tensor (batch, time, channel)
            input to convolve. 2d or 4d tensors are expected.

        Returns
        -------
        wx : torch.Tensor
            The convolved outputs.
        """
        if not self.skip_transpose:
            x = x.transpose(1, -1)

        if self.unsqueeze:
            x = x.unsqueeze(1)

        if self.padding == "same":
            x = self._manage_padding(
                x, self.kernel_size, self.dilation, self.stride
            )

        elif self.padding == "causal":
            num_pad = (self.kernel_size - 1) * self.dilation
            x = F.pad(x, (num_pad, 0))

        elif self.padding == "valid":
            pass

        else:
            raise ValueError(
                "Padding must be 'same', 'valid' or 'causal'. Got "
                + self.padding
            )

        wx = self.conv(x)

        if self.unsqueeze:
            wx = wx.squeeze(1)

        if not self.skip_transpose:
            wx = wx.transpose(1, -1)

        return wx

    def _manage_padding(self, x, kernel_size: int, dilation: int, stride: int):
        """This function performs zero-padding on the time axis
        such that their lengths is unchanged after the convolution.

        Arguments
        ---------
        x : torch.Tensor
            Input tensor.
        kernel_size : int
            Size of kernel.
        dilation : int
            Dilation used.
        stride : int
            Stride.

        Returns
        -------
        x : torch.Tensor
            The padded outputs.
        """

        # Detecting input shape
        L_in = self.in_channels

        # Time padding
        padding = get_padding_elem(L_in, stride, kernel_size, dilation)

        # Applying padding
        x = F.pad(x, padding, mode=self.padding_mode)

        return x

    def _check_input_shape(self, shape):
        """Checks the input shape and returns the number of input channels."""

        if len(shape) == 2:
            self.unsqueeze = True
            in_channels = 1
        elif self.skip_transpose:
            in_channels = shape[1]
        elif len(shape) == 3:
            in_channels = shape[2]
        else:
            raise ValueError(
                "conv1d expects 2d, 3d inputs. Got " + str(len(shape))
            )

        # Kernel size must be odd
        if not self.padding == "valid" and self.kernel_size % 2 == 0:
            raise ValueError(
                "The field kernel size must be an odd number. Got %s."
                % (self.kernel_size)
            )

        return in_channels

    def remove_weight_norm(self):
        """Removes weight normalization at inference if used during training."""
        self.conv = nn.utils.remove_weight_norm(self.conv)
        
        

class Sequential(torch.nn.ModuleDict):
    """A sequence of modules with potentially inferring shape on construction.

    If layers are passed with names, these can be referenced with dot notation.

    Arguments
    ---------
    *layers : tuple
        Layers to be applied in sequence.
    input_shape : iterable
        A list or tuple of ints or None, representing the expected shape of an
        input tensor. None represents a variable-length dimension. If no
        ``input_shape`` is passed, no shape inference will be performed.
    **named_layers : dict
        The inputs are treated as a list of layers to be
        applied in sequence. The output shape of each layer is used to
        infer the shape of the following layer. If a tuple is returned,
        only the shape of the first element is used to determine input
        shape of the next layer (e.g. RNN returns output, hidden).

    Example
    -------
    >>> inputs = torch.rand(10, 40, 50)
    >>> model = Sequential(input_shape=inputs.shape)
    >>> model.append(Linear, n_neurons=100, layer_name="layer1")
    >>> model.append(Linear, n_neurons=200, layer_name="layer2")
    >>> outputs = model(inputs)
    >>> outputs.shape
    torch.Size([10, 40, 200])
    >>> outputs = model.layer1(inputs)
    >>> outputs.shape
    torch.Size([10, 40, 100])
    """

    def __init__(self, *layers, input_shape=None, **named_layers):
        super().__init__()

        # Make sure either layers or input_shape is passed
        if not layers and input_shape is None and not named_layers:
            raise ValueError("Must pass either layers or input shape")

        # Keep track of what layers need "lengths" passed
        self.length_layers = []

        # Replace None dimensions with arbitrary value
        self.input_shape = input_shape
        if input_shape and None in input_shape:
            self.input_shape = list(input_shape)
            for i, dim in enumerate(self.input_shape):
                # To reduce size of dummy tensors, use 1 for batch dim
                if i == 0 and dim is None:
                    dim = 1

                # Use 64 as nice round arbitrary value, big enough that
                # halving this dimension a few times doesn't reach 1
                self.input_shape[i] = dim or 256

        # Append non-named layers
        for layer in layers:
            self.append(layer)

        # Append named layers
        for name, layer in named_layers.items():
            self.append(layer, layer_name=name)

    def append(self, layer, *args, layer_name=None, **kwargs):
        """Add a layer to the list of layers, inferring shape if necessary.

        Arguments
        ---------
        layer : A torch.nn.Module class or object
            If the layer is a class, it should accept an argument called
            ``input_shape`` which will be inferred and passed. If the layer
            is a module object, it is added as-is.
        *args : tuple
            These are passed to the layer if it is constructed.
        layer_name : str
            The name of the layer, for reference. If the name is in use,
            ``_{count}`` will be appended.
        **kwargs : dict
            These are passed to the layer if it is constructed.
        """

        # Compute layer_name
        if layer_name is None:
            layer_name = str(len(self))
        elif layer_name in self:
            index = 0
            while f"{layer_name}_{index}" in self:
                index += 1
            layer_name = f"{layer_name}_{index}"

        # Check if it needs to be constructed with input shape
        if self.input_shape:
            argspec = inspect.getfullargspec(layer)
            if "input_shape" in argspec.args + argspec.kwonlyargs:
                input_shape = self.get_output_shape()
                layer = layer(*args, input_shape=input_shape, **kwargs)

        # Finally, append the layer.
        try:
            self.add_module(layer_name, layer)
        except TypeError:
            raise ValueError(
                "Must pass `input_shape` at initialization and use "
                "modules that take `input_shape` to infer shape when "
                "using `append()`."
            )

    def get_output_shape(self):
        """Returns expected shape of the output.

        Computed by passing dummy input constructed with the
        ``self.input_shape`` attribute.

        Returns
        -------
        Expected shape of the output after all layers applied.
        """
        with torch.no_grad():
            dummy_input = torch.zeros(self.input_shape)
            dummy_output = self(dummy_input)
        return dummy_output.shape

    def forward(self, x):
        """Applies layers in sequence, passing only the first element of tuples.

        Arguments
        ---------
        x : torch.Tensor
            The input tensor to run through the network.

        Returns
        -------
        x : torch.Tensor
            Output after all layers are applied.
        """
        for layer in self.values():
            x = layer(x)
            if isinstance(x, tuple):
                x = x[0]

        return x
    
    

class FilmTemporalBlocksSequential(Sequential):
    """
    A wrapper for the temporal-block layer to replicate it

    Arguments
    ---------
    input_shape : tuple
        Expected shape of the input.
    H : int
        The number of intermediate channels.
    P : int
        The kernel size in the convolutions.
    R : int
        The number of times to replicate the multilayer Temporal Blocks.
    X : int
        The number of layers of Temporal Blocks with different dilations.
    norm type : str
        The type of normalization, in ['gLN', 'cLN'].
    causal : bool
        To use causal or non-causal convolutions, in [True, False].

    Example
    -------
    >>> x = torch.randn(14, 100, 10)
    >>> H, P, R, X = 10, 5, 2, 3
    >>> TemporalBlocks = TemporalBlocksSequential(
    ...     x.shape, H, P, R, X, 'gLN', False
    ... )
    >>> y = TemporalBlocks(x)
    >>> y.shape
    torch.Size([14, 100, 10])
    """

    def __init__(
        self, 
        input_shape, 
        H, 
        P, 
        R, 
        X, 
        norm_type, 
        causal, 
        cond_dim, 
        film_mode='none', 
        film_n_layer=2,
        film_scale=True,
        film_where='before1x1'
    ):
        super().__init__(input_shape=input_shape)
        assert film_mode in ['none', 'layer', 'block']
        print(f'Use FiLM at (every) {film_mode}.')
        self.film_mode = film_mode
        for r in range(R):
            for x in range(X):
                dilation = 2 ** x
                if film_mode == 'layer' or (film_mode == 'block' and x == 0):
                    film_this_layer = True 
                else:
                    film_this_layer = False

                self.append(
                    FilmTemporalBlock,
                    out_channels=H,
                    kernel_size=P,
                    stride=1,
                    padding="same",
                    dilation=dilation,
                    norm_type=norm_type,
                    causal=causal,
                    cond_dim=cond_dim,
                    film_this_layer=film_this_layer,
                    film_n_layer=film_n_layer,
                    film_scale=film_scale,
                    film_where=film_where,
                    layer_name=f"filmtemporalblock_{r}_{x}",
                )

    def forward(self, x, cond_embed=None):
        """Applies layers in sequence, passing only the first element of tuples.

        Arguments
        ---------
        x : torch.Tensor
            The input tensor to run through the network.
        """
        for layer in self.values():
            x = layer(x, cond_embed)
            if isinstance(x, tuple):
                x = x[0]

        return x


class MaskNet(nn.Module):
    """
    Arguments
    ---------
    N : int
        Number of filters in autoencoder.
    B : int
        Number of channels in bottleneck 1 × 1-conv block.
    H : int
        Number of channels in convolutional blocks.
    P : int
        Kernel size in convolutional blocks.
    X : int
        Number of convolutional blocks in each repeat.
    R : int
        Number of repeats.
    C : int
        Number of speakers.
    norm_type : str
        One of BN, gLN, cLN.
    causal : bool
        Causal or non-causal.
    mask_nonlinear : str
        Use which non-linear function to generate mask, in ['softmax', 'relu'].

    Example:
    ---------
    >>> N, B, H, P, X, R, C = 11, 12, 2, 5, 3, 1, 2
    >>> MaskNet = MaskNet(N, B, H, P, X, R, C)
    >>> mixture_w = torch.randn(10, 11, 100)
    >>> est_mask = MaskNet(mixture_w)
    >>> est_mask.shape
    torch.Size([2, 10, 11, 100])
    """

    def __init__(
        self,
        N,
        B,
        H,
        P,
        X,
        R,
        C,
        norm_type="gLN",
        causal=False,
        mask_nonlinear="relu",
        cond_dim=768,
        film_mode='none',
        film_n_layer=2,
        film_scale=True,
        film_where='before1x1'
    ):
        super(MaskNet, self).__init__()

        # Hyper-parameter
        self.C = C
        self.mask_nonlinear = mask_nonlinear

        # Components
        # [M, K, N] -> [M, K, N]
        self.layer_norm = ChannelwiseLayerNorm(N)

        # [M, K, N] -> [M, K, B]
        self.bottleneck_conv1x1 = Conv1d(
            in_channels=N, out_channels=B, kernel_size=1, bias=False,
        )

        # [M, K, B] -> [M, K, B]
        in_shape = (None, None, B)
        self.temporal_conv_net = FilmTemporalBlocksSequential(
            in_shape, H, P, R, X, norm_type, causal, 
            cond_dim=cond_dim, film_mode=film_mode, 
            film_n_layer=film_n_layer, film_scale=film_scale,
            film_where=film_where
        )

        # [M, K, B] -> [M, K, C*N]
        self.mask_conv1x1 = Conv1d(
            in_channels=B, out_channels=C * N, kernel_size=1, bias=False
        )

    def forward(self, mixture_w, cond_embed=None):
        """Keep this API same with TasNet.

        Arguments
        ---------
        mixture_w : Tensor
            Tensor shape is [M, K, N], M is batch size.

        Returns
        -------
        est_mask : Tensor
            Tensor shape is [M, K, C, N].
        """
        mixture_w = mixture_w.permute(0, 2, 1)
        M, K, N = mixture_w.size()
        
        y = self.layer_norm(mixture_w)
        if torch.any(torch.isnan(y)):
            print('LayerNorm NaN')
        if torch.any(torch.isinf(y)):
            print('LayerNorm inf')

        y = self.bottleneck_conv1x1(y)
        if torch.any(torch.isnan(y)):
            print('BottleNeck NaN')
        if torch.any(torch.isinf(y)):
            print('BottleNeck inf')

        y = self.temporal_conv_net(y, cond_embed)
        if torch.any(torch.isnan(y)):
            print('TCN NaN')
        if torch.any(torch.isinf(y)):
            print('TCN inf')

        score = self.mask_conv1x1(y)
        if torch.any(torch.isnan(score)):
            print('MaskConv NaN')
        if torch.any(torch.isinf(score)):
            print('MaskConv inf')

        # score = self.network(mixture_w)  # [M, K, N] -> [M, K, C*N]
        score = score.contiguous().reshape(
            M, K, self.C, N
        )  # [M, K, C*N] -> [M, K, C, N]

        # [M, K, C, N] -> [C, M, N, K]
        score = score.permute(2, 0, 3, 1)

        if self.mask_nonlinear == "softmax":
            est_mask = F.softmax(score, dim=2)
        elif self.mask_nonlinear == "sigmoid":
            est_mask = F.sigmoid(score)
        elif self.mask_nonlinear == "relu":
            est_mask = F.relu(score)
        else:
            raise ValueError("Unsupported mask non-linear function")
        return est_mask


class FilmTemporalBlock(torch.nn.Module):
    """The conv1d compound layers used in Masknet.

    Arguments
    ---------
    input_shape : tuple
        The expected shape of the input.
    out_channels : int
        The number of intermediate channels.
    kernel_size : int
        The kernel size in the convolutions.
    stride : int
        Convolution stride in convolutional layers.
    padding : str
        The type of padding in the convolutional layers,
        (same, valid, causal). If "valid", no padding is performed.
    dilation : int
        Amount of dilation in convolutional layers.
    norm type : str
        The type of normalization, in ['gLN', 'cLN'].
    causal : bool
        To use causal or non-causal convolutions, in [True, False].

    Example:
    ---------
    >>> x = torch.randn(14, 100, 10)
    >>> TemporalBlock = TemporalBlock(x.shape, 10, 11, 1, 'same', 1)
    >>> y = TemporalBlock(x)
    >>> y.shape
    torch.Size([14, 100, 10])
    """

    def __init__(
        self,
        input_shape,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        norm_type="gLN",
        causal=False,
        cond_dim=768,
        film_this_layer=False,
        film_n_layer=2,
        film_scale=True,
        film_where='before1x1'
    ):
        super().__init__()
        M, K, B = input_shape

        self.layers = Sequential(input_shape=input_shape)

        # [M, K, B] -> [M, K, H]
        self.layers.append(
            Conv1d,
            out_channels=out_channels,
            kernel_size=1,
            bias=False,
            layer_name="conv",
        )

        self.layers.append(nn.PReLU(), layer_name="act")
        self.layers.append(
            choose_norm(norm_type, out_channels), layer_name="norm"
        )

        # [M, K, H] -> [M, K, B]
        self.layers.append(
            DepthwiseSeparableConv,
            out_channels=B,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            norm_type=norm_type,
            causal=causal,
            layer_name="DSconv",
        )

        self.film_this_layer = film_this_layer
        if self.film_this_layer:
            self.film = FiLM(
                in_dim=cond_dim,
                out_dim= out_channels if film_where == 'after1x1' else B,
                n_layer=film_n_layer,
                scale=film_scale
            )
            print(f'Initialized a FiLM {film_where}.')

        self.film_where = film_where

    def forward(self, x, cond_embed=None):
        """
        Arguments
        ---------
        x : Tensor
            Tensor shape is [M, K, B].

        Returns
        -------
        x : Tensor
            Tensor shape is [M, K, B].
        """

        if (cond_embed == None) or (self.film_this_layer == False):
            y = self.layers(x) + x

        elif self.film_where == 'before1x1':
            x_cond = self.film(x, cond_embed)
            y = self.layers(x_cond) + x_cond

        elif self.film_where == 'rightbefore1x1':
            x_cond = self.film(x, cond_embed)
            y = self.layers(x_cond) + x

        elif self.film_where == 'after1x1':
            x_ = self.layers.conv(x)
            x_cond = self.film(x_, cond_embed)
            y = self.layers.DSconv(self.layers.norm(self.layers.act(x_cond))) + x
            
        return y


class DepthwiseSeparableConv(Sequential):
    """Building block for the Temporal Blocks of Masknet in ConvTasNet.

    Arguments
    ---------
    input_shape : tuple
        Expected shape of the input.
    out_channels : int
        Number of output channels.
    kernel_size : int
        The kernel size in the convolutions.
    stride : int
        Convolution stride in convolutional layers.
    padding : str
        The type of padding in the convolutional layers,
        (same, valid, causal). If "valid", no padding is performed.
    dilation : int
        Amount of dilation in convolutional layers.
    norm type : str
        The type of normalization, in ['gLN', 'cLN'].
    causal : bool
        To use causal or non-causal convolutions, in [True, False].

    Example
    -------
    >>> x = torch.randn(14, 100, 10)
    >>> DSconv = DepthwiseSeparableConv(x.shape, 10, 11, 1, 'same', 1)
    >>> y = DSconv(x)
    >>> y.shape
    torch.Size([14, 100, 10])

    """

    def __init__(
        self,
        input_shape,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        norm_type="gLN",
        causal=False,
    ):
        super().__init__(input_shape=input_shape)

        batchsize, time, in_channels = input_shape

        # [M, K, H] -> [M, K, H]
        if causal:
            paddingval = dilation * (kernel_size - 1)
            padding = "causal"
            default_padding = "same"
        else:
            default_padding = 0

        self.append(
            Conv1d,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=False,
            layer_name="conv_0",
            default_padding=default_padding,
        )

        if causal:
            self.append(Chomp1d(paddingval), layer_name="chomp")

        self.append(nn.PReLU(), layer_name="act")
        self.append(choose_norm(norm_type, in_channels), layer_name="act")

        # [M, K, H] -> [M, K, B]
        self.append(
            Conv1d,
            out_channels=out_channels,
            kernel_size=1,
            bias=False,
            layer_name="conv_1",
        )


class Chomp1d(nn.Module):
    """This class cuts out a portion of the signal from the end.

    It is written as a class to be able to incorporate it inside a sequential
    wrapper.

    Arguments
    ---------
    chomp_size : int
        The size of the portion to discard (in samples).

    Example
    -------
    >>> x = torch.randn(10, 110, 5)
    >>> chomp = Chomp1d(10)
    >>> x_chomped = chomp(x)
    >>> x_chomped.shape
    torch.Size([10, 100, 5])
    """

    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        """
        Arguments
        x : Tensor
            Tensor shape is [M, Kpad, H].

        Returns
        -------
        x : Tensor
            Tensor shape is [M, K, H].
        """
        return x[:, : -self.chomp_size, :].contiguous()


def choose_norm(norm_type, channel_size):
    """This function returns the chosen normalization type.

    Arguments
    ---------
    norm_type : str
        One of ['gLN', 'cLN', 'batchnorm'].
    channel_size : int
        Number of channels.

    Example
    -------
    >>> choose_norm('gLN', 10)
    GlobalLayerNorm()
    """

    if norm_type == "gLN":
        return GlobalLayerNorm(channel_size)
    elif norm_type == "cLN":
        return ChannelwiseLayerNorm(channel_size)
    else:
        return nn.BatchNorm1d(channel_size)


class ChannelwiseLayerNorm(nn.Module):
    """Channel-wise Layer Normalization (cLN).

    Arguments
    ---------
    channel_size : int
        Number of channels in the normalization dimension (the third dimension).

    Example
    -------
    >>> x = torch.randn(2, 3, 3)
    >>> norm_func = ChannelwiseLayerNorm(3)
    >>> x_normalized = norm_func(x)
    >>> x.shape
    torch.Size([2, 3, 3])
    """

    def __init__(self, channel_size):
        super(ChannelwiseLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.Tensor(1, 1, channel_size))  # [1, 1, N]
        self.beta = nn.Parameter(torch.Tensor(1, 1, channel_size))  # [1, 1, N]
        self.reset_parameters()

    def reset_parameters(self):
        """Resets the parameters."""
        self.gamma.data.fill_(1)
        self.beta.data.zero_()

    def forward(self, y):
        """
        Args:
            y: [M, K, N], M is batch size, N is channel size, K is length
        Returns:
            cLN_y: [M, K, N]
        """

        t = y.dtype
        y = y.type(torch.float32)

        mean = torch.mean(y, dim=2, keepdim=True)  # [M, K, 1]
        var = torch.var(y, dim=2, keepdim=True, unbiased=False)  # [M, K, 1]
        cLN_y = self.gamma * (y - mean) / torch.pow(var + EPS, 0.5) + self.beta

        return cLN_y.type(t)


class GlobalLayerNorm(nn.Module):
    """Global Layer Normalization (gLN).

    Arguments
    ---------
    channel_size : int
        Number of channels in the third dimension.

    Example
    -------
    >>> x = torch.randn(2, 3, 3)
    >>> norm_func = GlobalLayerNorm(3)
    >>> x_normalized = norm_func(x)
    >>> x.shape
    torch.Size([2, 3, 3])
    """

    def __init__(self, channel_size):
        super(GlobalLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.Tensor(1, 1, channel_size))  # [1, 1, N]
        self.beta = nn.Parameter(torch.Tensor(1, 1, channel_size))  # [1, 1, N]
        self.reset_parameters()

    def reset_parameters(self):
        """Resets the parameters."""
        self.gamma.data.fill_(1)
        self.beta.data.zero_()

    def forward(self, y):
        """
        Arguments
        ---------
        y : Tensor
            Tensor shape [M, K, N]. M is batch size, N is channel size, and K is length.

        Returns
        -------
        gLN_y : Tensor
            Tensor shape [M, K. N]
        """

        # t = y.dtype
        # y = y.type(torch.float32)

        # Compute layer norm in full precision
        mean = y.mean(dim=1, keepdim=True).mean(
            dim=2, keepdim=True
        )  # [M, 1, 1]
        var = (
            (torch.pow(y - mean, 2))
            .mean(dim=1, keepdim=True)
            .mean(dim=2, keepdim=True)
        )
        gLN_y = self.gamma * (y - mean) / torch.pow(var + EPS, 0.5) + self.beta

        # return gLN_y.type(t)
        return gLN_y
