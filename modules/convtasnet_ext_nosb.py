'''
Copied from speechbrain's conv-tasnet implementation.
Reworked for extraction.
'''

""" Implementation of a popular speech separation model.
"""
import torch
import torch.nn as nn
import speechbrain as sb
import torch.nn.functional as F

from speechbrain.processing.signal_processing import overlap_and_add

from .film import FiLM

EPS = 1e-8

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

    
    

class FilmTemporalBlocksSequential(nn.Module):
    """ Blocks of Film Temporal with naming """
    def __init__(self, channels, num_repeats=3, num_layers=8):
        super().__init__()
        self.blocks = nn.ModuleDict()
        for r in range(num_repeats):
            for l in range(num_layers):
                block_name = f'filmtemporalblock_{r}_{l}'
                self.blocks[block_name] = FilmTemporalBlock(channels, channels, r, l)

    def forward(self, x):
        for block in self.blocks.values():
            x = block(x)
        return x

class MaskNet(nn.Module):
    """ Full model adaptation to use named layers for loading state_dict """
    def __init__(self, N, B, H, P, X, R, C):
        super().__init__()
        self.layer_norm = nn.LayerNorm(N)  # Placeholder for ChannelwiseLayerNorm
        self.bottleneck_conv1x1 = nn.Conv1d(N, B, 1)
        self.temporal_conv_net = FilmTemporalBlocksSequential(B, R, X)
        self.mask_conv1x1 = nn.Conv1d(B, C * N, 1)

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.bottleneck_conv1x1(x)
        x = self.temporal_conv_net(x)
        x = self.mask_conv1x1(x)
        return x

class FilmTemporalBlock(nn.Module):
    """ Custom Temporal Block aligning with SpeechBrain naming """
    def __init__(self, in_channels, out_channels, num_blocks, block_index):
        super().__init__()
        self.layers = nn.ModuleDict({
            'conv': nn.Conv1d(in_channels, 512, kernel_size=1, stride=1, bias=False),
            'act': nn.PReLU(num_parameters=1),
            'norm': GlobalLayerNorm(512),  # GroupNorm as a placeholder for gLN/cLN
            'DSconv': DepthwiseSeparableConv(512 ,out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1)
        })
        self.name = f'filmtemporalblock_{num_blocks}_{block_index}'

    def forward(self, x):
        x = self.layers['conv'](x)
        x = self.layers['act'](x)
        x = self.layers['norm'](x)
        x = self.layers['DSconv'](x)
        return x

class DepthwiseSeparableConv(nn.Module):
    """ Depthwise Separable Convolution as defined in your state_dict """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation):
        super().__init__()
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, dilation=dilation, bias=False)
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.activation = nn.PReLU(num_parameters=1)
        self.norm = GlobalLayerNorm(out_channels)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.activation(x)
        return self.norm(x)


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



