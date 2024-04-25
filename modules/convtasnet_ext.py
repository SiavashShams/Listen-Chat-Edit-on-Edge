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


class FilmTemporalBlocksSequential(sb.nnet.containers.Sequential):
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
        self.bottleneck_conv1x1 = sb.nnet.CNN.Conv1d(
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
        self.mask_conv1x1 = sb.nnet.CNN.Conv1d(
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

        self.layers = sb.nnet.containers.Sequential(input_shape=input_shape)

        # [M, K, B] -> [M, K, H]
        self.layers.append(
            sb.nnet.CNN.Conv1d,
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


class DepthwiseSeparableConv(sb.nnet.containers.Sequential):
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
            sb.nnet.CNN.Conv1d,
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
            sb.nnet.CNN.Conv1d,
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
