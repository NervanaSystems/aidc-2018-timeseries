from ngraph.frontends.neon.layer import Convolution
from ngraph.frontends.neon import GaussianInit, Rectlin, Sequential
from ngraph.frontends.neon.model import ResidualModule
from topologies.custom_neon_classes import DilatedCausalConv, Dropout2D


def dilated_causal_conv_layer(kernel_size, n_filters, stride, dilation, init=GaussianInit(0, 0.01)):
    # define dilated causal convolution layer
    conv_layer = DilatedCausalConv(filter_shape=(kernel_size, n_filters),
                             filter_init=init,
                             strides=stride,
                             dilation=dilation,
                             padding='causal',
                             batch_norm=False)

    return [conv_layer]

## define temporal block
def temporal_block(out_channels, kernel_size, stride, dilation, dropout=0.2):
    # conv layer
    conv_layer = dilated_causal_conv_layer(kernel_size, out_channels, stride, dilation)

    # relu
    relu_layer = Rectlin()

    # dropout
    dropout_layer = Dropout2D(dropout)

    return conv_layer + [relu_layer, dropout_layer]



## define residual block
def residual_block(in_channels, out_channels, kernel_size, dilation, dropout=0.2, stride=1):
    # define two temporal blocks
    tb = []
    for i in range(2):
        tb += temporal_block(out_channels, kernel_size, stride, dilation, dropout=dropout)
    main_path = Sequential(tb)

    # sidepath
    if in_channels != out_channels:
        side_path = Sequential([Convolution(filter_shape=(1, out_channels), filter_init=GaussianInit(0, 0.01), strides=1, dilation=1, padding='same', batch_norm=False)])
    else:
        side_path = None

    # combine both
    return ResidualModule(main_path, side_path)

## define tcn
def tcn(n_features_in, hidden_sizes, kernel_size=7, dropout=0.2):
    # loop and define multiple residual blocks
    n_hidden_layers = len(hidden_sizes)

    layers = []
    for i in range(n_hidden_layers):
        dilation_size = 2 ** i
        in_channels = n_features_in if i==0 else hidden_sizes[i-1]
        out_channels = hidden_sizes[i]
        layers += [residual_block(in_channels, out_channels, kernel_size, dilation=dilation_size, dropout=dropout), Rectlin()]

    # define model
    model = Sequential(layers)

    return model

