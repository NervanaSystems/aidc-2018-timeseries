from ngraph.frontends.neon import Sequential, Convolution, Affine, BatchNorm
from ngraph.frontends.neon import KaimingInit, Rectlin, Identity


def define_model(out_axis, filter_shapes=[5], n_filters=[32], init=KaimingInit()):
    assert len(filter_shapes) == len(n_filters)

    layers = []
    for e, (f, n) in enumerate(zip(filter_shapes, n_filters)):
        layers.append(Convolution(filter_shape=(f, n), filter_init=init, strides=1, padding="valid", dilation=1, activation=Rectlin(), batch_norm=True))

    affine_layer = Affine(weight_init=init, bias_init=init,
                          activation=Identity(), axes=out_axis)

    model = Sequential(layers + [affine_layer])

    return model