import ngraph as ng
from ngraph.frontends.neon.layer import Layer
from ngraph.frontends.neon.graph import SubGraph
from ngraph.frontends.neon.axis import shadow_axes_map, reorder_spatial_axes
from ngraph.frontends.neon.layer import ConvBase, Convolution, LABELS
from ngraph.frontends.neon import GaussianInit
import six


class Dropout2D(Layer):
    """
    Layer for stochastically zero-out entire channels to prevent overfitting
    Arguments:
        keep (float):  Number between 0 and 1 that indicates probability of any particular
                       activation being kept.  Defaults to 0.5.
    Example:
        .. code-block:: python
        # Place a Dropout layer between two conv layers
        layers = [
            Convolution(nout=2048, activation=Rectlin()),
            Dropout2D(keep=0.6), # zeroes about 820 channels
            Convolution(nout=2048, activation=Rectlin())
        ]
    """
    def __init__(self, keep=0.5, **kwargs):
        super(Dropout2D, self).__init__(**kwargs)
        self.keep = keep
        self.mask = None

    @SubGraph.scope_op_creation
    def __call__(self, in_obj, **kwargs):
        if Layer.inference_mode:
            return self.keep * in_obj
        else:
            if self.mask is None:
                in_axes = in_obj.axes.sample_axes()
                channel_axes = ng.make_axes([in_axes.channel_axis()])
                self.mask = ng.persistent_tensor(axes=channel_axes).named('channel_mask')
            self.mask = ng.uniform(self.mask, low=0.0, high=1.0) <= self.keep
            return self.mask * in_obj


class DilatedCausalConvBase(ConvBase):
    def __init__(self, *args, **kwargs):
        super(DilatedCausalConvBase, self).__init__(*args, **kwargs)

        self.weight_norm = True
        self.W = None

        self.g = None
        self.v = None

    @SubGraph.scope_op_creation
    def __call__(self, in_obj, channel_axes="C", spatial_axes=("D", "H", "W"), **kwargs):
        """
        Arguments:
            in_obj (Op): Input op
            channel_axes (str): name of the expected channel axis type - defaults to "C"
            spatial_axes (tuple): names of expected depth, height and width axis types - defaults
                                  to "D", "H", and "W"
        """
        if isinstance(spatial_axes, dict):
            spatial_axes = tuple(spatial_axes.get(name, name) for name in ("D", "H", "W"))
        elif isinstance(spatial_axes, tuple):
            if len(spatial_axes) < 3:
                raise ValueError("spatial_axes must have length 3 (e.g. ('D', 'H', 'W'))")
            spatial_axes = tuple(name if name else default for name, default in zip(spatial_axes, ("D", "H", "W")))

        orig_axes = in_obj.axes
        in_obj = reorder_spatial_axes(in_obj, channel_axes, spatial_axes)
        channel_axes = in_obj.axes.get_by_names(channel_axes)
        spatial_axes = in_obj.axes.get_by_names(*spatial_axes)

        filter_axes = self._filter_axes(channel_axes, spatial_axes)

        # mark 'K' as a shadow axis for the initializers.
        axes_map = shadow_axes_map(filter_axes.find_by_name('K'))
        filter_axes = ng.make_axes([axis if axis.name != 'K' else list(axes_map.keys())[0] for axis in filter_axes])

        if not self.initialized:
            if not self.weight_norm:
                self.W = ng.variable(axes=filter_axes, initial_value=self.init, metadata={"label": LABELS["weight"]}).named("W")
            else:
                self.v = ng.variable(axes=filter_axes, initial_value=self.init, metadata={"label": LABELS["weight"]}).named("v")
                out_axes = ng.make_axes([filter_axes.get_by_names("K__NG_SHADOW")])
                v_norm = ng.mean(ng.square(self.v), out_axes=out_axes)
                self.g = ng.variable(axes=out_axes, initial_value=self.init, metadata={"label": LABELS["weight"]}).named("g")
                self.W = self.g * self.v * ng.reciprocal(ng.sqrt(v_norm + 1e-3))
        else:
            if filter_axes != self.W.axes:
                raise ValueError(("{layer_name} layer has already been initialized with an "
                                  "input object which has resulted in filter axes: "
                                  "{existing_filter_axes}. This new input object has axes: "
                                  "{input_axes}, which implies the need for filter axes: "
                                  "{new_filter_axes} which are different than the existing "
                                  "filter axes.").format(layer_name=self.name, existing_filter_axes=self.W.axes, input_axes=in_obj.axes, new_filter_axes=filter_axes, ))

        output = ng.map_roles(self._conv_op(in_obj, channel_axes, spatial_axes), axes_map)
        # Reorder the output to match the input order
        output_axis_order = ng.make_axes([output.axes.find_by_name(ax.name)[0] for ax in orig_axes])
        # Remove introduced axes. If their length is > 1, then perhaps they should be kept
        slices = [0 if (ax not in orig_axes) and ax.length == 1 else slice(None) for ax in output.axes]
        output = ng.tensor_slice(output, slices)
        # New axes with length > 1 may have been introduced. Add them to the end.
        output_axis_order = output_axis_order | output.axes
        return ng.axes_with_order(output, output_axis_order)


def make_dilated_causal_conv(filter_shape, init, strides, padding, dilation, **kwargs):
    default_filter_shape = {k: 1 for k in "DHWK"}
    if isinstance(filter_shape, (list, tuple)):
        if (len(filter_shape) < 2) or (len(filter_shape) > 4):
            raise ValueError("If filter_shape is a list, its length should be between 2 and 4, "
                             "specifying the filter size for 1 to 3 spatial dimensions and the "
                             "number of filters. Provided: {}".format(filter_shape))
        axis_names = {2: "WK", 3: "HWK", 4: "DHWK"}[len(filter_shape)]
        default_filter_shape.update(list(zip(axis_names, filter_shape)))
        filter_shape = default_filter_shape
    else:
        axis_names = filter_shape.keys()
    if isinstance(strides, int):
        strides = {k: strides for k in axis_names if k != "K"}
    if isinstance(padding, (int, six.string_types, tuple)):
        padding = {k: padding for k in axis_names if k != "K"}
    if isinstance(dilation, int):
        dilation = {k: dilation for k in axis_names if k != "K"}

    return DilatedCausalConvBase(filter_shape, init, strides, padding, dilation, **kwargs)


class DilatedCausalConv(Convolution):
    def __init__(self, filter_shape, filter_init, strides=1, padding=0, dilation=1, bias_init=None, activation=None, batch_norm=False, **kwargs):
        super(DilatedCausalConv, self).__init__(filter_shape, filter_init, strides=strides, padding=padding, dilation=dilation, bias_init=bias_init, activation=activation, batch_norm=batch_norm, **kwargs)
        self._make_dilated_causal_conv_layer(filter_shape, filter_init, strides, padding, dilation, **kwargs)

        self.weight_norm = True

    def _make_dilated_causal_conv_layer(self, filter_shape, filter_init, strides, padding, dilation, **kwargs):
        self.conv = make_dilated_causal_conv(filter_shape, filter_init, strides, padding, dilation, **kwargs)