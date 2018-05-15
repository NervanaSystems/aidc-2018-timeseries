import ngraph as ng
from ngraph.frontends.neon import Sequential, Recurrent, LSTM, Affine, SubGraph
from ngraph.frontends.neon import GlorotInit, Tanh, Logistic, Identity
from ngraph.frontends.neon.layer import get_steps

def define_recurrent_layers(out_axes=None, celltype='RNN', recurrent_units=[32], init=GlorotInit(), return_sequence=True):
    layers = []
    for e, i in enumerate(recurrent_units):
        layer_return_sequence = e < len(recurrent_units) - 1 or return_sequence
        if celltype == 'RNN':
            layers.append(Recurrent(nout=i, init=init, backward=False, activation=Tanh(),
                                    return_sequence=layer_return_sequence))
        elif celltype == 'LSTM':
            layers.append(LSTM(nout=i, init=init, backward=False, activation=Tanh(), gate_activation=Logistic(),
                               return_sequence=layer_return_sequence))
    if out_axes is not None:
        affine_layer = Affine(weight_init=init, bias_init=init,
                              activation=Identity(), axes=out_axes)
        layers.append(affine_layer)
    return layers

class RecurrentEncoder(Sequential):
    """
    This wrapper returns the final hidden states of all layers, allowing us to build multilayer seq2seq models.
    """
    def __init__(self, celltype='RNN', recurrent_units=[32], init=GlorotInit(), bottleneck=False, *args, **kwargs):
        layers = define_recurrent_layers(celltype=celltype,
                                         recurrent_units=recurrent_units,
                                         init=init,
                                         return_sequence=True)
        super(RecurrentEncoder, self).__init__(layers, *args, **kwargs)
        self.bottleneck = bottleneck

    @SubGraph.scope_op_creation
    def __call__(self, in_obj, combine=False, **kwargs):
        final_states = []
        for l in self.layers:
            in_obj = l(in_obj, **kwargs)
            recurrent_axis = in_obj.axes.recurrent_axis()
            final_state = get_steps(in_obj, recurrent_axis, backward=False)[-1]
            final_states.append(final_state)

        if self.bottleneck:
            final_states = final_states[::-1]

        if combine:
            if len(final_states) == 1:
                return final_states[0]
            else:
                batch_axis = final_states[0].axes.batch_axis()
                axes_list = [(state.axes - [batch_axis])[0] for state in final_states]
                combined = ng.ConcatOp(final_states, axes_list)
                return combined
        else:
            return final_states


class RecurrentDecoder(Sequential):
    """
    This wrapper allows us to pass initial states into all layers of a multilayer decoder.
    It also allows an affine readout layer to be placed at the end.
    """
    def __init__(self, out_axes=None, celltype='RNN', recurrent_units=[32], init=GlorotInit(), *args, **kwargs):
        layers = define_recurrent_layers(out_axes=out_axes,
                                         celltype=celltype,
                                         recurrent_units=recurrent_units,
                                         init=init,
                                         return_sequence=True)
        super(RecurrentDecoder, self).__init__(layers, *args, **kwargs)
        self.celltype = celltype
        self.recurrent_units = recurrent_units

    @SubGraph.scope_op_creation
    def __call__(self, inference, *args, **kwargs):
        if inference:
            return self.run_inference(*args, **kwargs)
        else:
            return self.run_training(*args, **kwargs)

    def run_training(self, in_obj, init_states, **kwargs):
        if self.celltype == 'LSTM':
            init_states = [(state, ng.constant(0., state.axes)) for state in init_states]

        for i, l in enumerate(self.layers):
            if i < len(init_states):
                in_obj = l(in_obj, init_state=init_states[i], **kwargs)
            else:
                in_obj = l(in_obj, **kwargs)
        return in_obj

    def run_inference(self, out_axes, init_states, **kwargs):
        if self.celltype == 'LSTM':
            init_states = [(state, ng.constant(0., state.axes)) for state in init_states]

        one_time_axis = ng.make_axis(1, name="REC")
        time_axis = out_axes.recurrent_axis()
        batch_axis = out_axes.batch_axis()
        feature_axis = (out_axes - [time_axis, batch_axis])[0]

        outputs = [ng.constant(0., [batch_axis, one_time_axis, feature_axis])]
        hidden_states = init_states

        for timestep in range(time_axis.length):
            in_obj = outputs[-1]

            # Compute the next hidden/cell states for the recurrent layers
            next_hidden_states = []
            for i, l in enumerate(self.layers[:-1]):
                if i < len(hidden_states):
                    init_state = hidden_states[i]
                else:
                    init_state = None

                if self.celltype == 'LSTM':
                    h, c = l(in_obj, init_state=init_state, return_cell_state=True)
                    in_obj = h

                    h = ng.slice_along_axis(h, one_time_axis, 0)
                    c = ng.slice_along_axis(c, one_time_axis, 0)
                    next_hidden_states.append((h, c))
                else:
                    h = l(in_obj, init_state=init_state)
                    in_obj = h

                    h = ng.slice_along_axis(h, one_time_axis, 0)
                    next_hidden_states.append((h, c))
            hidden_states = next_hidden_states

            # Compute the output of the affine layer
            in_obj = self.layers[-1](in_obj)
            outputs.append(in_obj)

        # Get rid of the initial 0 input
        outputs = outputs[1:]
        outputs = [ng.slice_along_axis(output, one_time_axis, 0) for output in outputs]
        outputs = ng.stack(outputs, time_axis)
        outputs = ng.axes_with_order(outputs, out_axes)
        return outputs


def define_model(out_axes=None, celltype='RNN', recurrent_units=[32], init=GlorotInit(), return_sequence=True):
    layers = define_recurrent_layers(out_axes=out_axes,
                                     celltype=celltype,
                                     recurrent_units=recurrent_units,
                                     init=init,
                                     return_sequence=return_sequence)
    return Sequential(layers)


def encode_and_decode(encoder, decoder, encoder_inputs, decoder_inputs):
    encoded_states = encoder(encoder_inputs, combine=False)
    decoded = decoder(inference=False, in_obj=decoder_inputs, init_states=encoded_states)
    return decoded


def encode_and_generate(encoder, decoder, encoder_inputs, out_axes):
    encoded_states = encoder(encoder_inputs, combine=False)
    return decoder(inference=True, out_axes=out_axes, init_states=encoded_states)
