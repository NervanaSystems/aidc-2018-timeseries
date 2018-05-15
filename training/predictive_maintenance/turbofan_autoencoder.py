from datasets.turbofan import TurboFan
import ngraph as ng
from ngraph.frontends.neon import Layer, Affine, Identity
from ngraph.frontends.neon import GlorotInit, RMSProp
from ngraph.frontends.neon.layer import get_steps
from ngraph.frontends.neon import ArrayIterator
from utils.arguments import default_argparser
import os
from topologies import recurrent_model
from training.timeseries_trainer import TimeseriesTrainer
import numpy as np


parser = default_argparser()
parser.add_argument('--modeltype', default='LSTM', choices=['RNN', 'LSTM'],
                    help='type of model to use (RNN, LSTM)')
parser.add_argument('--skip', default=1, type=int, help='skip length for sliding window')
parser.add_argument('--n_hidden', default="128,256", type=str, help='hidden layers sizes in the encoder')
parser.add_argument('--bottleneck', default=False, action='store_true',
                    help='whether to use a bottleneck in the encoder-decoder model.')
parser.add_argument('--backward', default=False, action='store_true',
                    help='whether to reverse the target sequence in the autoencoder')
parser.add_argument('--lr', type=float, default=4e-3,
                    help='initial learning rate (default: 4e-3)')
parser.add_argument('--grad_clip_value', type=float, default=None,
                    help='value to clip each element of gradient')
parser.add_argument('--tensorboard_dir', type=str, default='./tensorboard',
                    help='directory to save tensorboard summary to')
args = parser.parse_args()
if args.predict_seq:
    raise ValueError("predict sequence is not available for turbofan use case")

if not os.path.exists(args.results_dir):
    os.mkdir(args.results_dir)

# Plot the inference / generation results
if args.save_plots:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        args.save_plots = False

# Define initialization
init_uni = GlorotInit()

batch_size = args.batch_size
seq_len = args.seq_len
no_epochs = args.epochs
output_dim = 1

dataset = TurboFan(data_dir="../../data/", T=args.seq_len, skip=args.skip, max_rul_predictable=130)
feature_dim = dataset.n_features


if args.save_plots:
    dataset.plot_sample(args.results_dir, trajectory_id=10)

# Build input data iterables
# Yields an input array of Shape (batch_size, seq_len, input_feature_dim)
train_samples = len(dataset.train['X']['data'])
num_iterations_per_epoch = train_samples // batch_size
num_iterations = (no_epochs * train_samples) // batch_size

# Name and create axes
batch_axis = ng.make_axis(length=batch_size, name="N")
time_axis = ng.make_axis(length=seq_len, name="REC")
feature_axis = ng.make_axis(length=feature_dim, name="F")
out_axis = ng.make_axis(length=1, name="Fo")

in_axes = ng.make_axes([batch_axis, time_axis, feature_axis])
rul_axes = ng.make_axes([batch_axis, out_axis])

# Build placeholders for the created axes
inputs = dict(X=ng.placeholder(in_axes),
              y=ng.placeholder(rul_axes))

Xs = get_steps(inputs['X'], time_axis)
if args.backward:
    target_steps = Xs[::-1]
    target = ng.stack(target_steps, time_axis)
else:
    target_steps = Xs
    target = inputs['X']

previous_steps = [ng.constant(0., [batch_axis, feature_axis])] + [target_steps[i] for i in range(seq_len - 1)]
previous = ng.stack(previous_steps, time_axis)

# define model
encoder_recurrent_units = list(map(int, args.n_hidden.split(",")))
if args.bottleneck:
    decoder_recurrent_units = encoder_recurrent_units[::-1]
else:
    decoder_recurrent_units = encoder_recurrent_units
encoder = recurrent_model.RecurrentEncoder(celltype=args.modeltype,
                                           recurrent_units=encoder_recurrent_units,
                                           bottleneck=args.bottleneck)
decoder = recurrent_model.RecurrentDecoder(out_axes=(feature_axis,), celltype=args.modeltype,
                                           recurrent_units=decoder_recurrent_units)

affine_layer = Affine(weight_init=init_uni, bias_init=init_uni, activation=Identity(),
                      axes=[out_axis])

# Optimizer
optimizer = RMSProp(gradient_clip_value=args.grad_clip_value, learning_rate=args.lr)


def predictions(encoder, affine_layer, inputs):
    encoded = encoder(inputs, combine=True)
    preds = affine_layer(encoded)
    preds = ng.axes_with_order(preds, rul_axes)
    return preds


def build_seq2seq_computations():
    # Training loss, optimizer
    train_decoded = recurrent_model.encode_and_decode(encoder, decoder,
                                                      inputs['X'], previous)
    train_loss = ng.squared_L2(target - train_decoded)
    batch_cost = ng.sequential([optimizer(train_loss), ng.mean(train_loss, out_axes=())])
    train_computation = ng.computation(batch_cost, "all")

    # Evaluation loss
    with Layer.inference_mode_on():
        eval_decoded = recurrent_model.encode_and_generate(encoder, decoder, inputs['X'], in_axes)
        eval_loss = ng.mean(ng.squared_L2(target - eval_decoded), out_axes=())
    loss_computation = ng.computation([eval_loss], "all")
    return train_computation, loss_computation


def build_regressor_computations():
    train_preds = predictions(encoder, affine_layer, inputs['X'])
    train_loss = ng.squared_L2(train_preds - inputs['y'])

    # Cost calculation
    batch_cost = ng.sequential([optimizer(train_loss), ng.mean(train_loss, out_axes=())])
    train_computation = ng.computation(batch_cost, "all")

    with Layer.inference_mode_on():
        eval_preds = predictions(encoder, affine_layer, inputs['X'])
        eval_loss = ng.mean(ng.squared_L2(eval_preds - inputs['y']), out_axes=())
    loss_computation = ng.computation([eval_loss], "all")

    return train_computation, loss_computation


def build_generator_computation():
    with Layer.inference_mode_on():
        generated = recurrent_model.encode_and_generate(encoder, decoder, inputs['X'], in_axes)
    return ng.computation([generated], "all")

def build_regressor_prediction():
    with Layer.inference_mode_on():
        eval_preds = predictions(encoder, affine_layer, inputs['X'])
    return ng.computation([eval_preds], "all")

def plot_generated(trainer):
    # Get a batch from the train set
    train_set_one_epoch = ArrayIterator(dataset.train, batch_size, shuffle=False)
    gen_series = trainer.predict(train_set_one_epoch, num_batches=1)
    train_set_one_epoch.reset()

    # Get an example from the batch
    gen_series = gen_series[4]

    if args.backward:
        # If args.backward is set, the autoencoder would have produced the input sequence in reverse.
        # We flip it again to match the true series
        gen_series = gen_series[::-1, :]

    true_series = next(train_set_one_epoch)['X'][4]

    # Plot the true and generated values of each series
    ncols = int(np.ceil((dataset.n_sensors + dataset.n_operating_modes) * 1.0 // 3))
    fig, ax = plt.subplots(ncols, 3)
    fig.set_figheight(20)
    fig.set_figwidth(10)

    for i in range(dataset.n_operating_modes):
        plt.subplot(ncols, 3, i + 1)
        if i == 0:
            plt.plot(true_series[:, i], label="true", color="blue")
        else:
            plt.plot(true_series[:, i], color="blue")
        if i == 0:
            plt.plot(gen_series[:, i], label="gen", color="red")
        else:
            plt.plot(gen_series[:, i], color="red")
        plt.title("Operating mode {}".format(i + 1))

    for i in range(dataset.n_sensors):
        plt.subplot(ncols, 3, dataset.n_operating_modes + i + 1)
        plt.plot(true_series[:, dataset.n_operating_modes + i], color="blue")
        plt.plot(gen_series[:, dataset.n_operating_modes + i], color="red")
        plt.title("Sensor {}".format(i + 1))
    fig.legend()

    plt.tight_layout()
    fig.savefig(os.path.join(args.results_dir, "generated_series.png"))


train_set = ArrayIterator(dataset.train, batch_size, total_iterations=num_iterations, shuffle=True)
test_set = ArrayIterator(dataset.test, batch_size)
train_set_one_epoch = ArrayIterator(dataset.train, batch_size, shuffle=False)

seq2seq_train_computation, seq2seq_loss_computation = build_seq2seq_computations()
seq2seq_gen_sequence_computation = build_generator_computation()

regressor_train_computation, regressor_loss_computation = build_regressor_computations()
regressor_predictions = build_regressor_prediction()

ae_trainer = TimeseriesTrainer(optimizer, seq2seq_train_computation, seq2seq_loss_computation, seq2seq_gen_sequence_computation, inputs,
                               model_graph=[encoder, decoder])
ae_trainer.summary()
ae_trainer.train(train_set, test_set, n_epochs=args.epochs/50, log_interval=args.log_interval, save_plots=True, results_dir=args.results_dir)

if args.save_plots:
    plot_generated(ae_trainer)

print('Start training the regression model')
reg_trainer = TimeseriesTrainer(optimizer, regressor_train_computation, regressor_loss_computation, regressor_predictions, inputs, model_graph=[encoder, decoder], tensorboard_dir=args.tensorboard_dir)
reg_trainer.train(train_set, test_set, n_epochs=args.epochs, log_interval=args.log_interval, save_plots=True, results_dir=args.results_dir)
