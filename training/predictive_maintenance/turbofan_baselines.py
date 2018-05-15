"""
To run this script, use the command
python ./turbofan_baselines.py --batch_size 512 --seq_len 100 --modeltype LSTM --n_hidden 75,75 --epochs 200 --log_interval 100 --lr 0.002 --grad_clip_value 0.4 --save_plots --results_dir ./ -b gpu
"""
from datasets.turbofan import TurboFan
import ngraph as ng
from ngraph.frontends.neon import Layer
from ngraph.frontends.neon import GlorotInit, RMSProp, Sequential, Rectlin
from ngraph.frontends.neon import ArrayIterator
from utils.arguments import default_argparser
import os
from topologies import recurrent_model, convolutional_model
from training.timeseries_trainer import TimeseriesTrainer

parser = default_argparser()
parser.add_argument('--modeltype', default='LSTM', choices=['RNN', 'CNN', 'LSTM'],
                        help='type of model to use (RNN, CNN, LSTM)')
parser.add_argument('--skip', default=1, type=int, help='skip length for sliding window')
parser.add_argument('--n_hidden', default="128,256", type=str, help='hidden layers sizes')
parser.add_argument('--filter_shape', default="3,3", type=str, help='filter shape for cnn')
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

from datetime import datetime
out_folder = os.path.join(args.results_dir, "results-turbofan-LSTM-{}".format(datetime.strftime(datetime.now(), "%Y-%m-%d_%H%M%S")))
if not os.path.exists(out_folder):
    os.mkdir(out_folder)

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
    dataset.plot_sample(out_folder, trajectory_id=10)

# Build input data iterables
# Yields an input array of Shape (batch_size, seq_len, input_feature_dim)
train_samples = len(dataset.train['X']['data'])
num_iterations = (no_epochs * train_samples) // batch_size

train_set = ArrayIterator(dataset.train, batch_size, total_iterations=num_iterations, shuffle=True)
train_set_one_epoch = ArrayIterator(dataset.train, batch_size, shuffle=False)
test_set = ArrayIterator(dataset.test, batch_size)

# Name and create axes
batch_axis = ng.make_axis(length=batch_size, name="N")
time_axis = ng.make_axis(length=seq_len, name="REC")
feature_axis = ng.make_axis(length=feature_dim, name="F")
out_axis = ng.make_axis(length=output_dim, name="Fo")

in_axes = ng.make_axes([batch_axis, time_axis, feature_axis])
out_axes = ng.make_axes([batch_axis, out_axis])

# Build placeholders for the created axes
inputs = dict(X=ng.placeholder(in_axes), y=ng.placeholder(out_axes),
              iteration=ng.placeholder(axes=()))
preds_inputs = dict(X=inputs['X'])

# define model
n_hidden = list(map(int, args.n_hidden.split(",")))
filter_shape = list(map(int, args.filter_shape.split(",")))
if args.modeltype in ["RNN", "LSTM"]:
    seq1 = Sequential(recurrent_model.define_model(out_axis, celltype=args.modeltype, recurrent_units=n_hidden, return_sequence=args.predict_seq).layers + [Rectlin()])
elif args.modeltype == "CNN":
    seq1 = convolutional_model.define_model(out_axis, filter_shapes=filter_shape, n_filters=n_hidden)
    layers_modified = [lambda op: ng.map_roles(op, {'REC': 'W', 'F': 'C'})] + seq1.layers + [Rectlin()]
    seq1 = Sequential(layers_modified)

# Optimizer
optimizer = RMSProp(learning_rate=args.lr, gradient_clip_value=args.grad_clip_value)

# Define the loss function (squared L2 loss)
fwd_prop = seq1(inputs['X'])
train_loss = ng.squared_L2(fwd_prop - inputs['y'])

# Cost calculation
batch_cost = ng.sequential([optimizer(train_loss), ng.mean(train_loss, out_axes=())])
train_computation = ng.computation(batch_cost, "all")

# Forward prop of test set
# Required for correct functioning of batch norm and dropout layers during inference mode
with Layer.inference_mode_on():
    preds = seq1(inputs['X'])
    preds = ng.axes_with_order(preds, out_axes)
eval_loss = ng.mean(ng.squared_L2(preds - inputs['y']), out_axes=())
eval_computation = ng.computation([eval_loss], "all")
predict_computation = ng.computation([preds], "all")

trainer = TimeseriesTrainer(optimizer, train_computation, eval_computation, predict_computation, inputs, model_graph=[seq1],
                            tensorboard_dir=args.tensorboard_dir)
trainer.summary()


print("Starting training")
trainer.train(train_set, test_set, n_epochs=args.epochs, log_interval=args.log_interval, save_plots=args.save_plots, results_dir=out_folder)


if args.save_plots:
    # Compute the predictions on the training and test sets for visualization
    train_preds = trainer.predict(train_set_one_epoch)
    train_target = dataset.train['y']['data']

    test_preds = trainer.predict(test_set)
    test_target = dataset.test['y']['data']

    # Visualize the model's predictions on the training and test sets
    plt.figure()
    plt.scatter(train_preds[:, 0], train_target[:, 0])
    plt.xlabel('Training Predictions')
    plt.ylabel('Training Targets')
    plt.title('Predictions on training set')
    plt.savefig(os.path.join(out_folder, 'preds_training_output.png'))

    plt.figure()
    plt.scatter(test_preds[:, 0], test_target[:, 0])
    plt.xlabel('Validation Predictions')
    plt.ylabel('Validation Targets')
    plt.title('Predictions on validation set')
    plt.savefig(os.path.join(out_folder, 'preds_validation_output.png'))
