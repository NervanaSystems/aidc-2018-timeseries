"""
This script replicates some of the experiments run in the paper:
Bai, Shaojie, J. Zico Kolter, and Vladlen Koltun. "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling." arXiv preprint arXiv:1803.01271 (2018).
for the synthetic "adding" data
To compare with the original implementation, run
python ./adding_with_tcn.py --batch_size 32 --dropout 0.0 --epochs 20 --ksize 6 --levels 7 --seq_len 200 --log_interval 100 --nhid 27 --lr 0.002 --save_plots --results_dir ./ -b gpu
python ./adding_with_tcn.py --batch_size 32 --dropout 0.0 --epochs 20 --ksize 7 --levels 7 --seq_len 400 --log_interval 100 --nhid 27 --lr 0.002 --save_plots --results_dir ./ -b gpu
python ./adding_with_tcn.py --batch_size 32 --dropout 0.0 --epochs 20 --ksize 8 --levels 8 --seq_len 600 --log_interval 100 --nhid 24 --lr 0.002 --save_plots --results_dir ./ -b gpu
python ./adding_with_tcn.py --batch_size 32 --dropout 0.0 --epochs 20 --levels 2 --seq_len 200 --nhid 77 --modeltype LSTM --grad_clip_value 50 --save_plots --lr 0.002 --results_dir ./ --log_interval 1000 -b gpu
python ./adding_with_tcn.py --batch_size 32 --dropout 0.0 --epochs 20 --levels 2 --seq_len 400 --nhid 77 --modeltype LSTM --grad_clip_value 50 --save_plots --lr 0.002 --results_dir ./ --log_interval 1000 -b gpu
python ./adding_with_tcn.py --batch_size 32 --dropout 0.0 --epochs 20 --levels 1 --seq_len 600 --nhid 130 --modeltype LSTM --grad_clip_value 5 --save_plots --lr 0.002 --results_dir ./ --log_interval 1000 -b gpu
"""
from topologies.temporal_convolutional_network import tcn
from ngraph.frontends.neon.layer import Affine
from ngraph.frontends.neon import Identity, GaussianInit
from ngraph.frontends.neon import ArrayIterator, Sequential
import ngraph as ng
from ngraph.frontends.neon import Adam, GradientDescentMomentum, Layer
from training.timeseries_trainer import TimeseriesTrainer
from topologies import recurrent_model
import argparse
from datasets.adding import Adding
import os
from utils.arguments import default_argparser

parser = default_argparser()
parser.add_argument('--dropout', type=float, default=0.0,
                    help='dropout applied to layers (default: 0.0)')
parser.add_argument('--ksize', type=int, default=7,
                    help='kernel size (default: 7)')
parser.add_argument('--levels', type=int, default=8,
                    help='# of levels (default: 8)')
parser.add_argument('--lr', type=float, default=4e-3,
                    help='initial learning rate (default: 4e-3)')
parser.add_argument('--nhid', type=int, default=30,
                    help='number of hidden units per layer (default: 30)')
parser.add_argument('--grad_clip_value', type=float, default=None,
                    help='value to clip each element of gradient')
parser.add_argument('--modeltype', default='TCN', choices=['TCN', 'LSTM'],
                        help='type of model to use (TCN, LSTM)')
args = parser.parse_args()


n_features = 2
hidden_sizes = [args.nhid]*args.levels
kernel_size = args.ksize
n_classes = 1
dropout = 1.0 - args.dropout  # fraction to keep
seq_len = args.seq_len
n_train = 50000
n_val = 1000
batch_size = args.batch_size
n_epochs = args.epochs
num_iterations = int(n_train * n_epochs * 1.0 / batch_size)


adding_dataset = Adding(T=seq_len, n_train=n_train, n_test=n_val)
train_iterator = ArrayIterator(adding_dataset.train, batch_size, total_iterations=num_iterations, shuffle=True)
test_iterator = ArrayIterator(adding_dataset.test, batch_size)

# Name and create axes
batch_axis = ng.make_axis(length=batch_size, name="N")
time_axis = ng.make_axis(length=seq_len, name="REC")
feature_axis = ng.make_axis(length=n_features, name="F")
out_axis = ng.make_axis(length=n_classes, name="Fo")

in_axes = ng.make_axes([batch_axis, feature_axis, time_axis])
out_axes = ng.make_axes([batch_axis, out_axis])

# Build placeholders for the created axes
inputs = dict(X=ng.placeholder(in_axes), y=ng.placeholder(out_axes),
              iteration=ng.placeholder(axes=()))

# define model
if args.modeltype == "TCN":
     # take only the last timepoint of output sequence to predict sum
    last_timepoint = [lambda op: ng.tensor_slice(op, [slice(seq_len-1, seq_len, 1) if ax.name == "W" else slice(None) for ax in op.axes])]
    affine_layer = Affine(axes=out_axis, weight_init=GaussianInit(0, 0.01), activation=Identity())

    model = Sequential([lambda op: ng.map_roles(op, {'REC': 'W', 'F': 'C'})] + tcn(n_features, hidden_sizes, kernel_size=kernel_size, dropout=dropout).layers + last_timepoint + [affine_layer])
elif args.modeltype == "LSTM":
    model = recurrent_model.define_model(out_axis, celltype=args.modeltype, recurrent_units=hidden_sizes, return_sequence=False)

# Optimizer
if args.modeltype == "TCN":
    optimizer = Adam(learning_rate=args.lr, gradient_clip_value=args.grad_clip_value)
else:
    optimizer = GradientDescentMomentum(learning_rate=args.lr, gradient_clip_value=args.grad_clip_value)

# Define the loss function (squared L2 loss)
fwd_prop = model(inputs['X'])
train_loss = ng.squared_L2(fwd_prop - inputs['y'])
with Layer.inference_mode_on():
    preds = model(inputs['X'])
    preds = ng.axes_with_order(preds, out_axes)
eval_loss = ng.mean(ng.squared_L2(preds - inputs['y']), out_axes=())
eval_computation = ng.computation([eval_loss], "all")
predict_computation = ng.computation([preds], "all")


# Cost calculation
batch_cost = ng.sequential([optimizer(train_loss), ng.mean(train_loss, out_axes=())])
train_computation = ng.computation(batch_cost, "all")

trainer = TimeseriesTrainer(optimizer, train_computation, eval_computation, predict_computation, inputs, model_graph=[model], tensorboard_dir="./tfboard")
trainer.summary()

out_folder = os.path.join(args.results_dir, "results-adding-{}-modeltype-{}-batch_size-{}-dropout-{}-ksize-{}-levels-{}-seq_len-{}-nhid".format(args.modeltype, batch_size, args.dropout, kernel_size, args.levels, seq_len, args.nhid))
if not os.path.exists(out_folder):
    os.mkdir(out_folder)
trainer.train(train_iterator, test_iterator, n_epochs=args.epochs, log_interval=args.log_interval, save_plots=args.save_plots, results_dir=out_folder)
