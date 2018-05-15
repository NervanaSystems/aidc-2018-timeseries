"""
This script replicates some of the experiments run in the paper:
Bai, Shaojie, J. Zico Kolter, and Vladlen Koltun. "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling." arXiv preprint arXiv:1803.01271 (2018).
for music data
To compare with the original implementation, run
python ./music_forecasting_with_tcn.py --batch_size 32 --dropout 0.5 --epochs 2000 --ksize 3 --levels 2 --seq_len 100 --log_interval 2 --nhid 150 --lr 0.002 --grad_clip_value 0.4 --save_plots --results_dir ./ -b gpu
python ./music_forecasting_with_tcn.py --batch_size 32 --dropout 0.2 --epochs 2000 --levels 2 --seq_len 100 --log_interval 2 --nhid 200 --lr 0.002 --grad_clip_value 1 --save_plots --results_dir ./ -b gpu --modeltype LSTM

python ./music_forecasting_with_tcn.py --dataset Nott --batch_size 32 --dropout 0.2 --epochs 2000 --ksize 6 --levels 4 --seq_len 250 --log_interval 2 --nhid 150 --lr 0.002 --grad_clip_value 0.4 --save_plots --results_dir ./ -b gpu
python ./music_forecasting_with_tcn.py --dataset Nott --batch_size 32 --dropout 0.1 --epochs 2000 --levels 3 --seq_len 250 --log_interval 2 --nhid 280 --lr 0.004 --grad_clip_value 0.5 --save_plots --results_dir ./ -b gpu --modeltype LSTM
"""
from topologies.temporal_convolutional_network import tcn
from ngraph.frontends.neon import ArrayIterator
import ngraph as ng
from ngraph.frontends.neon import Adam, GradientDescentMomentum, Layer, Affine, Logistic, GaussianInit, Sequential
from topologies import recurrent_model
from training.timeseries_trainer import TimeseriesTrainer
from datasets.music import Music
import os
from utils.arguments import default_argparser

parser = default_argparser()
parser.add_argument('--datadir', type=str, default="../data/",
                    help='dir to download data if not already present')
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
parser.add_argument('--dataset', default='JSB', choices=['JSB', 'Nott'],
                        help='type of data to use (JSB, Nott)')
args = parser.parse_args()


hidden_sizes = [args.nhid]*args.levels
kernel_size = args.ksize
dropout = 1 - args.dropout # amount to keep
seq_len = args.seq_len
batch_size = args.batch_size
n_epochs = args.epochs

music_dataset = Music(data_dir=args.datadir, seq_len=seq_len, dataset=args.dataset)
seq_len = music_dataset.seq_len
n_train = music_dataset.train['X']['data'].shape[0]
num_iterations = int(n_train * n_epochs * 1.0 / batch_size)
n_features = music_dataset.train['X']['data'].shape[2]

train_iterator = ArrayIterator(music_dataset.train, batch_size, total_iterations=num_iterations, shuffle=True)
test_iterator = ArrayIterator(music_dataset.test, batch_size)


# Name and create axes
batch_axis = ng.make_axis(length=batch_size, name="N")
time_axis = ng.make_axis(length=seq_len, name="REC")
feature_axis = ng.make_axis(length=n_features, name="F")
out_axis = ng.make_axis(length=n_features, name="Fo")

in_axes = ng.make_axes([batch_axis, time_axis, feature_axis])
out_axes = ng.make_axes([batch_axis, time_axis, out_axis])

# Build placeholders for the created axes
inputs = dict(X=ng.placeholder(in_axes), y=ng.placeholder(out_axes),
              iteration=ng.placeholder(axes=()))

# define model
if args.modeltype == "TCN":
    affine_layer = Affine(axes=out_axis, weight_init=GaussianInit(0, 0.01), activation=Logistic())
    model = Sequential([lambda op: ng.map_roles(op, {'F': 'C', 'REC': 'W'})] + tcn(n_features, hidden_sizes, kernel_size=kernel_size, dropout=dropout).layers + [lambda op: ng.map_roles(op, {'C': 'F', 'W': 'REC'})] + [affine_layer])
elif args.modeltype == "LSTM":
    model = Sequential(recurrent_model.define_model(out_axis, celltype=args.modeltype, recurrent_units=hidden_sizes, return_sequence=True).layers + [Logistic()])

# Optimizer
if args.modeltype == "TCN":
    optimizer = Adam(learning_rate=args.lr, gradient_clip_value=args.grad_clip_value)
else:
    optimizer = GradientDescentMomentum(learning_rate=args.lr, gradient_clip_value=args.grad_clip_value)

# Define the loss function (categorical cross entropy, since each musical key on the piano is encoded as a binary value)
fwd_prop = model(inputs['X'])
fwd_prop = ng.axes_with_order(fwd_prop, out_axes)
train_loss = ng.cross_entropy_binary(fwd_prop, inputs['y'])

with Layer.inference_mode_on():
    preds = model(inputs['X'])
    preds = ng.axes_with_order(preds, out_axes)
eval_loss = ng.mean(ng.cross_entropy_binary(preds, inputs['y']), out_axes=())
eval_computation = ng.computation([eval_loss], "all")
predict_computation = ng.computation([preds], "all")


# Cost calculation
batch_cost = ng.sequential([optimizer(train_loss), ng.mean(train_loss, out_axes=())])
train_computation = ng.computation(batch_cost, "all")

trainer = TimeseriesTrainer(optimizer, train_computation, eval_computation, predict_computation, inputs, model_graph=[model], tensorboard_dir="./tfboard")
trainer.summary()

out_folder = os.path.join(args.results_dir, "results-music-{}-dataset-{}-modeltype-{}-batch_size-{}-dropout-{}-ksize-{}-levels-{}-seq_len-{}-nhid".format(args.dataset, args.modeltype, batch_size, args.dropout, kernel_size, args.levels, seq_len, args.nhid))
if not os.path.exists(out_folder):
    os.mkdir(out_folder)

trainer.train(train_iterator, test_iterator, n_epochs=args.epochs, log_interval=args.log_interval, save_plots=args.save_plots, results_dir=out_folder)
