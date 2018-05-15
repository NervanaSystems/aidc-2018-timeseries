from ngraph.frontends.neon import NgraphArgparser
import os

def default_argparser():
    # parse the command line arguments
    parser = NgraphArgparser(__doc__)
    parser.add_argument('--predict_seq', default=False, dest='predict_seq', action='store_true',
                        help='If given, seq_len future timepoints are predicted')
    parser.add_argument('--look_ahead', type=int,
                        help="Number of time steps to start predicting from",
                        default=1)
    parser.add_argument('--seq_len', type=int,
                        help="Number of time points in each input sequence",
                        default=32)
    parser.add_argument('--log_interval', type=int, default=100, help="frequency, in number of iterations, after which loss is evaluated")
    parser.add_argument('--save_plots', action="store_true", help="save plots to disk")
    parser.add_argument('--results_dir', type=str, help="Directory to write results to", default='./')
    parser.add_argument('--resume', type=str, default=None, help="weights of the model to resume training with")
    parser.set_defaults()

    return parser