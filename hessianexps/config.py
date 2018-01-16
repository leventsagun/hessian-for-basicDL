import argparse
import pandas as pd

def parse_command_line_arguments():
    parser = argparse.ArgumentParser()

    # Data Generation
    parser.add_argument('--data-type', type=str, choices=['blob', 'circle', 'moon'], default='blob',
                        help='Type of random data generation pattern.')
    parser.add_argument('--num-samples', type=int, default=100,
                        help='Number of training samples')
    parser.add_argument('--input-dim', type=int, default=2,
                        help='Dimension of the input space')
    parser.add_argument('--num-classes', type=int, default=2,
                        help='Number of classes in generated data. '
                             'Taken into account only by classifiers and data generators that support multi-class')
    parser.add_argument('--cov-factor', type=float, default=1.,
                        help='Multiplier for the covariance of the data')
    parser.add_argument('--data-seed', type=int, default=None,
                        help='Seed for random number generation of data generators')

    # Model
    parser.add_argument('--classifier', type=str, choices=['logreg', 'fullconn'], default='logreg',
                        help='Type of classifier. Logistic Regression, Fully-Connected NN.')
    parser.add_argument('--layer-sizes', nargs='*', type=int, default=[10, 5],
                        help='Number of units in hidden layers. '
                             'First layer will have --input-dim units. Last layer will have --num-classes units.')

    # Training
    parser.add_argument('--batch-size', type=int, default=0,
                        help='Number of samples in a training batch. '
                             '0 means whole dataset')
    parser.add_argument('--learning-rate', type=float, default=0.1,
                        help='Learning rate')
    parser.add_argument('--stopping-grad-norm', type=float, default=1e-4,
                        help='Stop training if grad_norm becomes smaller than this threshold')
    parser.add_argument('--max-iterations', type=int, default=1000000,
                        help='Cancel training after maximum number of iteration steps')

    # Results
    parser.add_argument('--hessian-calc-period', type=int, default=0,
                        help='Calculate hessian at every N iteration. '
                             '0 means do not calculate at intermediate iterations.')
    parser.add_argument('--results-folder', type=str, default='/Users/leventsagun/Dropbox/calismalar/second_order_explorations_NIPS2017',
                        help='Folder in which to put all results folders')
    parser.add_argument('--experiment-folder', type=str, default='defaults',
                        help='Folder in which to write results files')

    # Hessian analysis
    parser.add_argument('--top-evals', type=int, default=2,
                        help='Find top k evals')
    parser.add_argument('--bottom-evals', type=int, default=2,
                        help='Find bottom k evals')

    args = parser.parse_args()
   
    args.results_folder = args.results_folder + '/' + args.experiment_folder
    
    if args.classifier == 'logreg' and args.data_type == 'blob' and args.num_classes != 2:
        raise Exception('LogReg for more than 2 classes is not implemented yet.')
    return args
