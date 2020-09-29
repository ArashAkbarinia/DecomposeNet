import argparse


def parse_arguments(args):
    parser = argparse.ArgumentParser(description='VQ-VAE')
    model_parser = parser.add_argument_group('Model Parameters')
    model_parser.add_argument('--model', default='vqvae')
    model_parser.add_argument(
        '--batch-size', type=int, default=32,
        help='input batch size for training (default: 32)'
    )
    model_parser.add_argument(
        '--target_size', type=int, help='image target size'
    )
    model_parser.add_argument(
        '-j', '--workers', type=int, default=18,
        help='Number of workers for image generator (default: 1)'
    )
    model_parser.add_argument('--num_channels', type=int,
                              help='number of input channels')
    model_parser.add_argument('--hidden', type=int,
                              help='number of hidden channels')
    model_parser.add_argument('-k', '--k', type=int, dest='k',
                              help='number of vectors in embedding space')
    model_parser.add_argument('-kl', '--kl', type=int, dest='kl',
                              help='length of vector in space')
    model_parser.add_argument('--lr', type=float, default=None,
                              help='learning rate')
    model_parser.add_argument('--vq_coef', type=float, default=None,
                              help='vq coefficient in loss')
    model_parser.add_argument('--commit_coef', type=float, default=None,
                              help='commitment coefficient in loss')
    model_parser.add_argument('--kl_coef', type=float, default=None,
                              help='kl-divergence coefficient in loss')
    parser.add_argument(
        '--resume', type=str, default=None,
        help='The path to previous training to be resumed (default: None).'
    )
    parser.add_argument(
        '-ft', '--fine_tune', type=str, default=None,
        help='The path to weights to be fine-tuned (default: None).'
    )
    parser.add_argument('--colour_space', type=str, default=None,
                        help='The type of output colour space.')

    training_parser = parser.add_argument_group('Training Parameters')
    training_parser.add_argument('--dataset', default='imagenet')
    training_parser.add_argument(
        '--data_dir', type=str, default=None,
        help='The path to the data directory (default: None)'
    )
    training_parser.add_argument(
        '--train_dir', type=str, default=None,
        help='The path to the train directory (default: None)'
    )
    training_parser.add_argument(
        '--validation_dir',
        type=str,
        default=None,
        help='The path to the validation directory (default: None)'
    )
    training_parser.add_argument(
        '--epochs', type=int, default=90, metavar='N',
        help='number of epochs to train (default: 10)'
    )
    training_parser.add_argument(
        '--start_epoch', type=int, default=0, metavar='N',
        help='The initial epoch (default: 0)'
    )
    training_parser.add_argument('--max-epoch-samples', type=int, default=50000,
                                 help='max num of samples per epoch')
    training_parser.add_argument('--no-cuda', action='store_true',
                                 default=False, help='enables CUDA training')
    training_parser.add_argument('--seed', type=int, default=1, metavar='S',
                                 help='random seed (default: 1)')
    training_parser.add_argument('--gpus', default='0',
                                 help='gpus used for training - e.g 0,1,3')

    logging_parser = parser.add_argument_group('Logging Parameters')
    logging_parser.add_argument(
        '--log-interval', type=int, default=10,
        help='how many batches to wait before logging training status'
    )
    logging_parser.add_argument('--results-dir', metavar='RESULTS_DIR',
                                default='./results', help='results dir')
    logging_parser.add_argument('--save-name', default='',
                                help='saved folder')
    logging_parser.add_argument('--data-format', default='json',
                                help='in which format to save the data')

    return parser.parse_args(args)
