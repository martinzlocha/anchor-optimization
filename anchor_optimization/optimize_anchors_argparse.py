
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
from .optimize_anchors import anchors_optimize
import argparse
import sys


def parse_args(args):
    """ Parse the arguments.
    """

    parser = argparse.ArgumentParser(
        description='Optimize RetinaNet anchor configuration')
    parser.add_argument(
        'annotations',
        help='Path to CSV file containing annotations for anchor optimization.')
    parser.add_argument(
        '--scales',
        type=int,
        help='Number of scales.',
        default=3)
    parser.add_argument(
        '--ratios',
        type=int,
        help='Number of ratios, has to be an odd number.',
        default=3)
    parser.add_argument(
        '--include-stride',
        action='store_true',
        help='Should stride of the anchors be taken into account. Setting this to false will give '
        'more accurate results however it is much slower.')
    parser.add_argument(
        '--objective',
        type=str,
        default='focal',
        help='Function used to weight the difference between the target and proposed anchors. '
        'Options: focal, avg, ce.')
    parser.add_argument(
        '--popsize',
        type=int,
        default=15,
        help='The total population size multiplier used by differential evolution.')
    parser.add_argument(
        '--no-resize',
        help='Disable image resizing.',
        dest='resize',
        action='store_false')
    parser.add_argument(
        '--image-min-side',
        help='Rescale the image so the smallest side is min_side.',
        type=int,
        default=800)
    parser.add_argument(
        '--image-max-side',
        help='Rescale the image if the largest side is larger than max_side.',
        type=int,
        default=1333)
    parser.add_argument(
        '--seed',
        type=int,
        help='Seed value to use for differential evolution.')
    parser.add_argument(
        '--mutation',
        type=float,
        help='The mutation constant. If specified as a float it should be in the range [0, 2] '
        'see the documentation of "scipy.optimize.differential_evolution()". Another option '
        'is to use a tuple which then enables "dithering" which can speed the convergence significantly',
        default=0.5)
    parser.add_argument(
        '--threads',
        type=int,
        help='The number of threads to run in parallel. If > 1 then "updating" is set to "deferred" by default',
        default=1)
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Should information be printed out in the console.')

    return parser.parse_args(args)


def main(args=None):
    # parse arguments

    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    include_stride = False
    if args.include_stride:
        include_stride = True

    resize = False
    if args.resize:
        resize = True

    verbose = False
    if args.verbose:
        verbose = True

    seed = None
    if args.seed:
        seed = args.seed

    res_argp = anchors_optimize(annotations=args.annotations,
                                ratios=args.ratios,
                                scales=args.scales,
                                objective=args.objective,
                                popsize=args.popsize,
                                mutation=args.mutation,
                                image_min_side=args.image_min_side,
                                image_max_side=args.image_max_side,
                                # default SIZES values based on keras-retinanet
                                SIZES=[32, 64, 128, 256, 512],
                                # default STRIDES values based on keras-retinanet
                                STRIDES=[8, 16, 32, 64, 128],
                                include_stride=include_stride,
                                resize=resize,
                                threads=args.threads,
                                verbose=verbose,
                                seed=seed)


if __name__ == '__main__':
    main()
