import argparse
from os import path
from pylpr.data import generate_files
from pylpr.model import LPR_model

def main():
    parser = argparse.ArgumentParser(prog="pyLPR", description="LPRules model setup")
    group_args = parser.add_argument_group('ARGS')
    group_args.add_argument('dataset_dir', type=str, metavar='path',
                        help="Relative path to dataset directory")
    group_options = parser.add_argument_group('OPTIONS')
    group_options.add_argument('-f', '--files_update', action='store_true',
                        help="Update files containing unique entities and relations (default: False)")
    group_options.add_argument('--rules_load', action='store_true',
                        help='Load rules from file instead of generating new rules (default: False)')
    group_options.add_argument('--rules_file', default='rules.csv', type=str,
                        help="Path to file with final rules (default: \'rules.csv\')")
    group_options.add_argument('--column_generation', action='store_true',
                        help='Use column generation to find rules (default: False)')
    group_options.add_argument('--rules_file_temp', default='rules.npy', type=str,
                        help="Path to rules rules used during training (default: \'rules.npy\')")
    group_options.add_argument('--skip_writing', action='store_true',
                        help="Skip writing rules to file (default: False)")
    group_options.add_argument('--skip_neg', action='store_true',
                        help="Skips neg claculation for rules (default: False)")
    group_options.add_argument('--skip_weight', action='store_true',
                        help="Skips help calculation for rules (default: False)")
    group_options.add_argument('-S', '--solver', default='PULP_CBC_CMD', type=str,
                               help="Solver for linear problem (default: CBC)")
    group_options.add_argument('-i', '--iterations', default=20, type=int,
                               help="Iterations of solving LP per relation (default: 20)")
    group_options.add_argument('-c', '--cores', default=None, type=int,
                               help="Number of cores used for processing (default: All)")
    group_options.add_argument('-l', '--max_length', default=4, type=int,
                               help="Max rule length (default: 4)")
    group_options.add_argument('--seed', default=12345, type=int,
                               help="Seed for random number generator (default: 12345)")

    args = parser.parse_args()

    dataset_dir_path = path.abspath(path.join(path.dirname(__file__), args.dataset_dir))

    if args.files_update:
        print('Updating entities and relations files... ', end='')
        generate_files(dataset_dir_path)
        print('done')

    tradeoff = [0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06]
    # lpr = LPR_model(dataset_dir_path, tradeoff, args)
    # lpr.fit()
    # lpr.predict()
    lpr = LPR_model(dataset_dir_path, tradeoff, args)
    lpr.fit()

if __name__ == "__main__":
    main()
