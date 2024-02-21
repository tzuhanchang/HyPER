import argparse
import uproot

from HyPER.utils import Settings
from HyPER.graphs import BuildTorchGraphs


def argparser():
    parser = argparse.ArgumentParser(description='Build Graph datasets.')
    parser.add_argument('-f', '--file',   type=str, required=True, help='ROOT input file.')
    parser.add_argument('-c', '--config', type=str, required=True, help='Configuration/settings file.')
    parser.add_argument('-o', '--output', type=str, required=True, help='Graphs output directory.')
    parser.add_argument('-t', '--tree',   type=str, required=False, default='nominal_Loose', help='Tree name of the given ROOT dataset')
    parser.add_argument('--truth', type=bool, required=False, default=True, help='Build truth graphs along with the source graphs.')
    return parser.parse_args()


def ConfiguredBuild():
    args = argparser()

    # load settings:
    settings = Settings()
    settings.load(args.config)
    settings.show(graphs=True)

    f = uproot.open(args.file)[args.tree]
    dataset = BuildTorchGraphs(f.arrays(), settings, build_target=args.truth)
    dataset.save_to(args.output)


if __name__ == '__main__':
    ConfiguredBuild()