import torch
import argparse
import warnings

from HyPER.training import Train
from HyPER.utils import Settings


def argparser():
    parser = argparse.ArgumentParser(description='Build Graph datasets.')
    parser.add_argument('-c', '--config', type=str, required=True, help='Configuration/settings file.')
    parser.add_argument('--resume', type=str, required=False, default=None, help='Resume training state from the checkpoint.')
    return parser.parse_args()


if __name__ == '__main__':
    args = argparser()

    settings = Settings()
    if args.config is None:
        warnings.warn(UserWarning, "Using default settings, make sure they are what you wanted.")
    else:
        settings.load(args.config)
    settings.show()

    if settings.in_memory_dataset is False:
        torch.multiprocessing.set_sharing_strategy('file_system')

    Train(settings=settings, ckpt_path=args.resume)