#!/usr/bin/env python3
"""Project-local entry point for qualified data, teacher caches and C/R training."""
import argparse
import sys
from _bundle import setup
setup()
from tools.trajectory_adaptation import main as adaptation_main


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] in ('cache', 'train') and not any(
            option in arguments for option in ('-h', '--help')):
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument('--device', default='cuda:0')
        parser.add_argument('--execute', action='store_true')
        selected, _ = parser.parse_known_args(arguments[1:])
        if selected.execute:
            import torch
            device = torch.device(selected.device)
            if device.type == 'cuda':
                torch.cuda.set_device(device)
    return adaptation_main(arguments)

if __name__ == '__main__':
    main()
