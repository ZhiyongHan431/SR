#!/usr/bin/env python3
"""Project-local entry point for the CLIP semantic prompt: text-slot probe and projector training."""
import argparse
import sys
from _bundle import setup
setup()
from tools.train_semantic_prompt import main as semantic_main


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--execute', action='store_true')
    selected, _ = parser.parse_known_args(arguments)
    if selected.execute:
        import torch
        device = torch.device(selected.device)
        if device.type == 'cuda':
            torch.cuda.set_device(device)
    return semantic_main(arguments)


if __name__ == '__main__':
    main()
