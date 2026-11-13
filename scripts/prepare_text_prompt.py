#!/usr/bin/env python3
"""Project-local entry point: cache the umT5 features of a real prompt for the SR text slot."""
from _bundle import setup
setup()
from tools.prepare_sr_text_prompt import main

if __name__ == '__main__':
    main()
