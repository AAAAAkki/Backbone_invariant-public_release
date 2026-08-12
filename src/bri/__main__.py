"""Main module for Backbone Rigid Invariant (BRI) evaluation tool.

This module provides the main functionality for running BRI evaluation tasks,
including data cleaning and BRI computation. It handles command-line arguments,
logging setup, and parallel processing of tasks.

Example:
    To run the cleaning task:
        $ python main.py --task clean --input-data path/to/data

    To run the BRI computation:
        $ python main.py --task bri --input-data path/to/data
"""

import bri.main as main

if __name__ == "__main__":
    main.main()
