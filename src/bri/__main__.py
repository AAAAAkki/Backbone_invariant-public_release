"""Entry point for Backbone Rigid Invariant (BRI) evaluation tool.

This module allows running the BRI CLI via ``python -m bri``.

Example::

    $ python -m bri inv pdbs/ output/
"""

from bri.cli import cli

if __name__ == "__main__":
    cli()
