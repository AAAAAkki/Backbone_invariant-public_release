"""Backbone Rigid Invariant (BRI) - A Python Package for Protein Structure Analysis

The BRI package provides tools for analyzing protein structures using geometric
descriptors based on backbone rigid invariants. It enables:

- Extraction and cleaning of protein structure data from PDB/mmCIF files
- Computation of backbone rigid invariants (BRI)
- Statistical analysis of BRI distributions
- Structural comparison using BRI descriptors
- Visualization of BRI patterns through diagrams and barcodes

Main Components
--------------
====================      ==============================================
Module                    Description
====================      ==============================================
:mod:`.structure`         Protein structure data model (Entry/Chain/Residue/Atom)
:mod:`.invariant`         BRI and LAI computation
:mod:`.invariant_compare` Structural comparison and near-duplicate search
:mod:`.filter`            Structure validation and cleaning
:mod:`.clash_filter`      Atom-clash detection
:mod:`.base`              Core utilities and mathematical functions
====================      ==============================================

Example Usage
------------
>>> from bri import ProteinEntry, ProteinChain
>>> # Load a structure by PDB ID (also accepts a file path or URL)
>>> entry = ProteinEntry.from_cif("1hho")
>>> # Compute the Backbone Rigid Invariant for all chains
>>> invariants = entry.get_entry_invariant()
>>> # Work with a single chain
>>> chain = ProteinChain.from_cif("1hho", model_id=1, chain_id="A")
>>> bri = chain.get_invariant()
>>> chain.generate_BID()

For more information and documentation, visit:
https://backbone-rigid-invariant.readthedocs.io
"""

__version__ = "1.3.1"
__author__ = "Ziqiu Jiang"
__maintainer__ = "Ziqiu Jiang"
__email__ = "jiangziqiu02@gmail.com"
__license__ = "CC-BY-NC-SA-4.0"
__copyright__ = "Copyright 2024, Ziqiu Jiang"

import importlib

# routing map: {"function_name": "relative.module_path"}
_LAZY_ROUTER = {
    "integrated_chainwise_filter": ".filter",
    "group_invariant_compare": ".invariant_compare",
}

# __all__ = list(_LAZY_ROUTER.keys())


# lazy loader
def __getattr__(name: str):
    if name in _LAZY_ROUTER:
        module_path = _LAZY_ROUTER[name]
        module = importlib.import_module(module_path, package=__name__)
        return getattr(module, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


from .structure import Residue, ProteinChain, ProteinEntry
