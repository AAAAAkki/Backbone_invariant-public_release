# Backbone Rigid Invariant

## Introduction

**Backbone Rigid Invariant (BRI)** is a Python package for describing the 3D shape
of a protein backbone with numbers that **do not depend on how the protein is
oriented in space**. Because the descriptors are rotation- and translation-invariant,
two structures can be compared directly — no superposition required.

This package enables:

1. **Cleaning** a PDB/mmCIF dataset (from [RCSB](https://www.rcsb.org/)) with a defined
   set of rules to obtain qualified protein chains.
2. **Computing** Backbone Rigid Invariants (BRI, coordinate-based) and Length Angle
   Invariants (LAI, bond lengths / bond angles / torsion angles) for a given chain.
3. **Comparing** chains quickly — computing distances between invariants to find
   near-duplicate protein structures.

## Getting started

### Prerequisites

Python **>= 3.10**.

### Installation

Install from PyPI:

```shell
pip install backbone-rigid-invariant
```

Install from a built distribution in this repository:

```shell
pip install backbone_rigid_invariant-<version>.tar.gz
# or
pip install backbone_rigid_invariant-<version>-py3-none-any.whl
```

Then import the package with `import bri`.

### A quick example

Load a structure, compute its invariants, and visualise them:

```python
from bri import ProteinEntry, ProteinChain

# Load an entry by PDB ID (fetched from RCSB) or from a local file path / URL.
entry = ProteinEntry.from_cif("1hho")

# Compute the Backbone Rigid Invariant for every chain in the entry.
invariants = entry.get_entry_invariant()

# Or work with a single chain directly.
chain = ProteinChain.from_cif("1hho", model_id=1, chain_id="A")
bri = chain.get_invariant()                 # -> pandas.DataFrame, one row per residue
lai = chain.get_invariant(invariant_type="lai")  # add bond/torsion-angle columns
```

For a complete, step-by-step introduction, see the tutorial notebooks in [`examples/`](examples/).

<!-- ### Command-line pipelines -->
<!---->
<!-- Common workflows are also exposed through a command-line interface: -->
<!---->
<!-- ```shell -->
<!-- python -m bri --task clean     --input-data entry_ids.txt --output results/ -->
<!-- python -m bri --task bri       --input-data results/chains_cleaned.csv --output results/bri/ -->
<!-- python -m bri --task duplicate --input-data results/chains_cleaned.csv --output results/ -->
<!-- ``` -->

<!-- Run `python -m bri --help` to list all tasks.  -->
<!-- The pipelines are documented in the [usage guide](https://backbone-rigid-invariant.readthedocs.io). -->

## Documentation

Full API documentation is hosted at <https://backbone-rigid-invariant.readthedocs.io>.

<!-- ## Version history -->

> **Note on the commit history.** 
<!-- > This repository was transferred from a private project.  -->
> The Git history was reset when preparing the public release, so the log starts from the initial public commit. 
<!-- > The full record of changes is available in [CHANGELOG.md](CHANGELOG.md).  -->

<!-- ## License -->
<!---->
<!-- This project is licensed under -->
<!-- [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC-BY-NC-SA-4.0)](LICENSE). -->

## Citation

If you use this package in your research, please cite:

> Olga Anosova, Alexey Gorelov, William Jeffcott, Ziqiu Jiang, and Vitaliy Kurlin.
> *A Complete and Bi-Continuous Invariant of Protein Backbones under Rigid Motion.*
> MATCH Communications in Mathematical and in Computer Chemistry, 94(1):97–134, 2025.
> DOI: [10.46793/match.94-1.097A](https://doi.org/10.46793/match.94-1.097A)

BibTeX:

```bibtex
@article{anosova2025complete,
  title   = {A Complete and Bi-Continuous Invariant of Protein Backbones under Rigid Motion},
  author  = {Anosova, Olga and Gorelov, Alexey and Jeffcott, William and Jiang, Ziqiu and Kurlin, Vitaliy},
  journal = {MATCH Communications in Mathematical and in Computer Chemistry},
  volume  = {94},
  number  = {1},
  pages   = {97--134},
  year    = {2025},
  doi     = {10.46793/match.94-1.097A}
}
```
