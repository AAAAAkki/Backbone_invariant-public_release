# Examples

Three tutorial notebooks introduce the package step by step. They are designed to
be read in order — each one builds on the previous.

1. **[tutorial-1.ipynb](tutorial-1.ipynb)** — *Loading structures & computing invariants.*

   Load a protein structure (by PDB ID, file path, or URL), inspect its chains and
   residues, compute the Backbone Rigid Invariant (BRI) and Local Angle Invariant
   (LAI), and visualise a single chain with the Backbone Invariant Diagram (BID) and
   Backbone Invariant Barcode (BIB).

2. **[tutorial-2.ipynb](tutorial-2.ipynb)** — *Comparing multiple chains.*

   Compare the backbone geometry of two or more chains, quantify how different they
   are, and visualise the results as distance matrices and projections (including an
   NMR ensemble example).

3. **[tutorial-3.ipynb](tutorial-3.ipynb)** — *Finding similar structures with a k-d tree.*

   Scale comparison up to database size: index a set of structures and query for the
   nearest neighbours of a given chain, instead of brute-force all-pairs comparison.

## Usage

> Make sure Python ≥ 3.10 is available and the dependencies in
> [requirements.txt](../requirements.txt) (or the `bri` package) are installed.

Run a notebook with:

```bash
jupyter notebook tutorial-1.ipynb
```

or open it in [JupyterLab](https://jupyter.org/) / VS Code / Google Colab.
