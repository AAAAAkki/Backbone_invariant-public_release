Use of Pipelines
================

This package provides convenient functions to streamline common workflows for
cleaning protein chains, computing Backbone Rigid Invariants, and searching for
near-duplicates in groups.

This section introduces the available pipelines and demonstrates how they can be
used directly from the command line.

Overview
--------

The pipelines are designed to:

* Clean protein chain data from mmCIF files
* Compute Backbone Rigid Invariants (BRIs) in a reproducible manner
* Search for near-duplicate structures within or across datasets

All pipelines are exposed through a command-line interface (CLI), making them
easy to integrate into shell scripts and larger workflows.

.. caution::

   The pipelines default to utilizing all available CPU cores for computation. This may
   put a strain on the computer, so please evaluate the available resources before
   running these pipelines.

General Command-Line Usage
--------------------------

After installing the package, the main entry point can be accessed via::

    python -m bri --task <pipeline> [OPTIONS]

Where:

* ``<pipeline>`` specifies the workflow to execute
* ``[OPTIONS]`` control input files, output locations, and algorithm-specific
  parameters

You can list all available pipelines with::

    python -m bri --help

Pipeline 1: Cleaning Protein Chains
-----------------------------------

This pipeline applies standard cleaning steps, including selection of chains and
removal of chains broken by missing atoms and residues, and atom clashes.

Example
^^^^^^^
::

    python -m bri --task clean \
        --input-data data_list/entry_ids.txt \
        --output results/

Options
^^^^^^^

* ``--input-data``: Path to a directory or ``.txt`` file containing accessible protein
  structure data.
* ``--output``: Directory to save cleaned protein chains.
* ``--max-samples``: Number of entries to use from the input data. Default is to use all.
* ``--process``: Whether to show a progress bar. Disabled by default.

.. note::
   The cleaned-chains output ``chains_cleaned.csv`` contains one row per chain with
   the following columns:

   * pdb_id: entry name of the mmCIF file
   * entity_id: labelled entity id of the chain
   * model_id: model id (``pdbx_PDB_model_num``) of the chain
   * chain_id: chain id (``label_asym_id``)
   * start_residue: first residue number (``label_seq_id``) of the cleaned chain
   * chain_length: length of the cleaned chain
   * auth_chain_id: author-assigned chain id (``auth_asym_id``)
   * auth_seq_id_start: first author-assigned residue number (``auth_seq_id``)
   * auth_seq_id_end: last author-assigned residue number (``auth_seq_id``)
   * seq: amino acid sequence of the cleaned chain

   Entity information is not always present in protein structures (especially
   computed structures); those fields may be blank in such cases.

Outputs
^^^^^^^

* ``chains_cleaned.csv``: cleaned protein chains, written to the specified output
  directory.
* ``chains_filtered.csv``: chains removed during cleaning, written to the same
  directory.

Pipeline 2: Computing Backbone Rigid Invariants
-----------------------------------------------

This pipeline computes Backbone Rigid Invariants (BRIs) for cleaned protein
chains.

Example
^^^^^^^
::

    python -m bri --task bri \
        --input-data results/chains_cleaned.csv \
        --output results/bri/

Options
^^^^^^^

* ``--input-data``: A ``.csv`` file containing cleaned protein chains, a ``.txt`` file
  containing accessible entries, or a directory of mmCIF/bcif files.
* ``--output``: Directory to save BRIs separately for each chain, or a single file for
  all chains.
* ``--max-samples``: Number of entries to use from the input data. Default is to use all.
* ``--process``: Whether to show a progress bar. Disabled by default.
* ``--invariant-type``: Type of invariant to compute: ``bri`` (coordinate-based, the
  default) or ``lai`` (adds bond-length, bond-angle, and torsion-angle columns).

Outputs
^^^^^^^

* A single file containing all BRIs, or one file per chain saved separately in the
  target directory.

Pipeline 3: Searching for Near-Duplicates
-----------------------------------------

This pipeline searches for near-duplicate protein structures based on their BRIs.

Example
^^^^^^^
::

    python -m bri --task duplicate \
        --input-data results/chains_cleaned.csv \
        --output results/duplicates

Options
^^^^^^^

* ``--input-data``: A ``.csv`` file containing cleaned protein chains.
* ``--output``: If not specified, the full results of pairwise distances are saved
  into a directory grouped by chain length. Two summary tables are always saved,
  even without an output directory:

  1. ``L_inf_lt1_<date>.csv``: all pairs of structures with L-infinity distances
     less than 1.0 Ångström.
  2. ``L_inf_lt001_eq_seq_<date>.csv``: all pairs of structures with the same amino
     acid sequences and L-infinity distances less than 0.01 Ångström.
* ``--max-samples``: Number of entries to use from the input data. Default is to use all.

Outputs
^^^^^^^

* ``.csv`` files listing pairs of near-duplicate structures.

Additional directory-level tasks
--------------------------------

Besides the three pipelines above, the CLI offers tasks that operate on a directory
of previously computed invariant files. They share the ``--input-data`` (directory)
and ``--output`` options:

* ``inv``: Compute invariants for all structure files in a directory.
* ``compare``: Build distance matrices from a directory of invariant files.
* ``proj``: Produce statistical projections (BRI / LAI) from invariant files.
* ``plot``: Generate comparison-curve plots against a reference structure
  (additionally accepts ``--structure`` and ``--offset``).

See ``python -m bri --help`` for the authoritative list of tasks and options.

Chaining Pipelines
------------------

The pipelines are designed to be composable. For example, cleaning, invariant
computation, and duplicate search can be executed sequentially::

    python -m bri --task clean     --input-data entry_ids.txt              --output results/
    python -m bri --task bri       --input-data results/chains_cleaned.csv --output results/bri/
    python -m bri --task duplicate --input-data results/chains_cleaned.csv --output results/

Notes
-----

* Logs are printed and also saved to a ``logs/`` directory in the current working
  directory for troubleshooting.
