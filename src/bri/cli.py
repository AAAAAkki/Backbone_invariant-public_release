"""Command-line interface for the BRI evaluation tool.

This module provides click-based subcommands for cleaning protein structure
files, computing backbone rigid invariants (BRI), building distance matrices,
and generating statistical projections and comparison plots.

Example::

    $ bri inv pdbs/ output/
    $ bri pipe pdbs/ output/ -n 4
"""

from __future__ import annotations

import datetime
import functools
import json
import logging
import multiprocessing as mp
from pathlib import Path
import sys
import time
from typing import Any

import click
import pandas as pd
import tqdm

from bri.structure import ProteinEntry
from bri.invariant import InvariantType
from bri.invariant_compare import group_invariant_compare
from bri.filter import entry_integrated_cleaning, MINI_ENTRY_CLEAN_COL
from bri.workflow import (
    compute_dir_invariants,
    compute_distance_matrix,
    scatter_projection,
    comparison_plots,
)

# ─── Logging ────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("BRI")

_DEFAULT_N_PROCESS = max(1, mp.cpu_count() // 2)


def _begin_task(name: str) -> float:
    """Set up file logging, log the task header, and return the start time.

    :param name: Name of the task being started.
    :return: Monotonic start time (for elapsed-time logging).
    """
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    handler = logging.FileHandler(log_dir / f"eval_{timestamp}.log", encoding="utf-8")
    handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(handler)

    logger.info(f"Start at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 40)
    logger.info(f"Available CPUs: {mp.cpu_count()}")
    logger.info(f"Start running task: {name}")
    return time.time()


# ═══════════════════════════════════════════════════════════════════
#  Input loading
# ═══════════════════════════════════════════════════════════════════


def load_input(
    input_path: str | Path, start: int = 0, end: int | None = None
) -> list[str] | list[dict[str, str | int]] | list[Path] | pd.DataFrame:
    """Load input data from a ``.txt``, ``.json``, ``.csv`` file or directory.

    :param input_path: Path to the input file or directory.
    :param start: Starting index for the data slice.
    :param end: Number of items to take (relative to *start*).
    :return: Processed input data (list or DataFrame).
    :raises ValueError: If the file type is not supported.
    """
    end = end + start if end else None
    ext = Path(input_path).suffix.lower()

    if ext == ".txt":
        with open(input_path, "r") as file:
            text = file.read().strip("\n")
            return text.split(",")[start:end]
    elif ext == ".json":
        with open(input_path, "r") as file:
            return json.load(file)[start:end]
    elif ext == ".csv":
        return pd.read_csv(input_path, na_values="", keep_default_na=False).iloc[
            start:end, :
        ]
    elif ext == "":
        all_paths = list(Path(input_path).glob("*"))
        return all_paths[start:end]
    else:
        raise ValueError(f"Unsupported file type: {ext}")


# ═══════════════════════════════════════════════════════════════════
#  clean task helpers
# ═══════════════════════════════════════════════════════════════════


def _clean_worker(
    path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, int] | None:
    """Execute the cleaning task on a single PDB/mmCIF file.

    :param path: Path to the structure file.
    :return: Tuple of (clean_set, dirty_set, chain_num), or None on failure.
    """
    try:
        return entry_integrated_cleaning(str(path))
    except Exception as e:
        logger.warning(f"Failed to clean {path}. Exception: {e}")
        return None


def _save_clean_results(
    result: list[tuple[pd.DataFrame, pd.DataFrame, int] | None],
    output_path: str | None = None,
) -> None:
    """Save cleaning results as cleaned / filtered chain CSVs.

    :param result: List of (clean_set, dirty_set, chain_num) tuples.
    :param output_path: Directory for the output CSVs. Defaults to ``"."``.
    """
    logger.info("Post-processing cleaning results")
    clean_result = [i for i in result if i is not None]
    clean = [i[0] for i in clean_result]
    dirty = [i[1] for i in clean_result]
    chain_num = sum(i[2] for i in clean_result)
    clean_df = pd.concat(clean, ignore_index=True)
    dirty_df = pd.concat(dirty, ignore_index=True)

    output_path = output_path or "."
    clean_df.to_csv(Path(output_path) / "chains_cleaned.csv", index=False)
    dirty_df.to_csv(Path(output_path) / "chains_filtered.csv", index=False)
    logger.info(f"Task completed. Cleaned {len(clean_df)} chains out of {chain_num}.")


# ═══════════════════════════════════════════════════════════════════
#  bri (entry-level) task helpers
# ═══════════════════════════════════════════════════════════════════


def _pre_process_bri(
    input_data: pd.DataFrame | list[str],
) -> list[str] | dict[str, str | int]:
    """Pre-process input data for BRI computation.

    :param input_data: Raw input data (DataFrame or list).
    :return: Processed data ready for BRI computation.
    """
    logger.info("Pre-processing data for BRI computation task")
    if isinstance(input_data, pd.DataFrame):
        if set(MINI_ENTRY_CLEAN_COL).issubset(set(input_data.columns)):
            data_df = input_data.loc[:, MINI_ENTRY_CLEAN_COL]
            return data_df.to_dict("records")
        else:
            logger.error("Cannot find required data for BRI computation")
            sys.exit(1)
    elif isinstance(input_data, list):
        return input_data
    else:
        logger.error("Unsupported input data type for BRI computation")
        sys.exit(1)


def _bri_worker(
    input_id: dict[str, str | int] | str | Path,
    output: str | Path | None = None,
    invariant_type: InvariantType = "bri",
) -> pd.DataFrame | None:
    """Compute Backbone Rigid Invariants for a single chain or entry.

    :param input_id: Dict with keys ``pdb_id``, ``chain_id`` (and optionally
        ``start_residue``, ``chain_length``), or a PDB ID / file path string.
    :param output: Output directory (saves per-chain CSVs) or ``None``.
    :param invariant_type: ``'bri'`` (coordinate-based) or ``'lai'`` (bond
        lengths, angles, torsions).
    :return: Computed invariant DataFrame, or None on failure.
    """
    save_path = None
    if output is not None and Path(output).is_dir():
        save_path = Path(output)

    if isinstance(input_id, dict):
        pdb_id = str(input_id.get("pdb_id", "unknown"))
        chain_id = str(input_id.get("chain_id", "?"))
        try:
            entry = ProteinEntry.from_cif(pdb_id)
            chain = entry[chain_id]
            start = input_id.get("start_residue")
            length = input_id.get("chain_length")
            if start is not None:
                chain = chain.slice_residues(
                    int(start), int(length) if length else None
                )
            invariant = chain.get_invariant(invariant_type=invariant_type)
            if save_path is not None and not invariant.empty:
                chain.save_invariant(
                    save_path / f"{pdb_id}_{chain_id}.csv",
                    invariant_type=invariant_type,
                )
            return invariant
        except Exception as e:
            logger.warning(f"Failed to compute invariant for {pdb_id}-{chain_id}: {e}")
            return None

    try:
        path_str = str(input_id)
        if path_str.lower().endswith(".pdb"):
            entry = ProteinEntry.from_pdb(path_str)
        else:
            entry = ProteinEntry.from_cif(path_str)
        invariant = entry.get_entry_invariant(invariant_type=invariant_type)
        if save_path is not None and not invariant.empty:
            entry.save_invariant(
                save_path / f"{entry.pdb_id}.csv",
                invariant_type=invariant_type,
            )
        return invariant
    except Exception as e:
        logger.warning(f"Failed to compute invariant for {input_id}: {e}")
        return None


def _save_bri_results(
    result: list[pd.DataFrame | None], output_path: str | None = None
) -> None:
    """Save BRI computation results to a file or log the count.

    :param result: List of invariant DataFrames (may contain ``None``).
    :param output_path: File path (``.csv`` / ``.parquet``) or directory.
    """
    logger.info("Post-processing BRI computation results")
    valid_results = [r for r in result if r is not None]
    if valid_results:
        output_path = output_path or "bri_results.csv"
        save_loc = Path(output_path)
        if save_loc.suffix:
            bri_results = pd.concat(valid_results)
            if save_loc.suffix == ".parquet":
                bri_results.to_parquet(save_loc, index=False)
            else:
                bri_results.to_csv(save_loc, index=False)
            logger.info(f"Saved {len(valid_results)} BRIs to {save_loc}")
        else:  # Results already saved per-chain during computation
            logger.info(f"Saved {len(valid_results)} BRIs to {save_loc}")
    else:
        logger.warning("No valid BRI results to save")


# ═══════════════════════════════════════════════════════════════════
#  duplicate task helpers
# ═══════════════════════════════════════════════════════════════════


def _pre_process_duplicate(input_data: pd.DataFrame) -> pd.DataFrame:
    """Pre-process input data for duplicate detection.

    :param input_data: Input DataFrame with a ``seq`` column.
    :return: DataFrame with the ``seq`` column removed.
    """
    logger.info("Pre-processing data for duplicate detection task")
    return input_data.drop(columns=["seq"])


def _process_invariant_with_chain_id(
    pdb_id: str, chain_filter: list[dict[str, int | str]]
) -> pd.DataFrame:
    """Compute BRI for chains identified by dicts.

    :param pdb_id: Entry ID of the input chains.
    :param chain_filter: List of dicts with keys ``model_id``, ``chain_id``,
        ``start_residue``, and ``chain_length``.
    :return: Invariant DataFrame with an added ``pdb_id`` column, or an empty
        DataFrame if no chain yields a computable invariant.
    """
    invs = []
    entry = ProteinEntry.from_cif(pdb_id)
    for records in chain_filter:
        m_id, c_id = int(records["model_id"]), str(records["chain_id"])
        c = entry.get_chains(model_id=m_id, chain_id=c_id)[0]
        c = c.slice_residues(
            int(records["start_residue"]), chain_length=int(records["chain_length"])
        )
        if c.num_residues < 1:
            continue
        invs.append(c.get_invariant())

    if not invs:
        return pd.DataFrame()

    output = pd.concat(invs, ignore_index=True)
    output["pdb_id"] = pdb_id
    return output


def _duplicate_worker(dataset: pd.DataFrame, pool: mp.Pool) -> pd.DataFrame:
    """Detect near-duplicate structures within a dataset.

    :param dataset: DataFrame of chain records grouped by ``chain_length``.
    :param pool: Multiprocessing pool for parallel invariant computation.
    :return: DataFrame of pairwise comparison results.
    """
    compare_results = []

    chain_groups = dataset.groupby("chain_length")
    for length, chain_group in chain_groups:
        try:
            args = [
                (eid, filters.to_dict("records"))
                for eid, filters in chain_group.groupby("pdb_id")
            ]
            if len(args) < 2 and len(args[0][1]) < 2:
                logger.info(
                    f"Skipped length {length}: fewer than 2 comparable chains after filtering."
                )
                continue
            invariants_same_length = pool.starmap(
                _process_invariant_with_chain_id, args
            )
            invariants_same_length = list(invariants_same_length)

            fmt_bri_group = pd.concat(invariants_same_length, ignore_index=True)
            fmt_bri_group = fmt_bri_group[
                [
                    "pdb_id",
                    "model_id",
                    "chain_id",
                    "residue_id",
                    "residue_label",
                    "x(N)",
                    "y(N)",
                    "z(N)",
                    "x(A)",
                    "y(A)",
                    "z(A)",
                    "x(C)",
                    "y(C)",
                    "z(C)",
                    "chain_length",
                ]
            ]

            compare_result = group_invariant_compare(fmt_bri_group, seq_compare=True)
            compare_result["chain_length"] = length
            compare_results.append(compare_result)
        except Exception as e:
            logger.warning(
                f"Failed duplicate detection on {len(chain_group)} chains having {length} residues"
            )
            logger.warning(f"{e}")

    if not compare_results:
        logger.warning("No duplicate comparisons could be produced.")
        return pd.DataFrame()
    return pd.concat(compare_results, ignore_index=True)


def _save_duplicate_results(
    result: pd.DataFrame, output_path: str | None = None
) -> None:
    """Save duplicate detection results.

    :param result: DataFrame of pairwise comparison results.
    :param output_path: Directory for full per-length results (optional).
    """
    logger.info("Post-processing duplicate detection results")
    timestr = datetime.datetime.now().strftime("%d%m%y")

    # Concise thresholded summaries
    result_lt1 = result[result["distance"] < 1]
    result_lt1.to_csv(f"L_inf_lt1_{timestr}.csv", index=False)
    result_lt001_eq = result[(result["distance"] < 0.01) & (result["seq_diff"] < 1)]
    result_lt001_eq.to_csv(f"L_inf_lt001_eq_seq{timestr}.csv", index=False)
    logger.info(
        f"Concise BRI distance results saved to L_inf_lt1_{timestr}.csv, "
        f"L_inf_lt001_eq_seq{timestr}.csv"
    )

    # Full per-length tables (only when an output directory is given)
    if output_path:
        logger.info(f"Saving all BRI distance results to {output_path}")
        columns = (
            "pdb_id1,model_id1,chain_id1,"
            "pdb_id2,model_id2,chain_id2,"
            "L_inf_invariant,seq_diff,seq1,seq2"
        ).split(",")
        for length, results in result.groupby("chain_length"):
            file_path = Path(output_path) / f"BRI_L_inf_{length}.csv"
            results.to_csv(file_path, index=False, columns=columns)


# ═══════════════════════════════════════════════════════════════════
#  Multiprocessing helper
# ═══════════════════════════════════════════════════════════════════


def _run_pool(task, dataset, show_progress: bool = False):
    """Map *task* over *dataset* using a multiprocessing pool.

    :param task: Callable applied to each item in *dataset*.
    :param dataset: Iterable of work items.
    :param show_progress: If True, wrap the iteration in a tqdm progress bar.
    :return: List of results.
    """
    pool = mp.Pool(mp.cpu_count())
    try:
        if show_progress:
            result = tqdm.tqdm(pool.imap_unordered(task, dataset), total=len(dataset))
        else:
            result = pool.map(task, dataset)
        result = list(result)
    finally:
        pool.close()
    return result


# ═══════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════


@click.group()
def cli():
    """BRI evaluation tool.

    A suite of commands for cleaning protein structure files, computing
    backbone rigid invariants (BRI), building distance matrices, and
    generating statistical projections and comparison plots.

    Run ``bri <command> --help`` for details on each subcommand.
    """
    pass


# ─── Directory-level commands ──────────────────────────────────────


@cli.command("inv")
@click.argument(
    "input_dir", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.argument(
    "output_dir", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.option(
    "-n",
    "--n-process",
    default=_DEFAULT_N_PROCESS,
    show_default=True,
    type=int,
    help="Number of CPU cores to utilize for parallel processing.",
)
def cmd_inv(input_dir: Path, output_dir: Path, n_process: int):
    """Calculate invariants from raw PDB/mmCIF files.

    Computes Backbone Rigid Invariants (BRI), Length Angle Invariants (LAI),
    and Backbone Torsion Invariants (BTI) and saves them as CSVs.

    \b
    INPUT_DIR: Directory containing source .pdb / .cif / .bcif files.
    OUTPUT_DIR: Directory where the resulting invariant CSVs will be saved.
    """
    start_time = _begin_task("inv")
    count = compute_dir_invariants(input_dir, output_dir, n_process)
    logger.info(f"Invariant computation completed: {count} file(s) processed.")
    logger.info(f"Completed with total time: {(time.time() - start_time):.2f}s")


@cli.command("compare")
@click.argument(
    "input_dir", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.argument(
    "output_dir", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
def cmd_compare(input_dir: Path, output_dir: Path):
    """Calculate RMS and Chebyshev (L-infinity) distance matrices.

    Measures structural distances between all processed protein chain
    invariants in the input directory and compiles them into symmetric
    distance matrices.

    \b
    INPUT_DIR: Directory containing computed invariant CSVs (e.g., from 'inv').
    OUTPUT_DIR: Directory where the distance matrices will be saved.
    """
    start_time = _begin_task("compare")
    compute_distance_matrix(input_dir, output_dir)
    logger.info("Distance matrix computation completed.")
    logger.info(f"Completed with total time: {(time.time() - start_time):.2f}s")


@cli.command("proj")
@click.argument(
    "input_dir", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.argument(
    "output_dir", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
def cmd_proj(input_dir: Path, output_dir: Path):
    """Calculate statistical summaries and scatter projections.

    Computes mean and standard deviation for the invariants and generates
    2D scatter plots to visualize their distributions across the dataset.

    \b
    INPUT_DIR: Directory containing computed invariant CSVs.
    OUTPUT_DIR: Directory where the statistical CSVs and plots will be saved.
    """
    start_time = _begin_task("proj")
    scatter_projection(input_dir, output_dir)
    logger.info("Statistical projection completed.")
    logger.info(f"Completed with total time: {(time.time() - start_time):.2f}s")


@cli.command("plot")
@click.argument(
    "input_dir", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.argument(
    "output_dir", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.option(
    "-s",
    "--structure",
    type=str,
    default=None,
    help="Reference structure for comparison (PDB ID, 'pdb-model-chain', or file path).",
)
@click.option(
    "-ofs",
    "--offset",
    type=int,
    default=0,
    show_default=True,
    help="Residue-ID offset for aligning the reference structure.",
)
def cmd_plot(input_dir: Path, output_dir: Path, structure: str | None, offset: int):
    """Generate comparison plots against a reference.

    Creates high-resolution overlay plots comparing the invariant sequences
    of the computed models, and optionally against a known experimental
    reference structure.

    \b
    INPUT_DIR: Directory containing computed invariant CSVs.
    OUTPUT_DIR: Directory where the comparison PNG plots will be saved.
    """
    start_time = _begin_task("plot")
    _ = comparison_plots(input_dir, output_dir, reference=structure, offset=offset)
    logger.info("Comparison plot generation completed.")
    logger.info(f"Completed with total time: {(time.time() - start_time):.2f}s")


@cli.command("pipe")
@click.argument(
    "input_dir", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.argument(
    "output_dir", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.option(
    "-n",
    "--n-process",
    default=_DEFAULT_N_PROCESS,
    show_default=True,
    type=int,
    help="Number of CPU cores to utilize for the invariant computation step.",
)
def cmd_pipe(input_dir: Path, output_dir: Path, n_process: int):
    """Execute the full pipeline: invariants, matrices, and projections.

    Sequentially executes 'inv', 'compare', and 'proj'. Ideal for processing
    a raw batch of structure files through to final statistical projections
    in a single run.

    \b
    INPUT_DIR: Directory containing raw source .pdb / .cif / .bcif files.
    OUTPUT_DIR: Directory where all resulting CSVs, matrices, and plots will
    be saved.
    """
    start_time = _begin_task("pipe")
    _ = compute_dir_invariants(input_dir, output_dir, n_process)
    _ = compute_distance_matrix(output_dir, output_dir)
    _ = scatter_projection(output_dir, output_dir)
    logger.info("Full pipeline completed.")
    logger.info(f"Completed with total time: {(time.time() - start_time):.2f}s")


# ─── Entry-level commands ──────────────────────────────────────────


@cli.command("clean")
@click.argument("input_data", type=click.Path(exists=True))
@click.option(
    "-o",
    "--output",
    type=str,
    default=".",
    show_default=True,
    help="Directory for the cleaned / filtered chain CSVs.",
)
@click.option(
    "--max-samples",
    type=int,
    default=None,
    help="Maximum number of input entries (default: all).",
)
@click.option(
    "--process/--no-process",
    default=False,
    help="Show a real-time progress bar.",
)
def cmd_clean(input_data: str, output: str, max_samples: int | None, process: bool):
    """Clean and filter protein structure files.

    Applies integrated cleaning to each input entry, separating valid chains
    from noisy ones. Results are saved as ``chains_cleaned.csv`` and
    ``chains_filtered.csv``.

    \b
    INPUT_DATA: Path to a .txt, .json, or .csv file listing entries, or a
    directory of structure files.
    """
    start_time = _begin_task("clean")
    logger.info(f"Processing {input_data} as input")
    dataset = load_input(input_data, 0, max_samples)

    result = _run_pool(_clean_worker, dataset, show_progress=process)
    _save_clean_results(result, output)
    logger.info(f"Completed with total time: {(time.time() - start_time):.2f}s")


@cli.command("bri")
@click.argument("input_data", type=click.Path(exists=True))
@click.option(
    "-o",
    "--output",
    type=str,
    default=None,
    help="Output file (.csv / .parquet) or directory for per-chain CSVs.",
)
@click.option(
    "--invariant-type",
    type=click.Choice(["bri", "lai"]),
    default="bri",
    show_default=True,
    help="Invariant type: 'bri' (coordinate-based) or 'lai' (bond lengths, angles, torsions).",
)
@click.option(
    "--max-samples",
    type=int,
    default=None,
    help="Maximum number of input entries (default: all).",
)
@click.option(
    "--process/--no-process",
    default=False,
    help="Show a real-time progress bar.",
)
def cmd_bri(
    input_data: str,
    output: str | None,
    invariant_type: InvariantType,
    max_samples: int | None,
    process: bool,
):
    """Compute backbone rigid invariants for entries or chains.

    Accepts a .txt/.json/.csv file listing entries. Each entry can be a dict
    with keys ``pdb_id``, ``chain_id``, ``start_residue``, ``chain_length``,
    or a PDB ID / file path.

    \b
    INPUT_DATA: Path to a .txt, .json, or .csv file listing entries, or a
    directory of structure files.
    """
    start_time = _begin_task("bri")
    logger.info(f"Processing {input_data} as input")
    dataset = load_input(input_data, 0, max_samples)
    dataset = _pre_process_bri(dataset)

    worker = functools.partial(
        _bri_worker, output=output, invariant_type=invariant_type
    )
    result = _run_pool(worker, dataset, show_progress=process)
    _save_bri_results(result, output)
    logger.info(f"Completed with total time: {(time.time() - start_time):.2f}s")


@cli.command("duplicate")
@click.argument("input_data", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "-o",
    "--output",
    type=str,
    default=None,
    help="Directory for full per-length duplicate results (optional).",
)
@click.option(
    "--max-samples",
    type=int,
    default=None,
    help="Maximum number of input entries (default: all).",
)
def cmd_duplicate(input_data: str, output: str | None, max_samples: int | None):
    """Detect near-duplicate structures within a dataset.

    Compares all chains of the same length using BRI distance and sequence
    distance, producing concise summary CSVs of close matches.

    \b
    INPUT_DATA: Path to a .csv file with chain metadata (must include 'seq').
    """
    start_time = _begin_task("duplicate")
    logger.info(f"Processing {input_data} as input")
    dataset = load_input(input_data, 0, max_samples)
    dataset = _pre_process_duplicate(dataset)

    pool = mp.Pool(mp.cpu_count())
    try:
        result = _duplicate_worker(dataset, pool)
    finally:
        pool.close()
    _save_duplicate_results(result, output)
    logger.info(f"Completed with total time: {(time.time() - start_time):.2f}s")


if __name__ == "__main__":
    cli()
