from __future__ import annotations

import argparse
import datetime
import functools
import json
import logging
import multiprocessing as mp
from pathlib import Path
import sys
import time
from typing import Any

import pandas as pd
import tqdm

from bri.structure import ProteinEntry, ProteinChain
from bri.invariant_compare import group_invariant_compare
from bri.filter import (
    entry_integrated_cleaning,
    MINI_ENTRY_CLEAN_COL,
)
from bri.workflow import (
    compute_dir_invariants,
    compute_distance_matrix,
    scatter_projection,
)

TASK_TYPE = ["clean", "bri", "inv", "compare", "proj", "plot", "duplicate"]


# Set up basic logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("BRI")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for the BRI evaluation tool.

    :return: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description="BRI evaluation tool")

    parser.add_argument(
        "--task",
        type=str,
        choices=TASK_TYPE,
        required=True,
        help="Type of task to run",
    )

    parser.add_argument(
        "--input-data",
        type=str,
        default=None,
        help="Dataset path",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output results path",
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of input entries (default: all)",
    )

    parser.add_argument(
        "--process",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Show the real-time process of task",
    )

    parser.add_argument(
        "--invariant-type",
        type=str,
        default="bri",
        choices=["bri", "lai"],
        help="Invariant type for BRI computation: 'bri' (coordinate-based) or "
        "'lai' (bond lengths, angles, torsions).  Default: 'bri'.",
    )

    parser.add_argument(
        "--structure",
        type=str,
        default=None,
        help="Reference structure for comparison plots (PDB ID, "
        "'pdb-model-chain', or file path).",
    )

    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Residue-ID offset for aligning the reference structure.",
    )

    return parser.parse_args()


def input_process(
    input_path: str | Path, start: int = 0, end: int | None = None
) -> list[str] | list[Any] | pd.DataFrame:
    """Process input files and return data in appropriate format.

    :param input_path: Path to input file (.txt, .json, or .csv)
    :param start: Starting index for data slice
    :param end: Ending index for data slice
    :return: Processed input data
    :raises ValueError: If file type is not supported
    """
    end = end + start if end else None
    ext = Path(input_path).suffix
    ext = ext.lower()

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


def pre_process_clean(input_data: Any) -> Any:
    """Pre-process input data for cleaning task.

    :param input_data: Input data to be processed
    :return: Processed data ready for cleaning
    """
    logger.info("Pre-processing data for cleaning task")
    # cleaning-specific pre-processing
    return input_data


def pre_process_bri(input_data: Any) -> Any:
    """Pre-process input data for BRI computation task.

    :param input_data: Input data to be processed
    :return: Processed data ready for BRI computation
    """
    logger.info("Pre-processing data for BRI computation task")
    # BRI-specific pre-processing
    if isinstance(input_data, pd.DataFrame):
        if set(MINI_ENTRY_CLEAN_COL).issubset(set(input_data.columns)):
            return input_data[MINI_ENTRY_CLEAN_COL].to_dict("records")
        else:
            logger.error(f"Cannot find required data for BRI computation")
            sys.exit(1)

    elif isinstance(input_data, list):
        return input_data

    # TODO pre-process for json(dict) input
    else:
        logger.error(f"Unsupported input data type for BRI computation")
        sys.exit(1)


def pre_process_duplicate(input_data: pd.DataFrame):
    """Pre-process input data for duplicate detection task.

    :param input_data: Input data to be processed
    :return: Processed data ready for duplicate detection
    """
    logger.info("Pre-processing data for duplicate detection task")

    cleaned_input = input_data.drop(columns=["seq"])

    return cleaned_input


def post_process_clean(result: Any, output_path: str | None = None) -> None:
    """Post-process results after cleaning task.

    :param result: Result data from cleaning task
    :param output_path: Optional path to save results
    """
    logger.info("Post-processing cleaning results")
    # remove None type
    clean_result = [i for i in result if i is not None]
    # separate cleaned chains and noisy chains
    clean = [i[0] for i in clean_result]
    dirty = [i[1] for i in clean_result]
    chain_num = sum([i[2] for i in clean_result])
    clean_result = pd.concat(clean, ignore_index=True)
    dirty_result = pd.concat(dirty, ignore_index=True)

    output_path = output_path or "."
    clean_result.to_csv(Path(output_path) / "chains_cleaned.csv", index=False)
    dirty_result.to_csv(Path(output_path) / "chains_filtered.csv", index=False)

    # Post-process results
    logger.info(
        f"Task completed. Cleaned {len(clean_result)} chains out of {chain_num}."
    )


def post_process_bri(result: list[pd.DataFrame], output_path: str | None = None):
    """Post-process results after BRI computation task.

    :param result: Result data from BRI computation
    :param output_path: Optional path to save results
    """
    logger.info("Post-processing BRI computation results")
    # Filter out None results
    valid_results = [r for r in result if r is not None]
    if valid_results:
        output_path = output_path or "bri_results.csv"  # set save location
        save_loc = Path(output_path)
        if save_loc.suffix:
            bri_results = pd.concat(valid_results)
            if save_loc.suffix == ".parquet":
                bri_results.to_parquet(save_loc, index=False)
            else:
                bri_results.to_csv(save_loc, index=False)
            logger.info(f"Saved {len(valid_results)} BRIs to {save_loc}")

        else:  # Results already saved
            logger.info(f"Saved {len(valid_results)} BRIs to {save_loc}")

    else:
        logger.warning("No valid BRI results to save")


def post_process_duplicate(result: pd.DataFrame, output_path: str | None = None):
    """Post-process results after duplicate detection task.

    :param result: Result data from duplicate detection
    :param output_path: Optional path to save results
    """
    logger.info("Post-processing duplicate detection results")
    timestr = datetime.datetime.now().strftime("%d%m%y")

    # threshold
    result1 = result[result["distance"] < 1]
    result1.to_csv(f"L_inf_lt1_{timestr}.csv", index=False)
    result001_seq_e = result[(result["distance"] < 0.01) & (result["seq_diff"] < 1)]
    result001_seq_e.to_csv(f"L_inf_lt001_eq_seq{timestr}.csv", index=False)
    logger.info(
        f"Concise BRI distance results saved to L_inf_lt1_{timestr}.csv, L_inf_lt001_eq_seq{timestr}.csv"
    )

    # save all results by chain length only if output path specified
    if output_path:
        logger.info(f"Saving all BRI distance results to {output_path}")
        columns = "pdb_id1,model_id1,chain_id1,pdb_id2,model_id2,chain_id2,L_inf_invariant,seq_diff,seq1,seq2".split(
            ","
        )
        results_group = result.groupby("chain_length")
        for length, results in results_group:
            file_path = Path(output_path) / f"BRI_L_inf_{length}.csv"
            results.to_csv(file_path, index=False, columns=columns)


def clean_task(path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, int] | None:
    """Execute the cleaning task on a given PDB/mmCIF file.

    :param path: Path to the PDB/mmCIF file
    :return: Tuple of (clean_set, dirty_set, chain_num), or None on failure.
    """
    try:
        result = entry_integrated_cleaning(str(path))
    except Exception as e:
        logger.warning(f"Failed to clean {path}. Exception: {e}")
        return None

    return result


def bri_task(
    input_id: dict | str | Path, paras: argparse.Namespace
) -> pd.DataFrame | None:
    """Compute Backbone Rigid Invariants for a given chain or entry.

    :param input_id: Dictionary with keys ``pdb_id``, ``chain_id`` (and
        optionally ``start_residue``, ``chain_length``), or a PDB ID / file
        path string.
    :param paras: Program arguments (must have ``invariant_type`` and
        ``output`` attributes).
    :return: Computed invariant as DataFrame, or None on failure.
    """
    save_path = None
    if paras.output is not None and Path(paras.output).is_dir():
        save_path = Path(paras.output)

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
            invariant = chain.get_invariant(invariant_type=paras.invariant_type)
            if save_path is not None and not invariant.empty:
                chain.save_invariant(
                    save_path / f"{pdb_id}_{chain_id}.csv",
                    invariant_type=paras.invariant_type,
                )
            return invariant
        except Exception as e:
            logger.warning(f"Failed to compute invariant for {pdb_id}-{chain_id}: {e}")
            return None

    elif isinstance(input_id, (str, Path)):
        try:
            path_str = str(input_id)
            if path_str.lower().endswith(".pdb"):
                entry = ProteinEntry.from_pdb(path_str)
            else:
                entry = ProteinEntry.from_cif(path_str)
            invariant = entry.get_entry_invariant(invariant_type=paras.invariant_type)
            if save_path is not None and not invariant.empty:
                entry.save_invariant(
                    save_path / f"{entry.pdb_id}.csv",
                    invariant_type=paras.invariant_type,
                )
            return invariant
        except Exception as e:
            logger.warning(f"Failed to compute invariant for {input_id}: {e}")
            return None


def process_invariant_with_chain_id(pdb_id: str, chain_filter: list[dict]):
    """Compute BRI for a single chain identified by a dict.

    :param pdb_id: Entry id of input chains.
    :param chain_id: Dict with keys ``model_id`` and ``chain_id``.
    :return: Invariant DataFrame with an added ``pdb_id`` column, or an empty
        DataFrame if the chain cannot be loaded or has no computable invariant
        (so that one bad chain does not abort the whole length-group).
    """
    invs = []
    entry = ProteinEntry.from_cif(pdb_id)
    for records in chain_filter:
        m_id, c_id = int(records["model_id"]), str(records["chain_id"])
        c = entry.get_chain(model_id=m_id, chain_id=c_id)
        c = c.slice_residues(
            records["start_residue"], chain_length=records["chain_length"]
        )
        if c.num_residues < 1:
            continue
        invs.append(c.get_invariant())
    output = pd.DataFrame()
    if not invs:
        return output

    output = pd.concat(invs, ignore_index=True)
    output["pdb_id"] = pdb_id
    return output


def duplicate_task(dataset: pd.DataFrame, pool):
    """Detect duplicates in a given structure.

    :param path: Path to the PDB/mmCIF file
    :return: Duplicate detection results, or None if detection fails
    """
    compare_results = []

    # get invariants by length
    chain_groups = dataset.groupby("chain_length")
    for length, chain_group in chain_groups:
        # apply multiprocessing
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
                process_invariant_with_chain_id, args
            )  # get invariants in group
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
            ]  # choose columns

            # compare invariants
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
    compare_results = pd.concat(compare_results, ignore_index=True)
    return compare_results


# ═══════════════════════════════════════════════════════════════
#  CLI task wrappers for directory-level workflows
# ═══════════════════════════════════════════════════════════════


def pre_process_inv(input_data: Any) -> Path:
    """Pre-process input for the ``inv`` task.

    :param input_data: Directory path (str or Path) or list thereof.
    :return: Resolved :class:`Path` to the input directory.
    """
    logger.info("Pre-processing data for invariant computation")
    if isinstance(input_data, (str, Path)):
        return Path(input_data)
    if isinstance(input_data, list) and len(input_data) >= 1:
        return Path(input_data[0])
    logger.error("Unsupported input for inv task; expected a directory path.")
    sys.exit(1)


def inv_task(input_dir: Path, paras: argparse.Namespace) -> int:
    """Compute invariants for all PDB files in a directory.

    :param input_dir: Directory of ``.pdb`` files.
    :param paras: Program arguments (must have ``output``).
    :return: Number of files processed.
    """
    output_dir = (
        Path(paras.output) if paras.output else input_dir / "compute_invariants"
    )
    return compute_dir_invariants(input_dir, output_dir)


def post_process_inv(result: int, output_path: str | None = None) -> None:
    """Post-process results after the ``inv`` task."""
    logger.info(f"Invariant computation completed: {result} file(s) processed.")


def pre_process_compare(input_data: Any) -> Path:
    """Pre-process input for the ``compare`` task."""
    logger.info("Pre-processing data for distance matrix computation")
    if isinstance(input_data, (str, Path)):
        return Path(input_data)
    if isinstance(input_data, list) and len(input_data) >= 1:
        return Path(input_data[0])
    logger.error("Unsupported input for compare task.")
    sys.exit(1)


def compare_task(input_dir: Path, paras: argparse.Namespace) -> None:
    """Compute distance matrices from invariant CSVs.

    :param input_dir: Directory of invariant ``*_inv.csv`` files.
    :param paras: Program arguments (must have ``output``).
    """
    output_dir = Path(paras.output) if paras.output else input_dir
    compute_distance_matrix(input_dir, output_dir)


def post_process_compare(result: None, output_path: str | None = None) -> None:
    """Post-process results after the ``compare`` task."""
    logger.info("Distance matrix computation completed.")


def pre_process_proj(input_data: Any) -> Path:
    """Pre-process input for the ``proj`` task."""
    logger.info("Pre-processing data for statistical projection")
    if isinstance(input_data, (str, Path)):
        return Path(input_data)
    if isinstance(input_data, list) and len(input_data) >= 1:
        return Path(input_data[0])
    logger.error("Unsupported input for proj task.")
    sys.exit(1)


def proj_task(input_dir: Path, paras: argparse.Namespace) -> None:
    """Compute statistical projections from invariant CSVs.

    :param input_dir: Directory of invariant ``*_inv.csv`` files.
    :param paras: Program arguments (must have ``output``).
    """
    output_dir = Path(paras.output) if paras.output else input_dir
    scatter_projection(input_dir, output_dir)


def post_process_proj(result: None, output_path: str | None = None) -> None:
    """Post-process results after the ``proj`` task."""
    logger.info("Statistical projection completed.")


def pre_process_plot(input_data: Any) -> Path:
    """Pre-process input for the ``plot`` task."""
    logger.info("Pre-processing data for comparison plots")
    if isinstance(input_data, (str, Path)):
        return Path(input_data)
    if isinstance(input_data, list) and len(input_data) >= 1:
        return Path(input_data[0])
    logger.error("Unsupported input for plot task.")
    sys.exit(1)


def plot_task(input_dir: Path, paras: argparse.Namespace) -> None:
    """Generate comparison-curve plots from invariant CSVs.

    :param input_dir: Directory of invariant ``*_inv.csv`` files.
    :param paras: Program arguments (must have ``output``, ``structure``,
        and ``offset`` attributes).
    """
    from bri.workflow import comparison_plots

    output_dir = Path(paras.output) if paras.output else input_dir
    comparison_plots(
        input_dir,
        output_dir,
        reference=paras.structure,
        offset=paras.offset,
    )


def post_process_plot(result: None, output_path: str | None = None) -> None:
    """Post-process results after the ``plot`` task."""
    logger.info("Comparison plot generation completed.")


def main():
    args = parse_args()

    # Set up file logging only after parsing args (not on --help)
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"eval_{timestamp}.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    logger.info(f"Start at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 40)
    logger.info(f"Available CPUs: {mp.cpu_count()}")

    # Parse input data
    logger.info(f"Processing {args.input_data} as input")
    dataset = input_process(args.input_data, 0, args.max_samples)

    # Pre-process based on task type
    pre_process_func = globals().get(f"pre_process_{args.task}")
    if pre_process_func:
        dataset = pre_process_func(dataset)
    else:
        logger.error(f"Pre-process not defined for task '{args.task}'")

    # Assign task
    task = globals().get(f"{args.task}_task")
    if task is None:
        logger.error(f"Task function '{args.task}_task' not found")
        sys.exit(1)
    # For BRI and directory-level tasks, bind the full argument namespace.
    if args.task in ("bri", "inv", "compare", "proj", "plot"):
        task = functools.partial(task, paras=args)

    # Task start
    logger.info(f"Start running task: {args.task}")
    start_time = time.time()

    if args.task in ("inv", "compare", "proj", "plot"):
        # Directory-level tasks: dataset is a single Path, not an iterable.
        result = task(dataset)

    elif args.task != "duplicate":
        pool = mp.Pool(mp.cpu_count())
        if args.process:
            result = tqdm.tqdm(pool.imap_unordered(task, dataset), total=len(dataset))
        else:
            result = pool.map(task, dataset)

        result = list(result)
        pool.close()
    else:
        pool = mp.Pool(mp.cpu_count())
        result = task(dataset, pool)
        pool.close()

    # Post-process based on task type
    post_process_func = globals().get(f"post_process_{args.task}")
    if post_process_func:
        post_process_func(result, args.output)

    logger.info(f"Completed with total time: {(time.time() - start_time):.2f}s")
