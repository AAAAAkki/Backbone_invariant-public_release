# -*- coding = utf-8 -*-
"""High-level workflow helpers.

This module collects the importable, user-friendly functions that drive the
common batch workflows — computing backbone rigid invariants for a folder of
PDB/mmCIF files, building pairwise distance matrices from the results, and
producing statistical scatter projections.

They can be called directly::

    from bri.workflow import compute_dir_invariants, compute_distance_matrix

    compute_dir_invariants(Path("pdbs/"), Path("out/inv/"))
    compute_distance_matrix(Path("out/inv/"), Path("out/dist/"))
"""

from __future__ import annotations

import functools
import logging
import multiprocessing as mp
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bri.structure import ProteinEntry, ProteinChain
from bri.invariant_compare import group_invariant_compare

from bri.invariant import (
    get_invariant as _compute_invariant,
    LAI_COLUMNS,
    BRI_COLUMNS,
    INVARIANT_META_COLUMNS,
)

logger = logging.getLogger("BRI")

__all__ = [
    "get_data",
    "compute_and_save_invariant",
    "compute_dir_invariants",
    "distance_table_to_matrix",
    "compute_distance_matrix",
    "compute_invariant_stats",
    "plot_invariant_scatter",
    "scatter_projection",
    "plot_invariant_curves",
    "comparison_plots",
]


def get_data(folder: Path) -> pd.DataFrame:
    """Load and concatenate all invariant CSV files from a directory.

    Each CSV is expected to have been produced by
    :func:`compute_and_save_invariant` and therefore to contain BRI and LAI
    columns. Missing values (NaN at chain boundaries) are preserved.

    :param folder: Directory containing the CSV files.
    :return: All CSVs concatenated into one invariant table.
    :raises ValueError: If no CSV files are found in *folder*.
    """
    files = list(folder.glob("*.csv"))
    if not files:
        raise ValueError(f"No CSV files found in {folder}")
    data = []
    for f in files:
        df = pd.read_csv(f)
        if "chain_length" not in df.columns:
            df["chain_length"] = len(df)
        data.append(df)
    return pd.concat(data, ignore_index=True)


def compute_and_save_invariant(file: Path, output_dir: Path) -> int:
    """Compute and save the invariants for a single structure file.

    Computes the full backbone rigid invariant (BRI together with the derived
    LAI bond lengths, bond angles and torsion angles) for every polypeptide
    chain and writes one ``<stem>_inv.csv`` per input file.

    :param file: Path to a ``.pdb``, ``.cif`` or ``.bcif`` file.
    :param output_dir: Directory to write the result CSV into.
    :return: ``0`` on success (the constant return enables use as a
        :func:`multiprocessing.Pool.map` worker).
    """
    if file.suffix.lower() == ".pdb":
        entry = ProteinEntry.from_pdb(str(file))
    else:
        entry = ProteinEntry.from_cif(str(file))

    chain_dfs = []
    for chain in entry.chains:
        if not chain.polypeptide or chain.is_empty:
            continue
        full = _compute_invariant(chain.to_dataframe(backbone_only=True), angle=True)
        chain_dfs.append(full)

    if chain_dfs:
        result = pd.concat(chain_dfs, ignore_index=True)
        result["pdb_id"] = entry.pdb_id
    else:
        result = pd.DataFrame()
    result.to_csv(output_dir / f"{file.stem}_inv.csv", index=False)
    return 0


def compute_dir_invariants(
    input_dir: Path, output_dir: Path, n_process: int | None = None
) -> int:
    """Compute and save invariants for every structure file in a directory.

    Uses :func:`multiprocessing.Pool` for parallelism.

    :param input_dir: Directory containing ``.pdb``, ``.cif`` and ``.bcif``
        files.
    :param output_dir: Directory to write the result CSVs into (created if
        missing).
    :param n_process: Number of worker processes. Defaults to half the CPU
        count.
    :return: Number of files processed.
    """
    struct_files = (
        list(input_dir.glob("*.pdb"))
        + list(input_dir.glob("*.cif"))
        + list(input_dir.glob("*.bcif"))
    )
    if not struct_files:
        logger.warning(f"No .pdb / .cif files found in {input_dir}")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    if n_process is None:
        n_process = max(1, mp.cpu_count() // 2)

    logger.info(
        f"Computing invariants for {len(struct_files)} file(s) using {n_process} worker(s)."
    )
    pool = mp.Pool(n_process)
    worker = functools.partial(compute_and_save_invariant, output_dir=output_dir)
    results = pool.map(worker, struct_files)
    pool.close()
    pool.join()
    return len(results)


def distance_table_to_matrix(
    distance_table: pd.DataFrame,
    id_col1: str = "pdb_id1",
    id_col2: str = "pdb_id2",
    distance_col: str = "distance",
) -> pd.DataFrame:
    """Convert a long distance table into a symmetric distance matrix.

    :param distance_table: Table with the pair-ID columns and the distance
        column named below.
    :param id_col1: Column name for the first ID of a pair. Default
        ``"pdb_id1"``.
    :param id_col2: Column name for the second ID of a pair. Default
        ``"pdb_id2"``.
    :param distance_col: Column name for the distance value. Default
        ``"distance"``.
    :return: A square, symmetric matrix with IDs as both row and column labels
        and zeros on the diagonal.
    """
    try:
        import natsort

        all_ids = natsort.natsorted(
            set(distance_table[id_col1].unique())
            | set(distance_table[id_col2].unique())
        )
    except ModuleNotFoundError:
        all_ids = sorted(
            set(distance_table[id_col1].unique())
            | set(distance_table[id_col2].unique())
        )

    matrix = pd.DataFrame(index=all_ids, columns=all_ids, dtype=float)

    for _, row in distance_table.iterrows():
        id1, id2, dist = row[id_col1], row[id_col2], row[distance_col]
        matrix.loc[id1, id2] = dist
        matrix.loc[id2, id1] = dist

    np.fill_diagonal(matrix.values, 0.0)
    return matrix


def _load_invariant_data(
    data: pd.DataFrame | list[pd.DataFrame] | Path | str,
) -> pd.DataFrame:
    """Normalise the supported input types into one invariant DataFrame.

    :param data: A DataFrame, a list of DataFrames (which are concatenated),
        or a directory path (CSVs loaded via :func:`get_data`).
    :return: A single concatenated invariant DataFrame.
    """
    if isinstance(data, (str, Path)):
        return get_data(Path(data))
    if isinstance(data, list):
        return pd.concat(data, ignore_index=True)
    return data


def compute_distance_matrix(
    data: pd.DataFrame | list[pd.DataFrame] | Path | str,
    output_dir: Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Calculate BRI distance matrices (Chebyshev and RMS).

    Accepts invariant data as a DataFrame, a list of DataFrames, or a
    directory of ``*_inv.csv`` files. Chains of different lengths cannot be
    compared by BRI, so each length group produces its own pair of matrices.

    :param data: Invariant DataFrame, list of DataFrames, or directory of CSVs.
    :param output_dir: If given, save each matrix as
        ``distance_matrix_BRI_*.csv``. If ``None``, matrices are returned but
        not saved.
    :return: Mapping ``{filename: matrix}``, one entry per (metric,
        chain-length) group.
    """
    data = _load_invariant_data(data)
    if len(data) == 0:
        logger.warning("No invariant data found; cannot compute distance matrices.")
        return {}

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    matrices: dict[str, pd.DataFrame] = {}
    metrics = ["chebyshev", "rms"]
    for length, group in data.groupby("chain_length"):
        for metric in metrics:
            try:
                dist = group_invariant_compare(group, metric=metric)
            except Exception:
                logger.debug(
                    "Skipping %s metric for length %s (not enough data)",
                    metric,
                    length,
                )
                continue
            if dist is None or len(dist) == 0:
                continue
            matrix = distance_table_to_matrix(dist)
            filename = f"distance_matrix_BRI_{metric}_len{int(length)}.csv"
            matrices[filename] = matrix
            if output_dir is not None:
                matrix.to_csv(output_dir / filename)
                logger.info(f"Saved distance matrix: {filename}")
    return matrices


def compute_invariant_stats(
    data: pd.DataFrame,
    value_cols: list[str],
    group_col: str = "pdb_id",
) -> pd.DataFrame:
    """Compute per-group mean and standard deviation of invariant columns.

    Groups the invariant data by *group_col* (one group per chain or model)
    and computes the mean and standard deviation of each column in
    *value_cols*.

    :param data: Invariant DataFrame.
    :param value_cols: Columns to summarise.
    :param group_col: Column identifying each chain / model. Default
        ``"pdb_id"``.
    :return: Summary with ``{col}_mean`` and ``{col}_std`` columns (plus the
        group column).
    """
    available = [c for c in value_cols if c in data.columns]
    stats = data.groupby(group_col)[available].agg(["mean", "std"]).reset_index()
    # Flatten the MultiIndex produced by .agg(["mean", "std"]).
    stats.columns = [f"{col}_{stat}" if stat else col for col, stat in stats.columns]
    return stats


def plot_invariant_scatter(
    data: pd.DataFrame,
    value_cols: list[str],
    group_col: str = "pdb_id",
    color: str = "steelblue",
    edge_color: str = "navy",
    figsize: tuple[float, float] | None = None,
):
    """Scatter-plot mean vs. standard deviation for each invariant column.

    Each point represents one chain. Tight clusters mark backbone regions that
    are consistent across chains (rigid); spread-out points mark regions that
    vary (flexible).

    :param data: Invariant DataFrame.
    :param value_cols: Columns to project (one scatter panel each).
    :param group_col: Column identifying each chain / model. Default
        ``"pdb_id"``.
    :param color: Fill colour of the scatter points.
    :param edge_color: Edge colour of the scatter points.
    :param figsize: Figure size; inferred from the panel count if ``None``.
    :return: The figure containing the scatter panels.
    """
    import matplotlib.pyplot as plt

    cols = [c for c in value_cols if c in data.columns]
    if not cols:
        raise ValueError("None of the requested columns are present in the data.")

    stats = compute_invariant_stats(data, cols, group_col)

    n = len(cols)
    n_cols = min(n, 3)
    n_rows = (n + n_cols - 1) // n_cols
    if figsize is None:
        figsize = (n_cols * 4.5, n_rows * 4)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)

    for idx, col in enumerate(cols):
        row, col_idx = divmod(idx, n_cols)
        ax = axes[row][col_idx]
        ax.scatter(
            stats[f"{col}_mean"],
            stats[f"{col}_std"],
            alpha=0.6,
            s=30,
            c=color,
            edgecolors=edge_color,
            linewidth=0.3,
        )
        ax.set_title(col, fontsize=10)
        ax.set_xlabel("mean", fontsize=9)
        ax.set_ylabel("std (flexibility)", fontsize=9)
        ax.grid(alpha=0.2)

    # Hide any unused panels.
    for idx in range(n, n_rows * n_cols):
        row, col_idx = divmod(idx, n_cols)
        axes[row][col_idx].set_visible(False)

    fig.tight_layout()
    return fig


def scatter_projection(
    data: pd.DataFrame | list[pd.DataFrame] | Path | str,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Compute statistical summaries and scatter plots from invariant data.

    Accepts invariant data as a DataFrame, a list of DataFrames, or a
    directory of ``*_inv.csv`` files.

    :param data: Invariant DataFrame, list of DataFrames, or directory of CSVs.
    :param output_dir: If given, save summary CSVs and scatter PNGs there. If
        ``None``, the figures are returned instead of being saved.
    :return: Dict with keys ``"bri_stats"`` / ``"lai_stats"`` (summary
        :class:`~pandas.DataFrame` objects) and ``"bri_fig"`` / ``"lai_fig"``
        (matplotlib figures, present only when *output_dir* is ``None``).
    """
    import matplotlib.pyplot as plt

    # Determine a name prefix for output files.
    if isinstance(data, (str, Path)):
        name = Path(data).name
    else:
        name = "invariants"

    data = _load_invariant_data(data)

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {}

    # --- BRI projection ---
    bri_cols = [c for c in BRI_COLUMNS if c in data.columns]
    if bri_cols:
        stats = compute_invariant_stats(data, bri_cols)
        results["bri_stats"] = stats

        fig = plot_invariant_scatter(data, bri_cols)
        if output_dir is not None:
            stats.to_csv(output_dir / f"{name}_BRI_proj.csv", index=False)
            fig.savefig(output_dir / f"{name}_BRI_proj.png", dpi=300)
            plt.close(fig)
        else:
            results["bri_fig"] = fig

    # --- LAI projection ---
    lai_cols = [c for c in LAI_COLUMNS if c in data.columns]
    if lai_cols:
        stats = compute_invariant_stats(data, lai_cols)
        results["lai_stats"] = stats

        fig = plot_invariant_scatter(data, lai_cols)
        if output_dir is not None:
            stats.to_csv(output_dir / f"{name}_LAI_proj.csv", index=False)
            fig.savefig(output_dir / f"{name}_LAI_proj.png", dpi=300)
            plt.close(fig)
        else:
            results["lai_fig"] = fig

    if output_dir is not None:
        logger.info(f"Saved projection CSVs and plots to {output_dir}")
    return results


def _apply_y_limits(ax, col: str):
    """Apply sensible y-axis limits based on invariant column type."""
    if col.startswith(("x(", "y(", "z(")):
        ax.set_ylim(-2.3, 2.3)
        ax.set_yticks([-2, -1, 0, 1, 2])
    elif col == "tau(CN)":
        ax.set_ylim(-100, 280)
        ax.set_yticks([-90, 0, 90, 270])
    elif col.startswith("tau"):
        ax.set_ylim(-200, 200)
    elif col.startswith("angle"):
        ax.set_ylim(0, 180)
        ax.set_yticks([0, 45, 90, 135, 180])
    elif col.startswith("length"):
        ax.set_ylim(0, 2.5)
        ax.set_yticks([0, 1, 2])


def _unwrap_torsion(values, col: str) -> pd.Series:
    """Shift *tau(CN)* values below −90° by +360° to avoid the wrap-around."""
    if col == "tau(CN)":
        values = values.copy()
        values[values < -90] += 360
    return values


def _resolve_reference(
    reference: ProteinChain | str | None,
) -> ProteinChain | None:
    """Resolve a reference specification to a :class:`ProteinChain`.

    Accepts a ``ProteinChain`` directly, a PDB ID (``"1hho"``), a
    ``"pdb_id-model-chain"`` string (``"1hho-1-A"``), or a local file path.
    """
    if isinstance(reference, ProteinChain):
        return reference
    if not isinstance(reference, str):
        return None

    parts = reference.split("-")
    if len(parts) >= 3:
        identifier, model_id, chain_id = parts[0], int(parts[1]), parts[2]
    elif len(parts) == 2:
        identifier, model_id, chain_id = parts[0], int(parts[1]), "A"
    else:
        identifier, model_id, chain_id = reference, 1, "A"

    path = Path(identifier)
    if path.exists() and path.is_file():
        if path.suffix.lower() == ".pdb":
            return ProteinChain.from_pdb(
                str(path), model_id=model_id, chain_id=chain_id
            )
        return ProteinChain.from_cif(str(path), model_id=model_id, chain_id=chain_id)
    return ProteinChain.from_cif(identifier, model_id=model_id, chain_id=chain_id)


def plot_invariant_curves(
    data: pd.DataFrame | list[pd.DataFrame] | Path | str,
    columns: list[str],
    reference: ProteinChain | str | None = None,
    offset: int = 0,
    figsize: tuple[float, float] = (15, 8),
):
    """Overlay-plot invariant values along the residue sequence.

    Each chain in *data* is drawn as a thin gray line.  If a *reference*
    chain is provided it is overlaid as a red line with markers, making it
    easy to spot where the dataset deviates from the reference.

    :param data: Invariant DataFrame, list of DataFrames, or directory of CSVs.
    :param columns: Invariant columns to plot (one subplot per column).
    :param reference: Reference chain — a :class:`ProteinChain`, PDB ID,
        ``"pdb_id-model-chain"`` string, or file path.
    :param offset: Residue-ID offset for aligning the reference.
    :param figsize: Figure size.
    :return: Matplotlib :class:`~matplotlib.figure.Figure`.
    """
    import matplotlib.pyplot as plt

    data = _load_invariant_data(data)
    available = [c for c in columns if c in data.columns]
    if not available:
        raise ValueError("None of the requested columns are present in the data.")

    # Group by chain so each is drawn as a separate gray line.
    chain_keys = [k for k in ("pdb_id", "model_id", "chain_id") if k in data.columns]
    groups = (
        list(data.groupby(chain_keys, sort=False)) if chain_keys else [(None, data)]
    )
    max_len = max(g["residue_id"].max() for _, g in groups)

    fig, axes = plt.subplots(len(available), 1, figsize=figsize, sharex=True)
    if len(available) == 1:
        axes = [axes]

    # --- Dataset chains (gray) ---
    for _, group_df in groups:
        if "residue_id" in group_df.columns:
            residues = group_df["residue_id"].astype(int).tolist()
        else:
            residues = list(range(1, len(group_df) + 1))
        for i, col in enumerate(available):
            values = pd.to_numeric(group_df.get(col), errors="coerce")
            values = _unwrap_torsion(values, col)
            axes[i].plot(residues, values, color="lightgray", alpha=0.4)

    # --- Reference (red) ---
    ref_chain = _resolve_reference(reference)
    if ref_chain is not None:
        bri = ref_chain.get_invariant()
        lai = ref_chain.get_invariant("lai")
        ref_inv = pd.merge(bri, lai, on=INVARIANT_META_COLUMNS)

        res_ids = ref_inv["residue_id"].astype(int)
        # Detect discontinuities in residue to draw separate segments.
        gap_positions = np.flatnonzero(res_ids.diff().fillna(0) > 1)
        segment_starts = [0, *gap_positions.tolist()]
        segment_ends = [*gap_positions.tolist(), len(ref_inv)]

        if offset:
            res_ids = res_ids - offset

        for i, col in enumerate(available):
            if col not in ref_inv.columns:
                continue
            values = pd.to_numeric(ref_inv[col], errors="coerce")
            values = _unwrap_torsion(values, col)
            for s, e in zip(segment_starts, segment_ends):
                axes[i].plot(
                    res_ids.iloc[s:e],
                    values.iloc[s:e],
                    color="darkred",
                    linewidth=1.2,
                    marker="o",
                    markersize=1.7,
                )
        max_len = int(max(max_len, res_ids.max()))

    # --- Formatting ---
    for i, col in enumerate(available):
        axes[i].set_xlim(-1, max_len + 2)
        axes[i].set_ylabel(col, fontsize=9)
        _apply_y_limits(axes[i], col)
        axes[i].set_xticks(list(range(1, max_len, 10)))
        axes[i].grid(alpha=0.3)

    axes[-1].set_xlabel("Residue number")
    fig.tight_layout()
    return fig


def comparison_plots(
    data: pd.DataFrame | list[pd.DataFrame] | Path | str,
    output_dir: Path | None = None,
    reference: ProteinChain | str | None = None,
    offset: int = 0,
) -> dict[str, Any]:
    """Generate BRI and LAI comparison-curve plots.

    For each invariant type present in *data*, overlays every chain as a
    gray line and (optionally) a reference chain as a red line.

    :param data: Invariant DataFrame, list of DataFrames, or directory of CSVs.
    :param output_dir: If given, save ``BRI_comparison.png`` and
        ``LAI_comparison.png`` there.  If ``None``, figures are returned.
    :param reference: Reference chain for overlay (see
        :func:`plot_invariant_curves`).
    :param offset: Residue-ID offset for the reference.
    :return: Dict ``{"bri": fig, "lai": fig}`` (only when *output_dir* is
        ``None``).
    """
    import matplotlib.pyplot as plt

    data = _load_invariant_data(data)
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {}

    for label, cols in [("BRI", BRI_COLUMNS), ("LAI", LAI_COLUMNS)]:
        available = [c for c in cols if c in data.columns]
        if not available:
            continue
        fig = plot_invariant_curves(data, available, reference, offset)
        if output_dir is not None:
            fig.savefig(output_dir / f"{label}_comparison.png", dpi=300)
            plt.close(fig)
            logger.info(f"Saved {label}_comparison.png")
        else:
            results[label.lower()] = fig

    return results
