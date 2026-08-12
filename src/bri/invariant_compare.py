"""Compare protein structures through their Backbone Rigid Invariants.

Because the BRI is rotation/translation-invariant, two chains can be compared
*without superposition*: their structural distance reduces to a distance
between their BRI vectors. This module provides efficient nearest-neighbour
search over those vectors:

- :func:`group_invariant_compare` — pairwise comparison of a set of chains
  (all pairs, or every chain in one set against another). Returns the
  L-infinity (Chebyshev) or RMS distance for structure, and optionally the
  Hamming distance for sequence.
- :func:`neighbour_distance_search_ckdtree` /
  :func:`neighbour_distance_search_RMSD` — query one chain against a database.
- :func:`get_theo_lipschitz_constant` / :func:`get_e_lipschitz_constant` —
  bound how much a coordinate perturbation can change the BRI, the theoretical
  guarantee that BRI distances are meaningful.

Chains of *different lengths* cannot be compared by BRI; the comparison
functions therefore operate within groups of equal ``chain_length``.

Example::

    >>> from bri.structure import ProteinEntry
    >>> from bri.invariant_compare import group_invariant_compare
    >>> entry = ProteinEntry.from_cif("2k4p")        # NMR ensemble, 20 models
    >>> bri = entry.get_entry_invariant()
    >>> pairs = group_invariant_compare(bri, metric="chebyshev")  # all-pairs distance
"""

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree  # pyright:ignore[reportAttributeAccessIssue]
from sklearn.neighbors import BallTree

from .structure import ProteinChain
from .invariant import invariant_ext, get_invariant, BRI_COLUMNS
from .base.math_base import IntArray, FloatArray, get_RMSD

__all__ = [
    "extract_chain_info",
    "generate_index_table",
    "convert_index_table",
    "coordinate_value_reshape",
    "group_invariant_compare",
    "neighbour_distance_search_RMSD",
    "neighbour_distance_search_ckdtree",
    "get_theo_lipschitz_constant",
    "get_e_lipschitz_constant",
]


def extract_chain_info(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse per-residue invariants into one row per chain.

    :param df: Per-residue invariant DataFrame with a ``residue_label`` column.
    :return: One row per chain (``pdb_id``, ``model_id``, ``chain_id``,
        ``chain_length``) with its residue sequence joined into ``seq``.
    """

    df["residue_label_ascii"] = df["residue_label"].apply(ord)
    chains = (
        df.groupby(["pdb_id", "model_id", "chain_id", "chain_length"], sort=False)
        .agg({"residue_label": lambda x: "".join(x)})
        .reset_index()
    )
    chains = chains.rename(columns={"residue_label": "seq"})
    return chains


def organise_dup_output(
    neighbours: tuple[FloatArray, IntArray], chains: pd.DataFrame
) -> pd.DataFrame:
    """Turn a single-query nearest-neighbour result into a readable table.

    :param neighbours: ``(distances, indices)`` tuple returned by a tree query
        for one query point.
    :param chains: Chain-information table whose row order matches the
        neighbour indices.
    :return: One row per neighbour with its ``distance`` and chain identifiers.
    """
    res = []
    for distance, idx in zip(neighbours[0][0], neighbours[1][0]):
        row = {}
        chain = chains.iloc[idx, :]

        row["distance"] = distance
        row["pdb_id"] = chain["pdb_id"]
        row["model_id"] = chain["model_id"]
        row["chain_id"] = chain["chain_id"]

        res.append(row)

    df_output = pd.DataFrame(res)
    return df_output


def generate_index_table(
    neighbours: tuple[FloatArray, IntArray], reduce: bool = True
) -> pd.DataFrame:
    """Turn a tree-query result into a tidy pair table.

    Converts the ``(distances, indices)`` tuple returned by ``tree.query()``
    into a DataFrame with columns ``distance``, ``idx_1``, ``idx_2``.

    :param neighbours: ``(distances, indices)`` arrays from ``tree.query()`` —
        ``distances[i]`` holds the distances for query point *i* and
        ``indices[i]`` the matching neighbour indices.
    :param reduce: If ``True`` (default), keep only the upper triangle
        (``idx_1 < idx_2``) so each unordered pair appears once — appropriate
        when a set is queried against itself.
    :return: Columns ``distance``, ``idx_1``, ``idx_2``; filtered to
        ``idx_1 < idx_2`` when *reduce* is ``True``.
    """
    distances, indices = neighbours
    n_chains = len(distances)

    # Build list of rows, filtering to idx_1 < idx_2 during construction for efficiency
    rows = []
    for i in range(n_chains):
        for dist, idx in zip(distances[i], indices[i]):
            if dist == np.inf:
                break
            if reduce:  # Only include pairs where idx_1 < idx_2
                if i < idx:
                    rows.append({"distance": dist, "idx_1": i, "idx_2": idx})
            else:
                rows.append({"distance": dist, "idx_1": i, "idx_2": idx})

    if not rows:
        return pd.DataFrame(columns=["distance", "idx_1", "idx_2"])
    return pd.DataFrame(rows)


def convert_index_table(
    idx_table: pd.DataFrame, chains: pd.DataFrame, other: pd.DataFrame, keys: list[str]
) -> pd.DataFrame:
    """Resolve the chain identifiers behind a nearest-neighbour index table.

    Replaces the opaque integer indices produced by a kd-tree / ball-tree query
    with the actual chain identifiers (PDB id, model, chain, ...) of the two
    chains in each pair. Vectorised with pandas merges.

    :param idx_table: DataFrame with columns ``distance``, ``idx_1``,
        ``idx_2`` (as produced by :func:`generate_index_table`).
    :param chains: Chain-information table whose row order matches the
        ``idx_1`` indices.
    :param other: Chain-information table whose row order matches the
        ``idx_2`` indices.
    :param keys: Column names to copy from *chains* / *other* for each member
        of a pair.
    :return: Columns ``distance`` followed by ``{key}1`` and ``{key}2`` for
        every key in *keys*.
    """
    # Use merge operations for efficiency instead of iterrows()
    result = idx_table[["distance", "idx_1", "idx_2"]].copy()

    # Merge with chains to get information for idx_1
    chains_subset = chains[keys].reset_index(drop=True)
    chains_subset.columns = [f"{key}1" for key in keys]
    chains_subset["idx_1"] = chains_subset.index
    result = result.merge(chains_subset, on="idx_1", how="left")

    # Merge with other to get information for idx_2
    other_subset = other[keys].reset_index(drop=True)
    other_subset.columns = [f"{key}2" for key in keys]
    other_subset["idx_2"] = other_subset.index
    result = result.merge(other_subset, on="idx_2", how="left")

    # Reorder columns: distance first, then keys with suffix 1, then keys with suffix 2
    col_order = ["distance"] + [f"{key}1" for key in keys] + [f"{key}2" for key in keys]
    return result.loc[:, col_order]


def coordinate_value_reshape(
    df: pd.DataFrame, chain_len: int, cols: list[str] = BRI_COLUMNS
) -> FloatArray:
    """Reshape per-residue invariant values into one long vector per chain.

    :param df: Per-residue invariant DataFrame for a set of equal-length chains.
    :param chain_len: Number of residues per chain.
    :param cols: Columns to pack into the vector. Defaults to
        :data:`BRI_COLUMNS`.
    :return: Array of shape ``(n_chains, len(cols) * chain_len)`` with NaNs
        replaced by ``0.0``.
    """
    coordinate_value = df.loc[:, cols].to_numpy(dtype=float)
    n_chains = len(coordinate_value) // chain_len
    vector_len = len(cols) * chain_len
    coordinate_value = np.nan_to_num(coordinate_value, nan=0.0)
    shaped_value = np.reshape(coordinate_value, (n_chains, vector_len))
    return shaped_value


def group_invariant_compare(
    x: pd.DataFrame,
    y: pd.DataFrame | None = None,
    cols: list[str] = BRI_COLUMNS,
    seq_compare: bool = False,
    metric: str = "chebyshev",
):
    """Compare chains by their backbone rigid invariants.

    Computes, for every pair of chains, a structural distance from their BRI
    vectors using a fast nearest-neighbour search, and optionally a sequence
    distance (Hamming). When *y* is omitted, every chain in *x* is compared
    against every other chain in *x* (pairwise, upper triangle only); when *y*
    is given, every chain in *x* is compared against every chain in *y*.

    Only chains of the **same** ``chain_length`` can be compared; if *x*
    contains chains of mixed lengths, or is empty, the function returns
    ``None``.

    :param x: DataFrame of per-residue invariants for the base set, with
        columns ``pdb_id``, ``model_id``, ``chain_id``, ``residue_id``,
        ``residue_label``, ``chain_length`` and the comparable invariant values
        (see *cols*). As produced by
        :meth:`ProteinEntry.get_entry_invariant <bri.structure.ProteinEntry.get_entry_invariant>`.
    :param y: DataFrame of invariants for the comparison target, with the same
        schema as *x*. If ``None``, *x* is compared with itself.
    :param cols: Invariant columns to compare on. Defaults to
        :data:`BRI_COLUMNS` (the nine inter-residue BRI coordinates).
    :param seq_compare: If ``True``, also report the Hamming (residue-count)
        sequence difference for each pair. Default ``False``.
    :param metric: Structural distance metric:

        - ``"chebyshev"`` (default) — L-infinity distance via
          :class:`scipy.spatial.cKDTree`. The natural metric for the BRI.
        - ``"rms"`` — root-mean-square distance via
          :class:`sklearn.neighbors.BallTree`.

    :return: Pairwise results with ``distance`` and the identifiers of both
        chains (``pdb_id1``, ``model_id1``, ``chain_id1``, ``pdb_id2``, ...).
        When *seq_compare* is ``True``, the columns ``seq_diff``, ``seq1`` and
        ``seq2`` are added. Returns ``None`` if the inputs are empty or contain
        chains of mixed length.
    """
    same_col = ["pdb_id", "model_id", "chain_id"]
    seq_compare_col = ["pdb_id", "model_id", "chain_id", "seq"]
    reduce_matrix = False
    if y is None:
        y = x
        reduce_matrix = True

    chains_x = extract_chain_info(x)
    chains_y = extract_chain_info(y)
    n_base = len(chains_x)
    n_target = len(chains_y)
    if (n_base < 1) or (n_target < 1):  # no enough chains
        return pd.DataFrame()
    chain_lengths = list(chains_x["chain_length"].unique())
    if len(chain_lengths) > 1:  # chain length not equal
        return pd.DataFrame()
    chain_length = int(chain_lengths[0])

    # reshape atom coordinates in a chain into long vectors
    base_value = coordinate_value_reshape(x, chain_length, cols)
    search_value = coordinate_value_reshape(y, chain_length, cols)
    # nearest-neighbour search with the requested distance metric
    if metric == "chebyshev":
        tree = cKDTree(base_value)
        nn_res = tree.query(search_value, k=len(base_value), p=np.inf, workers=-1)
    else:  # "rms"
        tree = BallTree(base_value, metric=get_RMSD)
        nn_res = tree.query(search_value, k=len(base_value), return_distance=True)

    # Compute structural distance (BRI) comparisons
    idx_table = generate_index_table(nn_res, reduce_matrix)
    distance_res = convert_index_table(idx_table, chains_x, chains_x, same_col)

    if not seq_compare:
        return distance_res

    # Also compute sequence distance comparisons
    base_seq_value = coordinate_value_reshape(x, chain_length, ["residue_label_ascii"])
    targ_seq_value = coordinate_value_reshape(y, chain_length, ["residue_label_ascii"])
    seq_tree = BallTree(base_seq_value, metric="hamming")
    nn_res_seq = seq_tree.query(targ_seq_value, k=n_base, return_distance=True)

    idx_table_seq = generate_index_table(nn_res_seq, reduce_matrix)
    result_seq = convert_index_table(idx_table_seq, chains_x, chains_x, seq_compare_col)

    # Convert normalized Hamming distance to absolute sequence difference
    result_seq = result_seq.rename(columns={"distance": "seq_diff"})
    result_seq["seq_diff"] = (
        (result_seq["seq_diff"] * chain_length).round().astype("int")
    )

    # Merge structural and sequence distance results
    merge_keys = [f"{col}1" for col in same_col] + [f"{col}2" for col in same_col]
    output_full = distance_res.merge(result_seq, on=merge_keys, how="inner")
    return output_full


def neighbour_distance_search_ckdtree(
    target: pd.DataFrame,
    df: pd.DataFrame,
    seq_compare: bool = False,
    cols: list[str] = BRI_COLUMNS,
) -> pd.DataFrame:
    """Query the Chebyshev (L-infinity) distance of one chain against a set.

    Builds a kd-tree over the BRI vectors of *df* and returns, for the single
    query chain *target*, its L-infinity distance to every chain in *df*
    (sorted nearest-first). This is the building block of "find structures
    similar to this one".

    :param target: Per-residue BRI of one query chain (same schema as *df*).
    :param df: Per-residue BRIs of the reference set. All chains (target
        included) must share the same ``chain_length``.
    :param seq_compare: If ``True``, also report the Hamming sequence
        difference to each neighbour. Default ``False``.
    :param cols: Invariant columns to compare on. Defaults to
        :data:`BRI_COLUMNS`.
    :return: One row per reference chain with ``distance`` (L-infinity) and its
        identifiers (``pdb_id``, ``model_id``, ``chain_id``); plus ``seq_diff``
        when *seq_compare* is ``True``. Empty if no comparison is possible.
    """
    chains = extract_chain_info(df)
    n_neighbor = len(chains)
    if n_neighbor < 1:  # no neighbors
        return pd.DataFrame()
    chain_lengths = list(chains["chain_length"].unique())
    if len(chain_lengths) > 1:  # chain length not equal
        return pd.DataFrame()
    chain_length = chain_lengths[0]

    # reshape atom coordinates in a chain into long vectors
    coordinate_value = coordinate_value_reshape(df, chain_length, cols)
    target_value = coordinate_value_reshape(target, chain_length, cols)

    # Build kd_tree using L_infinite distance
    kd_tree = cKDTree(coordinate_value)
    neighbours = kd_tree.query(target_value, k=n_neighbor, p=np.inf)

    # organise output
    df_output = organise_dup_output(neighbours, chains)

    if not seq_compare:
        return df_output

    else:  # KNN with kd_tree and hamming distance
        df_seq_value = coordinate_value_reshape(
            df, chain_length, ["residue_label_ascii"]
        )
        target["residue_label_ascii"] = target["residue_label"].apply(ord)
        target_seq_value = coordinate_value_reshape(
            target, chain_length, ["residue_label_ascii"]
        )
        tree = BallTree(df_seq_value, metric="hamming")
        # make distance matrix with NN
        neighbours = tree.query(target_seq_value, k=n_neighbor, return_distance=True)

        df_seq_diff = organise_dup_output(neighbours, chains)
        df_seq_diff.columns = ["seq_diff", "pdb_id", "model_id", "chain_id"]
        df_seq_diff["seq_diff"] = np.around(df_seq_diff["seq_diff"] * chain_length)
        df_seq_diff["seq_diff"] = df_seq_diff["seq_diff"].astype("int")
        output = df_output.merge(df_seq_diff, on=["pdb_id", "model_id", "chain_id"])
        return output


def neighbour_distance_search_RMSD(
    target: pd.DataFrame,
    df: pd.DataFrame,
    seq_compare: bool = False,
    cols: list[str] = BRI_COLUMNS,
) -> pd.DataFrame:
    """Query the RMS distance of one chain against a set.

    Same as :func:`neighbour_distance_search_ckdtree` but uses the
    root-mean-square (Euclidean-over-coordinates) distance via a
    :class:`sklearn.neighbors.BallTree`, instead of the Chebyshev distance.

    :param target: Per-residue BRI of one query chain (same schema as *df*).
    :param df: Per-residue BRIs of the reference set. All chains (target
        included) must share the same ``chain_length``.
    :param seq_compare: If ``True``, also report the Hamming sequence
        difference to each neighbour. Default ``False``.
    :param cols: Invariant columns to compare on. Defaults to
        :data:`BRI_COLUMNS`.
    :return: One row per reference chain with ``distance`` (RMS) and its
        identifiers (``pdb_id``, ``model_id``, ``chain_id``); plus ``seq_diff``
        when *seq_compare* is ``True``. Empty if no comparison is possible.
    """
    chains = extract_chain_info(df)
    n_neighbor = len(chains)
    if n_neighbor < 1:  # no neighbors
        return pd.DataFrame()
    chain_lengths = list(chains["chain_length"].unique())
    if len(chain_lengths) > 1:  # chain length not equal
        return pd.DataFrame()
    chain_length = chain_lengths[0]

    # reshape atom coordinates in a chain into long vectors
    coordinate_value = coordinate_value_reshape(df, chain_length, cols)
    target_value = coordinate_value_reshape(target, chain_length, cols)

    # make distance matrix with BallTree using RMSD
    tree = BallTree(coordinate_value, metric=get_RMSD)
    neighbours = tree.query(target_value, k=n_neighbor, return_distance=True)

    # organise output
    df_output = organise_dup_output(neighbours, chains)
    if not seq_compare:
        return df_output

    else:  # KNN with kd_tree and hamming distance
        df_seq_value = coordinate_value_reshape(
            df, chain_length, ["residue_label_ascii"]
        )
        target["residue_label_ascii"] = target["residue_label"].apply(ord)
        target_seq_value = coordinate_value_reshape(
            target, chain_length, ["residue_label_ascii"]
        )
        tree = BallTree(df_seq_value, metric="hamming")
        # make distance matrix with NN
        neighbours = tree.query(target_seq_value, k=n_neighbor, return_distance=True)

        df_seq_diff = organise_dup_output(neighbours, chains)
        df_seq_diff.columns = ["seq_diff", "pdb_id", "model_id", "chain_id"]
        df_seq_diff["seq_diff"] = np.around(df_seq_diff["seq_diff"] * chain_length)
        df_seq_diff["seq_diff"] = df_seq_diff["seq_diff"].astype("int")
        output = df_output.merge(df_seq_diff, on=["pdb_id", "model_id", "chain_id"])
        return output


def get_theo_lipschitz_constant(
    chain: ProteinChain, chain_perturbed: ProteinChain
) -> float:
    """Compute the theoretical Lipschitz constant of the BRI map.

    Given an original chain and a perturbed copy, returns an upper bound on how
    much the BRI can change per unit of coordinate perturbation, derived in
    closed form from the backbone bond lengths and angles. This is the
    analytic guarantee that small structural changes produce proportionally
    small BRI changes.

    :param chain: The reference :class:`~bri.structure.ProteinChain`.
    :param chain_perturbed: A (perturbed) :class:`~bri.structure.ProteinChain`
        of the same length.
    :return: The theoretical Lipschitz constant. Compare with
        :func:`get_e_lipschitz_constant` to check tightness.
    """

    bri_full = get_invariant(chain.to_dataframe(backbone_only=True), angle=True)
    bri_pert_full = get_invariant(
        chain_perturbed.to_dataframe(backbone_only=True), angle=True
    )

    bri_ext = invariant_ext(bri_full)
    bri_ext_perturb = invariant_ext(bri_pert_full)
    bri_ext["h_C"] = bri_ext["length(C)"] * np.sin(np.deg2rad(bri_ext["angle(A)"]))
    bri_ext_perturb["h_C"] = bri_ext_perturb["length(C)"] * np.sin(
        np.deg2rad(bri_ext_perturb["angle(A)"])
    )

    length_cols = ["length(N)", "length(A)", "length(C)"]
    L_val = max(
        bri_ext.loc[:, length_cols].max().max(),
        bri_ext_perturb.loc[:length_cols].max().max(),
    )

    L_AC = max(bri_ext["length(C)"].max(), bri_ext_perturb["length(C)"].max())
    l_NA = min(bri_ext["length(A)"].min(), bri_ext_perturb["length(A)"].min())
    l_h_C = min(bri_ext["h_C"].min(), bri_ext_perturb["h_C"].min())
    K = 1 / l_NA + 2 / l_h_C * (1 + 2 * L_AC / l_NA)

    return 2 * (1 + 2 * L_val * K)


def get_e_lipschitz_constant(
    chain: ProteinChain, chain_perturbed: ProteinChain
) -> float:
    """Compute the empirical Lipschitz constant of the BRI map.

    Measures the realised worst-case sensitivity for the given pair of chains:
    the ratio of the largest BRI change to the largest coordinate change. This
    is the empirical counterpart of :func:`get_theo_lipschitz_constant` and
    should be bounded above by it.

    :param chain: The reference :class:`~bri.structure.ProteinChain`.
    :param chain_perturbed: A (perturbed) :class:`~bri.structure.ProteinChain`
        of the same length.
    :return: The empirical Lipschitz constant
        ``max|ΔBRI| / max|Δcoordinates|``.
    """

    inv_cols = ["x(N)", "y(N)", "z(N)", "x(A)", "y(A)", "z(A)", "x(C)", "y(C)", "z(C)"]
    coordinate_cols = ["x", "y", "z"]

    linf_bri = (
        (chain.get_invariant()[inv_cols] - chain_perturbed.get_invariant()[inv_cols])
        .abs()
        .max()
        .max()
    )
    eps_max = (
        (
            chain.to_dataframe()[coordinate_cols]
            - chain_perturbed.to_dataframe()[coordinate_cols]
        )
        .abs()
        .max()
        .max()
    )
    return linf_bri / eps_max
