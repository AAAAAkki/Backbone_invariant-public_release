"""Filtering and validation of protein structures.

This module cleans and validates protein chains so that the Backbone Rigid
Invariant can be computed reliably. Each check returns the **residues or atoms
that fail it** as a :class:`~pandas.DataFrame` (empty when nothing fails), so
checks can be chained and the offending records inspected.

Per-chain checks
----------------

==================================  ==============================================
Function                            What it flags
==================================  ==============================================
:func:`disorder_check`              Atoms with alternate conformations / partial occupancy
:func:`clash_check`                 Backbone atoms closer than a distance threshold
:func:`gap_check`                   Backbone atom pairs farther apart than expected
:func:`residue_continuity_check`    Missing residue numbers (chain breaks)
:func:`residue_completeness_check`  Residues missing one of N, CA, C
:func:`standard_residue_check`      Non-standard (non-20-amino-acid) residues
:func:`residue_angel_check`         Residues with unusual N–Cα–C bond angles
==================================  ==============================================

Each ``entry_*`` counterpart applies the same check to every chain of a
:class:`~bri.structure.ProteinEntry`. The two integrated helpers
:func:`integrated_chainwise_filter` (one chain) and
:func:`entry_integrated_cleaning` (a whole entry by PDB id) run the full
pipeline in one call.
"""

from __future__ import annotations

import itertools
from typing import Any

import pandas as pd
import numpy as np

from .base.base_util import (
    basic_amino_acid_20,
    basic_amino_acid_20_s,
    amino_acid_short,
)
from .base.math_base import get_distance, get_angle
from .structure import ProteinChain, ProteinEntry, on_entry


atom: list[str] = ["N", "CA", "C", "N+1"]
min_test_set: list[tuple[str, str]] = list(itertools.combinations(atom, 2))
atom_position: dict[str, int] = {"N": 0, "CA": 1, "C": 2, "N+1": 3, "CA+1": 4, "C+1": 5}
atom_combination = itertools.combinations(atom_position.keys(), 2)
max_test_set: list[tuple[str, str]] = [
    (a1, a2)
    for a1, a2 in atom_combination
    if (abs(atom_position[a2] - atom_position[a1]) < 4)
]
max_test_set = [
    (a1, a2) for a1, a2 in max_test_set if not (a1.endswith("+1") and a2.endswith("+1"))
]
max_test_dict: dict[tuple[str, str], int] = {
    (a1, a2): abs(atom_position[a2] - atom_position[a1]) for a1, a2 in max_test_set
}
ENTRY_CLEAN_COL = [
    "pdb_id",
    "entity_id",
    "model_id",
    "chain_id",
    "start_residue",
    "chain_length",
    "auth_chain_id",
    "auth_seq_id_start",
    "auth_seq_id_end",
    "seq",
]
MINI_ENTRY_CLEAN_COL = [
    "pdb_id",
    "model_id",
    "chain_id",
    "start_residue",
    "chain_length",
]
DIRTY_COLS = [
    "pdb_id",
    "model_id",
    "chain_id",
    "residue_id",
    "residue_label",
    "type",
]


def _ensure_dataframe(
    chain: pd.DataFrame | ProteinChain, backbone_only: bool = True
) -> pd.DataFrame:
    """Accept either a :class:`~bri.structure.ProteinChain` or a DataFrame.

    A :class:`~bri.structure.ProteinChain` is converted via
    :meth:`~bri.structure.ProteinChain.to_dataframe`; a DataFrame is returned
    unchanged.
    """
    if isinstance(chain, ProteinChain):
        return chain.to_dataframe(backbone_only=backbone_only)
    return chain


@on_entry()
def entry_disorder_check(entry: ProteinEntry) -> pd.DataFrame:
    """Apply :func:`disorder_check` to every chain of an entry.

    :param entry: The :class:`~bri.structure.ProteinEntry` to check.
    :return: All disordered atoms across the entry (empty if none).
    """
    return disorder_check(entry)


@on_entry()
def entry_clash_check(entry: ProteinEntry, lower_bound: float = 1) -> pd.DataFrame:
    """Apply :func:`clash_check` to every chain of an entry.

    :param entry: The :class:`~bri.structure.ProteinEntry` to check.
    :param lower_bound: Minimum allowed distance between atoms, in Å. Default
        ``1``.
    :return: All clashing atoms across the entry (empty if none).
    """
    return clash_check(entry, lower_bound)


@on_entry()
def entry_gap_check(
    entry: ProteinEntry, upper_bound_coefficient: float = 2
) -> pd.DataFrame:
    """Apply :func:`gap_check` to every chain of an entry.

    :param entry: The :class:`~bri.structure.ProteinEntry` to check.
    :param upper_bound_coefficient: Multiplier on the expected bond length
        beyond which a gap is flagged. Default ``2``.
    :return: All atom gaps across the entry (empty if none).
    """
    return gap_check(entry, upper_bound_coefficient)


@on_entry()
def entry_residue_continuity_check(entry: ProteinEntry) -> pd.DataFrame:
    """Apply :func:`residue_continuity_check` to every chain of an entry.

    :param entry: The :class:`~bri.structure.ProteinEntry` to check.
    :return: One row per chain with a discontinuous (broken) sequence (empty
        if none).
    """
    return residue_continuity_check(entry)


@on_entry()
def entry_residue_completeness_check(entry: ProteinEntry) -> pd.DataFrame:
    """Apply :func:`residue_completeness_check` to every chain of an entry.

    :param entry: The :class:`~bri.structure.ProteinEntry` to check.
    :return: All incomplete residues across the entry (empty if none).
    """
    return residue_completeness_check(entry)


@on_entry()
def entry_standard_residue_check(
    entry: ProteinEntry, label_len: int = 3
) -> pd.DataFrame:
    """Apply :func:`standard_residue_check` to every chain of an entry.

    :param entry: The :class:`~bri.structure.ProteinEntry` to check.
    :param label_len: Expected length of the residue label: ``3`` for
        three-letter codes (default), ``1`` for one-letter codes.
    :return: All non-standard residues across the entry (empty if none).
    """
    return standard_residue_check(entry, label_len)


@on_entry()
def entry_residue_angel_check(
    entry: ProteinEntry, origin: str = "CA", points: tuple[str, str] = ("C", "N")
) -> pd.DataFrame:
    """Apply :func:`residue_angel_check` to every chain of an entry.

    :param entry: The :class:`~bri.structure.ProteinEntry` to check.
    :param origin: Atom taken as the vertex of the angle. Default ``"CA"``.
    :param points: Pair of atoms that, together with *origin*, define the
        angle. Default ``("C", "N")``.
    :return: Per-chain minimum and maximum N–Cα–C angles across the entry.
    """
    return residue_angel_check(entry, origin, points)


def entry_integrated_cleaning(
    pdb_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Run the full cleaning pipeline on one PDB entry.

    Downloads (or opens) the entry *pdb_id*, then applies
    :func:`integrated_chainwise_filter` to every chain and summarises the
    result. This is the high-level entry point used by the ``clean`` command-line
    pipeline.

    :param pdb_id: PDB identifier of the structure to clean (e.g. ``"1hho"``).
    :return: A tuple ``(clean_set, dirty_set, chain_number)`` where:

        - *clean_set* — one row per chain that passed all checks, with
          identifying columns (``pdb_id``, ``entity_id``, ``model_id``,
          ``chain_id``, ``start_residue``, ``chain_length``, ``auth_chain_id``,
          ``auth_seq_id_start``, ``auth_seq_id_end``, ``seq``).
        - *dirty_set* — one row per rejected residue/segment with the reason in
          the ``type`` column.
        - *chain_number* — total number of chains found in the entry.
    """
    entry = field_check(pdb_id)
    if (not isinstance(entry, ProteinEntry)) or (len(entry.chains) == 0):
        return (
            pd.DataFrame(),
            pd.DataFrame([{"pdb_id": pdb_id, "type": "Non-protein"}]),
            0,
        )

    chain_num = len(entry.chains)
    entry_integrated_filter = on_entry()(integrated_chainwise_filter)
    cleaning_res = entry_integrated_filter(entry)

    dirty_set = cleaning_res.loc[cleaning_res["type"] != "clean"]
    if "chain_length" in dirty_set.columns:
        dirty_set = dirty_set.loc[:, DIRTY_COLS + ["chain_length"]]
    else:
        dirty_set = dirty_set[:, DIRTY_COLS]
        dirty_set["chain_length"] = np.nan

    clean_set = cleaning_res.loc[cleaning_res["type"] == "clean"]

    if clean_set.empty:
        return clean_set, dirty_set, chain_num

    clean_set = (
        clean_set.groupby(["pdb_id", "model_id", "chain_id", "auth_chain_id"])
        .agg(
            start_residue=("residue_id", "min"),
            seq_max=("residue_id", "max"),
            auth_seq_id_start=("auth_residue_id", "min"),
            auth_seq_id_end=("auth_residue_id", "max"),
            residue_label=("residue_label", lambda x: "".join(x[::3])),
        )
        .reset_index()
    )
    clean_set["chain_length"] = clean_set.apply(
        lambda row: row.seq_max - row.start_residue + 1, axis=1
    )
    clean_set.start_residue = clean_set.start_residue.astype("int")
    clean_set.chain_length = clean_set.chain_length.astype("int")
    clean_set.auth_seq_id_start = clean_set.auth_seq_id_start.astype("int")
    clean_set.auth_seq_id_end = clean_set.auth_seq_id_end.astype("int")

    chain_entity_dict = {c.chain_id: c.entity_id for c in entry.chains}
    clean_set["entity_id"] = clean_set["chain_id"].map(chain_entity_dict)
    clean_set = clean_set.drop(columns=["seq_max"])
    clean_set = clean_set.rename(columns={"residue_label": "seq"})
    clean_set = clean_set[ENTRY_CLEAN_COL]
    return clean_set, dirty_set, chain_num


def integrated_chainwise_filter(
    chain: pd.DataFrame | ProteinChain,
) -> pd.DataFrame:
    """Run the full cleaning pipeline on a single chain.

    Applies the checks below in order, labelling each rejected residue/segment
    with a ``type`` and removing it before the next check runs:

    1. ``"disordered"`` — atoms with alternate conformations or partial occupancy.
    2. ``"chain-break"`` — missing residue numbers (sequence discontinuity).
    3. ``"non-standard"`` — residues outside the 20 canonical amino acids.
    4. ``"incomplete"`` — residues missing any of N, CA, C.
    5. ``"clash"`` — backbone atoms closer than 0.01 Å.
    6. A final ``"chain-break"`` re-check, since removing residues above can
       create new discontinuities.

    Residues that survive every step are returned with ``type == "clean"``.

    :param chain: A :class:`~bri.structure.ProteinChain` or a legacy flat
        DataFrame representing the atoms of a single chain.
    :return: Every input residue, tagged ``"clean"`` or with the first failing
        ``type``. The amino-acid sequence of chains rejected for a break is
        placed in the ``residue_label`` column.
    """

    chain_df = _ensure_dataframe(chain, backbone_only=True)

    dirty_result = pd.DataFrame()
    _orig_labels = chain_df["residue_label"]
    chain_df["residue_label"] = (
        chain_df["residue_label"]
        .map(amino_acid_short, na_action="ignore")
        .fillna(_orig_labels)
    )
    full_seq = ",".join(chain_df.drop_duplicates("residue_id")["residue_label"])

    disordered_atoms = disorder_check(chain_df)
    if not disordered_atoms.empty:
        # remove defect residues
        disordered_atoms["type"] = "disordered"
        dirty_result = sort_dirty_residues([dirty_result, disordered_atoms])
        chain_df = chain_df.loc[
            ~chain_df["residue_id"].isin(dirty_result["residue_id"])
        ]
        if chain_df.empty:
            return dirty_result

    # check chain break
    chain_break = residue_continuity_check(chain_df)
    if not chain_break.empty:
        chain_break["type"] = "chain-break"
        chain_break["residue_label"] = full_seq
        return sort_dirty_residues([dirty_result, chain_break])

    non_typical_atoms = standard_residue_check(chain_df, 1)
    defect_residue_atoms = residue_completeness_check(chain_df)
    if not non_typical_atoms.empty:
        non_typical_atoms["type"] = "non-standard"
        dirty_result = sort_dirty_residues([dirty_result, non_typical_atoms])
    if not defect_residue_atoms.empty:
        defect_residue_atoms["type"] = "incomplete"
        dirty_result = sort_dirty_residues([dirty_result, defect_residue_atoms])
    # remove defect residues (only if at least one check found issues)
    if not non_typical_atoms.empty or not defect_residue_atoms.empty:
        chain_df = chain_df.loc[
            ~chain_df["residue_id"].isin(dirty_result["residue_id"])
        ]
        if chain_df.empty:
            return dirty_result
        # check chain break after removing dirty residues
        chain_break = residue_continuity_check(chain_df)
        if not chain_break.empty:
            chain_break["type"] = "chain-break"
            chain_break["residue_label"] = full_seq
            return sort_dirty_residues([dirty_result, chain_break])

    clashed_atoms = clash_check(chain_df, 0.01)
    # gap_atoms = gap_check(chain)
    # remove defect residues
    if not clashed_atoms.empty:
        clashed_atoms.drop(columns=["x", "y", "z", "occupancy"], inplace=True)
        clashed_atoms["type"] = "clash"
        dirty_result = sort_dirty_residues([dirty_result, clashed_atoms])
        # if gap_atoms is not None:
        #     gap_atoms.drop(columns=['x', 'y', 'z', 'occupancy'], inplace=True)
        #     gap_atoms['type'] = 'gap'
        #     dirty_result = sort_dirty_residues([dirty_result, gap_atoms])
        # remove defect residues
        chain_df = chain_df.loc[
            ~chain_df["residue_id"].isin(dirty_result["residue_id"])
        ]
        if chain_df.empty:
            return dirty_result

    # final chain break check
    chain_break = residue_continuity_check(chain_df)
    if not chain_break.empty:
        chain_break["type"] = "chain-break"
        chain_break["residue_label"] = full_seq
        return sort_dirty_residues([dirty_result, chain_break])

    chain_df["type"] = "clean"
    return pd.concat([chain_df, dirty_result], ignore_index=True)


def disorder_check(
    chain: pd.DataFrame | ProteinChain,
) -> pd.DataFrame:
    """Find atoms whose conformation is ambiguous.

    An atom is flagged when it shares its ``(residue_id, atom)`` with another
    atom (true alternate locations) or has an occupancy that is not 1.0 (partial
    occupancy). Such atoms prevent an unambiguous BRI and are typically removed
    during cleaning.

    :param chain: A :class:`~bri.structure.ProteinChain` or a backbone atom
        DataFrame.
    :return: The disordered atoms (empty if none).
    """
    chain = _ensure_dataframe(chain, backbone_only=True)
    dup = chain.loc[
        chain.duplicated(["residue_id", "chain_id", "model_id", "atom"], keep=False), :
    ]
    partial = chain.loc[
        (~chain["occupancy"].str.startswith("1")) & (chain["occupancy"] != ".")
    ]
    if partial.empty and dup.empty:
        return pd.DataFrame()

    res = pd.concat([dup, partial], ignore_index=True)
    _ = res.drop_duplicates(inplace=True)
    return res


def clash_check(
    chain: pd.DataFrame | ProteinChain, lower_bound: float = 1
) -> pd.DataFrame:
    """Find backbone atom pairs closer than a distance threshold.

    Scans the relevant consecutive backbone atom pairs (e.g. N–CA, CA–C,
    C–N+1, N–N+1) and reports those whose distance falls below *lower_bound*.

    :param chain: A :class:`~bri.structure.ProteinChain` or a backbone atom
        DataFrame.
    :param lower_bound: Minimum allowed distance in Å. Default ``1``; pass a
        small value such as ``0.01`` to flag only severe overlaps.
    :return: The clashing atom pairs with their distance (empty if none).
    """
    chain = _ensure_dataframe(chain, backbone_only=True)
    output = []
    for symbols in min_test_set:
        symbol_next = []
        symbol_names = []
        for s in symbols:
            if s.endswith("+1"):
                s = s[:-2]
                symbol_next.append(s)
            symbol_names.append(s)
        # select chain with target atoms
        selected_atoms = chain.loc[chain["atom"].isin(symbol_names)]
        if len(symbol_next) and symbol_names[0] != symbol_names[1]:
            selected_atoms.loc[
                (selected_atoms["atom"].isin(symbol_next)), "residue_id"
            ] -= 1
        atom1, atom2 = symbols
        # get length
        chain_len = len(selected_atoms["residue_id"].unique())
        if chain_len == 0:
            continue

        # distance
        if symbol_names[0] != symbol_names[1]:
            chain_group = selected_atoms.groupby("residue_id")[["x", "y", "z"]]
            selected_atoms.loc[:, "distance"] = chain_group.diff(-1).apply(
                get_distance, axis=1
            )
            del chain_group
        else:
            selected_atoms.loc[:, "distance"] = (
                selected_atoms[["x", "y", "z"]].diff(-1).apply(get_distance, axis=1)
            )
        selected_atoms.loc[:, "atom1"] = atom1
        selected_atoms.loc[:, "atom2"] = atom2

        selected_atoms = selected_atoms.dropna(how="any")
        selected_atoms = selected_atoms[selected_atoms["distance"] < lower_bound]

        selected_atoms = selected_atoms.drop_duplicates(keep="first")
        output.append(selected_atoms)
    output = pd.concat(output, ignore_index=True)
    if output.empty:
        return pd.DataFrame()
    return output


def gap_check(
    chain: pd.DataFrame | ProteinChain,
    upper_bound_coefficient: float = 2,
) -> pd.DataFrame:
    """Find consecutive backbone atom pairs farther apart than expected.

    For each relevant backbone atom pair, compares its distance against the
    typical bond length times *upper_bound_coefficient* and flags pairs that
    exceed it — a sign of locally stretched or broken geometry.

    :param chain: A :class:`~bri.structure.ProteinChain` or a backbone atom
        DataFrame.
    :param upper_bound_coefficient: Multiplier on the expected bond length.
        Default ``2``.
    :return: The atom pairs flagged as gaps, with their distance (empty if
        none).
    """
    chain = _ensure_dataframe(chain, backbone_only=True)
    output = []
    for symbols, distance in max_test_dict.items():
        symbol_next = []
        symbol_names = []
        for s in symbols:
            if s.endswith("+1"):
                s = s[:-2]
                symbol_next.append(s)
            symbol_names.append(s)
        # select chain with target atoms
        selected_atoms = chain.loc[chain["atom"].isin(symbol_names)]
        if len(symbol_next) and symbol_names[0] != symbol_names[1]:
            selected_atoms.loc[
                (selected_atoms["atom"].isin(symbol_next)), "residue_id"
            ] -= 1
        atom1, atom2 = symbols
        # get length
        chain_len = len(selected_atoms["residue_id"].unique())
        if chain_len == 0:
            continue

        # distance
        if symbol_names[0] != symbol_names[1]:
            chain_group = selected_atoms.groupby("residue_id")[["x", "y", "z"]]
            selected_atoms.loc[:, "distance"] = chain_group.diff(-1).apply(
                get_distance, axis=1
            )
            del chain_group
        else:
            selected_atoms.loc[:, "distance"] = (
                selected_atoms[["x", "y", "z"]].diff(-1).apply(get_distance, axis=1)
            )
        selected_atoms.loc[:, "atom1"] = atom1
        selected_atoms.loc[:, "atom2"] = atom2

        selected_atoms = selected_atoms.dropna(how="any")
        selected_atoms = selected_atoms[
            selected_atoms["distance"] > distance * upper_bound_coefficient
        ]

        selected_atoms = selected_atoms.drop_duplicates(keep="first")
        output.append(selected_atoms)
    output = pd.concat(output, ignore_index=True)
    if output.empty:
        return pd.DataFrame()
    return output


def residue_continuity_check(
    chain: pd.DataFrame | ProteinChain,
) -> pd.DataFrame:
    """Detect missing residue numbers (chain breaks).

    Compares the residue numbers present in *chain* against the full integer
    range from the first to the last; any gap means the chain is discontinuous,
    which prevents a continuous BRI being computed across it.

    :param chain: A :class:`~bri.structure.ProteinChain` or a backbone atom
        DataFrame.
    :return: One row reporting the chain and the comma-separated list of
        missing residue numbers in ``missed_residues`` (empty if the chain is
        continuous).
    """
    chain = _ensure_dataframe(chain, backbone_only=True)
    start, end = chain["residue_id"].min(), chain["residue_id"].max() + 1
    reference = set(range(int(start), int(end)))
    missing = reference - set(chain["residue_id"].unique()) - {0}
    missing = np.array(list(missing), dtype="int")
    missing.sort()
    missing = missing.astype("str")
    if len(missing) == 0:
        return pd.DataFrame()
    sample = chain.iloc[[0]][["model_id", "chain_id"]]
    sample["missed_residues"] = ",".join(missing)
    sample["residue_id"] = 0
    sample["chain_length"] = len(reference) - len(missing)
    return sample


def residue_completeness_check(
    chain: pd.DataFrame | ProteinChain,
) -> pd.DataFrame:
    """Find residues that are missing a backbone atom.

    A complete residue must contain all three backbone atoms N, CA and C
    (three records per residue). Any residue with a different count is
    incomplete and cannot contribute a full BRI.

    :param chain: A :class:`~bri.structure.ProteinChain` or a backbone atom
        DataFrame.
    :return: The incomplete residues (empty if every residue has N, CA and C).
    """
    chain = _ensure_dataframe(chain, backbone_only=True)
    count_dict = chain.value_counts("residue_id", sort=True)
    residue_id = [k for k, v in count_dict.items() if v != 3]
    if len(residue_id) > 0:
        res = chain.loc[chain["residue_id"].isin(residue_id)]
        return res
    return pd.DataFrame()


def standard_residue_check(
    chain: pd.DataFrame | ProteinChain, label_length: int = 3
) -> pd.DataFrame:
    """Find residues that are not one of the 20 canonical amino acids.

    :param chain: A :class:`~bri.structure.ProteinChain` or a backbone atom
        DataFrame.
    :param label_length: Expected residue-label length in *chain*: ``3`` for
        three-letter codes such as ``"ALA"`` (default), ``1`` for one-letter
        codes such as ``"A"``.
    :return: The non-standard residues (empty if every residue is canonical).
    """
    chain = _ensure_dataframe(chain, backbone_only=True)
    if label_length == 3:
        standard_residues = basic_amino_acid_20
    else:
        standard_residues = basic_amino_acid_20_s

    non_standard = chain.loc[~chain["residue_label"].isin(standard_residues)]
    if non_standard.empty:
        return pd.DataFrame()
    return non_standard


def residue_angel_check(
    chain: pd.DataFrame | ProteinChain,
    origin: str = "CA",
    points: tuple[str, str] = ("C", "N"),
) -> pd.DataFrame:
    """Report the range of a backbone bond angle across the chain.

    Pivots the backbone atoms into one row per residue, computes the angle at
    *origin* between the two atoms in *points* for every residue, and returns
    the minimum and maximum observed values.

    :param chain: A :class:`~bri.structure.ProteinChain` or a backbone atom
        DataFrame.
    :param origin: Atom placed at the vertex of the angle. Default ``"CA"``.
    :param points: Pair of atoms that, with *origin*, define the angle.
        Default ``("C", "N")`` — the Cα–C / Cα–N angle.
    :return: One row per chain with the ``min_angle_NAC`` and
        ``max_angle_NAC`` values (in degrees).
    """
    chain = _ensure_dataframe(chain, backbone_only=True)
    dic = {"N": [4, 7], "CA": [7, 10], "C": [10, 13], "O": [13, None]}
    residue_table = chain.pivot(
        index=["model_id", "residue_label", "chain_id", "residue_id"],
        columns=["atom"],
        values=["x", "y", "z"],
    ).reset_index()
    residue_table.columns = [
        "_".join(col) if col[1] else col[0] for col in residue_table.columns
    ]
    residue_table = (
        residue_table.sort_values("residue_id")
        .iloc[:, [0, 1, 2, 3, 6, 10, 14, 5, 9, 13, 4, 8, 12, 7, 11, 15]]
        .reset_index(drop=True)
    )
    residue_data = residue_table.to_numpy()
    origin_indices = dic[origin]
    point1_indices = dic[points[0]]
    point2_indices = dic[points[1]]
    v_1 = (
        residue_data[:, point1_indices[0] : point1_indices[1]]
        - residue_data[:, origin_indices[0] : origin_indices[1]]
    )
    v_2 = (
        residue_data[:, point2_indices[0] : point2_indices[1]]
        - residue_data[:, origin_indices[0] : origin_indices[1]]
    )
    angles = [get_angle(v_1[i], v_2[i]) for i in range(len(v_1))]
    residue_table = residue_table[["model_id", "chain_id"]].drop_duplicates()
    residue_table["min_angle_NAC"] = min(angles)
    residue_table["max_angle_NAC"] = max(angles)

    return residue_table


@on_entry()
def residue_count(entry: ProteinChain) -> pd.DataFrame:
    """Count residues by amino-acid type for each chain.

    :param entry: A :class:`~bri.structure.ProteinChain` (or an
        :class:`~bri.structure.ProteinEntry`, via the :func:`~bri.structure.on_entry`
        decorator).
    :return: One row per chain: the chain identifiers, a column per residue
        label holding its count, and a ``chain_length`` total.
    """

    chain = _ensure_dataframe(entry, backbone_only=True)

    _ = chain.drop_duplicates(subset=["residue_id"], inplace=True)
    residue_count_dic = chain.value_counts("residue_label")
    res = chain.iloc[0][["model_id", "chain_id"]].to_dict()
    res.update(dict(residue_count_dic))
    res = {k: [v] for k, v in res.items()}
    del chain
    output = pd.DataFrame(res)
    output["chain_length"] = output.iloc[:, 2:].sum(axis=1)
    return output


def field_check(entry_id: str) -> ProteinEntry | dict[str, str]:
    """Load an entry, checking that it is a usable protein structure.

    A thin wrapper around
    :meth:`ProteinEntry.from_cif <bri.structure.ProteinEntry.from_cif>` that
    returns a small ``{"pdb_id", "type"}`` dict instead of raising when the
    entry cannot be loaded or contains no polypeptide.

    :param entry_id: PDB identifier to load.
    :return: The loaded :class:`~bri.structure.ProteinEntry`, or a dict with
        ``type="Non-protein"`` if the entry is absent or non-protein.
    """
    try:
        entry = ProteinEntry.from_cif(entry_id)
        if not entry.peptide:
            return {"pdb_id": entry_id, "type": "Non-protein"}
        return entry
    except Exception:
        return {"pdb_id": entry_id, "type": "Non-protein"}


def main_chain_check(file_dict: dict[str, Any]) -> bool:
    """Heuristic: whether a CIF record describes a clean main chain.

    Returns ``False`` when ``entity_poly`` is absent, and ``True`` when the
    structure carries no modified residues (``pdbx_struct_mod_residue`` missing
    or empty).

    :param file_dict: Dictionary of CIF categories for one entry.
    :return: ``True`` if the entry looks like a plain main chain, else
        ``False``.
    """
    if "entity_poly" not in file_dict:
        return False
    if "pdbx_struct_mod_residue" not in file_dict:
        return True
    if not file_dict["pdbx_struct_mod_residue"]:
        return True
    return False


# @on_entry()
# def residue_contiguity_wills_check(
#     entry: ProteinChain,
# ) -> pd.DataFrame:
#     """Check residue contiguity using Will's method.
#
#     :param chain: Chain to check for contiguity
#     :return: DataFrame of discontinuous residues if found, None otherwise
#     """
#
#     chain = _ensure_dataframe(entry, backbone_only=True)
#     residue_numbers = chain["residue_id"].unique()
#     all_consecutive = np.all(residue_numbers[1:] - residue_numbers[:-1] == 1)
#     rear = [False, *(~(residue_numbers[1:] - residue_numbers[:-1] == 1))]
#     front = [*(~(residue_numbers[1:] - residue_numbers[:-1] == 1)), False]
#     res = list(residue_numbers[front]) + list(residue_numbers[rear])
#     res = sorted(res)
#     if not all_consecutive:
#         return chain[chain["residue_id"].isin(res)]
#     return None


def sort_dirty_residues(
    dirty_set: list[pd.DataFrame],
) -> pd.DataFrame:
    """Concatenate, deduplicate and sort the per-check dirty-residue outputs.

    :param dirty_set: List of DataFrames produced by individual checks (each
        may carry a ``type`` label).
    :return: A single DataFrame with one row per rejected residue, sorted by
        ``residue_id`` and carrying the standard dirty columns (``DIRTY_COLS``
        plus ``chain_length``).
    """
    dirty = pd.concat(dirty_set, ignore_index=True)
    if dirty.empty:
        return dirty
    if "chain_length" in dirty.columns:
        dirty = dirty.loc[:, DIRTY_COLS + ["chain_length"]]
    else:
        dirty = dirty.loc[:, DIRTY_COLS]
        dirty["chain_length"] = np.nan
    _ = dirty.drop_duplicates(
        subset=["model_id", "chain_id", "residue_id"], inplace=True
    )
    dirty.sort_values("residue_id", inplace=True)
    return dirty
