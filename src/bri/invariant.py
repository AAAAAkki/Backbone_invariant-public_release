"""Invariant computation.

The **Backbone Rigid Invariant** (BRI) is a set of per-residue numbers that
describe the 3D shape of a protein backbone *independently of how the whole
structure is positioned or oriented in space*. Unlike a root-mean-square-deviation
comparison, BRI needs no prior superposition: two residues with the same BRI
have the same backbone geometry.

Each residue contributes 12 coordinate-based values (the columns in
:data:`BRI_COLUMNS`) that locate the backbone atoms (N, Cα, C) of the
*current* and *next* residue in a local frame anchored on the current residue.

From the backbone geometry one can also derive the **Length Angle Invariant**
(LAI) — bond lengths, bond angles, and torsion angles (columns in
:data:`LAI_COLUMNS`) — via :func:`invariant_ext`.

The most convenient entry point is :meth:`ProteinChain.get_invariant
<bri.structure.ProteinChain.get_invariant>`, which loads the chain and returns a
ready-to-use :class:`~pandas.DataFrame`. The lower-level :func:`get_invariant`
operates directly on an atom-coordinate DataFrame and is useful when the data
does not originate from a :class:`~bri.structure.ProteinChain`::

    >>> from bri.structure import ProteinChain
    >>> chain = ProteinChain.from_cif("1hho", model_id=1, chain_id="A")
    >>> bri = chain.get_invariant()                       # 12 BRI columns + metadata
    >>> lai = chain.get_invariant(invariant_type="lai")   # adds bond/torsion angles

.. seealso::

   :mod:`bri.structure` — the protein data model and the high-level
   :meth:`ProteinChain.get_invariant <bri.structure.ProteinChain.get_invariant>`
   method.
"""

from __future__ import annotations
from typing import Literal, Optional

import numpy as np
import pandas as pd

from .base.math_base import (
    get_distance,
    dot_product,
    cross_product,
    vector_norm,
    vector_round,
    get_angle,
    get_dihedral_angle,
    get_third_side_length,
    FloatArray,
)
from .base.base_util import amino_acid_short

dic: dict[str, list[Optional[int]]] = {
    "N": [4, 7],
    "CA": [7, 10],
    "C": [10, 13],
    "O": [13, None],
}

BTP_ATTR = [
    "|C-N-A|",
    "|N-A-C|",
    "|A-C-N|",
    "TP(NA)_x",
    "TP(NA)_y",
    "TP(AC)_x",
    "TP(AC)_y",
    "TP(CN)_x",
    "TP(CN)_y",
]

#: Type alias for invariant types. ``"bri"`` selects the 12 coordinate-based
#: Backbone Rigid Invariant columns (:data:`BRI_COLUMNS`); ``"lai"`` selects the
#: Length Angle Invariant columns (:data:`LAI_COLUMNS`) of bond lengths, bond
#: angles, and torsion angles.
InvariantType = Literal["bri", "lai"]

#: Metadata columns always included alongside BRI or LAI values. They identify
#: which chain and residue each row belongs to.
INVARIANT_META_COLUMNS = [
    "model_id",
    "chain_id",
    "residue_id",
    "residue_label",
    "chain_length",
]

#: Columns of the **Backbone Rigid Invariant** (BRI) — the 12 coordinate-based
#: values describing relative backbone atom positions in a per-residue local
#: frame. ``N``, ``A``, ``C`` denote the nitrogen, Cα, and carbonyl carbon;
#: a leading atom pair such as ``x(AN)`` is an intra-residue projected distance.
BRI_COLUMNS_FULL = [
    "x(AN)",
    "x(AC)",
    "y(AC)",
    "x(N)",
    "y(N)",
    "z(N)",
    "x(A)",
    "y(A)",
    "z(A)",
    "x(C)",
    "y(C)",
    "z(C)",
]

#: BRI columns used for structural comparison. The first three columns of
#: :data:`~bri.invariant.BRI_COLUMNS` (the intra-residue projected distances
#: ``x(AN)``, ``x(AC)``, ``y(AC)``) are local-only and excluded; the remaining
#: nine inter-residue coordinates carry the comparative backbone geometry.
BRI_COLUMNS = BRI_COLUMNS_FULL[3:]

#: Columns of the **Length Angle Invariant** (LAI) — bond lengths (Å), bond
#: angles (degrees), and torsion angles (degrees) derived from the BRI via
#: :func:`invariant_ext`. ``N``, ``A``, ``C`` denote the N–Cα, Cα–C, and
#: peptide C–N bonds respectively.
LAI_COLUMNS = [
    "length(N)",
    "length(A)",
    "length(C)",
    "angle(N)",
    "angle(A)",
    "angle(C)",
    "tau(NA)",
    "tau(AC)",
    "tau(CN)",
]


def get_invariant(chain: pd.DataFrame, angle: bool = False) -> pd.DataFrame:
    """Compute the Backbone Rigid Invariant (BRI) of a single chain.

    This is the core computation. It takes an atom-coordinate DataFrame for one
    protein chain and returns a per-residue table of the 12 coordinate-based
    BRI values (:data:`BRI_COLUMNS`) plus the three bond-length columns, with
    identifying metadata (``model_id``, ``chain_id``, ``residue_id``,
    ``residue_label``, ``chain_length``). Values are rounded to 3 decimals.

    Most users should call the higher-level
    :meth:`ProteinChain.get_invariant <bri.structure.ProteinChain.get_invariant>`
    instead, which loads and validates the chain for you. Use this function
    directly only when the coordinates do not come from a
    :class:`~bri.structure.ProteinChain`.

    :param chain: A DataFrame representing one chain's backbone atoms, with at
        least the columns ``model_id``, ``chain_id``, ``residue_id``,
        ``residue_label``, ``atom`` (one of ``"N"``, ``"CA"``, ``"C"``) and the
        Cartesian coordinates ``x``, ``y``, ``z``. Such a DataFrame is produced
        by :meth:`ProteinChain.to_dataframe <bri.structure.ProteinChain.to_dataframe>`.
    :param angle: If ``True``, also derive the bond-angle and torsion-angle
        columns of the Length Angle Invariant (:data:`LAI_COLUMNS`) via
        :func:`invariant_ext`. Default ``False``.
    :return: One row per residue with the BRI columns, bond lengths, and (if
        ``angle=True``) the LAI angle columns. Empty if the input has fewer
        than one complete residue.
    """
    if chain.empty:
        return pd.DataFrame()

    # reconstruct DataFrame format
    chain["residue_label"] = chain["residue_label"].map(amino_acid_short)
    chain["atom"] = chain["atom"].map({"CA": "A", "N": "N", "C": "C"})
    residue_table = chain.pivot(
        index=["model_id", "chain_id", "residue_id", "residue_label"],
        columns=["atom"],
        values=["x", "y", "z"],
    ).reset_index()
    residue_table.columns = [
        f"{col[0]}({col[1]})" if col[1] else col[0] for col in residue_table.columns
    ]
    residue_table = (
        residue_table.sort_values("residue_id")
        .iloc[:, [0, 1, 2, 3, 6, 9, 12, 4, 7, 10, 5, 8, 11]]
        .reset_index(drop=True)
    )
    columns = list(residue_table.columns)
    columns = columns[:4] + ["x(AN)", "x(AC)", "y(AC)"] + columns[4:]
    # data value extraction for computation
    length = len(residue_table)
    residue_table = residue_table.to_numpy()
    invariant = np.copy(residue_table)
    extra_info = np.copy(residue_table)

    # first row
    first = residue_table[0]
    invariant[0] = _row_trans(first, first, orthogonal=True)
    extra_info[0] = invariant[0]
    # rest row
    if length > 1:
        invariant[1:] = [
            _row_trans(r, lr, orthogonal=True)
            for r, lr in zip(residue_table[1:, :], residue_table[0:-1, :])
        ]
        extra_info[1:] = [
            _row_trans(r, r, orthogonal=True) for r in residue_table[1:, :]
        ]

    # construct DataFrame
    invariant = np.hstack(
        (invariant[:, :4], extra_info[:, [4, 10, 11]], invariant[:, 4:])
    )
    invariant[:, 4:] = vector_round(invariant[:, 4:].astype(float), 3)
    invariant = pd.DataFrame(invariant, columns=columns)

    # bond length
    for a in ("N", "A", "C"):
        invariant[f"length({a})"] = invariant[[f"x({a})", f"y({a})", f"z({a})"]].apply(
            get_distance, axis=1
        )
        invariant[f"length({a})"] = vector_round(invariant[f"length({a})"])

    invariant.iloc[0, invariant.columns.get_loc("length(A)")] = invariant.iloc[
        0, invariant.columns.get_loc("length(N)")
    ]
    invariant.iloc[0, invariant.columns.get_loc("length(N)")] = np.nan
    invariant["chain_length"] = length

    if angle:
        return invariant_ext(invariant)
    return invariant


def invariant_ext(invariant: pd.DataFrame) -> pd.DataFrame:
    """Extend a BRI with bond angles and torsion angles (the LAI columns).

    Adds the six :data:`LAI_COLUMNS` angle columns — ``angle(N)``,
    ``angle(A)``, ``angle(C)`` (bond angles at N, Cα, C) and ``tau(NA)``,
    ``tau(AC)``, ``tau(CN)`` (torsion angles about the N–Cα, Cα–C, and
    peptide C–N bonds) — computed directly from the coordinate-based BRI
    values. The bond-length columns are assumed to already be present.

    :param invariant: A BRI DataFrame containing at least the columns
        :data:`BRI_COLUMNS` and the bond-length columns ``length(N)``,
        ``length(A)``, ``length(C)`` (as produced by :func:`get_invariant`).
    :return: The same DataFrame with the bond-angle and torsion-angle columns
        added in place, rounded to two decimals.

    .. seealso:: :func:`get_invariant` — the main entry point, which calls this
        function when ``angle=True``.
    """

    tmp = invariant.loc[:, BRI_COLUMNS_FULL].astype("float")
    tmp[["y(AN)", "z(AN)", "z(AC)"]] = [0.0, 0.0, 0.0]

    AN_this = ["x(AN)", "y(AN)", "z(AN)"]
    AC_this = ["x(AC)", "y(AC)", "z(AC)"]
    C_0N = ["x(N)", "y(N)", "z(N)"]
    NA = ["x(A)", "y(A)", "z(A)"]
    AC = ["x(C)", "y(C)", "z(C)"]
    CN_2 = ["x(N_1)", "y(N_1)", "z(N_1)"]
    N_2A_2 = ["x(A_1)", "y(A_1)", "z(A_1)"]
    tmp[CN_2 + N_2A_2] = tmp[C_0N + NA].shift(-1)

    # angle(A), angle(N), angle(C)
    invariant["angle(N)"] = tmp.apply(
        lambda row: get_angle(-1 * row[C_0N], row[NA]), axis=1
    )
    invariant["angle(A)"] = tmp.apply(
        lambda row: get_angle(row[AN_this], row[AC_this]), axis=1
    )
    invariant["angle(C)"] = tmp.apply(
        lambda row: get_angle(-1 * row[AC_this], row[CN_2]), axis=1
    )
    invariant[["angle(A)", "angle(N)", "angle(C)"]] = invariant[
        ["angle(A)", "angle(N)", "angle(C)"]
    ].apply(vector_round, decimal=2)

    # tau(NA), tau(AC), tau(CN)
    invariant["tau(NA)"] = tmp.apply(
        lambda row: get_dihedral_angle(row[C_0N], row[NA], row[AC]), axis=1
    )
    invariant["tau(AC)"] = tmp.apply(
        lambda row: get_dihedral_angle(-1 * row[AN_this], row[AC_this], row[CN_2]),
        axis=1,
    )
    invariant["tau(CN)"] = tmp.apply(
        lambda row: get_dihedral_angle(row[AC_this], row[CN_2], row[N_2A_2]), axis=1
    )
    invariant[["tau(NA)", "tau(AC)", "tau(CN)"]] = invariant[
        ["tau(NA)", "tau(AC)", "tau(CN)"]
    ].apply(vector_round, decimal=2)
    return invariant


def get_invariant_summary(
    invariant: pd.DataFrame, bond_length: bool = True, min_and_max: bool = False
) -> pd.DataFrame:
    """Summarise a chain's BRI by its average and spread per column.

    Collapses a per-residue BRI DataFrame into a single one-row fingerprint of
    the chain: for every invariant column it reports the mean
    (``<col>_mean``) and the standard deviation (``<col>_dev``), and optionally
    the minimum and maximum (``<col>_min``, ``<col>_max``). A ``chain_length``
    column records the number of residues.

    The standard deviation is a useful flexibility indicator: small values
    mean the corresponding backbone geometry is repeated along the chain
    (typical of regular secondary structure), large values mean it varies
    (loops, turns).

    :param invariant: A BRI DataFrame for one chain (e.g. from
        :meth:`ProteinChain.get_invariant <bri.structure.ProteinChain.get_invariant>`).
    :param bond_length: Whether the input includes the bond-length columns
        (``length(N)``, ``length(A)``, ``length(C)``). Default ``True`` — set
        to ``False`` when the DataFrame was produced with ``angle=False``.
    :param min_and_max: If ``True``, also append ``<col>_min`` and
        ``<col>_max`` columns. Default ``False``.
    :return: One-row summary whose ``chain_length`` column gives the residue
        count.
    """
    columns = BRI_COLUMNS_FULL
    if bond_length:
        columns = BRI_COLUMNS_FULL + [
            "length(N)",
            "length(A)",
            "length(C)",
        ]
    invariant_data = invariant[columns].values.astype("float64")

    mean = vector_round(np.average(invariant_data[1:, 3:12], axis=0))
    dev = vector_round(np.std(invariant_data[1:, 3:12], axis=0))
    weak_mean = vector_round(np.average(invariant_data[:, 2:5], axis=0))
    weak_dev = vector_round(np.std(invariant_data[:, 2:5], axis=0))
    res = np.hstack((weak_mean, mean, weak_dev, dev)).reshape(1, -1)

    if bond_length:
        length_mean = np.average(invariant_data[:, 13:15], axis=0)
        length_mean = vector_round(
            np.hstack((np.average(invariant_data[1:, 12], axis=0), length_mean))
        )
        length_dev = np.std(invariant_data[:, 13:15], axis=0)
        length_dev = vector_round(
            np.hstack((np.std(invariant_data[1:, 12], axis=0), length_dev))
        )
        res = np.hstack(
            (weak_mean, mean, length_mean, weak_dev, dev, length_dev)
        ).reshape(1, -1)

    res_col = [col + "_mean" for col in columns] + [col + "_dev" for col in columns]

    if min_and_max:
        invariant_min = np.min(invariant_data[1:, 3:12], axis=0)
        invariant_max = np.max(invariant_data[1:, 3:12], axis=0)
        weak_invariant_min = np.min(invariant_data[:, 2:5], axis=0)
        weak_invariant_max = np.max(invariant_data[:, 2:5], axis=0)
        res = np.hstack(
            (res, weak_invariant_min, invariant_min, weak_invariant_max, invariant_max)
        ).reshape(1, -1)
        res_col = (
            res_col
            + [col + "_min" for col in columns]
            + [col + "_max" for col in columns]
        )

    summary = pd.DataFrame(data=res, columns=res_col)
    summary["chain_length"] = len(invariant)
    return summary


def _row_trans(
    row: pd.Series,
    last_row: Optional[pd.Series] = None,
    update_atoms: Optional[list[str]] = None,
    o: str = "CA",
    orthogonal: bool = True,
) -> pd.Series:
    """Re-express a residue's backbone coordinates in a local frame.

    Used internally by :func:`get_invariant` to convert absolute Cartesian
    coordinates into the translation/rotation-invariant local frame that
    defines the BRI. When *row* and *last_row* are the same residue, the frame
    is built from that residue; otherwise the frame is inherited from the
    previous residue (*last_row*), which is what makes consecutive residues
    comparable.

    :param row: Current residue's coordinates.
    :param last_row: Previous residue's coordinates, used to build the local
        basis. If ``None``, *row* itself is used.
    :param update_atoms: Atom names whose coordinates should be transformed.
        Defaults to ``["N", "CA", "C"]``.
    :param o: Origin atom of the local frame. Defaults to ``"CA"``.
    :param orthogonal: Whether to orthogonalise the second basis vector against
        the first (Gram–Schmidt). Defaults to ``True``.
    :return: A copy of *row* with the coordinates of the selected atoms
        replaced by their local-frame coordinates.
    """

    backbone_order = ("N", "CA", "C")
    if not update_atoms:
        update_atoms = ["N", "CA", "C"]
    if last_row is None:
        last_row = row
    # get the coordinate basis from the previous residue
    orthonormal_basis = _get_basis(last_row, orthogonal=orthogonal)
    # atoms to represent
    res_row = row.copy()
    if np.array_equal(row, last_row):
        # fix origin as CA
        origin = last_row[dic[o][0] : dic[o][1]]
        # iterate atoms N, CA, C
        for a in update_atoms:
            # compute vectors originate from CA, i.e., CA-N, CA-CA, CA-N
            p = (row[dic[a][0] : dic[a][1]] - origin).astype("float64")
            # compute new coordinates and construct results
            c1, c2, c3 = [dot_product(p, v) for v in orthonormal_basis]
            res_row[dic[a][0] : dic[a][1]] = [c1, c2, c3]
        return res_row

    for a in update_atoms:
        # select vector origin as the atom before the current one
        origin_atom = backbone_order[backbone_order.index(a) - 1]
        if a == "N":
            # compute vector origin for C-N+1
            origin = last_row[dic[origin_atom][0] : dic[origin_atom][1]]
        else:
            # select origins for vectors N+1-CA+1, CA+1-C+1
            origin = row[dic[origin_atom][0] : dic[origin_atom][1]]
        # compute vector
        p = (row[dic[a][0] : dic[a][1]] - origin).astype("float64")
        # compute new coordinates and construct results
        c1, c2, c3 = [dot_product(p, v) for v in orthonormal_basis]
        res_row[dic[a][0] : dic[a][1]] = [c1, c2, c3]

    return res_row


def _get_basis(
    row: pd.Series,
    o: str = "CA",
    v1_p: str = "N",
    v2_p: str = "C",
    orthogonal: bool = False,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Build a local orthonormal basis for one residue.

    Constructs three mutually perpendicular unit vectors anchored on atom *o*
    from the two direction vectors ``o → v1_p`` and ``o → v2_p``. This basis
    defines the local frame in which the BRI coordinates are expressed.

    :param row: A residue coordinate row containing the atoms named below.
    :param o: Origin atom of the basis. Defaults to ``"CA"``.
    :param v1_p: Atom defining the first basis direction (``o → v1_p``).
        Defaults to ``"N"``.
    :param v2_p: Atom defining the second basis direction (``o → v2_p``).
        Defaults to ``"C"``.
    :param orthogonal: If ``True``, orthogonalise the second vector against the
        first via Gram–Schmidt before normalising. Defaults to ``False``.
    :return: The three orthonormal basis vectors ``(v1, v2, v3)`` where ``v3``
        is the cross product of the (orthogonalised) ``v1`` and ``v2``.
    """

    # coordinates of CA
    origin = row[dic[o][0] : dic[o][1]]
    # vector CA-N and CA-C
    v1 = (row[dic[v1_p][0] : dic[v1_p][1]] - origin).astype("float64")
    v2 = (row[dic[v2_p][0] : dic[v2_p][1]] - origin).astype("float64")
    if orthogonal:
        v2 = v2 - (dot_product(v1, v2) / dot_product(v1, v1)) * v1
    # normalise v1, v2
    v1 = v1 / vector_norm(v1)
    v2 = v2 / vector_norm(v2)
    v3 = cross_product(v1, v2)
    return v1, v2, v3
