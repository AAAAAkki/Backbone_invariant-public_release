# -*- coding = utf-8 -*-
"""Protein structure data model with explicit atom connectivity.

This module provides a typed, hierarchical data model for protein structures
as an alternative to the flat DataFrame-based approach. Key features:

- Explicit atom identity and properties (element, charge, B-factor, occupancy)
- Explicit bond connectivity graph via Bond objects
- Hierarchical organization: Entry -> Chain -> Residue -> Atom
- Efficient NumPy-backed coordinate access for invariant computation
- Bidirectional conversion to/from the legacy DataFrame format

Design rationale
----------------
This module aims to achieve and represent:
  - Atom-level identity (atoms are rows, not entities)
  - Bond connectivity (which atoms are bonded to which)
  - Hierarchical relationships (residue/chain grouping is ad-hoc)

with a proper typed hierarchy:

.. code-block:: text

    ProteinEntry
      └── ProteinChain (1:n)
            ├── Residue (1:n, ordered)
            │     └── Atom (1:n, keyed by name)
            └── Bonds (0:n, referencing atoms by serial)

Bonds are auto-detected from known chemical topology based on the
Chemical Component Dictionary (CCD) when building from files. This makes
atom-centric queries natural:

    >>> chain = ProteinChain.from_cif("1abc", 1, "A")
    >>> ca = chain.residues[0].atoms["CA"]
    >>> bonded = chain.get_bonded(ca.serial)  # list of Atom objects
    >>> for a in bonded:
    ...     print(a.name, a.element, a.coord)

"""

from __future__ import annotations

import msgpack
from dataclasses import dataclass, field
from functools import lru_cache
from importlib.resources import files
from itertools import product
from pathlib import Path
from typing import Any, Iterator, get_args

import numpy as np
import pandas as pd

from biotite.structure.io.pdbx import (
    BinaryCIFBlock,
    BinaryCIFFile,
    CIFBlock,
    CIFFile,
    CIFCategory,
)

from bri.base.math_base import FloatArray
from bri.invariant import (
    InvariantType,
    BRI_COLUMNS_FULL,
    LAI_COLUMNS,
    INVARIANT_META_COLUMNS,
    get_invariant,
)

# ── Constants ───────────────────────────────────────────────
BACKBONE_ATOMS: frozenset[str] = frozenset({"N", "CA", "C"})

# Map from mmCIF column names to short internal names (shared with base_util).
COLUMN_MAP: dict[str, str] = {
    "group_PDB": "cate",
    "id": "id",
    "type_symbol": "type_symbol",
    "label_alt_id": "label_alt_id",
    "pdbx_PDB_ins_code": "pdbx_PDB_ins_code",
    "B_iso_or_equiv": "B_iso_or_equiv",
    "pdbx_formal_charge": "pdbx_formal_charge",
    "auth_comp_id": "auth_comp_id",
    "occupancy": "occupancy",
    "label_comp_id": "residue_label",
    "label_atom_id": "atom",
    "label_entity_id": "entity_id",
    "label_asym_id": "chain_id",
    "label_seq_id": "residue_id",
    "Cartn_x": "x",
    "Cartn_y": "y",
    "Cartn_z": "z",
    "pdbx_PDB_model_num": "model_id",
    "auth_asym_id": "auth_chain_id",
    "auth_seq_id": "auth_residue_id",
}
_PDB_ANNOT_MAP = {
    "chain_id": "label_asym_id",
    "res_id": "label_seq_id",
    "res_name": "label_comp_id",
    "ins_code": "pdbx_PDB_ins_code",
    "atom_name": "label_atom_id",
    "element": "type_symbol",
    "occupancy": "occupancy",
    "atom_id": "id",
    "altloc_id": "label_alt_id",
    "charge": "pdbx_formal_charge",
}


def _load_residue_bonds() -> dict[str, list[tuple[str, str, str]]]:
    """Load intra-residue bond topology from the PDB Chemical Component Dictionary.

    Parses ``data/aa-variants-v1.cif`` and returns a dict mapping each
    residue name to its non-hydrogen bond pairs (including backbone bonds
    N—CA, CA—C, C—O).
    """
    _ccd_path = files("bri.data").joinpath("aa-variants-v1.cif")
    ccd = CIFFile.read(_ccd_path)
    result: dict[str, list[tuple[str, str, str]]] = {}

    for name, block in ccd._blocks.items():
        block = CIFFile.deserialize(block)
        bond_cat = block.block.get("chem_comp_bond")
        if bond_cat is None:
            result[name] = []
            continue

        a1_arr = bond_cat["atom_id_1"].as_array()
        a2_arr = bond_cat["atom_id_2"].as_array()
        type_arr = bond_cat["value_order"].as_array()
        pairs = [(a1_arr[i], a2_arr[i], type_arr[i]) for i in range(bond_cat.row_count)]
        result[name] = pairs

    return result


# Intra-residue bond topology loaded from the PDB Chemical Component Dictionary.
RESIDUE_BONDS: dict[str, list[tuple[str, str, str]]] = _load_residue_bonds()

# Columns included by .from_cif().
CIF_COLUMNS = ["id", "type_symbol", "label_alt_id"]
# Columns produced by ProteinChain.to_dataframe().
LEGACY_COLUMNS = [
    "cate",
    "residue_label",
    "atom",
    "entity_id",
    "chain_id",
    "auth_chain_id",
    "residue_id",
    "auth_residue_id",
    "x",
    "y",
    "z",
    "occupancy",
    "model_id",
]


# ── Core Types ──────────────────────────────────────────────


@dataclass(slots=True)
class Atom:
    """A single atom with its 3D coordinates and physicochemical properties.

    Atoms are mutable to allow in-place coordinate perturbation without
    reallocating the entire structure. Use ``ProteinChain.clone()`` if
    you need a snapshot before mutation.

    :param serial: Unique atom serial number within the entry (from mmCIF
        ``atom_site.id``).
    :param name: Atom name as in PDB/mmCIF, e.g. ``"N"``, ``"CA"``, ``"CB"``,
        ``"OG"``.
    :param element: Chemical element symbol, e.g. ``"C"``, ``"N"``, ``"O"``,
        ``"S"``.
    :param x: Cartesian x coordinate in Angstrom.
    :param y: Cartesian y coordinate in Angstrom.
    :param z: Cartesian z coordinate in Angstrom.
    :param occupancy: Occupancy (0.0–1.0). Default 1.0.
    :param b_factor: B-factor (temperature factor). Default 0.0.
    :param alt_loc: Alternate conformation indicator.  If an atom is provides
        in more than one position, then a non-blank alternate alternate-location
        indicator must be used (``"A"``, ``"B"``) for each atomic position.
    """

    serial: int
    name: str
    element: str
    x: float
    y: float
    z: float
    occupancy: float = 1.0
    b_factor: float = 0.0
    alt_loc: str = "."
    _coord: FloatArray | None = field(default=None, repr=False)

    @property
    def coord(self) -> FloatArray:
        """Coordinates as a (3,) float64 array (cached)."""
        if self._coord is None:
            self._coord = np.array([self.x, self.y, self.z], dtype=np.float64)
        return self._coord

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Atom):
            return NotImplemented
        return self.serial == other.serial

    def __hash__(self) -> int:
        return hash(self.serial)


@dataclass(slots=True)
class Bond:
    """A covalent bond between two atoms, identified by their serial numbers.

    The bond direction is not meaningful — ``Bond(1, 2)`` is equivalent to
    ``Bond(2, 1)`` for equality and hashing.
    """

    a1: int
    a2: int
    order: str = "SING"  #: Bond type.

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Bond):
            return NotImplemented
        return (self.a1 == other.a1 and self.a2 == other.a2) or (
            self.a1 == other.a2 and self.a2 == other.a1
        )

    def __hash__(self) -> int:
        return hash(frozenset((self.a1, self.a2)))


@dataclass
class Residue:
    """A residue (typically an amino acid) within a protein chain.

    :param name: Three-letter residue name, e.g. ``"ALA"``, ``"GLY"``.
    :param seq_id: Sequence position (from mmCIF ``label_seq_id``).
    :param atoms: Dictionary mapping atom names to ``Atom`` objects.
    :param auth_seq_id: Author-denoted residue ID (may differ from ``seq_id``).
    """

    name: str
    seq_id: int
    atoms: dict[str, list[Atom]] = field(default_factory=dict)
    auth_seq_id: int | None = None

    # -- Convenience accessors (primary conformation) ---------------
    def iter_atoms(self) -> Iterator[Atom]:
        """Iterate over primary-conformation atoms (one per atom name)."""
        for lst in self.atoms.values():
            if lst:
                yield lst[0]

    @property
    def n(self) -> Atom | None:
        """Backbone amide nitrogen (``N``), primary conformation, or ``None``."""
        return self.get_atom("N")

    @property
    def ca(self) -> Atom | None:
        """Backbone Cα atom (``CA``), primary conformation, or ``None``."""
        return self.get_atom("CA")

    @property
    def c(self) -> Atom | None:
        """Backbone carbonyl carbon (``C``), primary conformation, or ``None``."""
        return self.get_atom("C")

    @property
    def has_complete_backbone(self) -> bool:
        """True if N, CA, C each have at least one conformation."""
        return all(self.atoms.get(k) for k in ("N", "CA", "C"))

    # -- Conformation-aware query API --------------------------------

    def get_atom(self, name: str, altloc: str = "") -> Atom | None:
        """Return the atom with *name* and *altloc*, or None.

        If *altloc* is empty, returns the first (primary) conformation.
        """
        lst = self.atoms.get(name)
        if not lst:
            return None
        if not altloc:
            return lst[0]
        for a in lst:
            if a.alt_loc == altloc:
                return a
        return None

    def get_altlocs(self, name: str) -> list[str]:
        """Return all alternate-location labels for an atom name."""
        lst = self.atoms.get(name)
        return [a.alt_loc for a in lst] if lst else []

    def has_alt_conformations(self, name: str) -> bool:
        """True if *name* has more than one conformation."""
        lst = self.atoms.get(name)
        return len(lst) > 1 if lst else False

    def has_any_alt_conformations(self) -> bool:
        """True if any atom in this residue has multiple conformations."""
        return any(len(lst) > 1 for lst in self.atoms.values())

    def iter_all_atoms(self) -> Iterator[Atom]:
        """Iterate over ALL atoms including all alternate conformations."""
        for lst in self.atoms.values():
            yield from lst

    def __repr__(self) -> str:
        parts = []
        for name, lst in self.atoms.items():
            if len(lst) == 1:
                parts.append(name)
            else:
                parts.append(f"{name}(×{len(lst)})")
        return f"Residue({self.name}, seq_id={self.seq_id}, atoms=[{', '.join(parts)}])"


@dataclass
class ProteinChain:
    """A single protein chain — an ordered sequence of residues with bond connectivity.

    This is the central data structure. It stores:

    - An ordered list of ``Residue`` objects.
    - A list of ``Bond`` objects representing all covalent bonds (intra- and inter-residue).
    - A serial-number index for O(1) atom lookup.

    Typical construction is via the ``from_cif`` classmethod, not the constructor directly.

    :param pdb_id: PDB/entry identifier, e.g. ``"1hho"``.
    :param model_id: Model number (1-indexed).
    :param chain_id: Chain identifier, e.g. ``"A"``.
    :param entity_id: Entity identifier from mmCIF.
    :param auth_chain_id: Author-denoted chain ID.
    :param residues: Ordered list of residues.
    :param bonds: List of covalent bonds.
    """

    pdb_id: str
    model_id: int
    chain_id: str
    entity_id: int = 0
    auth_chain_id: str = ""
    polypeptide: bool = False
    entity_type: str = ""
    residues: list[Residue] = field(default_factory=list)
    bonds: list[Bond] = field(default_factory=list)
    #: Serial -> (residue_idx, atom_name) for O(1) atom lookup.
    _serial_index: dict[int, tuple[int, str]] = field(default_factory=dict, repr=False)
    #: Unordered pair of serials -> Bond for O(1) pairwise bond lookup.
    _bond_index: dict[frozenset[int], Bond] = field(default_factory=dict, repr=False)
    #: Serial -> list of neighbor serials for graph traversal.
    _adjacency: dict[int, list[int]] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.bonds:
            self._rebuild_index()
        else:
            self._rebuild_serial_index()

    # -- Indexing --------------------------------------------------------

    def _rebuild_serial_index(self) -> None:
        """Build only the serial-number index (lightweight, no bond graph)."""
        self._serial_index.clear()
        for i, res in enumerate(self.residues):
            for name, atom_list in res.atoms.items():
                for atom in atom_list:
                    self._serial_index[atom.serial] = (i, name)

    def _rebuild_index(self) -> None:
        """Rebuild the serial-number, bond, and adjacency indices."""
        self._rebuild_serial_index()
        self._bond_index.clear()
        self._adjacency.clear()
        for i, res in enumerate(self.residues):
            for name, atom_list in res.atoms.items():
                for atom in atom_list:
                    self._serial_index[atom.serial] = (i, name)
        for bond in self.bonds:
            self._bond_index[frozenset((bond.a1, bond.a2))] = bond
            self._adjacency.setdefault(bond.a1, []).append(bond.a2)
            self._adjacency.setdefault(bond.a2, []).append(bond.a1)

    def bond_distance(self, a1: int, a2: int) -> int:
        """Return the minimum number of bonds between two atoms (graph distance).

        Returns 0 if *a1* and *a2* are the same atom, -1 if there is no path.

        Uses bidirectional BFS: searches from both atoms simultaneously,
        meeting in the middle. This is dramatically faster than unidirectional
        BFS when the path involves branching (e.g. side-chain to backbone).

        >>> chain.bond_distance(o.serial, ca.serial)
        2   # O–C–CA path
        """
        if a1 == a2:
            return 0

        adj = self._adjacency

        # Forward search: a1 -> ...
        f_visited: dict[int, int] = {a1: 0}
        f_queue: list[int] = [a1]

        # Backward search: a2 -> ...
        b_visited: dict[int, int] = {a2: 0}
        b_queue: list[int] = [a2]

        while f_queue and b_queue:
            # Expand the smaller frontier for balance.
            if len(f_queue) <= len(b_queue):
                nxt: list[int] = []
                for node in f_queue:
                    nd = f_visited[node] + 1
                    for nb in adj.get(node, ()):
                        if nb in b_visited:
                            return nd + b_visited[nb]
                        if nb not in f_visited:
                            f_visited[nb] = nd
                            nxt.append(nb)
                f_queue = nxt
            else:
                nxt = []
                for node in b_queue:
                    nd = b_visited[node] + 1
                    for nb in adj.get(node, ()):
                        if nb in f_visited:
                            return nd + f_visited[nb]
                        if nb not in b_visited:
                            b_visited[nb] = nd
                            nxt.append(nb)
                b_queue = nxt

        return -1

    def get_atom(self, serial: int) -> Atom | None:
        """Look up an atom by its serial number. O(1)."""
        idx = self._serial_index.get(serial)
        if idx is None:
            return None
        res_idx, name = idx
        for atom in self.residues[res_idx].atoms.get(name, ()):
            if atom.serial == serial:
                return atom
        return None

    def get_bond_between(self, a1: int, a2: int) -> Bond | None:
        """Return the bond between two atoms, or None. O(1).

        >>> bond = chain.get_bond_between(n.serial, ca.serial)
        >>> if bond:
        ...     print(bond.order)  # 1 = single, 2 = double, etc.
        """
        return self._bond_index.get(frozenset((a1, a2)))

    def get_bonded(self, serial: int) -> list[Atom]:
        """Return all atoms directly bonded to the given atom serial."""
        result: list[Atom] = []
        for neighbor in self._adjacency.get(serial, ()):
            a = self.get_atom(neighbor)
            if a is not None:
                result.append(a)
        return result

    def get_bonds_of(self, serial: int) -> list[Bond]:
        """Return all bonds involving the given atom serial."""
        return [b for b in self.bonds if b.a1 == serial or b.a2 == serial]

    def get_invariant(self, invariant_type: InvariantType = "bri") -> pd.DataFrame:
        """Compute the backbone rigid invariant from this chain.

        This is a convenience wrapper that converts the chain to a DataFrame,
        delegates to :func:`bri.invariant.get_invariant`, and returns a
        DataFrame containing the requested invariant values with metadata of
        residues in this chain.

        :param invariant_type: Which invariant to return:

            - ``"bri"`` — Backbone Rigid Invariant: the 12 coordinate-based
              values describing relative backbone atom positions (columns
              :data:`bri.invariant.BRI_COLUMNS`).
            - ``"lai"`` — Length Angle Invariant: bond lengths, bond angles,
              and torsion angles (columns :data:`bri.invariant.LAI_COLUMNS`).

        :return: A DataFrame with metadata columns (``model_id``,
            ``chain_id``, ``residue_id``, ``residue_label``, ``chain_length``)
            followed by either BRI or LAI columns. Empty if the invariant
            cannot be computed.
        """
        if invariant_type not in get_args(InvariantType):
            raise ValueError(
                f"invariant_type must be one of {InvariantType}, got {invariant_type!r}."
            )
        if not self.polypeptide:
            return pd.DataFrame()

        need_ext = invariant_type == "lai"

        df = self.to_dataframe(backbone_only=True)

        break_residues = df.loc[df["residue_id"].diff() > 1]  # Breaking check
        if not break_residues.empty:
            break_res_ids = [None] + break_residues["residue_id"].to_list() + [None]
            invs = []
            for s, e in zip(break_res_ids[:-1], break_res_ids[1:]):
                seg = self.slice_residues(s, e)
                invs.append(get_invariant(seg.to_dataframe(True), need_ext))
            inv = pd.concat(invs, ignore_index=True)
        else:
            inv = get_invariant(df, angle=need_ext)

        if inv.empty:
            return inv

        cols = [c for c in INVARIANT_META_COLUMNS if c in inv.columns]
        cols += BRI_COLUMNS_FULL if invariant_type == "bri" else LAI_COLUMNS
        return inv.loc[:, cols]

    def save_invariant(
        self, path: str | Path, invariant_type: InvariantType = "bri", **kwargs
    ) -> None:
        """Save the backbone rigid invariant of this chain as a CSV file.

        High-level convenience wrapper around :meth:`get_invariant` plus
        :meth:`pandas.DataFrame.to_csv`.

        :param path: Output file path (``.csv`` extension recommended).
        :param invariant_type: ``"bri"`` or ``"lai"`` — see :meth:`get_invariant`.
        :param kwargs: Additional keyword arguments forwarded to
            :meth:`pandas.DataFrame.to_csv` (e.g. ``sep``, ``index``).
        """
        path = Path(path)
        invariant = self.get_invariant(invariant_type=invariant_type)
        path.parent.mkdir(parents=True, exist_ok=True)
        invariant.to_csv(path, index=False, **kwargs)

    def match(self, **filters) -> bool:  # pyright:ignore[reportMissingParameterType]
        """Return True if this chain satisfies every filter.

        Each filter key is an attribute or property name. Its value is
        interpreted as:

        - a callable ``f(value) -> bool`` — an arbitrary per-attribute
          predicate (e.g. ``num_residues=lambda n: n > 100``);
        - a list / tuple / set / frozenset — membership
          (e.g. ``chain_id=["A", "B"]``);
        - any other value — exact equality (e.g. ``model_id=1``).

        Multiple filters combine by AND: a chain is selected only if it
        passes every one. An unknown attribute name raises ``AttributeError``.
        """
        for attr, want in filters.items():
            have = getattr(self, attr)
            if callable(want):
                if not want(have):
                    return False
            elif isinstance(want, (list, tuple, set, frozenset)):
                if have not in want:
                    return False
            elif have != want:
                return False
        return True

    # -- Backbone coordinate extraction ----------------------------------

    def get_backbone_mask(self) -> FloatArray:
        """Boolean mask of shape (n_residues,) — True where N, CA, C are all present."""
        return np.array([r.has_complete_backbone for r in self.residues])

    def get_backbone_xyz(self) -> FloatArray:
        """N, CA, C coordinates as an (n_residues, 3, 3) float64 array.

        Missing atoms are represented as NaN. The second axis follows the
        order (N, CA, C) and the third axis is (x, y, z).
        """
        n = len(self.residues)
        coords = np.full((n, 3, 3), np.nan, dtype=np.float64)
        for i, res in enumerate(self.residues):
            for j, name in enumerate(("N", "CA", "C")):
                lst = res.atoms.get(name)
                if lst:
                    coords[i, j] = lst[0].coord
        return coords

    def iter_backbone_atoms(self) -> Iterator[Atom]:
        """Iterate over primary backbone atoms (N, CA, C) in residue order."""
        for res in self.residues:
            for name in ("N", "CA", "C"):
                lst = res.atoms.get(name)
                if lst:
                    yield lst[0]

    # -- Mutation / cloning ----------------------------------------------

    def clone(self) -> ProteinChain:
        """Deep-copy this chain (atoms, bonds, residues — everything)."""
        new_residues: list[Residue] = []
        for res in self.residues:
            new_atoms: dict[str, list[Atom]] = {}
            for name, atom_list in res.atoms.items():
                new_list: list[Atom] = []
                for atom in atom_list:
                    new_list.append(
                        Atom(
                            serial=atom.serial,
                            name=atom.name,
                            element=atom.element,
                            x=atom.x,
                            y=atom.y,
                            z=atom.z,
                            occupancy=atom.occupancy,
                            b_factor=atom.b_factor,
                            alt_loc=atom.alt_loc,
                        )
                    )
                new_atoms[name] = new_list
            new_residues.append(
                Residue(
                    name=res.name,
                    seq_id=res.seq_id,
                    atoms=new_atoms,
                    auth_seq_id=res.auth_seq_id,
                )
            )
        new_bonds = [Bond(b.a1, b.a2, b.order) for b in self.bonds]
        return ProteinChain(
            pdb_id=self.pdb_id,
            model_id=self.model_id,
            chain_id=self.chain_id,
            entity_id=self.entity_id,
            auth_chain_id=self.auth_chain_id,
            polypeptide=self.polypeptide,
            entity_type=self.entity_type,
            residues=new_residues,
            bonds=new_bonds,
        )

    def perturb(self, radius: float, rng: np.random.Generator | None = None) -> None:
        """Apply random 3D perturbation to all atom coordinates in-place.

        Each atom is displaced by a random vector whose length is uniformly
        distributed in [0, *radius*].

        :param radius: Maximum displacement in Angstrom.
        :param rng: NumPy random generator (uses default entropy if ``None``).
        """
        # TODO: return a new object with perturbed atoms
        if rng is None:
            rng = np.random.default_rng()
        for res in self.residues:
            for atom_list in res.atoms.values():
                for atom in atom_list:
                    # Random direction on the unit sphere, scaled to [0, radius].
                    v = rng.normal(0, 1, 3)
                    v *= radius * rng.uniform(0, 1) ** (1 / 3) / np.linalg.norm(v)
                    atom.x += v[0]
                    atom.y += v[1]
                    atom.z += v[2]
                    atom._coord = None  # invalidate cached coordinate array

    def check_clean_status(self):
        """Check possible obstacles to invariant computing by running cleaning filters.

        Runs disorder, non-standard residue, and residue completeness checks to
        avoid fatal errors that may stop the computation of invariants.

        Stops when any check fails to save time.
        """
        if not self.polypeptide or self.is_empty:
            return

        import warnings
        from bri.filter import (
            disorder_check,
            standard_residue_check,
            residue_completeness_check,
        )

        df = self.to_dataframe(backbone_only=True)

        # Run essential checks
        disordered = disorder_check(df)
        if not disordered.empty:
            warnings.warn(
                f"{self} includes disordered atoms which may prevent the computation of invariants."
            )
            return

        non_std = standard_residue_check(df, label_length=3)
        if not non_std.empty:
            warnings.warn(
                f"{self} includes non-standard residues which may cause the loss of data."
            )
            return

        incomplete = residue_completeness_check(df)
        if not incomplete.empty:
            warnings.warn(
                f"{self} includes incomplete residues which may prevent the computation of invariants."
            )
            return

    def to_dataframe(self, backbone_only: bool = False) -> pd.DataFrame:
        """Convert to the legacy flat DataFrame format.

        :param backbone_only: If ``True``, only include backbone atoms (N, CA,
            C).
        :return: A DataFrame with columns matching the legacy
            ``Chain._coordinates`` schema (see :data:`LEGACY_COLUMNS`).
        """
        rows: list[dict[str, Any]] = []
        backbone_set = {"N", "CA", "C"} if backbone_only else None
        for res in self.residues:
            for name, atom_list in res.atoms.items():
                if backbone_set is not None and name not in backbone_set:
                    continue
                for atom in atom_list:
                    rows.append(
                        {
                            "cate": "ATOM",
                            "residue_label": res.name,
                            "atom": name,
                            "entity_id": self.entity_id,
                            "chain_id": self.chain_id,
                            "auth_chain_id": self.auth_chain_id,
                            "residue_id": res.seq_id,
                            "auth_residue_id": (
                                res.auth_seq_id
                                if res.auth_seq_id is not None
                                else res.seq_id
                            ),
                            "x": atom.x,
                            "y": atom.y,
                            "z": atom.z,
                            "occupancy": str(atom.occupancy),
                            "model_id": self.model_id,
                            "label_alt_id": atom.alt_loc,
                        }
                    )
        return pd.DataFrame(rows)

    @classmethod
    def from_cif(
        cls,
        path_or_id: str,
        model_id: int = 1,
        chain_id: str = "A",
        detect_bonds_flag: bool = False,
        check_clean_flag: bool = False,
    ) -> ProteinChain:
        """Build a ProteinChain from a CIF file path, URL, or PDB ID.

        The ``polypeptide`` flag is inferred from the entry's entity metadata.

        :param path_or_id: A local file path, an HTTP URL to a CIF file, or a
            4-character PDB ID.
        :param model_id: Model number to extract (default ``1``).
        :param chain_id: Chain identifier to extract (default ``"A"``).
        :param detect_bonds_flag: If ``True``, run bond detection after
            building.
        :param check_clean_flag: If ``True``, run
            :meth:`check_clean_status` after building to warn about residues
            that may prevent invariant computation.
        :return: The loaded :class:`ProteinChain`.
        """
        block, pdb_id = _load_cif_block(path_or_id)
        polypeptide, entity_type = _chain_entity_info(block, chain_id)
        return cls.from_cif_block(
            block,
            pdb_id,
            model_id,
            chain_id,
            polypeptide=polypeptide,
            entity_type=entity_type,
            detect_bonds_flag=detect_bonds_flag,
            check_clean_flag=check_clean_flag,
        )

    @classmethod
    def from_pdb(
        cls,
        file_path: str,
        model_id: int = 1,
        chain_id: str = "A",
        detect_bonds_flag: bool = False,
        check_clean_flag: bool = False,
    ) -> ProteinChain:
        """Build a ProteinChain from a PDB file.

        It reads a PDB file, extracts a single model and chain.

        :param file_path: Path to a PDB-format file.
        :param model_id: Model number to extract (1-indexed, default ``1``).
        :param chain_id: Chain identifier to extract (default ``"A"``).
        :param detect_bonds_flag: If ``True``, run bond detection after
            building.
        :param check_clean_flag: If ``True``, run
            :meth:`check_clean_status` after building to warn about residues
            that may prevent invariant computation.
        :return: The loaded :class:`ProteinChain`.
        """
        from biotite.structure.io.pdb import PDBFile

        path = Path(file_path)
        file = PDBFile.read(path)
        remark4 = file.get_remark(4)
        if remark4 is not None:
            pdb_id = remark4[0].split()[0]
        else:
            pdb_id = path.stem.upper()

        pdb_data = file.get_structure(
            extra_fields=["occupancy", "atom_id", "charge"], altloc="all"
        )

        # Extract the requested model.
        if model_id < 1 or model_id > pdb_data.shape[0]:
            raise ValueError(
                f"model_id={model_id} out of range (file has {pdb_data.shape[0]} model(s))."
            )
        model_struct = pdb_data[model_id - 1]
        annot = model_struct._annot

        # Build atom_data dict with COLUMN_MAP short names.
        atom_data: dict[str, Any] = {}
        for annot_key, cif_key in _PDB_ANNOT_MAP.items():
            if annot_key in annot:
                short_key = COLUMN_MAP.get(cif_key, cif_key)
                val = annot[annot_key]
                if annot_key == "altloc_id":
                    # PDB encodes "no altloc" as a space; normalise to "".
                    val = np.array([s.strip() for s in val])
                atom_data[short_key] = np.asarray(val)

        # Add coordinates.
        atom_data["x"] = model_struct.coord[:, 0].astype(np.float64)
        atom_data["y"] = model_struct.coord[:, 1].astype(np.float64)
        atom_data["z"] = model_struct.coord[:, 2].astype(np.float64)

        atom_data["model_id"] = np.full(len(annot["chain_id"]), model_id)

        # Filter to the requested chain.
        chain_id_arr = atom_data["chain_id"]
        mask = chain_id_arr == chain_id
        if not mask.any():
            raise ValueError(
                f"Chain '{chain_id}' not found in model {model_id} of {path}."
            )
        chain_atom_data = {k: v[mask] for k, v in atom_data.items()}

        # Infer polypeptide status from residue types (PDB lacks CIF entity metadata).
        from bri.base.base_util import basic_amino_acid_20

        residue_names = set(chain_atom_data["residue_label"])
        polypeptide = bool(residue_names & set(basic_amino_acid_20))

        chain = cls.from_dict(
            pdb_id=pdb_id,
            model_id=model_id,
            chain_id=chain_id,
            atom_data=chain_atom_data,
            polypeptide=polypeptide,
            entity_type="polypeptide(L)" if polypeptide else "",
            detect_bonds_flag=detect_bonds_flag,
            check_clean_flag=check_clean_flag,
        )
        return chain

    @classmethod
    def from_cif_block(
        cls,
        block: BinaryCIFBlock | CIFBlock,
        pdb_id: str,
        model_id: int,
        chain_id: str,
        entity_id: int = 0,
        auth_chain_id: str = "",
        polypeptide: bool = False,
        entity_type: str = "",
        detect_bonds_flag: bool = False,
        check_clean_flag: bool = False,
    ) -> ProteinChain:
        """Build a ProteinChain from a parsed mmCIF block.

        :param block: A mmCIF block from biotite (``CIFFile`` or
            ``BinaryCIFFile``).
        :param pdb_id: PDB identifier.
        :param model_id: Model number to extract.
        :param chain_id: Chain identifier to extract.
        :param entity_id: Entity identifier.
        :param auth_chain_id: Author chain identifier.
        :param detect_bonds_flag: If ``True``, run bond detection after
            building.
        :param check_clean_flag: If ``True``, run
            :meth:`check_clean_status` after building to warn about residues
            that may prevent invariant computation.
        :return: The loaded :class:`ProteinChain`.
        """
        cif_category: CIFCategory | None = block.get("atom_site", None)
        if cif_category is None:
            raise KeyError("No atom_site category in CIF block")

        # Columns to extract (keep as numpy arrays, no tolist()).
        cif_cols = [
            k for k, v in COLUMN_MAP.items() if v in LEGACY_COLUMNS or k in CIF_COLUMNS
        ]
        atom_data: dict[str, Any] = {}
        for col in cif_cols:
            if col in cif_category:
                try:
                    short_key = COLUMN_MAP.get(col, col)
                    atom_data[short_key] = cif_category[col].as_array()
                except Exception:
                    pass

        # Filter to the requested model and chain.
        model_arr = atom_data.get("model_id")
        chain_arr = atom_data.get("chain_id")
        if model_arr is None or chain_arr is None:
            raise KeyError("Missing model_id or chain_id in CIF block")

        mask = (model_arr == str(model_id)) & (chain_arr == chain_id)
        filtered = {k: v[mask] for k, v in atom_data.items()}

        return cls.from_dict(
            pdb_id=pdb_id,
            model_id=model_id,
            chain_id=chain_id,
            atom_data=filtered,
            entity_id=entity_id,
            auth_chain_id=auth_chain_id,
            polypeptide=polypeptide,
            entity_type=entity_type,
            detect_bonds_flag=detect_bonds_flag,
            check_clean_flag=check_clean_flag,
        )

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        pdb_id: str,
        model_id: int,
        chain_id: str,
        entity_id: int = 0,
        auth_chain_id: str = "",
        polypeptide: bool = False,
        entity_type: str = "",
        detect_bonds_flag: bool = False,
        check_clean_flag: bool = False,
    ) -> ProteinChain:
        """Build a ProteinChain from a legacy flat DataFrame.

        The DataFrame must contain at minimum the columns:
        ``residue_id``, ``residue_label``, ``atom``, ``x``, ``y``, ``z``.

        :param df: Legacy atom-table DataFrame.
        :param pdb_id: PDB identifier.
        :param model_id: Model number.
        :param chain_id: Chain identifier.
        :param entity_id: Entity identifier.
        :param auth_chain_id: Author chain identifier.
        :param detect_bonds_flag: If ``True``, run bond detection after
            building.
        :param check_clean_flag: If ``True``, run
            :meth:`check_clean_status` after building to warn about residues
            that may prevent invariant computation.
        :return: The loaded :class:`ProteinChain`.
        """
        residues: list[Residue] = []
        # Group by residue_id, preserving order of first appearance.
        seen: set[int] = set()
        for rid, group in df.groupby("residue_id", sort=False):
            try:
                rid = int(rid)  # pyright:ignore[reportArgumentType]
            except ValueError:
                # TODO: Load ligands/small molecules
                continue
            if rid in seen:
                continue
            seen.add(rid)
            first = group.iloc[0]
            atoms: dict[str, list[Atom]] = {}
            for _, row in group.iterrows():
                name = str(row["atom"])
                serial_val = row.get("id", row.get("serial", 0))
                try:
                    serial = int(serial_val)  # pyright:ignore[reportArgumentType]
                except (ValueError, TypeError):
                    serial = 0
                element = str(row.get("type_symbol", ""))
                try:
                    occ = float(row.get("occupancy", 1.0))  # pyright:ignore[reportArgumentType]
                except (ValueError, TypeError):
                    occ = 1.0
                atm = Atom(
                    serial=serial,
                    name=name,
                    element=element,
                    x=float(row["x"]),
                    y=float(row["y"]),
                    z=float(row["z"]),
                    occupancy=occ,
                    alt_loc=str(row.get("label_alt_id", "")),
                )
                atoms.setdefault(name, []).append(atm)
            auth_seq = row.get("auth_residue_id", None)
            if auth_seq is not None:
                try:
                    auth_seq_int = int(auth_seq)
                except (ValueError, TypeError):
                    auth_seq_int = None
            else:
                auth_seq_int = None
            residues.append(
                Residue(
                    name=str(first["residue_label"]),
                    seq_id=int(rid),
                    atoms=atoms,
                    auth_seq_id=auth_seq_int,
                )
            )

        chain = cls(
            pdb_id=pdb_id,
            model_id=model_id,
            chain_id=chain_id,
            entity_id=entity_id,
            auth_chain_id=auth_chain_id,
            polypeptide=polypeptide,
            entity_type=entity_type,
            residues=residues,
        )
        if detect_bonds_flag:
            chain.bonds = detect_bonds(chain)
            chain._rebuild_index()
        if check_clean_flag:
            _ = chain.check_clean_status()
        return chain

    @classmethod
    def from_dict(
        cls,
        pdb_id: str,
        model_id: int,
        chain_id: str,
        atom_data: dict[str, Any],
        entity_id: int = 0,
        auth_chain_id: str = "",
        polypeptide: bool = False,
        entity_type: str = "",
        detect_bonds_flag: bool = False,
        check_clean_flag: bool = False,
    ) -> ProteinChain:
        """Build a ProteinChain directly from column arrays.

        :param pdb_id: PDB identifier.
        :param model_id: Model number.
        :param chain_id: Chain identifier.
        :param atom_data: Dict mapping column names to equal-length arrays.
            **Required** keys: ``residue_id``, ``residue_label``, ``atom``,
            ``x``, ``y``, ``z``. Optional keys: ``id`` (serial),
            ``type_symbol`` (element), ``occupancy``, ``label_alt_id``,
            ``auth_residue_id``.
        :param entity_id: Entity identifier (default ``0``).
        :param auth_chain_id: Author chain identifier.
        :param polypeptide: Whether the chain is a polypeptide.
        :param entity_type: Entity type string.
        :param detect_bonds_flag: If ``True``, run bond detection after
            building.
        :param check_clean_flag: If ``True``, run
            :meth:`check_clean_status` after building to warn about residues
            that may prevent invariant computation.
        :return: The loaded :class:`ProteinChain`.
        """
        n_atoms = len(atom_data["residue_id"])

        # -- Pull columns, avoiding copies when dtype already matches --------
        def _as_arr(val: Any, dtype: Any = None):  # pyright: ignore[reportExplicitAny]
            arr = np.asarray(val) if not isinstance(val, np.ndarray) else val
            if dtype is not None and arr.dtype != dtype:
                return arr.astype(dtype, copy=False)
            return arr

        res_ids = _as_arr(atom_data["residue_id"])
        res_labels = _as_arr(atom_data["residue_label"])
        atom_names = _as_arr(atom_data["atom"])
        xs = _as_arr(atom_data["x"], np.float64)
        ys = _as_arr(atom_data["y"], np.float64)
        zs = _as_arr(atom_data["z"], np.float64)

        # Optional columns
        serials_arr = atom_data.get("id", atom_data.get("serial"))
        if serials_arr is None:
            serials_arr = np.arange(1, n_atoms + 1)
        else:
            serials_arr = _as_arr(serials_arr, np.int64)

        elements_arr = atom_data.get("type_symbol")
        if elements_arr is None:
            elements_arr = np.full(n_atoms, "", dtype=object)
        else:
            elements_arr = _as_arr(elements_arr)

        occupancies_arr = atom_data.get("occupancy")
        if occupancies_arr is None:
            occupancies_arr = np.ones(n_atoms, dtype=np.float64)
        else:
            occupancies_arr = _as_arr(occupancies_arr, np.float64)

        altlocs_arr = atom_data.get("label_alt_id")
        if altlocs_arr is None:
            altlocs_arr = np.full(n_atoms, ".", dtype=object)
        else:
            altlocs_arr = _as_arr(altlocs_arr)

        auth_rids_arr = atom_data.get("auth_residue_id")
        if auth_rids_arr is None:
            auth_rids_arr = res_ids

        # -- Residue grouping via unique and sort. Given that res_ids are ordered, no need to sort again
        unique_rids, first_occ = np.unique(res_ids, return_index=True)
        order = np.argsort(first_occ)
        ordered_rids = unique_rids[order]

        # Find boundaries where residue group changes
        boundary_mask = res_ids[1:] != res_ids[:-1]
        boundaries = np.flatnonzero(boundary_mask) + 1
        starts = np.concatenate([[0], boundaries])
        ends = np.concatenate([boundaries, [n_atoms]])

        residues: list[Residue] = []
        for g_idx, (start, end) in enumerate(zip(starts, ends)):
            rid = ordered_rids[g_idx]
            res_name = str(res_labels[start])

            atoms: dict[str, list[Atom]] = {}
            for idx in range(start, end):
                name = str(atom_names[idx])
                altloc = str(altlocs_arr[idx]).strip()

                atm = Atom(
                    serial=int(serials_arr[idx]),
                    name=name,
                    element=str(elements_arr[idx]),
                    x=float(xs[idx]),
                    y=float(ys[idx]),
                    z=float(zs[idx]),
                    occupancy=float(occupancies_arr[idx]),
                    alt_loc=altloc,
                )
                atoms.setdefault(name, []).append(atm)

            # Auth residue id
            auth_rid: int | None = None
            if auth_rids_arr is not None:
                try:
                    auth_rid = int(auth_rids_arr[start])
                except (ValueError, TypeError):
                    auth_rid = None

            residues.append(
                Residue(
                    name=res_name,
                    seq_id=int(rid),
                    atoms=atoms,
                    auth_seq_id=auth_rid,
                )
            )

        chain = cls(
            pdb_id=pdb_id,
            model_id=model_id,
            chain_id=chain_id,
            entity_id=entity_id,
            auth_chain_id=auth_chain_id,
            polypeptide=polypeptide,
            entity_type=entity_type,
            residues=residues,
        )
        if detect_bonds_flag:
            chain.bonds = detect_bonds(chain)
            chain._rebuild_index()
        if check_clean_flag:
            _ = chain.check_clean_status()
        return chain

    def slice_residues(
        self,
        start_seq_id: int | None = None,
        end_seq_id: int | None = None,
        chain_length: int | None = None,
    ) -> ProteinChain:
        """Return a new ProteinChain filtered to residues in [start, end), or a given
        length from the start.

        If neither of `end_seq_id` or `chain_length` is given, all residues from start
        will be included. However, if both `end_seq_id` and `chain_length` are given,
        an Exception will be raised.

        :param start_seq_id: Inclusive start residue seq_id. If ``None``, starts
            from the beginning of the chain.
        :param end_seq_id: Exclusive end residue seq_id. If ``None``, includes
            residues from start.
        :param chain_length: Length of the chain to include. If ``None``,
            includes residues from start.
        :return: A new chain containing only residues in the specified range.
        """
        if start_seq_id is None:
            start_seq_id = self.residues[0].seq_id
        if end_seq_id is None:
            if chain_length is None:
                end_seq_id = self.residues[-1].seq_id + 1
            else:
                end_seq_id = start_seq_id + chain_length
        elif chain_length is not None:
            raise ValueError("Cannot receive 3 arguments at the same time.")

        filtered = [r for r in self.residues if start_seq_id <= r.seq_id < end_seq_id]
        sliced_chain = ProteinChain(
            pdb_id=self.pdb_id,
            model_id=self.model_id,
            chain_id=self.chain_id,
            entity_id=self.entity_id,
            auth_chain_id=self.auth_chain_id,
            polypeptide=self.polypeptide,
            entity_type=self.entity_type,
            residues=filtered,
            bonds=list(self.bonds),
        )
        if sliced_chain.bonds:
            sliced_chain.bonds = detect_bonds(sliced_chain)
            sliced_chain._rebuild_index()

        return sliced_chain

    def generate_BID(self, invariant: pd.DataFrame | None = None):
        """Generate a Backbone Invariant Diagram (BID).

        Plots the nine coordinate-based BRI values along the chain as nine
        stacked line traces (one per atom coordinate), making it easy to spot
        regular regions (flat traces) versus loops and turns (spikes). N, Cα,
        and C coordinates are drawn in red, green, and blue respectively.

        :param invariant: A pre-computed BRI DataFrame. If ``None``, the BRI is
            computed internally via :meth:`get_invariant`.
        :return: The matplotlib :class:`~matplotlib.figure.Figure` containing
            the BID plot.
        """
        from matplotlib import pyplot as plt

        tick_step = 10
        colors = ("red", "green", "blue")
        fig, axes = plt.subplots(9, 1, sharex=True, figsize=(16, 8))

        if invariant is None:
            invariant = self.get_invariant()
        if invariant.empty:
            return fig

        residue_id = invariant.loc[:, "residue_id"]
        bri = invariant.loc[
            :, ["x(N)", "y(N)", "z(N)", "x(A)", "y(A)", "z(A)", "x(C)", "y(C)", "z(C)"]
        ]
        left, right = int(residue_id.min()), int(residue_id.max())
        tick_l = ((left + tick_step - 1) // tick_step) * tick_step
        tick_r = (right // tick_step + 1) * tick_step

        for i, col in enumerate(bri.columns):
            axes[i].plot(residue_id, bri[col], marker=".", color=colors[i % 3])

            axes[i].set_xlim(left - 0.5, right + 0.5)
            axes[i].set_xticks(range(tick_l, tick_r, tick_step))
            axes[i].set_ylim(-2, 2)
            axes[i].set_ylabel(col, fontsize=8, rotation=0)
            axes[i].grid(axis="y", alpha=0.4)

        axes[-1].set_xlabel("Residue number")
        fig.tight_layout()  # pyright:ignore[reportAttributeAccessIssue]
        return fig

    def generate_BIB(self, invariant: pd.DataFrame | None = None):
        """Generate a Backbone Invariant Barcode (BIB).

        Renders the same nine BRI coordinates as three colour-coded strips
        (one per backbone atom N, Cα, C), where the x, y, z components map to
        red, green, and blue intensity. This compact view exposes
        structure-wide patterns across hundreds of residues at a glance.

        :param invariant: A pre-computed BRI DataFrame. If ``None``, the BRI is
            computed internally via :meth:`get_invariant`.
        :return: The matplotlib :class:`~matplotlib.figure.Figure` containing
            the BIB plot.
        """
        from matplotlib import pyplot as plt

        tick_step = 10
        fig, axes = plt.subplots(3, 1, sharex=True, figsize=(16, 3))
        if invariant is None:
            invariant = self.get_invariant()
        if invariant.empty:
            return fig

        residue_id = invariant.loc[:, "residue_id"]
        bri = invariant[
            ["x(N)", "y(N)", "z(N)", "x(A)", "y(A)", "z(A)", "x(C)", "y(C)", "z(C)"]
        ].astype("float")
        bri = (bri + 2) / 4
        bri.iloc[0] = [1] * 9
        left, right = int(residue_id.min()), int(residue_id.max())
        tick_l = ((left + tick_step - 1) // tick_step) * tick_step
        tick_r = (right // tick_step + 1) * tick_step

        for i in range(3):
            data = bri.iloc[:, i * 3 : i * 3 + 3]

            axes[i].set_ylabel(bri.columns[i * 3 + 1][2], fontsize=8, rotation=0)
            axes[i].tick_params(axis="y", left=False, labelleft=False)
            axes[i].set_xticks(range(tick_l, tick_r, tick_step))
            axes[i].set_xlim(left - 0.5, right + 0.5)
            axes[i].imshow(
                data.values.reshape(1, -1, 3),
                aspect="auto",
                extent=(left - 0.5, right + 0.5, 0.5, -0.5),
                norm="linear",
            )
        axes[-1].set_xlabel("Residue number")

        fig.tight_layout()  # pyright:ignore[reportAttributeAccessIssue]
        return fig

    @property
    def is_empty(self) -> bool:
        """True if this chain has no residues."""
        return len(self.residues) == 0

    @property
    def num_residues(self) -> int:
        """Number of residues in the chain."""
        return len(self.residues)

    @property
    def num_atoms(self) -> int:
        """Total number of atoms across all residues (all conformations)."""
        return sum(sum(len(lst) for lst in r.atoms.values()) for r in self.residues)

    @property
    def num_backbone_atoms(self) -> int:
        """Count of N + CA + C atoms (max 3 per residue, primary conformation)."""
        return sum(
            sum(1 for k in ("N", "CA", "C") if r.atoms.get(k)) for r in self.residues
        )

    # -- Dunder methods --------------------------------------------------

    def __repr__(self) -> str:
        kind = "polypeptide" if self.polypeptide else "non-polypeptide"
        return (
            f"ProteinChain(pdb={self.pdb_id}, model={self.model_id}, "
            f"chain={self.chain_id}, {kind}, length={self.num_residues}, "
            f"atoms={self.num_atoms}"
        )

    def __len__(self) -> int:
        return self.num_residues

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ProteinChain):
            return NotImplemented
        return (
            self.pdb_id == other.pdb_id
            and self.model_id == other.model_id
            and self.chain_id == other.chain_id
        )

    def __hash__(self) -> int:
        return hash((self.pdb_id, self.model_id, self.chain_id))


def _chain_entity_info(
    block: BinaryCIFBlock | CIFBlock, chain_id: str
) -> tuple[bool, str]:
    """Infer peptide flag and entity type for *chain_id* from CIF metadata.

    Returns (peptide, entity_type).  Defaults to (False, "") if metadata
    is missing or the chain is not found.
    """
    peptide = False
    entity_type = ""

    entity_cat = block.get("entity", None)
    entity_poly_cat = block.get("entity_poly", None)
    struct_asym_cat = block.get("struct_asym", None)

    if entity_cat is None or struct_asym_cat is None:
        return peptide, entity_type

    # Map label_asym_id → entity_id via struct_asym.
    asym_ids = struct_asym_cat["id"].as_array().tolist()
    asym_entity_ids = struct_asym_cat["entity_id"].as_array().tolist()
    asym_to_entity: dict[str, int] = {}
    for aid, eid in zip(asym_ids, asym_entity_ids):
        try:
            asym_to_entity[aid] = int(eid)
        except (ValueError, TypeError):
            pass

    entity_id = asym_to_entity.get(chain_id)
    if entity_id is None:
        return peptide, entity_type

    # Look up entity type.
    entity_ids = [int(x) for x in entity_cat["id"].as_array().tolist()]
    if entity_id in entity_ids:
        idx = entity_ids.index(entity_id)
        entity_types = entity_cat["type"].as_array().tolist()
        if idx < len(entity_types):
            entity_type = str(entity_types[idx])
            peptide = entity_type.startswith("polypeptide")

    if not peptide and entity_poly_cat is not None:
        poly_entity_ids = [
            int(x) for x in entity_poly_cat["entity_id"].as_array().tolist()
        ]
        if entity_id in poly_entity_ids:
            idx = poly_entity_ids.index(entity_id)
            poly_types = entity_poly_cat["type"].as_array().tolist()
            if idx < len(poly_types):
                entity_type = str(poly_types[idx])
                peptide = entity_type.startswith("polypeptide")

    return peptide, entity_type


# ── Entry-level utilities ─────────────────────────────────────


def on_entry(chain_ids: list[str] | None = None, **kwargs):
    """Decorator: apply a chain-wise function to matching chains of a :class:`ProteinEntry`.

    Chains can be selected with keyword arguments matching :class:`ProteinChain` attributes.
    Scalar values require exact equality; list/tuple/set values check membership.

    :param chain_ids: Convenience shorthand for ``chain_id=chain_ids``.
    :param kwargs: Attribute filters (e.g. ``model_id=1``,
        ``chain_id=['A','B','C']``). Each key must be a field of
        :class:`ProteinChain`. When no filters are given, **all** chains in the
        entry are used.
    :return: A decorator that transforms a chain-wise function into an
        entry-wise function.

    Example::

        >>> from bri.invariant import get_invariant_from_structure
        >>> # all chains in the entry:
        >>> get_entry_invariants = on_entry()(get_invariant_from_structure)
        >>> df = get_entry_invariants(entry, ext=True)
        >>> # by chain IDs (shorthand):
        >>> on_entry(chain_ids=['A', 'B'])(func)
        >>> # by model and chain (flexible kwargs):
        >>> on_entry(model_id=1, chain_id=['A', 'B', 'C'])(func)
        >>> on_entry(model_id=[1, 2], chain_id=['A', 'B'])(func)
    """
    # Resolve chain_id from kwargs vs chain_ids shorthand.
    if chain_ids is not None:
        if "chain_id" in kwargs:
            raise ValueError(
                "Cannot specify both chain_ids and chain_id; use one or the other."
            )
        kwargs["chain_id"] = chain_ids

    def decorator(func_on_chain):
        def wrapper(entry: ProteinEntry, *args, **func_kwargs):
            if kwargs:
                chains = [c for c in entry.chains if c.match(**kwargs)]
            else:
                chains = list(entry.chains)

            results = []
            for chain in chains:
                result = func_on_chain(chain, *args, **func_kwargs)
                if result is not None:
                    results.append(result)
            if not results:
                return pd.DataFrame()
            output = pd.concat(results, ignore_index=True)
            output.insert(0, "pdb_id", entry.pdb_id)
            return output

        return wrapper

    return decorator


@dataclass
class ProteinEntry:
    """A PDB/mmCIF entry containing one or more protein chains.

    :param pdb_id: PDB entry identifier.
    :param chains: List of ``ProteinChain`` objects.
    :param metadata: Arbitrary entry-level metadata (entity info, deposition
        date, etc.).
    """

    pdb_id: str
    chains: list[ProteinChain] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)  # pyright:ignore[reportMissingTypeArgument]

    def __getitem__(self, key: str) -> ProteinChain:
        """Get a chain by its chain_id."""
        for c in self.chains:
            if c.chain_id == key:
                return c
        raise KeyError(f"Chain '{key}' not found in entry {self.pdb_id}")

    def get_chains(self, **filters) -> list[ProteinChain]:  # pyright:ignore[reportMissingParameterType]
        """Return every chain whose attributes satisfy *filters*.

        Filters are interpreted by :meth:`ProteinChain.match` (exact value,
        membership via list/tuple/set, or a per-attribute callable) and
        combined by AND. Returns an empty list when nothing matches.

        Examples
        --------
        >>> entry.get_chains(chain_id="A")
        >>> entry.get_chains(model_id=1, chain_id=["A", "B"])
        >>> entry.get_chains(polypeptide=True, num_residues=lambda n: n > 100)
        """
        return [c for c in self.chains if c.match(**filters)]

    def iter_chains(self) -> Iterator[ProteinChain]:
        """Iterate over the chains in this entry."""
        yield from self.chains

    def iter_residues(self) -> Iterator[Residue]:
        """Iterate over every residue across all chains."""
        for chain in self.chains:
            yield from chain.residues

    def iter_atoms(self) -> Iterator[Atom]:
        """Iterate over primary-conformation atoms in all chains."""
        for chain in self.chains:
            for residue in chain.residues:
                for atom_list in residue.atoms.values():
                    if atom_list:
                        yield atom_list[0]

    def iter_all_atoms(self) -> Iterator[Atom]:
        """Iterate over ALL atoms including alternate conformations."""
        for chain in self.chains:
            for residue in chain.residues:
                for atom_list in residue.atoms.values():
                    yield from atom_list

    @property
    def num_chains(self) -> int:
        """Number of chains in this entry."""
        return len(self.chains)

    @property
    def num_residues(self) -> int:
        """Total number of residues across all chains."""
        return sum(c.num_residues for c in self.chains)

    @property
    def num_atoms(self) -> int:
        """Total number of atoms across all chains (all conformations)."""
        return sum(c.num_atoms for c in self.chains)

    @property
    def peptide(self) -> bool:
        """True if the entry contains polypeptide entities."""
        return any([c.polypeptide for c in self.chains])

    def to_dataframe(self, backbone_only: bool = False) -> pd.DataFrame:
        """Convert all chains to a single legacy DataFrame.

        :param backbone_only: If ``True``, only include backbone atoms (N, CA,
            C).
        :return: One row per atom across all chains, with the columns listed in
            :data:`LEGACY_COLUMNS`.
        """
        dfs = [c.to_dataframe(backbone_only) for c in self.chains]
        if not dfs:
            return pd.DataFrame(columns=LEGACY_COLUMNS)
        return pd.concat(dfs, ignore_index=True)

    def get_entry_invariant(
        self, invariant_type: InvariantType = "bri", **chain_filters
    ) -> pd.DataFrame:
        """Compute backbone rigid invariants for the whole entry.

        Computes the invariant for every matching (and eligible) chain and
        concatenates the results into one DataFrame with a leading ``pdb_id``
        column.

        :param invariant_type: ``"bri"`` or ``"lai"`` — see
            :meth:`ProteinChain.get_invariant`.
        :param chain_filters: :class:`ProteinChain` attribute filters selecting
            which chains to include (e.g. ``model_id=1``, ``chain_id="A"``).
            Omit to process every chain.
        :return: Invariants for all matching chains, prefixed with a
            ``pdb_id`` column.
        """

        chains = [c for c in self.chains if c.match(**chain_filters)]
        invariants = [c.get_invariant(invariant_type=invariant_type) for c in chains]
        if not invariants:
            return pd.DataFrame()
        output = pd.concat(invariants, ignore_index=True)
        output.insert(0, "pdb_id", self.pdb_id)

        return output

    def save_invariant(
        self, path: str | Path, invariant_type: InvariantType = "bri", **chain_filters
    ) -> None:
        """Save backbone rigid invariants for the entry as a CSV file.

        High-level convenience wrapper around :meth:`get_entry_invariant` plus
        :meth:`pandas.DataFrame.to_csv`.

        :param path: Output file path.
        :param invariant_type: ``"bri"`` or ``"lai"`` — see
            :meth:`get_entry_invariant`.
        :param chain_filters: :class:`ProteinChain` attribute filters forwarded
            to :meth:`get_entry_invariant` (e.g. ``chain_id="A"``,
            ``model_id=1``).
        """
        path = Path(path)
        invariant = self.get_entry_invariant(
            invariant_type=invariant_type, **chain_filters
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        invariant.to_csv(path, index=False)

    # -- CIF parsing -----------------------------------------------------

    @classmethod
    def from_cif(
        cls,
        path_or_id: str,
        detect_bonds_flag: bool = False,
        check_clean_flag: bool = False,
    ) -> ProteinEntry:
        """Build a ProteinEntry from a CIF file path, URL, or PDB ID.

        Automatically detects and extracts all polymer chains.

        :param path_or_id: A local file path, an HTTP URL to a CIF file, or a
            4-character PDB ID.
        :param detect_bonds_flag: If ``True``, run bond detection after
            building.
        :param check_clean_flag: If ``True``, run
            :meth:`ProteinChain.check_clean_status` on every chain after
            building to warn about residues that may prevent invariant
            computation.
        :return: The loaded :class:`ProteinEntry`.
        """

        block, pdb_id = _load_cif_block(path_or_id)

        cif_category: CIFCategory | None = block.get("atom_site", None)
        if cif_category is None:
            return cls(pdb_id=pdb_id)

        # -- Extract atom data as numpy arrays (no DataFrame) ----------
        cif_cols = [
            k
            for k, v in COLUMN_MAP.items()
            if v in LEGACY_COLUMNS or k in ("id", "type_symbol", "label_alt_id")
        ]
        atom_data: dict[str, Any] = {}
        for col in cif_cols:
            if col in cif_category:
                try:
                    short_key = COLUMN_MAP.get(col, col)
                    atom_data[short_key] = cif_category[col].as_array()
                except Exception:
                    pass

        if not atom_data:
            return cls(pdb_id=pdb_id)

        # Extract entity info for metadata.
        metadata: dict[str, dict[str, Any]] = {}
        entity_cat = block.get("entity", None)
        if entity_cat is not None:
            metadata["entity"] = {
                "id": (
                    entity_cat["id"].as_array().tolist() if "id" in entity_cat else []
                ),
                "type": (
                    entity_cat["type"].as_array().tolist()
                    if "type" in entity_cat
                    else []
                ),
            }
        entity_poly_cat = block.get("entity_poly", None)
        if entity_poly_cat is not None:
            metadata["entity_poly"] = {
                "entity_id": (
                    entity_poly_cat["entity_id"].as_array().tolist()
                    if "entity_id" in entity_poly_cat
                    else []
                ),
                "type": (
                    entity_poly_cat["type"].as_array().tolist()
                    if "type" in entity_poly_cat
                    else []
                ),
            }

        # -- Group atoms by (model_id, chain_id) ------------------------
        model_arr = atom_data.get("model_id")
        chain_arr = atom_data.get("chain_id")
        res_id_arr = atom_data.get("residue_id")
        if model_arr is None or chain_arr is None or res_id_arr is None:
            return cls(pdb_id=pdb_id)

        # Identify atoms whose residue_id is not a valid integer ("." for missing values, water, ligands...).
        def _is_int_like(x: int | str) -> bool:
            try:
                _ = int(x)
                return True
            except (ValueError, TypeError):
                return False

        valid = np.array([_is_int_like(r) for r in res_id_arr])
        atom_data = {k: v[valid] for k, v in atom_data.items()}
        model_arr = atom_data["model_id"]
        chain_arr = atom_data["chain_id"]

        # Discover unique (model_id, chain_id) pairs in order of first appearance.
        pair_order: list[tuple[str, str]] = []
        for mi, ci in zip(model_arr, chain_arr):
            key = (mi, ci)
            if key not in pair_order:
                pair_order.append(key)

        # -- Build chains --------------------------------
        chains: list[ProteinChain] = []
        entity_arr = atom_data.get("entity_id")
        auth_chain_arr = atom_data.get("auth_chain_id")

        for key in pair_order:
            model_str, chain_id_str = key

            try:
                mid = int(model_str)
            except (ValueError, TypeError):
                continue

            mask = (model_arr == model_str) & (chain_arr == chain_id_str)
            indices = np.flatnonzero(mask)
            if len(indices) == 0:
                continue

            chain_atom_data = {k: v[mask] for k, v in atom_data.items()}

            first = indices[0]
            entity_id = 0
            if entity_arr is not None:
                try:
                    entity_id = int(entity_arr[first])
                except (ValueError, TypeError):
                    pass

            auth_chain = chain_id_str
            if auth_chain_arr is not None:
                try:
                    auth_chain = str(auth_chain_arr[first])
                except (ValueError, TypeError):
                    pass

            polypeptide, etype = _chain_entity_info(block, chain_id_str)

            chain = ProteinChain.from_dict(
                pdb_id=pdb_id,
                model_id=mid,
                chain_id=chain_id_str,
                atom_data=chain_atom_data,
                entity_id=entity_id,
                auth_chain_id=auth_chain,
                polypeptide=polypeptide,
                entity_type=etype,
                detect_bonds_flag=detect_bonds_flag,
            )
            if not chain.is_empty:
                chains.append(chain)

        if check_clean_flag:
            _ = [c.check_clean_status() for c in chains]

        return cls(pdb_id=pdb_id, chains=chains, metadata=metadata)

    @classmethod
    def from_pdb(
        cls,
        file_path: str,
        detect_bonds_flag: bool = False,
        check_clean_flag: bool = False,
    ):
        """Build a ProteinEntry from a PDB-format file.

        Reads every model and chain in the file. Because the legacy PDB format
        carries no entity metadata, the polypeptide flag is inferred from the
        residue types present.

        :param file_path: Path to a PDB-format file.
        :param detect_bonds_flag: If ``True``, run bond detection on every
            chain after building.
        :param check_clean_flag: If ``True``, run
            :meth:`ProteinChain.check_clean_status` on every chain after
            building to warn about residues that may prevent invariant
            computation.
        :return: The loaded :class:`ProteinEntry`.
        """

        from biotite.structure.io.pdb import PDBFile
        from bri.base.base_util import basic_amino_acid_20

        path = Path(file_path)
        file = PDBFile.read(path)
        remark4 = file.get_remark(4)
        if remark4 is not None:
            pdb_id = remark4[0].split()[0]
        else:
            pdb_id = path.stem.upper()

        pdb_data = file.get_structure(
            extra_fields=["occupancy", "atom_id", "charge"], altloc="all"
        )
        model_num = pdb_data.shape[0]

        chains: list[ProteinChain] = []
        for model_i in range(model_num):
            model_struct = pdb_data[model_i]
            annot = model_struct._annot

            # Build atom_data dict with COLUMN_MAP short names.
            atom_data: dict[str, Any] = {}
            for annot_key, cif_key in _PDB_ANNOT_MAP.items():
                if annot_key in annot:
                    short_key = COLUMN_MAP.get(cif_key, cif_key)
                    val = annot[annot_key]
                    if annot_key == "altloc_id":
                        # PDB encodes "no altloc" as a space; normalise to "".
                        val = np.array([s.strip() for s in val])
                    atom_data[short_key] = np.asarray(val)

            # Add coordinates.
            atom_data["x"] = model_struct.coord[:, 0].astype(np.float64)
            atom_data["y"] = model_struct.coord[:, 1].astype(np.float64)
            atom_data["z"] = model_struct.coord[:, 2].astype(np.float64)

            atom_data["model_id"] = np.full(len(annot["chain_id"]), model_i + 1)

            # Identify unique chain IDs for this model.
            chain_id_arr = atom_data["chain_id"]
            for cid in np.unique(chain_id_arr):
                mask = chain_id_arr == cid
                chain_atom_data = {k: v[mask] for k, v in atom_data.items()}

                # Infer polypeptide status from residue types.
                residue_names = set(chain_atom_data["residue_label"])
                polypeptide = bool(residue_names & set(basic_amino_acid_20))

                chain = ProteinChain.from_dict(
                    pdb_id=pdb_id,
                    model_id=model_i + 1,
                    chain_id=str(cid),
                    atom_data=chain_atom_data,
                    polypeptide=polypeptide,
                    entity_type="polypeptide(L)" if polypeptide else "",
                    detect_bonds_flag=detect_bonds_flag,
                )
                if not chain.is_empty:
                    chains.append(chain)

        if check_clean_flag:
            _ = [c.check_clean_status() for c in chains]

        return cls(pdb_id=pdb_id, chains=chains)

    def __repr__(self) -> str:
        return (
            f"ProteinEntry(pdb={self.pdb_id}, chains={self.num_chains}, "
            f"residues={self.num_residues}, atoms={self.num_atoms})"
        )


# ── CIF loading utilities ────────────────────────────────────


@lru_cache(maxsize=128)
def _load_cif_block(path_or_id: str) -> tuple[BinaryCIFBlock | CIFBlock, str]:
    """Load a mmCIF block from a path, URL, or PDB ID (cached).

    Returns (block, pdb_id).
    """
    from requests import get

    import biotite.database.rcsb as rcsb

    path = Path(path_or_id)
    pdb_id = path.stem.upper() if path.suffix else path_or_id.upper()

    if path.exists():
        try:
            file = CIFFile.read(path)
        except UnicodeDecodeError:
            file = BinaryCIFFile.read(path)
    elif path_or_id.startswith("http"):
        content = get(path_or_id).content
        try:
            file = CIFFile.deserialize(content.decode())
        except UnicodeDecodeError:
            file = BinaryCIFFile.deserialize(
                msgpack.unpackb(content, use_list=True, raw=False)
            )
    else:
        if len(pdb_id) < 5:
            file = CIFFile.read(rcsb.fetch(pdb_id, "cif"))
        else:
            file = BinaryCIFFile.read(rcsb.fetch(pdb_id, "bcif"))

    return file.block, pdb_id


# ── Bond Detection ───────────────────────────────────────────


def detect_bonds(chain: ProteinChain) -> list[Bond]:
    """Detect all covalent bonds in a chain using known chemical topology.

    Detects two categories of bonds:

    1. **Intra-residue bonds** (backbone + side-chain): Based on the
       topology defined in ``RESIDUE_BONDS``, which is loaded from the
       PDB Chemical Component Dictionary (``data/aa-variants-v1.cif``).
       Residues not in this dictionary receive no intra-residue bonds.
    2. **Peptide bonds** (between residues): C(i) → N(i+1).

    Uses a fast path when atoms have single conformations (the common case),
    avoiding dict/set construction for altloc matching.

    :param chain: The chain to detect bonds for. Atoms must already be placed
        in residues.
    :return: Detected bonds. The caller should assign them to ``chain.bonds``
        and call ``chain._rebuild_index()`` afterwards.
    """
    bonds: list[Bond] = []
    _rb_cache: dict[str, tuple[tuple[str, str, str], ...]] = {}

    for i, residue in enumerate(chain.residues):
        # -- Intra-residue bonds (backbone + side-chain, from CCD CIF) --
        rname = residue.name
        if rname not in _rb_cache:
            _rb_cache[rname] = RESIDUE_BONDS.get(rname, ())
        for a1_name, a2_name, bt in _rb_cache[rname]:
            a1_list = residue.atoms.get(a1_name, [])
            a2_list = residue.atoms.get(a2_name, [])
            if not a1_list or not a2_list:
                continue

            # Fast path: single conformation for both atoms
            if len(a1_list) == 1 and len(a2_list) == 1:
                bonds.append(Bond(a1_list[0].serial, a2_list[0].serial, bt))
                continue

            # Slow path: match by altloc
            a1_by_alt = {a.alt_loc: a for a in a1_list}
            a2_by_alt = {a.alt_loc: a for a in a2_list}
            common = set(a1_by_alt) & set(a2_by_alt)
            if common:
                for altloc in common:
                    bonds.append(
                        Bond(a1_by_alt[altloc].serial, a2_by_alt[altloc].serial, bt)
                    )
            else:
                for a1, a2 in product(a1_list, a2_list):
                    bonds.append(Bond(a1.serial, a2.serial, bt))

        # -- Inter-residue peptide bond: C(i) → N(i+1) --
        if i + 1 < len(chain.residues):
            c_list = residue.atoms.get("C", [])
            n_list = chain.residues[i + 1].atoms.get("N", [])

            if (
                (chain.residues[i + 1].seq_id > residue.seq_id + 1)
                or not c_list
                or not n_list
            ):
                continue

            # Fast path: single conformation for both atoms
            if len(c_list) == 1 and len(n_list) == 1:
                bonds.append(Bond(c_list[0].serial, n_list[0].serial))
                continue

            # Slow path: match by altloc
            c_by_alt = {a.alt_loc: a for a in c_list}
            n_by_alt = {a.alt_loc: a for a in n_list}
            common = set(c_by_alt) & set(n_by_alt)
            if common:
                for altloc in common:
                    bonds.append(Bond(c_by_alt[altloc].serial, n_by_alt[altloc].serial))
            else:
                for a1, a2 in product(c_list, n_list):
                    bonds.append(Bond(a1.serial, a2.serial))

    return bonds


def detect_disulfide_bonds(
    chain: ProteinChain,
    max_distance: float = 2.5,
) -> list[Bond]:
    """Detect disulfide bonds between CYS SG atoms by distance threshold.

    :param chain: The chain to scan.
    :param max_distance: Maximum SG–SG distance in Angstrom to consider as a
        disulfide bond.
    :return: Detected disulfide bonds.
    """
    sg_atoms: list[tuple[int, Atom]] = []
    for res in chain.residues:
        if res.name == "CYS" and "SG" in res.atoms:
            sg = res.atoms["SG"][0]  # primary conformation
            sg_atoms.append((res.seq_id, sg))

    bonds: list[Bond] = []
    for i in range(len(sg_atoms)):
        seq_i, atom_i = sg_atoms[i]
        for j in range(i + 1, len(sg_atoms)):
            seq_j, atom_j = sg_atoms[j]
            # Skip adjacent residues (already have backbone connectivity).
            if abs(seq_i - seq_j) <= 1:
                continue
            dist = np.linalg.norm(atom_i.coord - atom_j.coord)
            if dist <= max_distance:
                bonds.append(Bond(atom_i.serial, atom_j.serial))

    return bonds


# ── Perturbation utilities ───────────────────────────────────


def perturb_chain(
    chain: ProteinChain,
    radius: float,
    rng: np.random.Generator | None = None,
) -> ProteinChain:
    """Return a new ProteinChain with perturbed atom coordinates.

    :param chain: Source chain.
    :param radius: Maximum perturbation distance in Angstrom.
    :param rng: NumPy random generator.
    :return: A deep copy with perturbed coordinates.
    """
    new_chain = chain.clone()
    new_chain.perturb(radius, rng)
    return new_chain
