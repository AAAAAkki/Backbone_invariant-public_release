# -*- coding = utf-8 -*-
"""Multi-stage geometry and clash filtering for protein structures.

Stages 1 and 2 collect bond-length and bond-angle outliers (outside *n_sigma*
of the restraint mean) in terms of backbones and sidechains, and return them
as DataFrames whose columns mirror those with ``mean``, ``std``, and
``z_score`` from the restraint dictionary.  An entry **passes** a stage when
**all** its chains are free of outliers.

Stage 3 detects nonbonded atomic clashes across the full model on the
surviving entries.

Usage
-----
>>> from bri.clash_filter import run_pipeline

>>> res = run_pipeline(["1abc", "1def", …])
>>> res["stage1_outliers"]   # DataFrame — backbone bond-length outliers
>>> res["stage2_pass_ids"]   # entries that survived two geometry stages
>>> res["stage3_clashes"]    # DataFrame — nonbonded clash records
"""

from __future__ import annotations

import multiprocessing as mp
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree  # pyright: ignore[reportAttributeAccessIssue]

from bri.base.math_base import get_angle, vector_norm
from bri.base.restraint import (
    get_bond_angle_restraint,
    get_bond_length_restraint,
    ATOMIC_RADII,
)
from bri.structure import Atom, Residue, ProteinChain, ProteinEntry

# ── Constants ─────────────────────────────────────────────────

BACKBONE_ATOMS = ("N", "CA", "C")
_BACKBONE_ORDERED_ATOM_PAIRS = (("N", "CA"), ("CA", "C"), ("C", "N"))

# ==================================================================
#  Lightweight z-score predicates
# ==================================================================


def _check_bond_outlier(r1: Residue, a1: Atom, a2: Atom, n_sigma: float) -> bool:
    """Return True if the bond |z_score| > *n_sigma*."""
    r_bl = get_bond_length_restraint(r1.name, a1.name, a2.name)
    if r_bl is None:
        return False
    dist = float(vector_norm(a1.coord - a2.coord))
    mean, std = r_bl["mean"], r_bl["std"]
    return bool(abs((dist - mean) / std) > n_sigma)


def _check_angle_outlier(
    r2: Residue, a1: Atom, a2: Atom, a3: Atom, n_sigma: float
) -> bool:
    """Return True if the angle |z_score| > *n_sigma*."""
    r = get_bond_angle_restraint(r2.name, a1.name, a2.name, a3.name)
    if r is None:
        return False
    ba = a1.coord - a2.coord
    bc = a3.coord - a2.coord
    angle = get_angle(ba, bc)
    mean, std = r["mean"], r["std"]
    return bool(abs((angle - mean) / std) > n_sigma)


# ==================================================================
#  Base row builders
# ==================================================================


def _base_bond_row(
    chain: ProteinChain,
    a1: Atom,
    a2: Atom,
    r1: Residue,
    r2: Residue,
) -> dict[str, str | float | int]:
    """Common identification columns for a two-atom record."""
    r_bl = get_bond_length_restraint(r1.name, a1.name, a2.name)
    if r_bl is None:
        return {}
    dist = float(vector_norm(a1.coord - a2.coord))
    mean, std = r_bl["mean"], r_bl["std"]
    z = (dist - mean) / std
    return {
        "distance": round(dist, 3),
        "pdb_id": chain.pdb_id,
        "model_id": chain.model_id,
        "chain_id": chain.chain_id,
        "chain_length": chain.num_residues,
        "residue_id_1": r1.seq_id,
        "residue_label_1": r1.name,
        "atom_1": a1.name,
        "alt_1": a1.alt_loc,
        "residue_id_2": r2.seq_id,
        "residue_label_2": r2.name,
        "atom_2": a2.name,
        "alt_2": a2.alt_loc,
        "mean": round(mean, 3),
        "std": round(std, 4),
        "z_score": round(abs(z), 3),
        "ratio": round(dist / mean, 3),
        "defect": round(mean - dist, 3),
        "clash_type": "bond_outlier",
    }


def _base_angle_row(
    chain: ProteinChain,
    a1: Atom,
    a2: Atom,
    a3: Atom,
    r1: Residue,
    r2: Residue,
    r3: Residue,
) -> dict[str, str | float | int]:
    """Common identification columns for a three-atom record."""
    r = get_bond_angle_restraint(r2.name, a1.name, a2.name, a3.name)
    if r is None:
        return {}
    mean, std = r["mean"], r["std"]

    ba = a1.coord - a2.coord
    bc = a3.coord - a2.coord
    angle = get_angle(ba, bc)
    mean, std = r["mean"], r["std"]
    z = (angle - mean) / std
    return {
        "pdb_id": chain.pdb_id,
        "model_id": chain.model_id,
        "chain_id": chain.chain_id,
        "chain_length": chain.num_residues,
        "residue_id_1": r1.seq_id,
        "residue_label_1": r1.name,
        "atom_1": a1.name,
        "alt_1": a1.alt_loc,
        "residue_id_2": r2.seq_id,
        "residue_label_2": r2.name,
        "atom_2": a2.name,
        "alt_2": a2.alt_loc,
        "residue_id_3": r3.seq_id,
        "residue_label_3": r3.name,
        "atom_3": a3.name,
        "alt_3": a3.alt_loc,
        "angle": round(angle, 3),
        "mean": round(mean, 3),
        "std": round(std, 4),
        "z_score": round(abs(z), 3),
        "defect": round(angle - mean, 3),
        "clash_type": "angle_outlier",
    }


# ==================================================================
#  Per-chain outlier collectors  (return list[dict])
# ==================================================================


def _normalize_alt(alt: str) -> str:
    """Normalize an alt_loc label: blank → ``"."``."""
    return alt if alt and alt != "." else "."


def _alt_loc_compatible(alt1: str, alt2: str) -> bool:
    """Return True if two alt_loc labels are compatible (same, or at least one is ``"."``)."""
    a1 = _normalize_alt(alt1)
    a2 = _normalize_alt(alt2)
    return a1 == a2 or a1 == "." or a2 == "."


def _bond_is_backbone(a1: Atom, a2: Atom) -> bool:
    return (a1.name, a2.name) in _BACKBONE_ORDERED_ATOM_PAIRS


def _collect_chain_backbone_outliers(
    chain: ProteinChain, *, n_sigma: float, early_stop: bool = True
) -> tuple[list[dict[str, str | float | int]], list[dict[str, str | float | int]]]:
    """Check backbone bond lengths and bond angles in one pass.

    Uses a vectorized fast path when the chain has no alternate backbone conformations.
    """
    # Fast path: no alt conformations → vectorized NumPy
    if not _chain_has_backbone_alt_confs(chain):
        return _collect_chain_backbone_outliers_vectorized(
            chain, n_sigma=n_sigma, early_stop=early_stop
        )

    # Slow path: alt confs present → iterative scan
    bl_rows: list[dict[str, str | float | int]] = []
    ba_rows: list[dict[str, str | float | int]] = []

    # ── Flatten backbone atoms (including alt confs) into a list ──
    atoms_info: list[tuple[Residue, Atom]] = []
    for res in chain.residues:
        for name in BACKBONE_ATOMS:
            for atom in res.atoms.get(name, ()):
                atoms_info.append((res, atom))

    n = len(atoms_info)
    if n == 0:
        return bl_rows, ba_rows

    # ── Precompute next-compatible index ──
    next_compat: list[int | None] = [None] * n
    for i in range(n):
        _, pi = atoms_info[i]
        for j in range(i + 1, n):
            _, pj = atoms_info[j]
            if _alt_loc_compatible(pi.alt_loc, pj.alt_loc):
                next_compat[i] = j
                break

    # ── Precompute prev-compatible index (backward, for angle checks) ──
    prev_compat: list[int | None] = [None] * n
    for i in range(n - 1, -1, -1):
        _, pi = atoms_info[i]
        for j in range(i - 1, -1, -1):
            _, pj = atoms_info[j]
            if _alt_loc_compatible(pi.alt_loc, pj.alt_loc) and _bond_is_backbone(
                pj, pi
            ):
                prev_compat[i] = j
                break

    # ── Main scan: one pass over backbone atoms ──
    for i in range(n):
        rb, pb = atoms_info[i]

        # ── Bond-length check: pb → next compatible atom ──
        j = next_compat[i]
        if j is not None:
            r_next, a_next = atoms_info[j]
            if _bond_is_backbone(pb, a_next) and (r_next.seq_id - rb.seq_id) <= 1:
                if _check_bond_outlier(rb, pb, a_next, n_sigma):
                    bl_rows.append(_base_bond_row(chain, pb, a_next, rb, r_next))
                    if early_stop:
                        return bl_rows, ba_rows

        # ── Angle check: pa → pb → a_next  (pb as central atom) ──
        if j is None:
            continue
        k = prev_compat[i]
        if k is None:
            continue
        ra, pa = atoms_info[k]

        if (rb.seq_id - ra.seq_id) > 1:
            continue

        if (pa.name, pb.name, a_next.name) not in (
            ("N", "CA", "C"),
            ("CA", "C", "N"),
            ("C", "N", "CA"),
        ):
            continue

        if _check_angle_outlier(rb, pa, pb, a_next, n_sigma):
            ba_rows.append(_base_angle_row(chain, pa, pb, a_next, ra, rb, r_next))
            if early_stop:
                return bl_rows, ba_rows

    return bl_rows, ba_rows


def _chain_has_backbone_alt_confs(chain: ProteinChain) -> bool:
    """Return True if any backbone atom (N, CA, C) has alternate conformations."""
    for res in chain.residues:
        for name in BACKBONE_ATOMS:
            if res.has_alt_conformations(name):
                return True
    return False


def _collect_chain_backbone_outliers_vectorized(
    chain: ProteinChain, *, n_sigma: float, early_stop: bool = True
) -> tuple[list[dict[str, str | float | int]], list[dict[str, str | float | int]]]:
    """Vectorized backbone outlier detection using :meth:`ProteinChain.get_backbone_xyz`.

    Assumes **no** alternate backbone conformations.
    Computes all N–CA, CA–C, and C–N distances, plus N–CA–C, CA–C–N, and
    C–N–CA angles, in batch NumPy operations.
    """
    bl_rows: list[dict[str, str | float | int]] = []
    ba_rows: list[dict[str, str | float | int]] = []
    residues = chain.residues
    n_res = len(residues)
    if n_res < 1:
        return bl_rows, ba_rows

    xyz = chain.get_backbone_xyz()  # (n_res, 3, 3): N, CA, C
    mask = chain.get_backbone_mask()  # (n_res,) bool
    if not mask.any():
        return bl_rows, ba_rows

    n_xyz = xyz[:, 0, :]  # (n_res, 3)
    ca_xyz = xyz[:, 1, :]  # (n_res, 3)
    c_xyz = xyz[:, 2, :]  # (n_res, 3)

    # ── Vectorized intra-residue bond lengths ──
    n_ca_dist = np.linalg.norm(n_xyz - ca_xyz, axis=1)  # (n_res,)
    ca_c_dist = np.linalg.norm(ca_xyz - c_xyz, axis=1)  # (n_res,)

    # ── Vectorized inter-residue C–N peptide bond distances ──
    c_n_dist = np.full(n_res - 1, np.nan, dtype=np.float64)
    valid_pairs = mask[:-1] & mask[1:]
    if valid_pairs.any():
        c_n_dist[valid_pairs] = np.linalg.norm(
            c_xyz[:-1][valid_pairs] - n_xyz[1:][valid_pairs], axis=1
        )

    # ── Per-residue checks ──
    for i in range(n_res):
        if not mask[i]:
            continue
        res = residues[i]
        n_a, ca_a, c_a = res.n, res.ca, res.c
        if not all([n_a, ca_a, c_a]):
            continue

        # -- N–CA bond --
        r_bl = get_bond_length_restraint(res.name, "N", "CA")
        if (
            r_bl is not None
            and abs((float(n_ca_dist[i]) - r_bl["mean"]) / r_bl["std"]) > n_sigma
        ):
            bl_rows.append(_base_bond_row(chain, n_a, ca_a, res, res))
            if early_stop:
                return bl_rows, ba_rows

        # -- CA–C bond --
        r_bl = get_bond_length_restraint(res.name, "CA", "C")
        if (
            r_bl is not None
            and abs((float(ca_c_dist[i]) - r_bl["mean"]) / r_bl["std"]) > n_sigma
        ):
            bl_rows.append(_base_bond_row(chain, ca_a, c_a, res, res))
            if early_stop:
                return bl_rows, ba_rows

        # -- N–CA–C angle --
        if _check_angle_outlier(res, n_a, ca_a, c_a, n_sigma):
            ba_rows.append(_base_angle_row(chain, n_a, ca_a, c_a, res, res, res))
            if early_stop:
                return bl_rows, ba_rows

        # -- Inter-residue checks --
        if i + 1 >= n_res:
            continue
        next_res = residues[i + 1]
        if next_res.seq_id > res.seq_id + 1:
            continue
        if not mask[i + 1]:
            continue
        n_next, ca_next = next_res.n, next_res.ca
        if not n_next:
            continue

        # -- C–N peptide bond --
        r_bl = get_bond_length_restraint(res.name, "C", "N")
        if (
            r_bl is not None
            and abs((float(c_n_dist[i]) - r_bl["mean"]) / r_bl["std"]) > n_sigma
        ):
            bl_rows.append(_base_bond_row(chain, c_a, n_next, res, next_res))
            if early_stop:
                return bl_rows, ba_rows

        # -- CA–C–N angle --
        if _check_angle_outlier(res, ca_a, c_a, n_next, n_sigma):
            ba_rows.append(
                _base_angle_row(chain, ca_a, c_a, n_next, res, res, next_res)
            )
            if early_stop:
                return bl_rows, ba_rows

        # -- C–N–CA angle --
        if ca_next and _check_angle_outlier(next_res, c_a, n_next, ca_next, n_sigma):
            ba_rows.append(
                _base_angle_row(chain, c_a, n_next, ca_next, res, next_res, next_res)
            )
            if early_stop:
                return bl_rows, ba_rows

    return bl_rows, ba_rows


def _collect_chain_sidechain_outliers(
    chain: ProteinChain, *, n_sigma: float, early_stop: bool = True
) -> tuple[list[dict[str, str | float | int]], list[dict[str, str | float | int]]]:
    """Check non-backbone bond lengths **and** bond angles in one pass.

    Designed exclusively for side-chain / non-backbone geometry — backbone
    bonds and backbone-only angles are skipped.
    """
    bl_rows: list[dict[str, str | float | int]] = []
    ba_rows: list[dict[str, str | float | int]] = []

    # ── Pre-build atom → (atom, residue) lookups ──
    atom_map: dict[int, Atom] = {}
    atom_residue: dict[int, Residue] = {}
    for res in chain.residues:
        for atom_list in res.atoms.values():
            for atom in atom_list:
                atom_map[atom.serial] = atom
                atom_residue[atom.serial] = res

    # ── bond-length check: non-backbone bonds ──
    for bond in chain.bonds:
        a1 = atom_map.get(bond.a1)
        a2 = atom_map.get(bond.a2)
        if a1 is None or a2 is None:
            continue

        if _bond_is_backbone(a1, a2):
            continue

        if not _alt_loc_compatible(a1.alt_loc, a2.alt_loc):
            continue

        r1 = atom_residue.get(bond.a1)
        r2 = atom_residue.get(bond.a2)
        if r1 is None or r2 is None:
            continue

        if _check_bond_outlier(r1, a1, a2, n_sigma):
            bl_rows.append(_base_bond_row(chain, a1, a2, r1, r2))
            if early_stop:
                return bl_rows, ba_rows

    # ── bond-angle check: non-backbone 2-hop paths ──
    adj = chain._adjacency  # pyright: ignore[reportPrivateUsage]

    for a_serial, b_neighbors in adj.items():
        a_atom = atom_map.get(a_serial)
        if a_atom is None:
            continue
        a_bb = a_atom.name in BACKBONE_ATOMS

        for b_serial in b_neighbors:
            b_atom = atom_map.get(b_serial)
            if b_atom is None:
                continue
            b_bb = b_atom.name in BACKBONE_ATOMS

            for c_serial in adj.get(b_serial, ()):
                if c_serial == a_serial:
                    continue

                c_atom = atom_map.get(c_serial)
                if c_atom is None:
                    continue
                c_bb = c_atom.name in BACKBONE_ATOMS

                # canonicalise direction
                if a_serial > c_serial:
                    continue

                if a_bb and b_bb and c_bb:
                    continue

                if not (
                    _alt_loc_compatible(a_atom.alt_loc, b_atom.alt_loc)
                    and _alt_loc_compatible(b_atom.alt_loc, c_atom.alt_loc)
                ):
                    continue

                ra = atom_residue.get(a_serial)
                rb = atom_residue.get(b_serial)
                rc = atom_residue.get(c_serial)

                if ra is None or rb is None or rc is None:
                    continue

                if _check_angle_outlier(rb, a_atom, b_atom, c_atom, n_sigma):
                    ba_rows.append(
                        _base_angle_row(chain, a_atom, b_atom, c_atom, ra, rb, rc)
                    )
                    if early_stop:
                        return bl_rows, ba_rows

    return bl_rows, ba_rows


# ==================================================================
#  Entry-level workers  (picklable → multiprocessing)
# ==================================================================


def _collect_entry_backbone_outliers(
    entry_id: str, n_sigma: float, early_stop: bool = True
) -> tuple[str, list[dict[str, str | float | int]]]:
    """Return (entry_id, [outlier_rows]) using the combined chain scanner."""
    try:
        entry = ProteinEntry.from_cif(entry_id)
    except Exception:
        return (entry_id, [])

    rows: list[dict[str, str | float | int]] = []
    for chain in entry.chains:
        if not chain.polypeptide:
            continue
        bl, ba = _collect_chain_backbone_outliers(
            chain, n_sigma=n_sigma, early_stop=early_stop
        )
        rows.extend(bl)
        rows.extend(ba)

        if early_stop and rows:
            return (entry_id, rows)

    return (entry_id, rows)


def _collect_entry_sidechain_outliers(
    entry_id: str, n_sigma: float, early_stop: bool = True
) -> tuple[str, list[dict[str, str | float | int]]]:
    """Return (entry_id, [outlier_rows]) using the combined chain scanner."""
    try:
        entry = ProteinEntry.from_cif(entry_id, True)
    except Exception:
        return (entry_id, [])

    rows: list[dict[str, str | float | int]] = []
    for chain in entry.chains:
        if not chain.polypeptide:
            continue
        bl, ba = _collect_chain_sidechain_outliers(
            chain, n_sigma=n_sigma, early_stop=early_stop
        )
        rows.extend(bl)
        rows.extend(ba)

        if early_stop and rows:
            return (entry_id, rows)

    return (entry_id, rows)


# ── All-stages worker (single CIF load → all results) ──────────


def _process_entry_all_stages(
    entry_id: str,
    n_sigma: float = 5.0,
    early_stop: bool = True,
    clash_radius: float = 4.0,
    defect_threshold: float = 0.4,
) -> tuple[str, list[dict], list[dict], list[dict]]:  # pyright: ignore[reportMissingTypeArgument]
    """Load CIF once, run geometry outlier detection and atomic clash detection.

    Returns
    -------
    (entry_id, backbone outliers, sidechain outliers, serious outliers)
    """
    try:
        entry = ProteinEntry.from_cif(entry_id, detect_bonds_flag=True)
    except Exception:
        return (entry_id, [], [], [])

    rows1: list[dict[str, str | float | int]] = []
    rows2: list[dict[str, str | float | int]] = []

    # Single pass over chains: backbone then sidechain per chain
    for chain in entry.chains:
        if not chain.polypeptide:
            continue

        bl, ba = _collect_chain_backbone_outliers(
            chain, n_sigma=n_sigma, early_stop=early_stop
        )
        rows1.extend(bl)
        rows1.extend(ba)
        if early_stop and rows1:
            return (entry_id, rows1, rows2, [])

        s_bl, s_ba = _collect_chain_sidechain_outliers(
            chain, n_sigma=n_sigma, early_stop=early_stop
        )
        rows2.extend(s_bl)
        rows2.extend(s_ba)
        if early_stop and rows2:
            return (entry_id, rows1, rows2, [])

    # Stage 3: nonbonded clash detection on survivors only
    clash_rows = _detect_clashes_on_entry(
        entry, radius=clash_radius, defect_threshold=defect_threshold
    )
    return (entry_id, rows1, rows2, clash_rows)


# ==================================================================
#  Stage 3 — nonbonded atomic clashes (full model, KD-tree)
# ==================================================================


def _pair_key(a: int, b: int) -> int:
    """Encode an unordered pair of serials as a single 64-bit integer."""
    return (a << 32) | b if a <= b else (b << 32) | a


def _build_exclusion_set(chains: list[ProteinChain]) -> set[int]:
    """Return the set of atom-serial pairs (as packed 64-bit ints) that are ≤ 2 bonds apart."""
    excluded: set[int] = set()
    for chain in chains:
        for bond in chain.bonds:
            excluded.add(_pair_key(bond.a1, bond.a2))
        adj = chain._adjacency  # pyright: ignore[reportPrivateUsage]
        for a, neighbors in adj.items():
            for b in neighbors:
                for c in adj.get(b, ()):
                    if c != a:
                        excluded.add(_pair_key(a, c))
    return excluded


def _detect_clashes_on_entry(
    entry: ProteinEntry,
    radius: float = 4.0,
    defect_threshold: float = 0.4,
    early_stop: bool = True,
) -> list[dict[str, str | float | int]]:
    """Nonbonded clash detection on an already-loaded ``ProteinEntry``."""
    # ── Count and flatten atoms across all chains ──
    n_atoms = entry.num_atoms
    if n_atoms < 2:
        return []

    coords_arr = np.empty((n_atoms, 3), dtype=np.float64)
    atom_refs: list[tuple[ProteinChain, Atom, Residue] | None] = [None] * n_atoms

    idx = 0
    for chain in entry.chains:
        for res in chain.residues:
            for atom_list in res.atoms.values():
                for atom in atom_list:
                    coords_arr[idx] = atom.coord
                    atom_refs[idx] = (chain, atom, res)
                    idx += 1

    excluded = _build_exclusion_set(list(entry.chains))

    tree = cKDTree(coords_arr)
    pairs = tree.query_pairs(radius, output_type="ndarray")

    clashes: list[dict[str, str | float | int]] = []
    largest_defect_info: dict[str, str | float | int] = {}
    for i, j in pairs:
        serial_i = atom_refs[i][1].serial
        serial_j = atom_refs[j][1].serial

        if _pair_key(serial_i, serial_j) in excluded:
            continue

        chain_i, atom_i, res_i = atom_refs[i]
        chain_j, atom_j, res_j = atom_refs[j]

        if not _alt_loc_compatible(atom_i.alt_loc, atom_j.alt_loc):
            continue
        if chain_i.model_id != chain_j.model_id:
            continue

        e1, e2 = atom_i.element, atom_j.element
        r1 = ATOMIC_RADII.get(e1, 1.0)
        r2 = ATOMIC_RADII.get(e2, 1.0)
        vdw_sum = r1 + r2

        dist = float(np.linalg.norm(atom_i.coord - atom_j.coord))
        defect = vdw_sum - dist

        if not early_stop:
            clashes.append(
                {
                    "pdb_id": entry.pdb_id,
                    "model_id1": chain_i.model_id,
                    "model_id2": chain_j.model_id,
                    "chain_id1": chain_i.chain_id,
                    "chain_id2": chain_j.chain_id,
                    "residue_id1": res_i.seq_id,
                    "residue_label1": res_i.name,
                    "atom1": atom_i.name,
                    "alt1": atom_i.alt_loc,
                    "residue_id2": res_j.seq_id,
                    "residue_label2": res_j.name,
                    "atom2": atom_j.name,
                    "alt2": atom_j.alt_loc,
                    "distance": round(dist, 3),
                    "vdw_sum": round(vdw_sum, 3),
                    "defect": round(defect, 3),
                    "clash_type": "nonbonded_clash",
                }
            )

        else:
            if (not largest_defect_info) or float(
                largest_defect_info["defect"]
            ) < defect:
                largest_defect_info = {
                    "pdb_id": entry.pdb_id,
                    "model_id1": chain_i.model_id,
                    "model_id2": chain_j.model_id,
                    "chain_id1": chain_i.chain_id,
                    "chain_id2": chain_j.chain_id,
                    "residue_id1": res_i.seq_id,
                    "residue_label1": res_i.name,
                    "atom1": atom_i.name,
                    "alt1": atom_i.alt_loc,
                    "residue_id2": res_j.seq_id,
                    "residue_label2": res_j.name,
                    "atom2": atom_j.name,
                    "alt2": atom_j.alt_loc,
                    "distance": round(dist, 3),
                    "vdw_sum": round(vdw_sum, 3),
                    "defect": round(defect, 3),
                    "clash_type": "nonbonded_clash",
                }
                clashes = [largest_defect_info]
            if defect > defect_threshold:
                return clashes

    return clashes


def _detect_clashes_entry(
    entry_id: str,
    radius: float = 4.0,
    defect_threshold: float = 0.4,
    early_stop: bool = True,
) -> list[dict[str, str | float | int]]:
    """Return a serious nonbonded clash record for one entry (loads CIF)."""
    try:
        entry = ProteinEntry.from_cif(entry_id, detect_bonds_flag=True)
    except Exception:
        return []
    return _detect_clashes_on_entry(
        entry,
        radius=radius,
        defect_threshold=defect_threshold,
        early_stop=early_stop,
    )


# ==================================================================
#  Public API — individual stage functions  (each → DataFrame)
# ==================================================================


def _pmap(func: Callable, args: list[Any], n_jobs: int | None = None):  # pyright: ignore[reportMissingTypeArgument]
    """Parallel map helper."""
    n_procs = n_jobs or mp.cpu_count()
    with mp.Pool(n_procs) as pool:
        return pool.starmap(func, args)


def _rows_to_pass_ids(
    all_results: list[tuple[str, list[dict[str, str | float | int]]]],
    entry_ids: list[str],
) -> list[str]:
    """Return entry IDs that have *no* outlier rows."""
    failed: set[str] = {eid for eid, rows in all_results if rows}
    return [eid for eid in entry_ids if eid not in failed]


def filter_backbone(
    entry_ids: list[str],
    n_sigma: float = 5.0,
    n_jobs: int | None = None,
    early_stop: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """Stage 1 — backbone bond-length and angle outliers.

    Backbone bonds are those where **both** atoms are in ``{N, CA, C}``
    (intra-residue N–CA, CA–C; inter-residue peptide C–N). Backbone
    angles are 2-hop paths where **all three** atoms (A, B, C) are in
    ``{N, CA, C}`` are checked (e.g. N–CA–C, CA–C–N, C–N–CA).

    :param entry_ids: PDB IDs to process.
    :param n_sigma: A bond/angle is an outlier when its z-score magnitude
        exceeds this value. Default ``5.0``.
    :param n_jobs: Number of parallel workers. Defaults to all available CPUs.
    :param early_stop: If ``True``, stop scanning an entry as soon as the first
        outlier is found (faster when only a pass/fail verdict is needed).
        Default ``True``.
    :return: A pair ``(outliers_df, pass_ids)`` where *outliers_df* has one row
        per outlier bond/angle (with ``mean``, ``std`` and ``z_score`` columns
        alongside the identifying atoms) and *pass_ids* lists the entry IDs
        whose **every** chain is free of backbone outliers.
    """
    results = _pmap(
        _collect_entry_backbone_outliers,
        [(eid, n_sigma, early_stop) for eid in entry_ids],
        n_jobs,
    )
    all_rows: list[dict[str, str | float | int]] = []
    for _, rows in results:
        all_rows.extend(rows)
    return pd.DataFrame(all_rows), _rows_to_pass_ids(results, entry_ids)


def filter_nonbackbone(
    entry_ids: list[str],
    n_sigma: float = 5.0,
    n_jobs: int | None = None,
    early_stop: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """Stage 2 — non-backbone (side-chain) outliers.

    Checks non-backbone bonds (e.g. C=O, Cα–Cβ) and any 2-hop angle that
    involves at least one side-chain or carbonyl-oxygen atom (e.g. N–CA–CB,
    CA–C=O, CA–CB–CG).

    :param entry_ids: PDB IDs to process (typically the survivors of Stage 1).
    :param n_sigma: A bond/angle is an outlier when its z-score magnitude
        exceeds this value. Default ``5.0``.
    :param n_jobs: Number of parallel workers. Defaults to all available CPUs.
    :param early_stop: If ``True``, stop scanning an entry as soon as the first
        outlier is found. Default ``True``.
    :return: A pair ``(outliers_df, pass_ids)`` where *outliers_df* has one row
        per outlier side-chain bond/angle and *pass_ids* lists the entry IDs
        whose **every** chain is free of side-chain outliers.
    """
    results = _pmap(
        _collect_entry_sidechain_outliers,
        [(eid, n_sigma, early_stop) for eid in entry_ids],
        n_jobs,
    )
    all_rows: list[dict[str, str | float | int]] = []
    for _, rows in results:
        all_rows.extend(rows)
    return pd.DataFrame(all_rows), _rows_to_pass_ids(results, entry_ids)


def detect_atomic_clashes(
    entry_ids: list[str],
    radius: float = 4.0,
    defect_threshold: float = 0.0,
    n_jobs: int | None = None,
) -> pd.DataFrame:
    """Stage 3 — nonbonded clash detection across the full entry (all chains).

    Pairs whose graph distance is ≤ 2 (bonded or angle-constrained) are
    excluded.  The remaining pairs within *radius* Å are checked for van
    der Waals overlap: *defect* = vdw_sum − distance > *defect_threshold*.

    :param entry_ids: PDB IDs to process (typically the survivors of Stages
        1–2).
    :param radius: KD-tree search radius in Angstrom.
    :param defect_threshold: Minimum vdW overlap to report (``0.0`` for a
        clashscore‑0 subset).
    :param n_jobs: Number of parallel workers.
    :return: One row per detected nonbonded clash.
    """
    results = _pmap(
        _detect_clashes_entry,
        [(eid, radius, defect_threshold) for eid in entry_ids],
        n_jobs,
    )
    all_clashes: list[dict[str, str | float | int]] = []
    for clashes in results:
        all_clashes.extend(clashes)
    return pd.DataFrame(all_clashes)


# ==================================================================
#  Full pipeline
# ==================================================================


def run_pipeline(
    entry_ids: list[str],
    n_sigma: float = 5.0,
    clash_radius: float = 4.0,
    defect_threshold: float = 0.4,
    n_jobs: int | None = None,
    early_stop: bool = True,
) -> dict[str, list[str] | pd.DataFrame]:
    """Run all stages and return DataFrames plus passing-ID lists.

    All stages are computed in a **single** CIF load per entry inside one
    persistent process pool, so this is far cheaper than calling
    :func:`filter_backbone`, :func:`filter_nonbackbone` and
    :func:`detect_atomic_clashes` separately.

    :param entry_ids: PDB IDs to process.
    :param n_sigma: Outlier z-score threshold for Stages 1–2. Default ``5.0``.
    :param clash_radius: KD-tree search radius (Å) for Stage 3 clash detection.
        Default ``4.0``.
    :param defect_threshold: Minimum van-der-Waals overlap to report a clash.
        Default ``0.4``.
    :param n_jobs: Number of parallel workers. Defaults to all available CPUs.
    :param early_stop: If ``True``, stop scanning each entry as soon as its
        first outlier is found. Default ``True``.
    :return: Dict with keys:

        - ``backbone_outliers`` / ``sidechain_outliers`` —
          :class:`pandas.DataFrame` of outlier records for each geometry stage.
        - ``backbone_pass_ids`` / ``sidechain_pass_ids`` — entries that
          survived that cumulative stage.
        - ``distant_clashes`` — :class:`pandas.DataFrame` of nonbonded clash
          records.
    """
    n_procs = n_jobs or mp.cpu_count()
    args = [
        (eid, n_sigma, early_stop, clash_radius, defect_threshold) for eid in entry_ids
    ]
    with mp.Pool(n_procs) as pool:
        results = pool.starmap(_process_entry_all_stages, args)

    bb_rows: list[dict[str, str | float | int]] = []
    side_rows: list[dict[str, str | float | int]] = []
    distant_rows: list[dict[str, str | float | int]] = []
    bb_fail: set[str] = set()
    side_fail: set[str] = set()

    for eid, r1, r2, r3 in results:
        if r1:
            bb_rows.extend(r1)
            bb_fail.add(eid)
        if r2:
            side_rows.extend(r2)
            side_fail.add(eid)
        if r3:
            distant_rows.extend(r3)

    s1_pass = [eid for eid in entry_ids if eid not in bb_fail]
    s2_pass = [eid for eid in s1_pass if eid not in side_fail]

    return {
        "backbone_outliers": pd.DataFrame(bb_rows),
        "backbone_pass_ids": s1_pass,
        "sidechain_outliers": pd.DataFrame(side_rows),
        "sidechain_pass_ids": s2_pass,
        "distant_clashes": pd.DataFrame(distant_rows),
    }
