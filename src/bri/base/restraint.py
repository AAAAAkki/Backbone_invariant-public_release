# -*- coding = utf-8 -*-
"""Bond length and angle restraints for protein geometry validation.

Data sourced from Engh & Huber (1991, 2001) and other standard references.
Values are given as mean ± standard deviation in Angstrom (lengths) or degrees (angles).

Efficient lookup: all keys are canonicalised (sorted for bond pairs, symmetric for
angle triplets) so lookups are O(1) regardless of atom order.
"""

from __future__ import annotations

# ── Key canonicalisation ──────────────────────────────────────


def _bond_key(a1: str, a2: str) -> tuple[str, str]:
    """Canonical key for an unordered bond pair."""
    return (a1, a2) if a1 <= a2 else (a2, a1)


def _angle_key(a1: str, a2: str, a3: str) -> tuple[str, str, str]:
    """Canonical key for an angle triplet.  a2 is the central atom."""
    return (a1, a2, a3) if a1 <= a3 else (a3, a2, a1)


ATOMIC_RADII = {"O": 1.4, "N": 1.5, "C": 1.7, "H": 1.0, "S": 1.85}
# ── Bond length restraints (Angstrom) ─────────────────────────

#: Default bond lengths keyed by canonical atom pair.
DEFAULT_BOND_LENGTHS: dict[tuple[str, str], dict[str, float]] = {
    # Backbone
    _bond_key("N", "CA"): {"mean": 1.459, "std": 0.020},
    _bond_key("CA", "C"): {"mean": 1.525, "std": 0.026},
    _bond_key("C", "O"): {"mean": 1.229, "std": 0.019},
    _bond_key("CA", "CB"): {"mean": 1.532, "std": 0.031},
    _bond_key("C", "N"): {"mean": 1.336, "std": 0.023},  # peptide bond
}

#: Residue-specific bond length overrides.  Outer key is the 3-letter residue name.
RESIDUE_BOND_LENGTHS: dict[str, dict[tuple[str, str], dict[str, float]]] = {
    "ALA": {
        _bond_key("CA", "CB"): {"mean": 1.520, "std": 0.021},
    },
    "ARG": {
        _bond_key("CA", "CB"): {"mean": 1.535, "std": 0.022},
        _bond_key("CB", "CG"): {"mean": 1.521, "std": 0.027},
        _bond_key("CG", "CD"): {"mean": 1.515, "std": 0.025},
        _bond_key("CD", "NE"): {"mean": 1.460, "std": 0.017},
        _bond_key("NE", "CZ"): {"mean": 1.326, "std": 0.013},
        _bond_key("CZ", "NH1"): {"mean": 1.326, "std": 0.013},
        _bond_key("CZ", "NH2"): {"mean": 1.326, "std": 0.013},
    },
    "ASN": {
        _bond_key("CA", "CB"): {"mean": 1.527, "std": 0.026},
        _bond_key("CB", "CG"): {"mean": 1.506, "std": 0.023},
        _bond_key("CG", "OD1"): {"mean": 1.235, "std": 0.022},
        _bond_key("CG", "ND2"): {"mean": 1.324, "std": 0.025},
    },
    "ASP": {
        _bond_key("CA", "CB"): {"mean": 1.535, "std": 0.022},
        _bond_key("CB", "CG"): {"mean": 1.513, "std": 0.021},
        _bond_key("CG", "OD1"): {"mean": 1.249, "std": 0.023},
        _bond_key("CG", "OD2"): {"mean": 1.249, "std": 0.023},
    },
    "CYS": {
        _bond_key("CA", "CB"): {"mean": 1.526, "std": 0.013},
        _bond_key("CB", "SG"): {"mean": 1.812, "std": 0.016},
    },
    "GLN": {
        _bond_key("CA", "CB"): {"mean": 1.535, "std": 0.022},
        _bond_key("CB", "CG"): {"mean": 1.521, "std": 0.027},
        _bond_key("CG", "CD"): {"mean": 1.506, "std": 0.023},
        _bond_key("CD", "OE1"): {"mean": 1.235, "std": 0.022},
        _bond_key("CD", "NE2"): {"mean": 1.324, "std": 0.025},
    },
    "GLU": {
        _bond_key("CA", "CB"): {"mean": 1.535, "std": 0.022},
        _bond_key("CB", "CG"): {"mean": 1.517, "std": 0.019},
        _bond_key("CG", "CD"): {"mean": 1.515, "std": 0.015},
        _bond_key("CD", "OE1"): {"mean": 1.252, "std": 0.011},
        _bond_key("CD", "OE2"): {"mean": 1.252, "std": 0.011},
    },
    "GLY": {
        _bond_key("N", "CA"): {"mean": 1.456, "std": 0.015},
        _bond_key("CA", "C"): {"mean": 1.514, "std": 0.016},
        _bond_key("C", "O"): {"mean": 1.232, "std": 0.016},
        _bond_key("C", "N"): {"mean": 1.326, "std": 0.018},
    },
    "HIS": {  # Using HISE parameters as standard
        _bond_key("CA", "CB"): {"mean": 1.535, "std": 0.022},
        _bond_key("CB", "CG"): {"mean": 1.496, "std": 0.018},
        _bond_key("CG", "ND1"): {"mean": 1.383, "std": 0.022},
        _bond_key("CG", "CD2"): {"mean": 1.353, "std": 0.014},
        _bond_key("ND1", "CE1"): {"mean": 1.323, "std": 0.015},
        _bond_key("CD2", "NE2"): {"mean": 1.375, "std": 0.022},
        _bond_key("CE1", "NE2"): {"mean": 1.333, "std": 0.019},
    },
    "ILE": {
        _bond_key("CA", "CB"): {"mean": 1.544, "std": 0.023},
        _bond_key("CB", "CG1"): {"mean": 1.536, "std": 0.028},
        _bond_key("CB", "CG2"): {"mean": 1.524, "std": 0.031},
        _bond_key("CG1", "CD1"): {"mean": 1.500, "std": 0.069},
    },
    "LEU": {
        _bond_key("CA", "CB"): {"mean": 1.533, "std": 0.023},
        _bond_key("CB", "CG"): {"mean": 1.521, "std": 0.029},
        _bond_key("CG", "CD1"): {"mean": 1.514, "std": 0.037},
        _bond_key("CG", "CD2"): {"mean": 1.514, "std": 0.037},
    },
    "LYS": {
        _bond_key("CA", "CB"): {"mean": 1.535, "std": 0.022},
        _bond_key("CB", "CG"): {"mean": 1.521, "std": 0.027},
        _bond_key("CG", "CD"): {"mean": 1.520, "std": 0.034},
        _bond_key("CD", "CE"): {"mean": 1.508, "std": 0.025},
        _bond_key("CE", "NZ"): {"mean": 1.486, "std": 0.025},
    },
    "MET": {
        _bond_key("CA", "CB"): {"mean": 1.535, "std": 0.022},
        _bond_key("CB", "CG"): {"mean": 1.509, "std": 0.032},
        _bond_key("CG", "SD"): {"mean": 1.807, "std": 0.026},
        _bond_key("SD", "CE"): {"mean": 1.774, "std": 0.056},
    },
    "PHE": {
        _bond_key("CA", "CB"): {"mean": 1.535, "std": 0.022},
        _bond_key("CB", "CG"): {"mean": 1.509, "std": 0.017},
        _bond_key("CG", "CD1"): {"mean": 1.383, "std": 0.015},
        _bond_key("CG", "CD2"): {"mean": 1.383, "std": 0.015},
        _bond_key("CD1", "CE1"): {"mean": 1.388, "std": 0.020},
        _bond_key("CD2", "CE2"): {"mean": 1.388, "std": 0.020},
        _bond_key("CE1", "CZ"): {"mean": 1.369, "std": 0.019},
        _bond_key("CE2", "CZ"): {"mean": 1.369, "std": 0.019},
    },
    "PRO": {
        _bond_key("N", "CA"): {"mean": 1.468, "std": 0.017},
        _bond_key("CA", "C"): {"mean": 1.524, "std": 0.020},
        _bond_key("C", "O"): {"mean": 1.228, "std": 0.020},
        _bond_key("C", "N"): {"mean": 1.338, "std": 0.019},
        _bond_key("CA", "CB"): {"mean": 1.531, "std": 0.020},
        _bond_key("CB", "CG"): {"mean": 1.495, "std": 0.050},
        _bond_key("CG", "CD"): {"mean": 1.502, "std": 0.033},
        _bond_key("CD", "N"): {"mean": 1.474, "std": 0.014},
    },
    "SER": {
        _bond_key("CA", "CB"): {"mean": 1.525, "std": 0.015},
        _bond_key("CB", "OG"): {"mean": 1.418, "std": 0.013},
    },
    "THR": {
        _bond_key("CA", "CB"): {"mean": 1.529, "std": 0.026},
        _bond_key("CB", "OG1"): {"mean": 1.428, "std": 0.020},
        _bond_key("CB", "CG2"): {"mean": 1.519, "std": 0.033},
    },
    "TRP": {
        _bond_key("CA", "CB"): {"mean": 1.535, "std": 0.022},
        _bond_key("CB", "CG"): {"mean": 1.498, "std": 0.018},
        _bond_key("CG", "CD1"): {"mean": 1.363, "std": 0.014},
        _bond_key("CG", "CD2"): {"mean": 1.432, "std": 0.017},
        _bond_key("CD1", "NE1"): {"mean": 1.375, "std": 0.017},
        _bond_key("NE1", "CE2"): {"mean": 1.371, "std": 0.013},
        _bond_key("CD2", "CE2"): {"mean": 1.409, "std": 0.012},
        _bond_key("CD2", "CE3"): {"mean": 1.399, "std": 0.015},
        _bond_key("CE2", "CZ2"): {"mean": 1.393, "std": 0.017},
        _bond_key("CE3", "CZ3"): {"mean": 1.380, "std": 0.017},
        _bond_key("CZ2", "CH2"): {"mean": 1.369, "std": 0.019},
        _bond_key("CZ3", "CH2"): {"mean": 1.396, "std": 0.016},
    },
    "TYR": {
        _bond_key("CA", "CB"): {"mean": 1.535, "std": 0.022},
        _bond_key("CB", "CG"): {"mean": 1.512, "std": 0.015},
        _bond_key("CG", "CD1"): {"mean": 1.387, "std": 0.013},
        _bond_key("CG", "CD2"): {"mean": 1.387, "std": 0.013},
        _bond_key("CD1", "CE1"): {"mean": 1.389, "std": 0.015},
        _bond_key("CD2", "CE2"): {"mean": 1.389, "std": 0.015},
        _bond_key("CE1", "CZ"): {"mean": 1.381, "std": 0.013},
        _bond_key("CE2", "CZ"): {"mean": 1.381, "std": 0.013},
        _bond_key("CZ", "OH"): {"mean": 1.374, "std": 0.017},
    },
    "VAL": {
        _bond_key("CA", "CB"): {"mean": 1.543, "std": 0.021},
        _bond_key("CB", "CG1"): {"mean": 1.524, "std": 0.021},
        _bond_key("CB", "CG2"): {"mean": 1.524, "std": 0.021},
    },
}

# ── Bond angle restraints (degrees) ───────────────────────────
RESIDUE_BOND_ANGLES: dict[str, dict[tuple[str, str, str], dict[str, float]]] = {
    "ALA": {
        _angle_key("N", "CA", "CB"): {"mean": 110.1, "std": 1.4},
        _angle_key("CB", "CA", "C"): {"mean": 110.1, "std": 1.5},
    },
    "ARG": {
        _angle_key("N", "CA", "CB"): {"mean": 110.6, "std": 1.8},
        _angle_key("CB", "CA", "C"): {"mean": 110.4, "std": 2.0},
        _angle_key("CA", "CB", "CG"): {"mean": 113.4, "std": 2.2},
        _angle_key("CB", "CG", "CD"): {"mean": 111.6, "std": 2.6},
        _angle_key("CG", "CD", "NE"): {"mean": 111.8, "std": 2.1},
        _angle_key("CD", "NE", "CZ"): {"mean": 123.6, "std": 1.4},
        _angle_key("NE", "CZ", "NH1"): {"mean": 120.3, "std": 0.5},
        _angle_key("NE", "CZ", "NH2"): {"mean": 120.3, "std": 0.5},
        _angle_key("NH1", "CZ", "NH2"): {"mean": 119.4, "std": 1.1},
    },
    "ASN": {
        _angle_key("N", "CA", "CB"): {"mean": 110.6, "std": 1.8},
        _angle_key("CB", "CA", "C"): {"mean": 110.4, "std": 2.0},
        _angle_key("CA", "CB", "CG"): {"mean": 113.4, "std": 2.2},
        _angle_key("CB", "CG", "ND2"): {"mean": 116.7, "std": 2.4},
        _angle_key("CB", "CG", "OD1"): {"mean": 121.6, "std": 2.0},
        _angle_key("ND2", "CG", "OD1"): {"mean": 121.9, "std": 2.3},
    },
    "ASP": {
        _angle_key("N", "CA", "CB"): {"mean": 110.6, "std": 1.8},
        _angle_key("CB", "CA", "C"): {"mean": 110.4, "std": 2.0},
        _angle_key("CA", "CB", "CG"): {"mean": 113.4, "std": 2.2},
        _angle_key("CB", "CG", "OD1"): {"mean": 118.3, "std": 0.9},
        _angle_key("CB", "CG", "OD2"): {"mean": 118.3, "std": 0.9},
        _angle_key("OD1", "CG", "OD2"): {"mean": 123.3, "std": 1.9},
    },
    "CYS": {
        _angle_key("N", "CA", "CB"): {"mean": 110.8, "std": 1.5},
        _angle_key("CB", "CA", "C"): {"mean": 111.5, "std": 1.2},
        _angle_key("CA", "CB", "SG"): {"mean": 114.2, "std": 1.1},
    },
    "GLN": {
        _angle_key("N", "CA", "CB"): {"mean": 110.6, "std": 1.8},
        _angle_key("CB", "CA", "C"): {"mean": 110.4, "std": 2.0},
        _angle_key("CA", "CB", "CG"): {"mean": 113.4, "std": 2.2},
        _angle_key("CB", "CG", "CD"): {"mean": 111.6, "std": 2.6},
        _angle_key("CG", "CD", "OE1"): {"mean": 121.6, "std": 2.0},
        _angle_key("CG", "CD", "NE2"): {"mean": 116.7, "std": 2.4},
        _angle_key("OE1", "CD", "NE2"): {"mean": 121.9, "std": 2.3},
    },
    "GLU": {
        _angle_key("N", "CA", "CB"): {"mean": 110.6, "std": 1.8},
        _angle_key("CB", "CA", "C"): {"mean": 110.4, "std": 2.0},
        _angle_key("CA", "CB", "CG"): {"mean": 113.4, "std": 2.2},
        _angle_key("CB", "CG", "CD"): {"mean": 114.2, "std": 2.7},
        _angle_key("CG", "CD", "OE1"): {"mean": 118.3, "std": 2.0},
        _angle_key("CG", "CD", "OE2"): {"mean": 118.3, "std": 2.0},
        _angle_key("OE1", "CD", "OE2"): {"mean": 123.3, "std": 1.2},
    },
    "GLY": {
        _angle_key("N", "CA", "C"): {"mean": 113.1, "std": 2.5},
        _angle_key("CA", "C", "N"): {"mean": 116.2, "std": 2.0},
        _angle_key("CA", "C", "O"): {"mean": 120.6, "std": 1.8},
        _angle_key("O", "C", "N"): {"mean": 123.2, "std": 1.7},
        _angle_key("C", "N", "CA"): {"mean": 122.3, "std": 2.1},
    },
    "HIS": {
        _angle_key("N", "CA", "CB"): {"mean": 110.6, "std": 1.8},
        _angle_key("CB", "CA", "C"): {"mean": 110.4, "std": 2.0},
        _angle_key("CA", "CB", "CG"): {"mean": 113.6, "std": 1.7},
        _angle_key("CB", "CG", "ND1"): {"mean": 121.4, "std": 1.3},
        _angle_key("CB", "CG", "CD2"): {"mean": 129.7, "std": 1.6},
        _angle_key("CG", "ND1", "CE1"): {"mean": 105.7, "std": 1.3},
        _angle_key("ND1", "CE1", "NE2"): {"mean": 111.5, "std": 1.3},
        _angle_key("CE1", "NE2", "CD2"): {"mean": 107.1, "std": 1.1},
        _angle_key("NE2", "CD2", "CG"): {"mean": 106.7, "std": 1.2},
        _angle_key("CD2", "CG", "ND1"): {"mean": 108.8, "std": 1.4},
    },
    "ILE": {
        _angle_key("N", "CA", "CB"): {"mean": 110.8, "std": 2.3},
        _angle_key("CB", "CA", "C"): {"mean": 111.6, "std": 2.0},
        _angle_key("CA", "CB", "CG1"): {"mean": 111.0, "std": 1.9},
        _angle_key("CB", "CG1", "CD1"): {"mean": 113.9, "std": 2.8},
        _angle_key("CA", "CB", "CG2"): {"mean": 110.9, "std": 2.0},
        _angle_key("CG1", "CB", "CG2"): {"mean": 111.4, "std": 2.2},
    },
    "LEU": {
        _angle_key("N", "CA", "CB"): {"mean": 110.4, "std": 2.0},
        _angle_key("CB", "CA", "C"): {"mean": 110.2, "std": 1.9},
        _angle_key("CA", "CB", "CG"): {"mean": 115.3, "std": 2.3},
        _angle_key("CB", "CG", "CD1"): {"mean": 111.0, "std": 1.7},
        _angle_key("CB", "CG", "CD2"): {"mean": 111.0, "std": 1.7},
        _angle_key("CD1", "CG", "CD2"): {"mean": 110.5, "std": 3.0},
    },
    "LYS": {
        _angle_key("N", "CA", "CB"): {"mean": 110.6, "std": 1.8},
        _angle_key("CB", "CA", "C"): {"mean": 110.4, "std": 2.0},
        _angle_key("CA", "CB", "CG"): {"mean": 113.4, "std": 2.2},
        _angle_key("CB", "CG", "CD"): {"mean": 111.6, "std": 2.6},
        _angle_key("CG", "CD", "CE"): {"mean": 111.9, "std": 3.0},
        _angle_key("CD", "CE", "NZ"): {"mean": 111.7, "std": 2.3},
    },
    "MET": {
        _angle_key("N", "CA", "CB"): {"mean": 110.6, "std": 1.8},
        _angle_key("CB", "CA", "C"): {"mean": 110.4, "std": 2.0},
        _angle_key("CA", "CB", "CG"): {"mean": 113.3, "std": 1.7},
        _angle_key("CB", "CG", "SD"): {"mean": 112.4, "std": 3.0},
        _angle_key("CG", "SD", "CE"): {"mean": 100.2, "std": 1.6},
    },
    "PHE": {
        _angle_key("N", "CA", "CB"): {"mean": 110.6, "std": 1.8},
        _angle_key("CB", "CA", "C"): {"mean": 110.4, "std": 2.0},
        _angle_key("CA", "CB", "CG"): {"mean": 113.9, "std": 2.4},
        _angle_key("CB", "CG", "CD1"): {"mean": 120.8, "std": 0.7},
        _angle_key("CB", "CG", "CD2"): {"mean": 120.8, "std": 0.7},
        _angle_key("CD1", "CG", "CD2"): {"mean": 118.3, "std": 1.3},
        _angle_key("CG", "CD1", "CE1"): {"mean": 120.8, "std": 1.1},
        _angle_key("CG", "CD2", "CE2"): {"mean": 120.8, "std": 1.1},
        _angle_key("CD1", "CE1", "CZ"): {"mean": 120.1, "std": 1.2},
        _angle_key("CD2", "CE2", "CZ"): {"mean": 120.1, "std": 1.2},
        _angle_key("CE1", "CZ", "CE2"): {"mean": 120.0, "std": 1.8},
    },
    "PRO": {
        _angle_key("N", "CA", "CB"): {"mean": 103.3, "std": 1.2},
        _angle_key("CB", "CA", "C"): {"mean": 111.7, "std": 2.1},
        _angle_key("CA", "CB", "CG"): {"mean": 104.8, "std": 1.9},
        _angle_key("CB", "CG", "CD"): {"mean": 106.5, "std": 3.9},
        _angle_key("CG", "CD", "N"): {"mean": 103.2, "std": 1.5},
        _angle_key("CA", "N", "CD"): {"mean": 111.7, "std": 1.4},
        _angle_key("C", "N", "CD"): {"mean": 128.4, "std": 2.1},
    },
    "SER": {
        _angle_key("N", "CA", "CB"): {"mean": 110.5, "std": 1.5},
        _angle_key("CB", "CA", "C"): {"mean": 110.1, "std": 1.9},
        _angle_key("CA", "CB", "OG"): {"mean": 111.2, "std": 2.7},
    },
    "THR": {
        _angle_key("N", "CA", "CB"): {"mean": 110.3, "std": 1.9},
        _angle_key("CB", "CA", "C"): {"mean": 111.6, "std": 2.7},
        _angle_key("CA", "CB", "OG1"): {"mean": 109.0, "std": 2.1},
        _angle_key("CA", "CB", "CG2"): {"mean": 112.4, "std": 1.4},
        _angle_key("OG1", "CB", "CG2"): {"mean": 110.0, "std": 2.3},
    },
    "TRP": {
        _angle_key("N", "CA", "CB"): {"mean": 110.6, "std": 1.8},
        _angle_key("CB", "CA", "C"): {"mean": 110.4, "std": 2.0},
        _angle_key("CA", "CB", "CG"): {"mean": 113.7, "std": 1.9},
        _angle_key("CB", "CG", "CD1"): {"mean": 127.0, "std": 1.3},
        _angle_key("CB", "CG", "CD2"): {"mean": 126.6, "std": 1.3},
        _angle_key("CD1", "CG", "CD2"): {"mean": 106.3, "std": 0.8},
        _angle_key("CG", "CD1", "NE1"): {"mean": 110.1, "std": 1.0},
        _angle_key("CD1", "NE1", "CE2"): {"mean": 109.0, "std": 0.9},
        _angle_key("NE1", "CE2", "CD2"): {"mean": 107.3, "std": 1.0},
        _angle_key("CE2", "CD2", "CG"): {"mean": 107.3, "std": 0.8},
        _angle_key("CG", "CD2", "CE3"): {"mean": 133.9, "std": 0.9},
        _angle_key("NE1", "CE2", "CZ2"): {"mean": 130.4, "std": 1.1},
        _angle_key("CE3", "CD2", "CE2"): {"mean": 118.7, "std": 1.2},
        _angle_key("CD2", "CE2", "CZ2"): {"mean": 122.3, "std": 1.2},
        _angle_key("CE2", "CZ2", "CH2"): {"mean": 117.4, "std": 1.0},
        _angle_key("CZ2", "CH2", "CZ3"): {"mean": 121.6, "std": 1.2},
        _angle_key("CH2", "CZ3", "CE3"): {"mean": 121.2, "std": 1.1},
        _angle_key("CZ3", "CE3", "CD2"): {"mean": 118.8, "std": 1.3},
    },
    "TYR": {
        _angle_key("N", "CA", "CB"): {"mean": 110.6, "std": 1.8},
        _angle_key("CB", "CA", "C"): {"mean": 110.4, "std": 2.0},
        _angle_key("CA", "CB", "CG"): {"mean": 113.4, "std": 1.9},
        _angle_key("CB", "CG", "CD1"): {"mean": 121.0, "std": 0.6},
        _angle_key("CB", "CG", "CD2"): {"mean": 121.0, "std": 0.6},
        _angle_key("CD1", "CG", "CD2"): {"mean": 117.9, "std": 1.1},
        _angle_key("CG", "CD1", "CE1"): {"mean": 121.3, "std": 0.8},
        _angle_key("CG", "CD2", "CE2"): {"mean": 121.3, "std": 0.8},
        _angle_key("CD1", "CE1", "CZ"): {"mean": 119.8, "std": 0.9},
        _angle_key("CD2", "CE2", "CZ"): {"mean": 119.8, "std": 0.9},
        _angle_key("CE1", "CZ", "CE2"): {"mean": 119.8, "std": 1.6},
        _angle_key("CE1", "CZ", "OH"): {"mean": 120.1, "std": 2.7},
        _angle_key("CE2", "CZ", "OH"): {"mean": 120.1, "std": 2.7},
    },
    "VAL": {
        _angle_key("N", "CA", "CB"): {"mean": 111.5, "std": 2.2},
        _angle_key("CB", "CA", "C"): {"mean": 111.4, "std": 1.9},
        _angle_key("CA", "CB", "CG1"): {"mean": 110.9, "std": 1.5},
        _angle_key("CA", "CB", "CG2"): {"mean": 110.9, "std": 1.5},
        _angle_key("CG1", "CB", "CG2"): {"mean": 110.9, "std": 1.6},
    },
}
#: Default bond angles keyed by canonical atom triplet.
DEFAULT_BOND_ANGLES: dict[tuple[str, str, str], dict[str, float]] = {
    # Backbone
    _angle_key("N", "CA", "C"): {"mean": 111.0, "std": 2.7},
    _angle_key("CA", "C", "N"): {"mean": 117.2, "std": 2.2},
    _angle_key("CA", "C", "O"): {"mean": 120.1, "std": 2.1},
    _angle_key("O", "C", "N"): {"mean": 122.7, "std": 1.6},
    _angle_key("C", "N", "CA"): {"mean": 121.7, "std": 2.5},
    # Cβ branching
    _angle_key("N", "CA", "CB"): {"mean": 110.6, "std": 2.1},
    _angle_key("C", "CA", "CB"): {"mean": 110.6, "std": 2.3},
}


# ── Public lookup API ─────────────────────────────────────────


def get_bond_length_restraint(
    res_name: str, atom1: str, atom2: str
) -> dict[str, float] | None:
    """Return ``{"mean": …, "std": …}`` for a bond pair, or *None* if unknown.

    Checks residue-specific overrides first, then falls back to defaults.
    Atom order is canonicalised internally — callers may pass atoms in either order.
    """
    key = _bond_key(atom1, atom2)
    override = RESIDUE_BOND_LENGTHS.get(res_name, {})
    if key in override:
        return override[key]
    return DEFAULT_BOND_LENGTHS.get(key)


def get_bond_angle_restraint(
    res_name: str, atom1: str, atom2: str, atom3: str
) -> dict[str, float] | None:
    """Return ``{"mean": …, "std": …}`` for an angle triplet, or *None* if unknown.

    *atom2* is the central atom.  The key is canonicalised so both
    ``(N, CA, C)`` and ``(C, CA, N)`` match the same restraint.
    """
    key = _angle_key(atom1, atom2, atom3)
    override = RESIDUE_BOND_ANGLES.get(res_name, {})
    if key in override:
        return override[key]
    return DEFAULT_BOND_ANGLES.get(key)
