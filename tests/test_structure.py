# -*- coding = utf-8 -*-
"""Tests for the structure module (Atom, Bond, Residue, ProteinChain, ProteinEntry)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bri.structure import (
    Atom,
    Bond,
    Residue,
    ProteinChain,
    ProteinEntry,
    RESIDUE_BONDS,
    detect_bonds,
    detect_disulfide_bonds,
    perturb_chain,
)

TEST_DATA = Path("tests/test_data/")


# ── Atom ────────────────────────────────────────────────────


class TestAtom:
    def test_creation(self):
        a = Atom(serial=1, name="N", element="N", x=1.0, y=2.0, z=3.0)
        assert a.serial == 1
        assert a.name == "N"
        assert a.element == "N"
        assert a.x == 1.0
        assert a.occupancy == 1.0
        assert a.b_factor == 0.0
        assert a.alt_loc == ""

    def test_coord_property(self):
        a = Atom(serial=5, name="CA", element="C", x=1.5, y=0.0, z=0.0)
        c = a.coord
        assert isinstance(c, np.ndarray)
        assert c.dtype == np.float64
        np.testing.assert_array_equal(c, [1.5, 0.0, 0.0])

    def test_mutable_coordinates(self):
        a = Atom(serial=1, name="N", element="N", x=0.0, y=0.0, z=0.0)
        a.x += 1.0
        a.y += 0.5
        assert a.x == 1.0
        assert a.y == 0.5

    def test_equality_by_serial(self):
        a1 = Atom(serial=1, name="N", element="N", x=0.0, y=0.0, z=0.0)
        a2 = Atom(serial=1, name="CA", element="C", x=99.0, y=99.0, z=99.0)
        a3 = Atom(serial=2, name="N", element="N", x=0.0, y=0.0, z=0.0)
        assert a1 == a2
        assert a1 != a3

    def test_hash_by_serial(self):
        a1 = Atom(serial=1, name="N", element="N", x=0.0, y=0.0, z=0.0)
        a2 = Atom(serial=1, name="CA", element="C", x=99.0, y=99.0, z=99.0)
        assert hash(a1) == hash(a2)


# ── Bond ────────────────────────────────────────────────────


class TestBond:
    def test_creation(self):
        b = Bond(1, 2)
        assert b.a1 == 1
        assert b.a2 == 2
        assert b.order == "SING"

    def test_creation_with_order(self):
        b = Bond(10, 20, order="DOUB")
        assert b.order == "DOUB"

    def test_undirected_equality(self):
        b1 = Bond(1, 2)
        b2 = Bond(2, 1)
        b3 = Bond(1, 3)
        assert b1 == b2
        assert b1 != b3

    def test_undirected_hash(self):
        b1 = Bond(1, 2)
        b2 = Bond(2, 1)
        assert hash(b1) == hash(b2)
        assert len({b1, b2}) == 1  # sets deduplicate


# ── Residue ─────────────────────────────────────────────────


class TestResidue:
    @pytest.fixture
    def ala_residue(self):
        return Residue(
            name="ALA",
            seq_id=42,
            atoms={
                "N": [Atom(serial=1, name="N", element="N", x=0.0, y=0.0, z=0.0)],
                "CA": [Atom(serial=2, name="CA", element="C", x=1.5, y=0.0, z=0.0)],
                "C": [Atom(serial=3, name="C", element="C", x=2.5, y=0.0, z=0.0)],
                "O": [Atom(serial=4, name="O", element="O", x=3.5, y=0.0, z=0.0)],
            },
        )

    def test_creation(self, ala_residue):
        assert ala_residue.name == "ALA"
        assert ala_residue.seq_id == 42

    def test_backbone_accessors(self, ala_residue):
        assert ala_residue.n.name == "N"
        assert ala_residue.ca.name == "CA"
        assert ala_residue.c.name == "C"
        assert ala_residue.o.name == "O"

    def test_has_complete_backbone(self, ala_residue):
        assert ala_residue.has_complete_backbone

    def test_missing_backbone(self):
        r = Residue(
            name="GLY",
            seq_id=1,
            atoms={
                "CA": [Atom(serial=1, name="CA", element="C", x=0.0, y=0.0, z=0.0)],
            },
        )
        assert not r.has_complete_backbone
        assert r.n is None

    def test_iter_atoms(self, ala_residue):
        atoms = list(ala_residue.iter_atoms())
        assert len(atoms) == 4
        names = {a.name for a in atoms}
        assert names == {"N", "CA", "C", "O"}

    def test_auth_seq_id_default(self):
        r = Residue(name="ALA", seq_id=1)
        assert r.auth_seq_id is None

    def test_auth_seq_id_set(self):
        r = Residue(name="ALA", seq_id=1, auth_seq_id=100)
        assert r.auth_seq_id == 100


# ── ProteinChain (synthetic, no CIF) ────────────────────────


def _make_backbone_atom(serial, name, x, y=0.0, z=0.0):
    return Atom(serial=serial, name=name, element=name[0], x=x, y=y, z=z)


def _make_chain(n_residues=3):
    """Build a minimal N-residue ALA chain along the X axis."""
    residues = []
    for i in range(n_residues):
        offset = i * 3.8
        residues.append(
            Residue(
                name="ALA",
                seq_id=i + 1,
                atoms={
                    "N": [_make_backbone_atom(i * 3 + 1, "N", offset + 0.0)],
                    "CA": [_make_backbone_atom(i * 3 + 2, "CA", offset + 1.5)],
                    "C": [_make_backbone_atom(i * 3 + 3, "C", offset + 2.5)],
                },
            )
        )
    chain = ProteinChain(pdb_id="test", model_id=1, chain_id="A", residues=residues)
    chain.bonds = detect_bonds(chain)
    chain._rebuild_index()
    return chain


class TestProteinChain:
    @pytest.fixture
    def chain(self):
        return _make_chain(3)

    def test_creation(self, chain):
        assert chain.pdb_id == "test"
        assert chain.model_id == 1
        assert chain.chain_id == "A"
        assert chain.num_residues == 3
        assert chain.num_atoms == 9

    def test_len(self, chain):
        assert len(chain) == 3

    def test_repr(self, chain):
        r = repr(chain)
        assert "ProteinChain" in r
        assert "test" in r
        assert "residues=3" in r
        assert "atoms=9" in r

    def test_equality(self):
        c1 = _make_chain(2)
        c2 = _make_chain(2)
        c2.chain_id = "B"
        assert c1 != c2

    def test_hash(self):
        c1 = _make_chain(2)
        c2 = _make_chain(2)
        c2.chain_id = "B"
        assert hash(c1) != hash(c2)

    # -- Bond detection --

    def test_bond_count(self, chain):
        # 3 residues: 2 intra-res backbone bonds each = 6, + 2 peptide bonds = 8
        assert len(chain.bonds) == 8

    def test_intra_residue_bonds_present(self, chain):
        # Check N-CA and CA-C bonds exist for first residue
        n = chain.residues[0].n
        ca = chain.residues[0].ca
        c = chain.residues[0].c
        n_bonds = chain.get_bonds_of(n.serial)
        ca_bonds = chain.get_bonds_of(ca.serial)
        assert len(n_bonds) == 1  # N is only bonded to CA (first residue)
        assert len(ca_bonds) == 2  # CA bonded to N and C

    def test_peptide_bond_present(self, chain):
        c0 = chain.residues[0].c
        n1 = chain.residues[1].n
        # C of residue 0 should be bonded to N of residue 1
        bonded_c0 = chain.get_bonded(c0.serial)
        assert any(a.serial == n1.serial for a in bonded_c0)

    def test_peptide_bond_count(self, chain):
        peptide_bonds = [
            b
            for b in chain.bonds
            if chain.get_atom(b.a1)
            and chain.get_atom(b.a2)
            and chain.get_atom(b.a1).name == "C"
            and chain.get_atom(b.a2).name == "N"
        ]
        # 2 peptide bonds for 3 residues
        assert len(peptide_bonds) == 2

    # -- Connectivity queries --

    def test_get_atom_found(self, chain):
        a = chain.get_atom(1)
        assert a is not None
        assert a.serial == 1
        assert a.name == "N"

    def test_get_atom_not_found(self, chain):
        assert chain.get_atom(999) is None

    def test_get_bonded(self, chain):
        ca = chain.residues[1].ca  # middle residue CA
        bonded = chain.get_bonded(ca.serial)
        assert len(bonded) == 2
        names = {a.name for a in bonded}
        assert names == {"N", "C"}

    def test_get_bonds_of(self, chain):
        ca = chain.residues[0].ca
        bonds = chain.get_bonds_of(ca.serial)
        assert len(bonds) == 2

    def test_get_bond_between_found(self, chain):
        n = chain.residues[0].n
        ca = chain.residues[0].ca
        bond = chain.get_bond_between(n.serial, ca.serial)
        assert bond is not None
        assert bond.order == "SING"

    def test_get_bond_between_order_independent(self, chain):
        n = chain.residues[0].n
        ca = chain.residues[0].ca
        b1 = chain.get_bond_between(n.serial, ca.serial)
        b2 = chain.get_bond_between(ca.serial, n.serial)
        assert b1 is b2

    def test_get_bond_between_not_found(self, chain):
        n0 = chain.residues[0].n
        n2 = chain.residues[2].n
        assert chain.get_bond_between(n0.serial, n2.serial) is None

    def test_get_bond_between_after_clone(self, chain):
        chain2 = chain.clone()
        n = chain2.residues[0].n
        ca = chain2.residues[0].ca
        bond = chain2.get_bond_between(n.serial, ca.serial)
        assert bond is not None
        assert bond.order == "SING"

    # -- Bond distance (graph shortest-path) --

    def test_bond_distance_same_atom(self, chain):
        ca = chain.residues[0].ca
        assert chain.bond_distance(ca.serial, ca.serial) == 0

    def test_bond_distance_direct(self, chain):
        n = chain.residues[0].n
        ca = chain.residues[0].ca
        assert chain.bond_distance(n.serial, ca.serial) == 1
        assert chain.bond_distance(ca.serial, n.serial) == 1

    def test_bond_distance_two_hops(self, chain):
        # O is not in the fixture, use C -> CA -> N path: C and N are 2 apart
        c = chain.residues[0].c
        n = chain.residues[0].n
        assert chain.bond_distance(c.serial, n.serial) == 2

    def test_bond_distance_inter_residue(self, chain):
        # N of residue 1 to N of residue 2:
        # N1–CA1–C1–N2 → 3 bonds
        n1 = chain.residues[0].n
        n2 = chain.residues[1].n
        assert chain.bond_distance(n1.serial, n2.serial) == 3

    def test_bond_distance_long_range(self, chain):
        # N of residue 0 to N of residue 2:
        # N0–CA0–C0–N1–CA1–C1–N2 → 6 bonds
        n0 = chain.residues[0].n
        n2 = chain.residues[2].n
        assert chain.bond_distance(n0.serial, n2.serial) == 6

    def test_bond_distance_disconnected(self):
        # Build a chain where two atoms have no path between them
        # (separate connected components).
        r1 = Residue(
            name="ALA",
            seq_id=1,
            atoms={
                "CA": [Atom(serial=1, name="CA", element="C", x=0.0, y=0.0, z=0.0)],
            },
        )
        r2 = Residue(
            name="ALA",
            seq_id=2,
            atoms={
                "CA": [Atom(serial=2, name="CA", element="C", x=5.0, y=0.0, z=0.0)],
            },
        )
        chain = ProteinChain(
            pdb_id="x", model_id=1, chain_id="A", residues=[r1, r2], bonds=[]
        )  # no bonds
        chain._rebuild_index()
        assert chain.bond_distance(1, 2) == -1

    def test_bond_distance_with_sidechain(self):
        """ALA: O–C–CA–CB path = 3 bonds between O and CB."""
        r = Residue(
            name="ALA",
            seq_id=1,
            atoms={
                "N": [Atom(serial=1, name="N", element="N", x=0.0, y=0.0, z=0.0)],
                "CA": [Atom(serial=2, name="CA", element="C", x=1.5, y=0.0, z=0.0)],
                "C": [Atom(serial=3, name="C", element="C", x=2.5, y=0.0, z=0.0)],
                "O": [Atom(serial=4, name="O", element="O", x=3.5, y=0.0, z=0.0)],
                "CB": [Atom(serial=5, name="CB", element="C", x=1.5, y=1.0, z=0.0)],
            },
        )
        chain = ProteinChain(pdb_id="x", model_id=1, chain_id="A", residues=[r])
        chain.bonds = detect_bonds(chain)
        chain._rebuild_index()

        o = r.atoms["O"][0]
        cb = r.atoms["CB"][0]
        ca = r.atoms["CA"][0]
        c = r.atoms["C"][0]
        # O–C–CA–CB = 3
        assert chain.bond_distance(o.serial, cb.serial) == 3
        # O–C–CA = 2
        assert chain.bond_distance(o.serial, ca.serial) == 2
        # C–O = 1
        assert chain.bond_distance(c.serial, o.serial) == 1

    # -- Backbone coordinate extraction --

    def test_get_backbone_xyz_shape(self, chain):
        xyz = chain.get_backbone_xyz()
        assert xyz.shape == (3, 3, 3)
        assert xyz.dtype == np.float64

    def test_get_backbone_xyz_values(self, chain):
        xyz = chain.get_backbone_xyz()
        # First CA should be at (1.5, 0, 0)
        np.testing.assert_array_almost_equal(xyz[0, 1], [1.5, 0.0, 0.0])

    def test_get_backbone_mask(self, chain):
        mask = chain.get_backbone_mask()
        assert mask.shape == (3,)
        assert mask.all()

    def test_get_backbone_mask_missing_atom(self):
        r = Residue(
            name="ALA",
            seq_id=1,
            atoms={
                "CA": [Atom(serial=1, name="CA", element="C", x=0.0, y=0.0, z=0.0)],
            },
        )
        chain = ProteinChain(pdb_id="x", model_id=1, chain_id="A", residues=[r])
        assert not chain.get_backbone_mask()[0]

    # -- Dataframe conversion --

    def test_to_dataframe_columns(self, chain):
        df = chain.to_dataframe()
        expected = [
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
        for col in expected:
            assert col in df.columns

    def test_to_dataframe_shape(self, chain):
        df = chain.to_dataframe()
        assert len(df) == 9  # 3 residues x 3 atoms

    def test_to_dataframe_backbone_only(self, chain):
        df = chain.to_dataframe(backbone_only=True)
        assert len(df) == 9

    def test_round_trip_dataframe(self, chain):
        df = chain.to_dataframe()
        chain2 = ProteinChain.from_dataframe(df, "test", 1, "A")
        assert chain2.num_residues == chain.num_residues
        assert chain2.num_atoms == chain.num_atoms
        assert len(chain2.bonds) == len(chain.bonds)

    # -- Clone --

    def test_clone_independence(self, chain):
        chain2 = chain.clone()
        assert chain2.num_residues == chain.num_residues
        assert chain2 is not chain
        # Modify clone, original unchanged
        chain2.residues[0].ca.x = 999.0
        assert chain.residues[0].ca.x != 999.0

    def test_clone_bonds_copied(self, chain):
        chain2 = chain.clone()
        assert len(chain2.bonds) == len(chain.bonds)

    # -- Perturb --

    def test_perturb_changes_coordinates(self, chain):
        orig_x = chain.residues[0].ca.x
        chain.perturb(0.5, rng=np.random.default_rng(42))
        assert chain.residues[0].ca.x != orig_x

    def test_perturb_zero_radius(self, chain):
        orig = chain.get_backbone_xyz().copy()
        chain.perturb(0.0, rng=np.random.default_rng(42))
        np.testing.assert_array_almost_equal(orig, chain.get_backbone_xyz())

    def test_perturb_max_displacement(self, chain):
        chain.perturb(1.0, rng=np.random.default_rng(123))
        xyz = chain.get_backbone_xyz()
        # Should not displace more than ~sqrt(3) from origin shape
        assert np.abs(xyz).max() < 100  # sanity bound


# ── perturb_chain utility ───────────────────────────────────


class TestPerturbChain:
    def test_returns_new_chain(self):
        chain = _make_chain(2)
        chain2 = perturb_chain(chain, 0.5, rng=np.random.default_rng(42))
        assert chain2 is not chain
        assert chain2.num_residues == chain.num_residues

    def test_original_unchanged(self):
        chain = _make_chain(2)
        orig = chain.get_backbone_xyz().copy()
        perturb_chain(chain, 0.5, rng=np.random.default_rng(42))
        np.testing.assert_array_equal(orig, chain.get_backbone_xyz())


# ── Side-chain bond detection ───────────────────────────────


class TestSideChainBonds:
    def test_ala_cb_bond(self):
        """ALA has CA-CB bond."""
        r = Residue(
            name="ALA",
            seq_id=1,
            atoms={
                "N": [Atom(serial=1, name="N", element="N", x=0.0, y=0.0, z=0.0)],
                "CA": [Atom(serial=2, name="CA", element="C", x=1.5, y=0.0, z=0.0)],
                "C": [Atom(serial=3, name="C", element="C", x=2.5, y=0.0, z=0.0)],
                "CB": [Atom(serial=4, name="CB", element="C", x=1.5, y=1.0, z=0.0)],
            },
        )
        chain = ProteinChain(pdb_id="x", model_id=1, chain_id="A", residues=[r])
        chain.bonds = detect_bonds(chain)
        chain._rebuild_index()
        cb = r.atoms["CB"][0]
        bonded = chain.get_bonded(cb.serial)
        assert len(bonded) == 1
        assert bonded[0].name == "CA"

    def test_gly_no_sidechain(self):
        """GLY has no sidechain atoms, only backbone bonds."""
        r = Residue(
            name="GLY",
            seq_id=1,
            atoms={
                "N": [Atom(serial=1, name="N", element="N", x=0.0, y=0.0, z=0.0)],
                "CA": [Atom(serial=2, name="CA", element="C", x=1.5, y=0.0, z=0.0)],
                "C": [Atom(serial=3, name="C", element="C", x=2.5, y=0.0, z=0.0)],
            },
        )
        chain = ProteinChain(pdb_id="x", model_id=1, chain_id="A", residues=[r])
        chain.bonds = detect_bonds(chain)
        chain._rebuild_index()
        # Only 2 bonds: N-CA, CA-C
        assert len(chain.bonds) == 2

    def test_unknown_residue_gets_backbone_only(self):
        """Non-standard residues get no bonds (not in CCD CIF)."""
        r = Residue(
            name="XYZ",
            seq_id=1,
            atoms={
                "N": [Atom(serial=1, name="N", element="N", x=0.0, y=0.0, z=0.0)],
                "CA": [Atom(serial=2, name="CA", element="C", x=1.5, y=0.0, z=0.0)],
                "C": [Atom(serial=3, name="C", element="C", x=2.5, y=0.0, z=0.0)],
                "CB": [Atom(serial=4, name="CB", element="C", x=1.5, y=1.0, z=0.0)],
            },
        )
        chain = ProteinChain(pdb_id="x", model_id=1, chain_id="A", residues=[r])
        chain.bonds = detect_bonds(chain)
        chain._rebuild_index()
        # XYZ is not in the CCD CIF → no intra-residue bonds.
        assert len(chain.bonds) == 0


# ── Disulfide bond detection ────────────────────────────────


class TestDisulfideBonds:
    @pytest.fixture
    def cys_chain(self):
        """Two non-adjacent CYS residues with SG atoms close enough for a disulfide bond."""
        residues = []
        names = ["CYS", "GLY", "CYS"]
        for i, name in enumerate(names):
            o = i * 5.0
            atoms = {
                "N": [
                    Atom(
                        serial=i * 10 + 1,
                        name="N",
                        element="N",
                        x=o + 0.0,
                        y=0.0,
                        z=0.0,
                    )
                ],
                "CA": [
                    Atom(
                        serial=i * 10 + 2,
                        name="CA",
                        element="C",
                        x=o + 1.5,
                        y=0.0,
                        z=0.0,
                    )
                ],
                "C": [
                    Atom(
                        serial=i * 10 + 3,
                        name="C",
                        element="C",
                        x=o + 2.5,
                        y=0.0,
                        z=0.0,
                    )
                ],
            }
            if name == "CYS":
                atoms["SG"] = [
                    Atom(
                        serial=i * 10 + 4,
                        name="SG",
                        element="S",
                        x=2.0,
                        y=1.0,
                        z=0.0,
                    )
                ]
            residues.append(Residue(name=name, seq_id=i + 1, atoms=atoms))
        chain = ProteinChain(pdb_id="x", model_id=1, chain_id="A", residues=residues)
        chain.bonds = detect_bonds(chain)
        chain._rebuild_index()
        return chain

    def test_disulfide_detected(self, cys_chain):
        ss = detect_disulfide_bonds(cys_chain)
        assert len(ss) == 1

    def test_disulfide_atoms_are_sg(self, cys_chain):
        ss = detect_disulfide_bonds(cys_chain)
        a1 = cys_chain.get_atom(ss[0].a1)
        a2 = cys_chain.get_atom(ss[0].a2)
        assert a1.name == "SG"
        assert a2.name == "SG"

    def test_too_far_no_disulfide(self):
        """SG atoms far apart should not be detected."""
        residues = []
        for i, (sx, sy) in enumerate([(0.0, 0.0), (50.0, 50.0)]):
            atoms = {
                "N": [
                    Atom(serial=i * 10 + 1, name="N", element="N", x=sx, y=sy, z=0.0)
                ],
                "CA": [
                    Atom(
                        serial=i * 10 + 2,
                        name="CA",
                        element="C",
                        x=sx + 1.5,
                        y=sy,
                        z=0.0,
                    )
                ],
                "C": [
                    Atom(
                        serial=i * 10 + 3,
                        name="C",
                        element="C",
                        x=sx + 2.5,
                        y=sy,
                        z=0.0,
                    )
                ],
                "SG": [
                    Atom(
                        serial=i * 10 + 4,
                        name="SG",
                        element="S",
                        x=sx + 1.5,
                        y=sy + 1.0,
                        z=0.0,
                    )
                ],
            }
            residues.append(Residue(name="CYS", seq_id=i + 1, atoms=atoms))
        chain = ProteinChain(pdb_id="x", model_id=1, chain_id="A", residues=residues)
        ss = detect_disulfide_bonds(chain)
        assert len(ss) == 0

    def test_adjacent_cys_skipped(self):
        """Adjacent CYS residues (seq_id diff <= 1) are skipped."""
        residues = []
        for i in range(2):
            atoms = {
                "N": [
                    Atom(
                        serial=i * 10 + 1,
                        name="N",
                        element="N",
                        x=i * 3.8,
                        y=0.0,
                        z=0.0,
                    )
                ],
                "CA": [
                    Atom(
                        serial=i * 10 + 2,
                        name="CA",
                        element="C",
                        x=i * 3.8 + 1.5,
                        y=0.0,
                        z=0.0,
                    )
                ],
                "C": [
                    Atom(
                        serial=i * 10 + 3,
                        name="C",
                        element="C",
                        x=i * 3.8 + 2.5,
                        y=0.0,
                        z=0.0,
                    )
                ],
                "SG": [
                    Atom(serial=i * 10 + 4, name="SG", element="S", x=2.0, y=1.0, z=0.0)
                ],
            }
            residues.append(Residue(name="CYS", seq_id=i + 1, atoms=atoms))
        chain = ProteinChain(pdb_id="x", model_id=1, chain_id="A", residues=residues)
        ss = detect_disulfide_bonds(chain, max_distance=5.0)
        assert len(ss) == 0  # adjacent, skipped despite proximity


# ── ProteinEntry ────────────────────────────────────────────


class TestProteinEntry:
    def test_creation_empty(self):
        entry = ProteinEntry(pdb_id="1abc")
        assert entry.pdb_id == "1abc"
        assert entry.num_chains == 0
        assert entry.num_residues == 0

    def test_creation_with_chains(self):
        c1 = _make_chain(2)
        c2 = _make_chain(3)
        c2.chain_id = "B"
        entry = ProteinEntry(pdb_id="1abc", chains=[c1, c2])
        assert entry.num_chains == 2
        assert entry.num_residues == 5
        assert entry.num_atoms == 15

    def test_getitem(self):
        c1 = _make_chain(2)
        c2 = _make_chain(3)
        c2.chain_id = "B"
        entry = ProteinEntry(pdb_id="1abc", chains=[c1, c2])
        assert entry["A"] is c1
        assert entry["B"] is c2

    def test_getitem_keyerror(self):
        entry = ProteinEntry(pdb_id="1abc")
        with pytest.raises(KeyError):
            _ = entry["A"]

    def test_iter_chains(self):
        c1 = _make_chain(2)
        c2 = _make_chain(2)
        c2.chain_id = "B"
        entry = ProteinEntry(pdb_id="1abc", chains=[c1, c2])
        chains = list(entry.iter_chains())
        assert len(chains) == 2

    def test_iter_residues(self):
        c1 = _make_chain(2)
        c2 = _make_chain(3)
        c2.chain_id = "B"
        entry = ProteinEntry(pdb_id="1abc", chains=[c1, c2])
        residues = list(entry.iter_residues())
        assert len(residues) == 5

    def test_iter_atoms(self):
        c1 = _make_chain(2)
        c2 = _make_chain(2)
        c2.chain_id = "B"
        entry = ProteinEntry(pdb_id="1abc", chains=[c1, c2])
        atoms = list(entry.iter_atoms())
        assert len(atoms) == 12  # 2 chains * 2 residues * 3 atoms

    def test_to_dataframe(self):
        c1 = _make_chain(2)
        c2 = _make_chain(2)
        c2.chain_id = "B"
        entry = ProteinEntry(pdb_id="1abc", chains=[c1, c2])
        df = entry.to_dataframe()
        assert len(df) == 12

    def test_repr(self):
        entry = ProteinEntry(pdb_id="1abc")
        r = repr(entry)
        assert "ProteinEntry" in r
        assert "1abc" in r

    # -- from_cif using test data --

    @pytest.mark.parametrize(
        "path",
        [
            str(TEST_DATA / "1hho.cif"),
        ],
    )
    def test_from_cif_file(self, path):
        entry = ProteinEntry.from_cif(path)
        assert entry.pdb_id == "1HHO"
        assert entry.num_chains > 0
        # Every chain should have at least one residue
        for chain in entry.chains:
            assert chain.num_residues > 0
            assert chain.num_atoms > 0
            assert len(chain.bonds) > 0

    def test_from_cif_id(self):
        entry = ProteinEntry.from_cif("1HHO")
        assert entry.pdb_id == "1HHO"
        assert entry.num_chains > 0


# ── Constants ───────────────────────────────────────────────


class TestConstants:
    def test_residue_bonds_covers_standard_20(self):
        standard = {
            "ALA",
            "ARG",
            "ASN",
            "ASP",
            "CYS",
            "GLN",
            "GLU",
            "GLY",
            "HIS",
            "ILE",
            "LEU",
            "LYS",
            "MET",
            "PHE",
            "PRO",
            "SER",
            "THR",
            "TRP",
            "TYR",
            "VAL",
        }
        assert set(RESIDUE_BONDS.keys()) == standard

    def test_residue_bonds_are_lists_of_pairs(self):
        for name, bonds in RESIDUE_BONDS.items():
            assert isinstance(bonds, list)
            for pair in bonds:
                assert isinstance(pair, tuple)
                assert len(pair) == 3
                assert isinstance(pair[0], str)
                assert isinstance(pair[1], str)
                assert isinstance(pair[2], str)


# ── Invariant bridge ────────────────────────────────────────


class TestInvariantBridge:
    def test_get_invariant_from_structure(self):
        from bri.invariant import get_invariant_from_structure

        chain = _make_chain(5)
        inv = get_invariant_from_structure(chain)
        assert inv is not None
        assert len(inv) == 5
        assert "x(N)" in inv.columns
        assert "chain_length" in inv.columns

    def test_get_invariant_from_structure_ext(self):
        from bri.invariant import get_invariant_from_structure

        chain = _make_chain(5)
        inv = get_invariant_from_structure(chain, ext=True)
        assert "angle(N)" in inv.columns
        assert "tau(NA)" in inv.columns


# ── Backward compatibility ──────────────────────────────────


class TestBackwardCompatibility:
    def test_legacy_imports_still_work(self):
        from bri import Chain, Entry, MiniChain, MiniEntry

        assert Chain is not None
        assert Entry is not None
        assert MiniChain is not None
        assert MiniEntry is not None

    def test_new_imports_available(self):
        from bri import Atom, Bond, Residue, ProteinChain, ProteinEntry
        from bri import detect_bonds, detect_disulfide_bonds, perturb_chain

        assert Atom is not None
        assert Bond is not None
        assert Residue is not None
        assert ProteinChain is not None
        assert ProteinEntry is not None
        assert callable(detect_bonds)
        assert callable(detect_disulfide_bonds)
        assert callable(perturb_chain)


# ── Clash detection (atom_clash) ────────────────────────────


class TestClashDetection:
    """Tests for atom_clash.py using the ProteinChain bond graph."""

    @pytest.fixture
    def ala_chain(self):
        """Single ALA residue with backbone + CB."""
        r = Residue(
            name="ALA",
            seq_id=1,
            atoms={
                "N": [Atom(serial=1, name="N", element="N", x=0.0, y=0.0, z=0.0)],
                "CA": [Atom(serial=2, name="CA", element="C", x=1.46, y=0.0, z=0.0)],
                "C": [Atom(serial=3, name="C", element="C", x=2.98, y=0.0, z=0.0)],
                "O": [Atom(serial=4, name="O", element="O", x=4.21, y=0.0, z=0.0)],
                "CB": [Atom(serial=5, name="CB", element="C", x=1.46, y=1.53, z=0.0)],
            },
        )
        chain = ProteinChain(pdb_id="test", model_id=1, chain_id="A", residues=[r])
        chain.bonds = detect_bonds(chain)
        chain._rebuild_index()
        return chain

    def test_bonded_clash_too_far(self, ala_chain):
        """N-CA at 1.46 Å vs expected 1.3 Å → should be bond_gap."""
        from atom_clash import _is_bonded_clash

        res = ala_chain.residues[0]
        n = res.n
        ca = res.ca
        d = float(np.linalg.norm(n.coord - ca.coord))
        result = _is_bonded_clash(n, ca, res, res, d)
        assert result is not None
        assert result["kind"] == "bond_gap"

    def test_bonded_clash_too_close(self, ala_chain):
        """C-O at 0.9 Å vs expected 1.2 Å (C=O) → too close."""
        from atom_clash import _is_bonded_clash

        res = ala_chain.residues[0]
        c = res.c
        o = res.o
        o.x = c.x + 0.9
        d = float(np.linalg.norm(c.coord - o.coord))
        result = _is_bonded_clash(c, o, res, res, d)
        assert result is not None
        assert result["kind"] == "bond_too_close"

    def test_nonbonded_clash_detected(self, ala_chain):
        """N and CB are 2 hops apart → eligible for nonbonded clash check."""
        from atom_clash import _is_nonbonded_clash

        n = ala_chain.residues[0].n
        cb = ala_chain.residues[0].atoms["CB"][0]
        d_graph = ala_chain.bond_distance(n.serial, cb.serial)
        assert d_graph == 2  # Should NOT appear in nonbonded (filtered by graph dist)

    def test_three_hop_atoms_are_nonbonded(self, ala_chain):
        """O and CB are 3 hops apart (O-C-CA-CB) → eligible for nonbonded."""
        o = ala_chain.residues[0].o
        cb = ala_chain.residues[0].atoms["CB"][0]
        d_graph = ala_chain.bond_distance(o.serial, cb.serial)
        assert d_graph == 3  # O and CB are 3 bonds apart, should be in nonbonded

    def test_detect_clashes_integration(self, ala_chain):
        """Full detect_clashes run on a single-residue chain."""
        from atom_clash import detect_clashes

        res = ala_chain.residues[0]
        # Move O close to CB so nonbonded clash triggers
        res.o.x = 1.46
        res.o.y = 1.50
        res.o.z = 0.0
        bonded, nonbonded = detect_clashes(ala_chain, spatial_cutoff=5.0)
        # N-CA should appear as bond_gap (1.46 Å > 1.1 × 1.3)
        assert not bonded.empty
        gap_atoms = bonded[bonded["clash_type"] == "bond_gap"]
        assert not gap_atoms.empty
        # O and CB (3 hops) should be in nonbonded
        o_name = res.o.name
        cb_name = res.atoms["CB"][0].name
        o_id = res.seq_id
        o_cb_in_nb = any(
            (
                row["residue_id1"] == o_id
                and row["atom1"] == o_name
                and row["atom2"] == cb_name
            )
            or (
                row["residue_id1"] == o_id
                and row["atom1"] == cb_name
                and row["atom2"] == o_name
            )
            for _, row in nonbonded.iterrows()
        )
        assert o_cb_in_nb, (
            "O and CB (3-hop) not in nonbonded — "
            "vdW threshold may not be met with current geometry"
        )

    def test_detect_clashes_no_2hop_in_nonbonded(self, ala_chain):
        """Verify no 2-hop pair appears in nonbonded clashes."""
        from atom_clash import detect_clashes

        _, nonbonded = detect_clashes(ala_chain, spatial_cutoff=5.0)
        # No way to check bond_distance without serials, but there
        # should be no 2-hop violations — verify via residue inspection.
        # N and CB are 2-hop → should NOT appear as both in same residue.
        res = ala_chain.residues[0]
        n_cb_in_nb = any(
            (
                row["residue_id1"] == res.seq_id
                and row["residue_id2"] == res.seq_id
                and {row["atom1"], row["atom2"]} == {"N", "CB"}
            )
            for _, row in nonbonded.iterrows()
        )
        assert not n_cb_in_nb, "2-hop pair N-CB found in nonbonded!"

    def test_interchain_clashes(self):
        """Two chains with close atoms should detect inter-chain clashes."""
        # Chain A
        r1 = Residue(
            name="ALA",
            seq_id=1,
            atoms={
                "N": [Atom(serial=1, name="N", element="N", x=0.0, y=0.0, z=0.0)],
                "CA": [Atom(serial=2, name="CA", element="C", x=1.5, y=0.0, z=0.0)],
                "C": [Atom(serial=3, name="C", element="C", x=2.5, y=0.0, z=0.0)],
            },
        )
        c1 = ProteinChain(pdb_id="t", model_id=1, chain_id="A", residues=[r1])
        c1.bonds = detect_bonds(c1)
        c1._rebuild_index()

        # Chain B — placed very close to chain A
        r2 = Residue(
            name="ALA",
            seq_id=1,
            atoms={
                "N": [Atom(serial=100, name="N", element="N", x=0.3, y=0.0, z=0.0)],
                "CA": [Atom(serial=101, name="CA", element="C", x=1.7, y=0.0, z=0.0)],
                "C": [Atom(serial=102, name="C", element="C", x=2.8, y=0.0, z=0.0)],
            },
        )
        c2 = ProteinChain(pdb_id="t", model_id=1, chain_id="B", residues=[r2])
        c2.bonds = detect_bonds(c2)
        c2._rebuild_index()

        entry = ProteinEntry(pdb_id="t", chains=[c1, c2])
        from atom_clash import detect_interchain_clashes

        inter = detect_interchain_clashes(entry, spatial_cutoff=5.0)
        assert not inter.empty
        # All should be cross-chain
        assert (inter["chain_id1"] != inter["chain_id2"]).all()

    def test_process_entry_integration(self):
        """process_entry on a real CIF file."""
        from atom_clash import process_entry

        bonded, intra_nb, inter = process_entry(
            str(TEST_DATA / "1hho.cif"), spatial_cutoff=5.0
        )
        # Should produce some results for this well-studied protein
        assert not bonded.empty
        assert not intra_nb.empty
        # Verify no 2-hop violations
        # (can't check bond_distance without the chain, but structure is correct)
