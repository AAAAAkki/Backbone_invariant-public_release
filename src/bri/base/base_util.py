"""Base utilities for protein structure analysis.

This module provides basic utilities and data structures for protein structure analysis,
including file loading, data conversion, and structure representation.
"""

import json
import msgpack
import os
from pathlib import Path
from typing import overload, Literal, Optional, Any

import biotite.database.rcsb as rcsb
from biotite.structure.io.pdbx import (
    BinaryCIFBlock,
    BinaryCIFFile,
    CIFBlock,
    CIFFile,
    CIFCategory,
)
import pandas as pd
from requests import get


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

STANDARD_COL = [
    "model_id",
    "chain_id",
    "residue_id",
    "residue_label",
    "atom",
    "x",
    "y",
    "z",
    "occupancy",
]

_CIF_SOURCE = Literal["AF", "PDB"]
_SOURCE_MAP: dict[_CIF_SOURCE, str] = {
    "AF": "https://alphafold.ebi.ac.uk/files/",
    "PDB": "https://files.rcsb.org/view/",
}


@overload
def block2frame(
    block: BinaryCIFBlock | CIFBlock,
    category: str,
    columns: list[str],
    plain: Literal[True],
) -> Optional[list[dict[str, Any]]]: ...


@overload
def block2frame(
    block: BinaryCIFBlock | CIFBlock,
    category: str,
    columns: list[str],
    plain: Literal[False],
) -> Optional[dict[str, list[Any]]]: ...


def block2frame(
    block: BinaryCIFBlock | CIFBlock,
    category: str,
    columns: list[str],
    plain: bool = False,
) -> Optional[dict[str, list[Any]] | list[dict[str, Any]]]:
    """Convert the category in a Block into frame-like structure that can instantialize a `DataFrame` with selected columns

    :param block: Block of a `BinaryCIFFile` or `CIFFile`
    :param category: Assigned category
    :param columns: Selected columns existed in the target category
    :param plain: Return a `dict` of `list` of values following column names, otherwise a `list` of `dict` that each represents one row. Defaults to False
    :return: A frame-like data that can be used to generate `DataFrame`
    """
    cif_category: CIFCategory = block.get(category, None)
    if cif_category:
        data: dict[str, list[Any]] = {}
        for column in columns:
            if column == "details":
                continue
            data[column] = cif_category[column].as_array().tolist()
        res = data
        if plain:
            keys = data.keys()
            values = data.values()
            res = [dict(zip(keys, value_group)) for value_group in zip(*values)]
        return res

    return None


class StructureBase:
    """Base class for protein structure representation.

    This class provides basic functionality for loading and handling protein structure data.
    """

    extract_cols: dict[str, str] = {
        "cate": "str",
        "id": "str",
        "type_symbol": "str",
        "label_alt_id": "str",
        "residue_label": "str",
        "chain_id": "str",
        "entity_id": "int",
        "residue_id": "int",
        "pdbx_PDB_ins_code": "str",
        "x": "float64",
        "y": "float64",
        "z": "float64",
        "B_iso_or_equiv": "str",
        "pdbx_formal_charge": "str",
        "auth_residue_id": "float64",
        "auth_comp_id": "str",
        "auth_chain_id": "str",
        "atom": "str",
        "occupancy": "str",
        "model_id": "int",
    }

    def __init__(
        self,
        path: str,
        extra_keys: Optional[dict[str, str]] = None,
        data: Optional[pd.DataFrame] = None,
    ) -> None:
        """Initialize a StructureBase instance, where `path`, `perturb_radius`, `pdb_id`, `entity_info`, `coordinates`, `invariant` are assigned.

        :param path: An entry id or a url/path of a CIF file
        :param extra_keys: Additional values to be loaded from the CIF
        :param data: Pre-loaded coordinates data, defaults to None
        """
        self._path: str = path  #: Input path
        self._perturb_radius: float = 0.0  #: Radius of perturbation
        self._pdb_id: str = Path(path).stem.upper()  #: PDB entry id
        self._entity_info: dict[int, Any] = {}  #: Entity information
        self._invariant: Optional[pd.DataFrame] = (
            None  #: Computed backbone rigid invariant
        )

        # regulate data type
        if data is not None:
            cols = list(self.extract_cols.keys())
            self._coordinates: pd.DataFrame = data.loc[:, cols]
        else:
            # load mmCIF file
            self._file: BinaryCIFBlock | CIFBlock = self._load_cif_block()
            self._coordinates = self._get_coordinate_meta()

            # add additional category items
            if extra_keys is not None:
                self.add_extra_fields(extra_keys)

            self._entity_info = self._get_entity_meta()

    def _load_cif_block(self) -> BinaryCIFBlock | CIFBlock:
        """Load target mmCIF or Binary CIF file as a structural data block.

        :return: Content of target mmCIF file
        """
        path = Path(self.path)

        if path.exists():
            try:
                file = CIFFile.read(path)
            except UnicodeDecodeError:  # Target file is binary encoded
                file = BinaryCIFFile.read(path)
        else:
            if self.path.startswith("http"):  # url provided
                content = get(self.path).content
                try:
                    file = CIFFile.deserialize(content.decode())
                except UnicodeDecodeError:
                    file = BinaryCIFFile.deserialize(
                        msgpack.unpackb(content, use_list=True, raw=False)
                    )
            else:  # entry id
                if len(self.path) < 5:  # Standard 4-byte PDB entry id
                    file = CIFFile.read(rcsb.fetch(self.path, "cif"))
                else:
                    file = BinaryCIFFile.read(rcsb.fetch(self.path, "bcif"))

        return file.block

    def add_extra_fields(self, specified_items: dict[str, str]) -> None:
        """Add additional category content.

        :param specified_items: Dictionary mapping attribute names to CIF categories
        """
        for k, v in specified_items.items():
            fields = v.split(".")
            category = self._file.get(fields[0], None)
            value = None
            if category and len(fields) > 1:
                value = category.get(fields[1], None)
                if value is not None:
                    if category.row_count == 1:
                        value = value.as_item()
                    else:
                        value = list(value.as_array())
            elif category:
                value = block2frame(self._file, fields[0], list(category.keys()), True)
            # else: value = None

            self.__dict__.update({k: value})

    def _get_coordinate_meta(self) -> pd.DataFrame:
        """Extract coordinate metadata from the structure file.

        :return: DataFrame containing coordinate metadata
        """
        short_col_names = list(self.extract_cols.keys())
        extract_cols_l = [k for k, v in COLUMN_MAP.items() if v in short_col_names]
        res = pd.DataFrame(block2frame(self._file, "atom_site", extract_cols_l, False))
        res.rename(columns=COLUMN_MAP, inplace=True)
        res = res[res["cate"] == "ATOM"]

        # specify data types
        for col, np_type in self.extract_cols.items():
            res[col] = res[col].astype(np_type)

        return res.loc[:, short_col_names]

    def _get_entity_meta(self) -> dict[int, dict[str, Any]]:
        """Extract entity metadata from the structure file.

        :return: Dictionary containing entity metadata
        """
        output: dict[int, dict[str, Any]] = {}
        entity = block2frame(self._file, "entity", ["id", "type"], True)
        entity_poly = block2frame(
            self._file, "entity_poly", ["type", "pdbx_strand_id"], True
        )
        entity_poly_seq = block2frame(
            self._file, "entity_poly_seq", ["mon_id", "entity_id"], True
        )
        if (entity and entity_poly and entity_poly_seq) is None:
            return {}

        if entity and entity_poly and entity_poly_seq:
            entity_id = [int(e["id"]) for e in entity if e.get("type") == "polymer"]

            for id in entity_id:
                residues = [
                    i.get("mon_id")
                    for i in entity_poly_seq
                    if i.get("entity_id") == str(id)
                ]
                entity_info = {
                    "entity_poly_type": entity_poly[id - 1].get("type"),
                    "chain_id": entity_poly[id - 1].get("pdbx_strand_id"),
                    "chain_length": len(residues),
                    "pdbx_description": entity[id - 1].get("pdbx_description"),
                }
                output[id] = entity_info

        return output

    @property
    def path(self) -> str:
        """Get the path to the structure file.

        :return: Path to structure file
        """
        return self._path

    @property
    def pdb_id(self) -> str:
        """Get the PDB ID.

        :return: PDB ID
        """
        return self._pdb_id

    @property
    def perturb_radius(self) -> float:
        """Get the perturbation radius.

        :return: Perturbation radius
        """
        return self._perturb_radius

    @property
    def coordinates(self) -> pd.DataFrame:
        """Get the atomic coordinates.

        :return: DataFrame containing atomic coordinates
        """
        return self._coordinates

    @property
    def invariant(self) -> Optional[pd.DataFrame]:
        """Get the backbone rigid invariant.

        :return: DataFrame containing backbone rigid invariant
        """
        return self._invariant

    @property
    def entity_info(self) -> dict[int, Any]:
        """Get the entity information.

        :return: Dictionary containing entity information
        """
        return self._entity_info


# Reference: https://www.wwpdb.org/data/ccd
with open(
    os.path.dirname(os.path.abspath(__file__)) + "/chemical_component_dictionary.json",
    "r",
) as f:
    amino_acid_short = json.load(f)

basic_amino_acid_20 = (
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
)

basic_amino_acid_20_s = tuple([amino_acid_short.get(i) for i in basic_amino_acid_20])
