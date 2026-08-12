## PDB Data Cleaning

Here lists several cleaning operations, which are done in this project in sequence to obtain a clean PDB dataset for further progress. 

> Updated by 31 March 2024

In the cleaning process, all PDB data are accessed from .cif files for its strength in flexibility and extensibility. However, there also exists some data inconsistency that might affect our cleaning process and presentation of our filtering results.

In particular, a .cif file includes a field called 'entity_poly_seq', which clearly shows residues in sequence with chain id and entity id they make up. Number of residues in this field are strictly increased from '1' in each chain. While to obtain the atom coordinates, it is necessary to refer to another field called 'atom_site', where we found that residue indices can begin from a random value, even negative numbers. Moreover, residues specified in coordinate data is not consist with residues listed in 'entity_poly_seq' section. To guarantee the continuity of chains, entries that include chains with gaps in the middle are filtered out (see Rule 3), but chains with missing residue coordinates in the beginning or end of chains remains. As a result, some chains have shorter length compared to what they should have, as declared in RCSB achieve. 

Before the filtering rules, it is essential to clarify another operation on atom coordinates which is implemented before all the filters. In the field 'atom_site', it is noticed that any atom is label as either 'ATOM' or 'HETATM', where [ATOM record is used to identify proteins or nucleic acid atoms, and the HETATM record is used to identify atoms in small molecules]([PDB-101: Learn: Guide to Understanding PDB Data: Dealing with Coordinates (rcsb.org)](https://pdb101.rcsb.org/learn/guide-to-understanding-pdb-data/dealing-with-coordinates)), including ligands. Since this project is focusing on protein properties, coordinates from small molecules with 'HETATM' record and atoms from nucleic acid are all removed. This operation may also contribute to the loss of residue indices as well as the inconsistency of chain length, which are mentioned above. 

**Input** (Original PDB entry database): 217705 entries

1. Non-protein

   - Usually, protein data in the format of .cif from RCSB should include attributes which describe the classifications of their components, i.e., whether it is a protein , DNA or RNA. Since we are only focusing on protein data, those pure DAN / RNA data without any peptides (e.g., 100D, 2K4L, 196D) should be dropped. 

     Moreover, some data does not even have the field for polymer types, such as 3HYA, a kind of hyaluronic acid being archived in RCSB in a different format from normal .cif. This kind of data is also noticed to filter out.

2. Disordered Atoms

   - Disordered atom records can be seen as possible atom positions in one residue, with occupancy value < 1. 

     Any PDB entries containing disordered atoms are excluded from this project to avoid uncertainty.

3. Chain Break

   
      - In some conditions, protein data archived in RCSB does not have strictly contiguous residue index in chains, which means those residue indices are not increasing one by one in .cif file, or some residues are missing in coordinates data.
   
        These  PDB entries which happen to have missing residues are removed from the dataset we are focusing on.
   


4. Atom Clashes

   - Proteins are constructed with a basic back bond structure consisting of carbon, nitrogen, and oxygen atoms. In this stage, we are selecting 4 main atoms (alpha-group) which exist in all residues: a nitrogen (from amino group), a carbon, an alpha-carbon and an oxygen (from carboxyl group). We set this filter to avoid small angle between axes in the built coordinate basis.

     We picked pairs including N-CA, N-C, N-O, N-N+1, CA-C, CA-O, CA-N+1, C-O, C-N+1, O-N+1, where N, C, CA, O, are from the same residue and N+1 is from the next adjacent residue, and inspected the distances between pairs of atoms. We assume that there should be at least 1 Angstrom distance between any pair of atoms, so PDB entries containing atoms that violates this threshold are excluded to avoid clashes (atoms being too close).

2. Atom Gaps

   - Similar to the last examine, to avoid atoms being too spread out, we assumed that each bond between connected atoms should not exceed 2 Angstrom

     We checked every pair of atoms within distance of 3 bonds from the whole chain (N-CA, N-C, N-O, N-N+1, CA-C, CA-O, CA-N+1, CA-CA+1, C-O, C-N+1, C-CA+1, C-C+1, O-N+1, O-CA+1 where N, C, CA, O, are from the same residue and N+1 is from the next adjacent residue). Any PDB entries containing atoms that violates this threshold are excluded to avoid atoms being too spread.

4. Atom Missing

   - Even if chains in PDB are not broken by missing residues, there happen to exist incomplete residues, more specifically, alpha group atoms are missing. 

     PDB entries will be removed when this condition is detected.

5. Extra Residue

   - About 24 kinds of recognized amino acid are found to make up peptides in PDB entries from RCSB, sometimes also unknown residues. We are taking only 20 standard ones into account, and remove PDB entries which include extra ones in this step.

     The 20 standard amino acids is referring to *ALA, ARG, ASN, ASP, CYS, GLN, GLU, GLY, HIS, ILE, LEU, LYS, MET, PHE, PRO, SER, THR, TRP, TYR, VAL*

     While the 4 extra exceptional amino acids found in PDB coordinate data are *SEC(selenocysteine), PYL(pyrrolysine), ASX(asparagine/aspartate), GLX(glutamate)* and a kind of unrecognized residue annotated as *UNK(unknown)*.
