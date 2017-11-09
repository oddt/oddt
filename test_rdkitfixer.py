from distutils.version import LooseVersion

import rdkit
from rdkit import Chem

from nose.tools import assert_equal, assert_not_equal, assert_almost_equal

from rdkit_fixer import (AtomListToSubMol,
                         PreparePDBMol,
                         ExtractPocketAndLigand)


def test_atom_list_to_submol():
    mol = Chem.MolFromSmiles('CCCCC(=O)O')
    submol = AtomListToSubMol(mol, range(3, 7))
    assert_equal(submol.GetNumAtoms(), 4)
    assert_equal(submol.GetNumAtoms(), 4)
    assert_equal(submol.GetNumBonds(), 3)
    assert_equal(submol.GetBondBetweenAtoms(1, 2).GetBondType(),
                 rdkit.Chem.rdchem.BondType.DOUBLE)

    molfile = '2qwe_Sbridge.pdb'
    mol = Chem.MolFromPDBFile(molfile, sanitize=False, removeHs=False)
    assert_equal(mol.GetConformer().Is3D(), True)
    submol = AtomListToSubMol(mol, range(6), includeConformer=True)
    assert_equal(submol.GetConformer().Is3D(), True)

    # submol has residue info
    atom = submol.GetAtomWithIdx(0)
    info = atom.GetPDBResidueInfo()
    assert_equal(info.GetResidueName(), 'CYS')
    assert_equal(info.GetResidueNumber(), 92)

    # test multiple conformers
    mol.AddConformer(mol.GetConformer())
    assert_equal(mol.GetNumConformers(), 2)
    submol = AtomListToSubMol(mol, range(6), includeConformer=True)
    assert_equal(submol.GetNumConformers(), 2)


def test_multivalent_Hs():
    """Test if fixer deals with multivalent Hs"""

    # TODO: require mol without Hs in the future (rdkit v. 2018)
    molfile = '2c92_hypervalentH.pdb'
    mol = Chem.MolFromPDBFile(molfile, sanitize=False, removeHs=False)
    mol = PreparePDBMol(mol, residue_whitelist=[], removeHs=False)

    atom = mol.GetAtomWithIdx(84)
    assert_equal(atom.GetAtomicNum(), 1)  # is it H
    assert_equal(atom.GetDegree(), 1)  # H should have 1 bond

    for n in atom.GetNeighbors():  # Check if neighbor is from the same residue
        assert_equal(atom.GetPDBResidueInfo().GetResidueName(),
                     n.GetPDBResidueInfo().GetResidueName())

    # mol can be sanitized
    assert_equal(Chem.SanitizeMol(mol), Chem.SanitizeFlags.SANITIZE_NONE)


def test_HOH_bonding():
    """Test if fixer unbinds HOH"""

    molfile = '2vnf_bindedHOH.pdb'
    mol = Chem.MolFromPDBFile(molfile, sanitize=False, removeHs=False)
    # don't use templates and don't remove waters
    mol = PreparePDBMol(mol, residue_whitelist=[], removeHOHs=False)

    atom = mol.GetAtomWithIdx(5)
    assert_equal(atom.GetPDBResidueInfo().GetResidueName(), 'HOH')
    assert_equal(atom.GetDegree(), 0)  # HOH should have no bonds

    # mol can be sanitized
    assert_equal(Chem.SanitizeMol(mol), Chem.SanitizeFlags.SANITIZE_NONE)


def test_metal_bonding():
    """Test if fixer disconnects metals"""

    molfile = '1ps3_zn.pdb'
    mol = Chem.MolFromPDBFile(molfile, sanitize=False, removeHs=False)

    mol = PreparePDBMol(mol)

    atom = mol.GetAtomWithIdx(36)
    assert_equal(atom.GetAtomicNum(), 30)  # is it Zn
    assert_equal(atom.GetDegree(), 0)  # Zn should have no bonds
    assert_equal(atom.GetFormalCharge(), 2)
    assert_equal(atom.GetNumExplicitHs(), 0)

    # mol can be sanitized
    assert_equal(Chem.SanitizeMol(mol), Chem.SanitizeFlags.SANITIZE_NONE)


def test_interresidue_bonding():
    """Test if fixer removes wrong connections between residues"""

    molfile = '4e6d_residues.pdb'
    mol = Chem.MolFromPDBFile(molfile, sanitize=False, removeHs=False)

    mol = PreparePDBMol(mol)

    # check if O from PRO
    atom1 = mol.GetAtomWithIdx(11)
    assert_equal(atom1.GetAtomicNum(), 8)
    assert_equal(atom1.GetPDBResidueInfo().GetResidueName(), 'PRO')
    # ...and N from GLN
    atom2 = mol.GetAtomWithIdx(22)
    assert_equal(atom2.GetAtomicNum(), 7)
    assert_equal(atom2.GetPDBResidueInfo().GetResidueName(), 'GLN')
    # ...are not connected
    assert_equal(mol.GetBondBetweenAtoms(11, 22), None)

    # mol can be sanitized
    assert_equal(Chem.SanitizeMol(mol), Chem.SanitizeFlags.SANITIZE_NONE)


def test_intraresidue_bonding():
    """Test if fixer removes wrong connections within single residue"""

    molfile = '1idg_connectivity.pdb'
    mol = Chem.MolFromPDBFile(molfile, sanitize=False, removeHs=False)
    mol = PreparePDBMol(mol)

    # check if N and C from GLU20 are not connected
    atom1 = mol.GetAtomWithIdx(11)
    assert_equal(atom1.GetAtomicNum(), 7)
    assert_equal(atom1.GetPDBResidueInfo().GetResidueName(), 'GLU')
    assert_equal(atom1.GetPDBResidueInfo().GetResidueNumber(), 20)
    atom2 = mol.GetAtomWithIdx(13)
    assert_equal(atom2.GetAtomicNum(), 6)
    assert_equal(atom2.GetPDBResidueInfo().GetResidueName(), 'GLU')
    assert_equal(atom2.GetPDBResidueInfo().GetResidueNumber(), 20)

    assert_equal(mol.GetBondBetweenAtoms(11, 13), None)

    # mol can be sanitized
    assert_equal(Chem.SanitizeMol(mol), Chem.SanitizeFlags.SANITIZE_NONE)


def test_bondtype():
    """Test if fixer deals with non-standard residue and fixes bond types"""

    molfile = '3rsb_bondtype.pdb'
    mol = Chem.MolFromPDBFile(molfile, sanitize=False, removeHs=False)
    mol = PreparePDBMol(mol)

    # check if there is double bond between N and C from MSE
    atom1 = mol.GetAtomWithIdx(13)
    assert_equal(atom1.GetAtomicNum(), 6)
    assert_equal(atom1.GetPDBResidueInfo().GetResidueName(), 'MSE')
    atom2 = mol.GetAtomWithIdx(14)
    assert_equal(atom2.GetAtomicNum(), 8)
    assert_equal(atom2.GetPDBResidueInfo().GetResidueName(), 'MSE')

    # there is a bond and it is double
    bond = mol.GetBondBetweenAtoms(13, 14)
    assert_not_equal(bond, None)
    assert_almost_equal(bond.GetBondTypeAsDouble(), 2.0)

    # mol can be sanitized
    assert_equal(Chem.SanitizeMol(mol), Chem.SanitizeFlags.SANITIZE_NONE)


def test_ring():
    """Test if fixer adds missing bond in ring"""

    molfile = '4yzm_ring.pdb'
    mol = Chem.MolFromPDBFile(molfile, sanitize=False, removeHs=False)
    mol = PreparePDBMol(mol)

    # check if there is double bond between N and C from MSE
    atom1 = mol.GetAtomWithIdx(12)
    assert_equal(atom1.GetAtomicNum(), 6)
    assert_equal(atom1.GetPDBResidueInfo().GetResidueName(), 'PHE')
    atom2 = mol.GetAtomWithIdx(13)
    assert_equal(atom2.GetAtomicNum(), 6)
    assert_equal(atom2.GetPDBResidueInfo().GetResidueName(), 'PHE')

    # there is a bond and it is aromatic
    bond = mol.GetBondBetweenAtoms(12, 13)
    assert_not_equal(bond, None)
    assert_almost_equal(bond.GetBondTypeAsDouble(), 1.5)

    # mol can be sanitized
    assert_equal(Chem.SanitizeMol(mol), Chem.SanitizeFlags.SANITIZE_NONE)


def test_sulphur_bridge():
    """Test sulphur bridges retention"""

    molfile = '2qwe_Sbridge.pdb'
    mol = Chem.MolFromPDBFile(molfile, sanitize=False, removeHs=False)

    mol = PreparePDBMol(mol)

    atom1 = mol.GetAtomWithIdx(5)
    atom2 = mol.GetAtomWithIdx(11)
    bond = mol.GetBondBetweenAtoms(atom1.GetIdx(), atom2.GetIdx())
    assert_equal(atom1.GetPDBResidueInfo().GetName().strip(), 'SG')
    assert_equal(atom1.GetPDBResidueInfo().GetResidueNumber(), 92)
    assert_equal(atom2.GetPDBResidueInfo().GetName().strip(), 'SG')
    assert_equal(atom2.GetPDBResidueInfo().GetResidueNumber(), 417)
    assert_not_equal(bond, None)


def test_pocket_extractor():
    """Test extracting pocket and ligand"""

    molfile = '5ar7.pdb'
    mol = Chem.MolFromPDBFile(molfile, sanitize=False, removeHs=False)

    # there should be no pocket at 1A
    pocket, ligand = ExtractPocketAndLigand(mol, cutoff=1., confId=-1)
    assert_equal(pocket.GetNumAtoms(), 0)
    assert_equal(ligand.GetNumAtoms(), 26)

    # small pocket of 5A
    pocket, ligand = ExtractPocketAndLigand(mol, cutoff=12., confId=-1)
    assert_equal(pocket.GetNumAtoms(), 928)
    assert_equal(ligand.GetNumAtoms(), 26)

    # check if HOH is in pocket
    atom = pocket.GetAtomWithIdx(910)
    assert_equal(atom.GetAtomicNum(), 8)
    assert_equal(atom.GetPDBResidueInfo().GetResidueName(), 'HOH')

    # Prepare and sanitize pocket and ligand
    pocket = PreparePDBMol(pocket)
    ligand = PreparePDBMol(ligand)
    assert_equal(Chem.SanitizeMol(pocket), Chem.SanitizeFlags.SANITIZE_NONE)
    assert_equal(Chem.SanitizeMol(ligand), Chem.SanitizeFlags.SANITIZE_NONE)

    # Check atom/bond properies for both molecules
    bond = pocket.GetBondWithIdx(39)
    assert_equal(bond.GetIsAromatic(), True)
    assert_equal(bond.GetBeginAtom().GetPDBResidueInfo().GetResidueName(), 'TYR')

    atom = ligand.GetAtomWithIdx(22)
    assert_equal(atom.GetAtomicNum(), 7)
    assert_equal(atom.GetIsAromatic(), True)
    assert_equal(atom.GetPDBResidueInfo().GetResidueName(), 'SR8')

    # test if metal is in pocket
    molfile = '4p6p_lig_zn.pdb'
    mol = Chem.MolFromPDBFile(molfile, sanitize=False, removeHs=False)
    assert_equal(mol.GetNumAtoms(), 176)
    pocket, ligand = ExtractPocketAndLigand(mol, cutoff=5., confId=-1)
    assert_equal(pocket.GetNumAtoms(), 162)
    assert_equal(ligand.GetNumAtoms(), 14)

    atom = pocket.GetAtomWithIdx(153)
    assert_equal(atom.GetPDBResidueInfo().GetResidueName().strip(), 'ZN')
    atom = pocket.GetAtomWithIdx(160)
    assert_equal(atom.GetPDBResidueInfo().GetResidueName(), 'HOH')

    pocket, ligand = ExtractPocketAndLigand(mol, cutoff=5., expandResidues=False)
    assert_equal(pocket.GetNumAtoms(), 74)
    assert_equal(ligand.GetNumAtoms(), 14)

    atom = pocket.GetAtomWithIdx(65)
    assert_equal(atom.GetPDBResidueInfo().GetResidueName().strip(), 'ZN')
    atom = pocket.GetAtomWithIdx(73)
    assert_equal(atom.GetPDBResidueInfo().GetResidueName(), 'HOH')


def test_aromatic_ring():
    """Test aromaticity for partial matches"""

    # ring is complete and should be aromatic
    molfile = '5ar7_HIS.pdb'
    mol = Chem.MolFromPDBFile(molfile, sanitize=False, removeHs=False)
    mol = PreparePDBMol(mol)

    atom = mol.GetAtomWithIdx(6)
    assert_equal(atom.GetAtomicNum(), 7)
    info = atom.GetPDBResidueInfo()
    assert_equal(info.GetResidueName(), 'HIS')
    assert_equal(info.GetResidueNumber(), 246)
    assert_equal(info.GetName().strip(), 'ND1')
    assert_equal(atom.GetIsAromatic(), True)

    atom = mol.GetAtomWithIdx(9)
    assert_equal(atom.GetAtomicNum(), 7)
    info = atom.GetPDBResidueInfo()
    assert_equal(info.GetResidueName(), 'HIS')
    assert_equal(info.GetResidueNumber(), 246)
    assert_equal(info.GetName().strip(), 'NE2')
    assert_equal(atom.GetIsAromatic(), True)

    assert_equal(Chem.SanitizeMol(mol), Chem.SanitizeFlags.SANITIZE_NONE)

    # there is only one atom from the ring and it shouldn't be aromatic
    molfile = '3cx9_TYR.pdb'
    mol = Chem.MolFromPDBFile(molfile, sanitize=False, removeHs=False)
    mol = PreparePDBMol(mol)

    atom = mol.GetAtomWithIdx(14)
    assert_equal(atom.GetAtomicNum(), 6)
    info = atom.GetPDBResidueInfo()
    assert_equal(info.GetResidueName(), 'TYR')
    assert_equal(info.GetResidueNumber(), 138)
    assert_equal(info.GetName().strip(), 'CG')
    assert_equal(atom.GetIsAromatic(), False)
    assert_equal(Chem.SanitizeMol(mol), Chem.SanitizeFlags.SANITIZE_NONE)
