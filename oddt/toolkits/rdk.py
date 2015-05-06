#-*. coding: utf-8 -*-
## Copyright (c) 2008-2011, Noel O'Boyle; 2012, Adrià Cereto-Massagué; 2014, Maciej Wójcikowski;
## All rights reserved.
##
##  This file is part of Cinfony.
##  The contents are covered by the terms of the BSD license
##  which is included in the file LICENSE_BSD.txt.

"""
rdkit - A Cinfony module for accessing the RDKit from CPython

Global variables:
  Chem and AllChem - the underlying RDKit Python bindings
  informats - a dictionary of supported input formats
  outformats - a dictionary of supported output formats
  descs - a list of supported descriptors
  fps - a list of supported fingerprint types
  forcefields - a list of supported forcefields
"""

import os
from copy import copy
import gzip

import numpy as np

import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem import Descriptors
from rdkit import RDConfig

_descDict = dict(Descriptors.descList)

import rdkit.DataStructs
import rdkit.Chem.MACCSkeys
import rdkit.Chem.AtomPairs.Pairs
import rdkit.Chem.AtomPairs.Torsions
### ODDT ###
from rdkit.Chem.Lipinski import NumRotatableBonds
from rdkit.Chem.AllChem import ComputeGasteigerCharges
from rdkit.Chem.Pharm2D import Gobbi_Pharm2D,Generate

# trap errors since it's still new feature
try:
    from rdkit.Chem import CanonicalRankAtoms
except ImportError:
    pass

# PIL and Tkinter
try:
    import Tkinter as tk
    import Image as PIL
    import ImageTk as PILtk
except:
    PILtk = None

# Aggdraw
try:
    import aggdraw
    from rdkit.Chem.Draw import aggCanvas
except ImportError:
    aggdraw = None

fps = ['rdkit', 'layered', 'maccs', 'atompairs', 'torsions', 'morgan']
"""A list of supported fingerprint types"""
descs = _descDict.keys()
"""A list of supported descriptors"""

_formats = {'smi': "SMILES",
            'can': "Canonical SMILES",
            'mol': "MDL MOL file",
            'mol2': "Tripos MOL2 file",
            'sdf': "MDL SDF file",
            'inchi':"InChI",
            'inchikey':"InChIKey"}
_notinformats = ['can', 'inchikey']
_notoutformats = ['mol2']
if not Chem.INCHI_AVAILABLE:
    _notinformats += ['inchi']
    _notoutformats += ['inchi', 'inchikey']

informats = dict([(_x, _formats[_x]) for _x in _formats if _x not in _notinformats])
"""A dictionary of supported input formats"""
outformats = dict([(_x, _formats[_x]) for _x in _formats if _x not in _notoutformats])
"""A dictionary of supported output formats"""

base_feature_factory = AllChem.BuildFeatureFactory(os.path.join(RDConfig.RDDataDir,'BaseFeatures.fdef'))
""" Global feature factory based on BaseFeatures.fdef """

_forcefields = {'uff': AllChem.UFFOptimizeMolecule}
forcefields = _forcefields.keys()
"""A list of supported forcefields"""

def _filereader_mol2(filename):
    block = ''
    data = ''
    n = 0
    with gzip.open(filename) if filename.split('.')[-1] == 'gz' else open(filename) as f:
        for line in f:
            if line[:1] == '#':
                data += line
            elif line[:17] == '@<TRIPOS>MOLECULE':
                if n>0: #skip `zero` molecule (any preciding comments and spaces)
                    yield Molecule(source={'fmt': 'mol2', 'string': block})
                n += 1
                block = data
                data = ''
            block += line
        # open last molecule
        if block:
            yield Molecule(source={'fmt': 'mol2', 'string': block})
    
def _filereader_sdf(filename):
    block = ''
    n = 0
    with gzip.open(filename) if filename.split('.')[-1] == 'gz' else open(filename) as f:
        for line in f:
            block += line
            if line[:4] == '$$$$':
                yield Molecule(source={'fmt': 'sdf', 'string': block})
                n += 1
                block = ''
        if block: # open last molecule if any
            yield Molecule(source={'fmt': 'sdf', 'string': block})

def readfile(format, filename, *args, **kwargs):
    """Iterate over the molecules in a file.

    Required parameters:
       format - see the informats variable for a list of available
                input formats
       filename

    You can access the first molecule in a file using the next() method
    of the iterator:
        mol = readfile("smi", "myfile.smi").next()

    You can make a list of the molecules in a file using:
        mols = list(readfile("smi", "myfile.smi"))

    You can iterate over the molecules in a file as shown in the
    following code snippet:
    >>> atomtotal = 0
    >>> for mol in readfile("sdf", "head.sdf"):
    ...     atomtotal += len(mol.atoms)
    ...
    >>> print atomtotal
    43
    """
    if not os.path.isfile(filename):
        raise IOError, "No such file: '%s'" % filename
    format = format.lower()
    # Eagerly evaluate the supplier functions in order to report
    # errors in the format and errors in opening the file.
    # Then switch to an iterator...
    if format=="sdf":
        return _filereader_sdf(filename)
    elif format=="mol":
        def mol_reader():
            yield Molecule(Chem.MolFromMolFile(filename, *args, **kwargs))
        return mol_reader()
    elif format=="pdb":
        def mol_reader():
            yield Molecule(Chem.MolFromPDBFile(filename, *args, **kwargs))
        return mol_reader()
    elif format=="mol2":
        return _filereader_mol2(filename)
    elif format=="smi":
        iterator = Chem.SmilesMolSupplier(filename, delimiter=" \t",
                                          titleLine=False, *args, **kwargs)
        def smi_reader():
            for mol in iterator:
                yield Molecule(mol)
        return smi_reader()
    elif format=='inchi' and Chem.INCHI_AVAILABLE:
        def inchi_reader():
            for line in open(filename, 'r'):
                mol = Chem.inchi.MolFromInchi(line.strip(), *args, **kwargs)
                yield Molecule(mol)
        return inchi_reader()
    else:
        raise ValueError, "%s is not a recognised RDKit format" % format

def readstring(format, string, **kwargs):
    """Read in a molecule from a string.

    Required parameters:
       format - see the informats variable for a list of available
                input formats
       string

    Example:
    >>> input = "C1=CC=CS1"
    >>> mymol = readstring("smi", input)
    >>> len(mymol.atoms)
    5
    """
    format = format.lower()
    if format=="mol" or format=="sdf":
        mol = Chem.MolFromMolBlock(string, **kwargs)
    elif format=="mol2":
        mol = Chem.MolFromMol2Block(string, **kwargs)
    elif format=="pdb":
        mol = Chem.MolFromPDBBlock(string, **kwargs)
    elif format=="smi":
        mol = Chem.MolFromSmiles(string, **kwargs)
    elif format=='inchi' and Chem.INCHI_AVAILABLE:
        mol = Chem.inchi.MolFromInchi(string, **kwargs)
    else:
        raise ValueError,"%s is not a recognised RDKit format" % format
#    if mol:
    return Molecule(mol)
#    else:
#        raise IOError, "Failed to convert '%s' to format '%s'" % (
#            string, format)

class Outputfile(object):
    """Represent a file to which *output* is to be sent.

    Required parameters:
       format - see the outformats variable for a list of available
                output formats
       filename

    Optional parameters:
       overwite -- if the output file already exists, should it
                   be overwritten? (default is False)

    Methods:
       write(molecule)
       close()
    """
    def __init__(self, format, filename, overwrite=False):
        self.format = format
        self.filename = filename
        if not overwrite and os.path.isfile(self.filename):
            raise IOError, "%s already exists. Use 'overwrite=True' to overwrite it." % self.filename
        if format=="sdf":
            self._writer = Chem.SDWriter(self.filename)
        elif format=="smi":
            self._writer = Chem.SmilesWriter(self.filename, isomericSmiles=True)
        elif format in ('inchi', 'inchikey') and Chem.INCHI_AVAILABLE:
            self._writer = open(filename, 'w')
        elif format in ('mol2'):
            self._writer = gzip.open(filename, 'w') if filename.split('.')[-1] == 'gz' else open(filename, 'w')
        else:
            raise ValueError,"%s is not a recognised RDKit format" % format
        self.total = 0 # The total number of molecules written to the file

    def write(self, molecule):
        """Write a molecule to the output file.

        Required parameters:
           molecule
        """
        if not self.filename:
            raise IOError, "Outputfile instance is closed."
        if self.format in ('inchi', 'inchikey', 'mol2'):
            self._writer.write(molecule.write(self.format) +'\n')
        else:
            self._writer.write(molecule.Mol)
        self.total += 1

    def close(self):
        """Close the Outputfile to further writing."""
        self.filename = None
        self._writer.flush()
        del self._writer

class Molecule(object):
    """Represent an rdkit Molecule.

    Required parameter:
       Mol -- an RDKit Mol or any type of cinfony Molecule

    Attributes:
       atoms, data, formula, molwt, title

    Methods:
       addh(), calcfp(), calcdesc(), draw(), localopt(), make3D(), removeh(),
       write()

    The underlying RDKit Mol can be accessed using the attribute:
       Mol
    """
    _cinfony = True

    def __init__(self, Mol = None, source = None, protein = False):
        if hasattr(Mol, "_cinfony"):
            a, b = Mol._exchange
            if a == 0:
                molecule = readstring("smi", b)
            else:
                molecule = readstring("mol", b)
            Mol = molecule.Mol

        self.Mol = Mol
        ### ODDT ###
        self.protein = protein
        self._atom_dict = None
        self._res_dict = None
        self._ring_dict = None
        self._coords = None
        self._charges = None
        # lazy
        self._source = source # dict with keys: n, fmt, string, filename
    
    # lazy Molecule parsing requires masked Mol
    @property
    def Mol(self):
        if not self._Mol and self._source:
            self._Mol = readstring(self._source['fmt'], self._source['string']).Mol
            self._source = None
        return self._Mol
        
    @Mol.setter
    def Mol(self, value):
        self._Mol = value
    
    @property
    def atoms(self): return [Atom(rdkatom) for rdkatom in self.Mol.GetAtoms()]
    @property
    def data(self): return MoleculeData(self.Mol)
    @property
    def molwt(self): return Descriptors.MolWt(self.Mol)
    @property
    def formula(self): return Descriptors.MolecularFormula(self.Mol)
    def _gettitle(self):
        # Note to self: maybe should implement the get() method for self.data
        if "_Name" in self.data:
            return self.data["_Name"]
        else:
            return ""
    def _settitle(self, val): self.Mol.SetProp("_Name", val)
    title = property(_gettitle, _settitle)
    @property
    def _exchange(self):
        if self.Mol.GetNumConformers() == 0:
            return (0, self.write("smi"))
        else:
            return (1, self.write("mol"))
    
    # cache frequently used properties and cache them in prefixed [_] variables
    @property
    def coords(self):
        if self._coords is None:
            self._coords = np.array([atom.coords for atom in self.atoms])
        return self._coords
    
    @property
    def charges(self):
        if self._charges is None:
            self._charges = np.array([atom.partialcharge for atom in self.atoms])
        return self._charges
    
    #### Custom ODDT properties ####
    @property
    def sssr(self):
        return [list(path) for path in list(Chem.GetSymmSSSR(self.Mol))]
    @property
    def num_rotors(self):
        return NumRotatableBonds(self.Mol)
    
    @property
    def canonic_order(self):
        """ Returns np.array with canonic order of heavy atoms in the molecule """
        tmp = self.clone
        tmp.removeh()
        return np.array(CanonicalRankAtoms(tmp.Mol), dtype=int)
    
    @property
    def atom_dict(self):
        # check cache and generate dicts
        if self._atom_dict is None:
            self._dicts()
        return self._atom_dict
        
    @property
    def res_dict(self):
        # check cache and generate dicts
        if self._res_dict is None:
            self._dicts()
        return self._res_dict
        
    @property
    def ring_dict(self):
        # check cache and generate dicts
        if self._ring_dict is None:
            self._dicts()
        return self._ring_dict
    
    @property
    def clone(self):
        return Molecule(copy(self.Mol))
    
    def clone_coords(self, source):
        self.Mol.RemoveAllConformers()
        for conf in source.Mol.GetConformers():
            self.Mol.AddConformer(conf)
        return self
    
    def _dicts(self):
        # Atoms
        atom_dtype = [('id', 'int16'),
                 # atom info
                 ('coords', 'float16', 3),
                 ('charge', 'float16'),
                 ('atomicnum', 'int8'),
                 ('atomtype','a4'),
                 ('hybridization', 'int8'),
                 ('neighbors', 'float16', (4,3)), # non-H neighbors coordinates for angles (max of 6 neighbors should be enough)
                 # residue info
                 ('resid', 'int16'),
                 ('resname', 'a3'),
                 ('isbackbone', 'bool'),
                 # atom properties
                 ('isacceptor', 'bool'),
                 ('isdonor', 'bool'),
                 ('isdonorh', 'bool'),
                 ('ismetal', 'bool'),
                 ('ishydrophobe', 'bool'),
                 ('isaromatic', 'bool'),
                 ('isminus', 'bool'),
                 ('isplus', 'bool'),
                 ('ishalogen', 'bool'),
                 # secondary structure
                 ('isalpha', 'bool'),
                 ('isbeta', 'bool')
                 ]

        a = []
        atom_dict = np.empty(self.Mol.GetNumAtoms(), dtype=atom_dtype)
        metals = [3,4,11,12,13,19,20,21,22,23,24,25,26,27,28,29,30,31,37,38,39,40,41,42,43,44,45,46,47,48,49,50,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,87,88,89,90,91,
    92,93,94,95,96,97,98,99,100,101,102,103]
        for i, atom in enumerate(self.atoms):
            
            atomicnum = atom.atomicnum
            partialcharge = atom.partialcharge
            coords = atom.coords
            atomtype = atom.Atom.GetProp("_TriposAtomType") if atom.Atom.HasProp("_TriposAtomType") else atom.Atom.GetSymbol()
#            if self.protein:
#                residue = pybel.Residue(atom.OBAtom.GetResidue())
#            else:
#                residue = False
            residue = None
            
            # get neighbors, but only for those atoms which realy need them
            neighbors = np.empty(4, dtype=[('coords', 'float16', 3),('atomicnum', 'int8')])
            neighbors.fill(np.nan)
            for n, nbr_atom in enumerate(atom.neighbors):
                neighbors[n] = (nbr_atom.coords, nbr_atom.atomicnum)
            atom_dict[i] = (atom.idx,
                      coords,
                      partialcharge,
                      atomicnum,
                      atomtype,
                      np.clip(atom.Atom.GetHybridization()-1, 0, 3),
                      neighbors['coords'],
                      # residue info
                      residue.idx if residue else 0,
                      residue.name if residue else '',
                      False, #residue.OBResidue.GetAtomProperty(atom.OBAtom, 2) if residue else False, # is backbone
                      # atom properties
                      False, #atom.OBAtom.IsHbondAcceptor(),
                      False, #atom.OBAtom.IsHbondDonor(),
                      False, #atom.OBAtom.IsHbondDonorH(),
                      atomicnum in metals,
                      False, #atomicnum == 6 and not (np.in1d(neighbors['atomicnum'], [6,1])).any(), #hydrophobe #doble negation, since nan gives False
                      atom.Atom.GetIsAromatic(),
                      atomtype in ['O3-', '02-' 'O-'], # is charged (minus)
                      atomtype in ['N3+', 'N2+', 'Ng+'], # is charged (plus)
                      atomicnum in [9,17,35,53], # is halogen?
                      False, # alpha
                      False # beta
                      )
        
        # Match features and mark them in atom_dict
        feats = base_feature_factory.GetFeaturesForMol(self.Mol)
        translate_feats = {'Donor':'isdonor',
                   'Acceptor':'isacceptor',
                   'NegIonizable':'isminus',
                   'PosIonizable':'isplus',
                   'Aromatic':'isaromatic',
                   'Hydrophobe':'ishydrophobe'
                   }
        feat_atom_ids = {}
        for f in feats:
            if f.GetFamily() in translate_feats:
                if not translate_feats[f.GetFamily()] in feat_atom_ids:
                    feat_atom_ids[translate_feats[f.GetFamily()]] = tuple()
                feat_atom_ids[translate_feats[f.GetFamily()]] += f.GetAtomIds()
        for key, aids in feat_atom_ids.items():
            atom_dict[key][np.array(aids)] = True
        
        ### FIX: remove acidic carbons from isminus group (they are part of smarts)
        atom_dict['isminus'][atom_dict['isminus'] & (atom_dict['atomicnum'] == 6)] = False
        
#        if self.protein:
#            # Protein Residues (alpha helix and beta sheet)
#            res_dtype = [('id', 'int16'),
#                         ('resname', 'a3'),
#                         ('N', 'float16', 3),
#                         ('CA', 'float16', 3),
#                         ('C', 'float16', 3),
#                         ('isalpha', 'bool'),
#                         ('isbeta', 'bool')
#                         ] # N, CA, C

#            b = []
#            for residue in self.residues:
#                backbone = {}
#                for atom in residue:
#                    if residue.OBResidue.GetAtomProperty(atom.OBAtom,1):
#                        if atom.atomicnum == 7:
#                            backbone['N'] = atom.coords
#                        elif atom.atomicnum == 6:
#                            if atom.type == 'C3':
#                                backbone['CA'] = atom.coords
#                            else:
#                                backbone['C'] = atom.coords
#                if len(backbone.keys()) == 3:
#                    b.append((residue.idx, residue.name, backbone['N'],  backbone['CA'], backbone['C'], False, False))
#            res_dict = np.array(b, dtype=res_dtype)
#            
#            # detect secondary structure by phi and psi angles
#            first = res_dict[:-1]
#            second = res_dict[1:]
#            psi = dihedral(first['N'], first['CA'], first['C'], second['N'])
#            phi = dihedral(first['C'], second['N'], second['CA'], second['C'])
#            # mark atoms belonging to alpha and beta
#            res_mask_alpha = np.where(((phi > -145) & (phi < -35) & (psi > -70) & (psi < 50))) # alpha
#            res_dict['isalpha'][res_mask_alpha] = True
#            for i in res_dict[res_mask_alpha]['id']:
#                atom_dict['isalpha'][atom_dict['resid'] == i] = True

#            res_mask_beta = np.where(((phi >= -180) & (phi < -40) & (psi <= 180) & (psi > 90)) | ((phi >= -180) & (phi < -70) & (psi <= -165))) # beta
#            res_dict['isbeta'][res_mask_beta] = True
#            atom_dict['isbeta'][np.in1d(atom_dict['resid'], res_dict[res_mask_beta]['id'])] = True

        # Aromatic Rings
        r = []
        for path in self.sssr:
            if self.Mol.GetAtomWithIdx(path[0]).GetIsAromatic():
                atom = atom_dict[atom_dict['id'] == path[0]]
                coords = atom_dict[np.in1d(atom_dict['id'], path)]['coords']
                centroid = coords.mean(axis=0)
                # get vector perpendicular to ring
                vector = np.cross(coords - np.vstack((coords[1:],coords[:1])), np.vstack((coords[1:],coords[:1])) - np.vstack((coords[2:],coords[:2]))).mean(axis=0) - centroid
                r.append((centroid, vector, atom['isalpha'], atom['isbeta']))
        ring_dict = np.array(r, dtype=[('centroid', 'float16', 3),('vector', 'float16', 3),('isalpha', 'bool'),('isbeta', 'bool'),])
        
        self._atom_dict = atom_dict
        self._ring_dict = ring_dict
        if self.protein:
            self._res_dict = res_dict
    
    def addh(self):
        """Add hydrogens."""
        self.Mol = Chem.AddHs(self.Mol)

    def removeh(self):
        """Remove hydrogens."""
        self.Mol = Chem.RemoveHs(self.Mol)

    def write(self, format="smi", filename=None, overwrite=False, **kwargs):
        """Write the molecule to a file or return a string.

        Optional parameters:
           format -- see the informats variable for a list of available
                     output formats (default is "smi")
           filename -- default is None
           overwite -- if the output file already exists, should it
                       be overwritten? (default is False)

        If a filename is specified, the result is written to a file.
        Otherwise, a string is returned containing the result.

        To write multiple molecules to the same file you should use
        the Outputfile class.
        """
        format = format.lower()
        # Use lazy molecule if possible
        if self._source and 'fmt' in self._source and self._source['fmt'] == format and self._source['string']:
            return self._source['string']
        if filename:
            if not overwrite and os.path.isfile(filename):
                raise IOError, "%s already exists. Use 'overwrite=True' to overwrite it." % filename
        if format=="smi" or format=="can":
            result = Chem.MolToSmiles(self.Mol, **kwargs)
        elif format=="mol":
            result = Chem.MolToMolBlock(self.Mol, **kwargs)
        elif format=="mol2":
            result = Chem.MolToMol2Block(self.Mol, **kwargs)
        elif format=="pdb":
            result = Chem.MolToPDBBlock(self.Mol, **kwargs)
        elif format in ('inchi', 'inchikey') and Chem.INCHI_AVAILABLE:
            result = Chem.inchi.MolToInchi(self.Mol, **kwargs)
            if format == 'inchikey':
                result = Chem.inchi.InchiToInchiKey(result, **kwargs)
        else:
            raise ValueError,"%s is not a recognised RDKit format" % format
        if filename:
            print >> open(filename, "w"), result
        else:
            return result

    def __iter__(self):
        """Iterate over the Atoms of the Molecule.

        This allows constructions such as the following:
           for atom in mymol:
               print atom
        """
        return iter(self.atoms)

    def __str__(self):
        return self.write()

    def calcdesc(self, descnames=[]):
        """Calculate descriptor values.

        Optional parameter:
           descnames -- a list of names of descriptors

        If descnames is not specified, all available descriptors are
        calculated. See the descs variable for a list of available
        descriptors.
        """
        if not descnames:
            descnames = descs
        ans = {}
        for descname in descnames:
            try:
                desc = _descDict[descname]
            except KeyError:
                raise ValueError, "%s is not a recognised RDKit descriptor type" % descname
            ans[descname] = desc(self.Mol)
        return ans

    def calcfp(self, fptype="rdkit", opt=None):
        """Calculate a molecular fingerprint.

        Optional parameters:
           fptype -- the fingerprint type (default is "rdkit"). See the
                     fps variable for a list of of available fingerprint
                     types.
           opt -- a dictionary of options for fingerprints. Currently only used
                  for radius and bitInfo in Morgan fingerprints.
        """
        if opt == None:
            opt = {}
        fptype = fptype.lower()
        if fptype=="rdkit":
            fp = Fingerprint(Chem.RDKFingerprint(self.Mol))
        elif fptype=="layered":
            fp = Fingerprint(Chem.LayeredFingerprint(self.Mol))
        elif fptype=="maccs":
            fp = Fingerprint(Chem.MACCSkeys.GenMACCSKeys(self.Mol))
        elif fptype=="atompairs":
            # Going to leave as-is. See Atom Pairs documentation.
            fp = Chem.AtomPairs.Pairs.GetAtomPairFingerprintAsIntVect(self.Mol)
        elif fptype=="torsions":
            # Going to leave as-is.
            fp = Chem.AtomPairs.Torsions.GetTopologicalTorsionFingerprintAsIntVect(self.Mol)
        elif fptype == "morgan":
            info = opt.get('bitInfo', None)
            radius = opt.get('radius', 4)
            fp = Fingerprint(Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(self.Mol,radius,bitInfo=info))
        elif fptype == "pharm2d":
            fp = Fingerprint(Generate.Gen2DFingerprint(self.Mol,Gobbi_Pharm2D.factory))
        else:
            raise ValueError, "%s is not a recognised RDKit Fingerprint type" % fptype
        return fp

    def draw(self, show=True, filename=None, update=False, usecoords=False):
        """Create a 2D depiction of the molecule.

        Optional parameters:
          show -- display on screen (default is True)
          filename -- write to file (default is None)
          update -- update the coordinates of the atoms to those
                    determined by the structure diagram generator
                    (default is False)
          usecoords -- don't calculate 2D coordinates, just use
                       the current coordinates (default is False)

        Aggdraw or Cairo is used for 2D depiction. Tkinter and
        Python Imaging Library are required for image display.
        """
        if not usecoords and update:
            AllChem.Compute2DCoords(self.Mol)
            usecoords = True
        mol = Chem.Mol(self.Mol.ToBinary()) # Clone
        if not usecoords:
            AllChem.Compute2DCoords(mol)

        if filename: # Note: overwrite is allowed
            Draw.MolToFile(mol, filename)
        if show:
            if not tk:
                errormessage = ("Tkinter or Python Imaging "
                                "Library not found, but is required for image "
                                "display. See installation instructions for "
                                "more information.")
                raise ImportError(errormessage)
            img = Draw.MolToImage(mol)
            root = tk.Tk()
            root.title((hasattr(self, "title") and self.title)
                       or self.__str__().rstrip())
            frame = tk.Frame(root, colormap="new", visual='truecolor').pack()
            imagedata = PILtk.PhotoImage(img)
            label = tk.Label(frame, image=imagedata).pack()
            quitbutton = tk.Button(root, text="Close", command=root.destroy).pack(fill=tk.X)
            root.mainloop()

    def localopt(self, forcefield = "uff", steps = 500):
        """Locally optimize the coordinates.

        Optional parameters:
           forcefield -- default is "uff". See the forcefields variable
                         for a list of available forcefields.
           steps -- default is 500

        If the molecule does not have any coordinates, make3D() is
        called before the optimization.
        """
        forcefield = forcefield.lower()
        if self.Mol.GetNumConformers() == 0:
            self.make3D(forcefield)
        _forcefields[forcefield](self.Mol, maxIters = steps)

    def make3D(self, forcefield = "uff", steps = 50):
        """Generate 3D coordinates.

        Optional parameters:
           forcefield -- default is "uff". See the forcefields variable
                         for a list of available forcefields.
           steps -- default is 50

        Once coordinates are generated, a quick
        local optimization is carried out with 50 steps and the
        UFF forcefield. Call localopt() if you want
        to improve the coordinates further.
        """
        forcefield = forcefield.lower()
        success = AllChem.EmbedMolecule(self.Mol)
        if success == -1: # Failed
            success = AllChem.EmbedMolecule(self.Mol,
                                            useRandomCoords = True)
            if success == -1:
                raise Error, "Embedding failed!"
        self.localopt(forcefield, steps)

class Atom(object):
    """Represent an rdkit Atom.

    Required parameters:
       Atom -- an RDKit Atom

    Attributes:
        atomicnum, coords, formalcharge

    The original RDKit Atom can be accessed using the attribute:
       Atom
    """

    def __init__(self, Atom):
        self.Atom = Atom
    @property
    def atomicnum(self): return self.Atom.GetAtomicNum()
    @property
    def coords(self):
        owningmol = self.Atom.GetOwningMol()
        if owningmol.GetNumConformers() == 0:
            raise AttributeError, "Atom has no coordinates (0D structure)"
        idx = self.Atom.GetIdx()
        atomcoords = owningmol.GetConformer().GetAtomPosition(idx)
        return (atomcoords[0], atomcoords[1], atomcoords[2])
    @property
    def formalcharge(self): return self.Atom.GetFormalCharge()
    ### ODDT ###
    @property
    def idx(self):
        return self.Atom.GetIdx()
    @property
    def neighbors(self):
        return [Atom(a) for a in self.Atom.GetNeighbors()]
    @property
    def partialcharge(self):
        if self.Atom.HasProp('_TriposPartialCharge'):
            return float(self.Atom.GetProp('_TriposPartialCharge'))
        if not self.Atom.HasProp('_GasteigerCharge'):
            ComputeGasteigerCharges(self.Atom.GetOwningMol())
        return float(self.Atom.GetProp('_GasteigerCharge').replace(',','.'))
    
    def __str__(self):
        if hasattr(self, "coords"):
            return "Atom: %d (%.2f %.2f %.2f)" % (self.atomicnum, self.coords[0],
                                                    self.coords[1], self.coords[2])
        else:
            return "Atom: %d (no coords)" % (self.atomicnum)

class Smarts(object):
    """A Smarts Pattern Matcher

    Required parameters:
       smartspattern

    Methods:
       findall(molecule)

    Example:
    >>> mol = readstring("smi","CCN(CC)CC") # triethylamine
    >>> smarts = Smarts("[#6][#6]") # Matches an ethyl group
    >>> print smarts.findall(mol)
    [(0, 1), (3, 4), (5, 6)]

    The numbers returned are the indices (starting from 0) of the atoms
    that match the SMARTS pattern. In this case, there are three matches
    for each of the three ethyl groups in the molecule.
    """
    def __init__(self,smartspattern):
        """Initialise with a SMARTS pattern."""
        self.rdksmarts = Chem.MolFromSmarts(smartspattern)
        if not self.rdksmarts:
            raise IOError, "Invalid SMARTS pattern."

    def findall(self,molecule):
        """Find all matches of the SMARTS pattern to a particular molecule.

        Required parameters:
           molecule
        """
        return molecule.Mol.GetSubstructMatches(self.rdksmarts)

class MoleculeData(object):
    """Store molecule data in a dictionary-type object

    Required parameters:
      Mol -- an RDKit Mol

    Methods and accessor methods are like those of a dictionary except
    that the data is retrieved on-the-fly from the underlying Mol.

    Example:
    >>> mol = readfile("sdf", 'head.sdf').next()
    >>> data = mol.data
    >>> print data
    {'Comment': 'CORINA 2.61 0041  25.10.2001', 'NSC': '1'}
    >>> print len(data), data.keys(), data.has_key("NSC")
    2 ['Comment', 'NSC'] True
    >>> print data['Comment']
    CORINA 2.61 0041  25.10.2001
    >>> data['Comment'] = 'This is a new comment'
    >>> for k,v in data.iteritems():
    ...    print k, "-->", v
    Comment --> This is a new comment
    NSC --> 1
    >>> del data['NSC']
    >>> print len(data), data.keys(), data.has_key("NSC")
    1 ['Comment'] False
    """
    def __init__(self, Mol):
        self._mol = Mol
    def _testforkey(self, key):
        if not key in self:
            raise KeyError, "'%s'" % key
    def keys(self):
        return self._mol.GetPropNames()
    def values(self):
        return [self._mol.GetProp(x) for x in self.keys()]
    def items(self):
        return zip(self.keys(), self.values())
    def __iter__(self):
        return iter(self.keys())
    def iteritems(self):
        return iter(self.items())
    def __len__(self):
        return len(self.keys())
    def __contains__(self, key):
        return self._mol.HasProp(key)
    def __delitem__(self, key):
        self._testforkey(key)
        self._mol.ClearProp(key)
    def clear(self):
        for key in self:
            del self[key]
    def has_key(self, key):
        return key in self
    def update(self, dictionary):
        for k, v in dictionary.iteritems():
            self[k] = v
    def __getitem__(self, key):
        self._testforkey(key)
        return self._mol.GetProp(key)
    def __setitem__(self, key, value):
        self._mol.SetProp(key, str(value))
    def __repr__(self):
        return dict(self.iteritems()).__repr__()

class Fingerprint(object):
    """A Molecular Fingerprint.

    Required parameters:
       fingerprint -- a vector calculated by one of the fingerprint methods

    Attributes:
       fp -- the underlying fingerprint object
       bits -- a list of bits set in the Fingerprint

    Methods:
       The "|" operator can be used to calculate the Tanimoto coeff. For example,
       given two Fingerprints 'a', and 'b', the Tanimoto coefficient is given by:
          tanimoto = a | b
    """
    def __init__(self, fingerprint):
        self.fp = fingerprint
    def __or__(self, other):
        return rdkit.DataStructs.FingerprintSimilarity(self.fp, other.fp)
    def __getattr__(self, attr):
        if attr == "bits":
            # Create a bits attribute on-the-fly
            return list(self.fp.GetOnBits())
        else:
            raise AttributeError, "Fingerprint has no attribute %s" % attr
    def __str__(self):
        return ", ".join([str(x) for x in _compressbits(self.fp)])
    @property
    def raw(self):
        return np.array(self.fp)

def _compressbits(bitvector, wordsize=32):
    """Compress binary vector into vector of long ints.

    This function is used by the Fingerprint class.

    >>> _compressbits([0, 1, 0, 0, 0, 1], 2)
    [2, 0, 2]
    """
    ans = []
    for start in range(0, len(bitvector), wordsize):
        compressed = 0
        for i in range(wordsize):
            if i + start < len(bitvector) and bitvector[i + start]:
                compressed += 2**i
        ans.append(compressed)

    return ans


if __name__=="__main__": #pragma: no cover
    import doctest
    doctest.testmod()

