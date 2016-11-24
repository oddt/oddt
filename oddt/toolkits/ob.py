from __future__ import print_function
# All functions using f2py need to be loaded before pybel/openbabel,
# otherwise it will segfault.
# See BUG report: https://github.com/numpy/numpy/issues/1746
from scipy.optimize import fmin_l_bfgs_b

from itertools import chain

import gzip
from base64 import b64encode
import six
import pybel
from pybel import *
import numpy as np
import openbabel as ob
from openbabel import OBAtomAtomIter, OBTypeTable

import oddt
from oddt.spatial import angle, angle_2v, dihedral

backend = 'ob'
# setup typetable to translate atom types
typetable = OBTypeTable()
typetable.SetFromType('INT')
typetable.SetToType('SYB')

# setup ElementTable
elementtable = ob.OBElementTable()

# hash OB!
pybel.ob.obErrorLog.StopLogging()


def _filereader_mol2(filename, opt=None):
    block = ''
    data = ''
    n = 0
    with gzip.open(filename) if filename.split('.')[-1] == 'gz' else open(filename) as f:
        for line in f:
            if line[:1] == '#':
                data += line
            elif line[:17] == '@<TRIPOS>MOLECULE':
                if n > 0:  # skip `zero` molecule (any preciding comments and spaces)
                    yield Molecule(source={'fmt': 'mol2', 'string': block, 'opt': opt})
                n += 1
                block = data
                data = ''
            block += line
        # open last molecule
        if block:
            yield Molecule(source={'fmt': 'mol2', 'string': block, 'opt': opt})


def _filereader_sdf(filename, opt=None):
    block = ''
    n = 0
    with gzip.open(filename) if filename.split('.')[-1] == 'gz' else open(filename) as f:
        for line in f:
            block += line
            if line[:4] == '$$$$':
                yield Molecule(source={'fmt': 'sdf', 'string': block, 'opt': opt})
                n += 1
                block = ''
        if block:  # open last molecule if any
            yield Molecule(source={'fmt': 'sdf', 'string': block, 'opt': opt})


def _filereader_pdb(filename, opt=None):
    block = ''
    n = 0
    with gzip.open(filename) if filename.split('.')[-1] == 'gz' else open(filename) as f:
        for line in f:
            block += line
            if line[:4] == 'ENDMDL':
                yield Molecule(source={'fmt': 'pdb', 'string': block, 'opt': opt})
                n += 1
                block = ''
        if block:  # open last molecule if any
            yield Molecule(source={'fmt': 'pdb', 'string': block, 'opt': opt})


def readfile(format, filename, opt=None, lazy=False):
    if format == 'mol2':
        if opt:
            opt['c'] = None
        else:
            opt = {'c': None}
    if lazy and format == 'mol2':
        return _filereader_mol2(filename, opt=opt)
    elif lazy and format == 'sdf':
        return _filereader_sdf(filename, opt=opt)
    elif lazy and format == 'pdb':
        return _filereader_pdb(filename, opt=opt)
    else:
        return pybel.readfile(format, filename, opt=opt)


class Molecule(pybel.Molecule):
    def __init__(self, OBMol=None, source=None, protein=False):
        # lazy
        self._source = source  # dict with keys: n, fmt, string, filename

        # call parent constructor
        super(Molecule, self).__init__(OBMol)

        self.protein = protein

        # ob.DeterminePeptideBackbone(molecule.OBMol)
        # percieve chains in residues
        # if len(res_dict) > 1 and not molecule.OBMol.HasChainsPerceived():
        #    print("Dirty HACK")
        #    molecule = pybel.readstring('pdb', molecule.write('pdb'))
        self._atom_dict = None
        self._res_dict = None
        self._ring_dict = None
        self._coords = None
        self._charges = None

    # lazy Molecule parsing requires masked OBMol
    @property
    def OBMol(self):
        if not self._OBMol and self._source:
            self._OBMol = readstring(self._source['fmt'], self._source['string'], opt=self._source['opt'] if 'opt' in self._source else {}).OBMol
            self._source = None
        return self._OBMol

    @OBMol.setter
    def OBMol(self, value):
        self._OBMol = value

    @property
    def atoms(self):
        return AtomStack(self.OBMol)

    @property
    def bonds(self):
        return BondStack(self.OBMol)

    # cache frequently used properties and cache them in prefixed [_] variables
    @property
    def coords(self):
        if self._coords is None:
            self._coords = np.array([atom.coords for atom in self.atoms], dtype=np.float32)
            self._coords.setflags(write=False)
        return self._coords

    @coords.setter
    def coords(self, new):
        new = np.asarray(new, dtype=np.float64)
        [a.OBAtom.SetVector(v[0], v[1], v[2]) for v, a in zip(new, self.atoms)]
        # clear cache
        self._coords = None
        self._atom_dict = None

    @property
    def charges(self):
        if self._charges is None:
            self._charges = np.array([atom.partialcharge for atom in self.atoms])
        return self._charges

    @property
    def smiles(self):
        return self.write('smi').split()[0]

    def write(self, format="smi", filename=None, overwrite=False, opt=None, size=None):
        format = format.lower()
        if format == 'png':
            size = size or (200, 200)
            format = '_png2'
            opt = opt or {}
            opt['w'] = size[0]
            opt['h'] = size[1]
        # Use lazy molecule if possible
        if self._source and 'fmt' in self._source and self._source['fmt'] == format and self._source['string']:
            return self._source['string']
        else:
            return super(Molecule, self).write(format=format, filename=filename, overwrite=overwrite, opt=opt)

    # Backport code implementing resudues (by me) to support older versions of OB (aka 'stable')
    @property
    def residues(self):
        return [Residue(res) for res in ob.OBResidueIter(self.OBMol)]

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        if oddt.ipython_notebook:
            if oddt.pandas.image_backend == 'png':
                return self._repr_png_()
            else:
                return self._repr_svg_()
        else:
            return super(Molecule, self).__repr__()

    # Custom ODDT properties #
    def __getattr__(self, attr):
        for desc in pybel._descdict.keys():
            if attr.lower() == desc.lower():
                return self.calcdesc([desc])[desc]
        raise AttributeError('Molecule has no such property: %s' % attr)

    @property
    def num_rotors(self):
        return self.OBMol.NumRotors()

    def _repr_svg_(self):
        return self.clone.write('svg', opt={'d': None}).replace('\n', '')

    def _repr_png_(self, size=200):
        string = self.clone.write('_png2', opt={'p': size,
                                                'd': None,
                                                't': None}
                                  )
        if six.PY3:  # bug in SWIG decoding
            string = string.encode('utf-8', errors='surrogateescape')
        return '<img src="data:image/png;base64,%s" alt="%s">' % (
            b64encode(string).decode('ascii'),
            self.title
        )

    @property
    def canonic_order(self):
        """ Returns np.array with canonic order of heavy atoms in the molecule """
        tmp = self.clone
        tmp.write('can')
        return np.array(tmp.data['SMILES Atom Order'].split(), dtype=int) - 1

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
        return Molecule(ob.OBMol(self.OBMol))

    def clone_coords(self, source):
        self.OBMol.SetCoordinates(source.OBMol.GetCoordinates())
        return self

    def _dicts(self):
        # Atoms
        atom_dtype = [('id', 'int16'),
                      # atom info
                      ('coords', 'float32', 3),
                      ('radius', 'float32'),
                      ('charge', 'float32'),
                      ('atomicnum', 'int8'),
                      ('atomtype', 'a4'),
                      ('hybridization', 'int8'),
                      ('neighbors', 'float32', (4, 3)),  # max of 4 neighbors should be enough
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
        atom_dict = np.empty(self.OBMol.NumAtoms(), dtype=atom_dtype)
        metals = [3, 4, 11, 12, 13, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
                  30, 31, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
                  50, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68,
                  69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83,
                  87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101,
                  102, 103]
        for i, atom in enumerate(self.atoms):

            atomicnum = atom.atomicnum
            # skip non-polar hydrogens for performance
#            if atomicnum == 1 and atom.OBAtom.IsNonPolarHydrogen():
#                continue
            atomtype = typetable.Translate(atom.type)  # sybyl atom type
            partialcharge = atom.partialcharge
            coords = atom.coords

            if self.protein:
                residue = Residue(atom.OBAtom.GetResidue())
            else:
                residue = False

            # get neighbors, but only for those atoms which realy need them
            neighbors = np.zeros(4, dtype=[('coords', 'float32', 3), ('atomicnum', 'int8')])
            neighbors['coords'].fill(np.nan)
            for n, nbr_atom in enumerate(atom.neighbors):
                # concider raising neighbors list to 6, but must do some benchmarks
                if n > 3:
                    break
                nbr_atomicnum = nbr_atom.atomicnum
                neighbors[n] = (nbr_atom.coords, nbr_atomicnum)
            atom_dict[i] = (atom.idx,
                            coords,
                            elementtable.GetVdwRad(atomicnum),
                            partialcharge,
                            atomicnum,
                            atomtype,
                            atom.OBAtom.GetHyb(),
                            neighbors['coords'],  # n_coords,
                            # residue info
                            residue.idx if residue else 0,
                            residue.name if residue else '',
                            residue.OBResidue.GetAtomProperty(atom.OBAtom, 2) if residue else False,  # is backbone
                            # atom properties
                            atom.OBAtom.IsHbondAcceptor(),
                            atom.OBAtom.IsHbondDonor(),
                            atom.OBAtom.IsHbondDonorH(),
                            atomicnum in metals,
                            atomicnum == 6 and np.in1d(neighbors['atomicnum'], [6, 1, 0]).all(),  # hydrophobe
                            atom.OBAtom.IsAromatic(),
                            atomtype in ['O3-', '02-' 'O-'] or atom.formalcharge < 0,  # is charged (minus)
                            atomtype in ['N3+', 'N2+', 'Ng+'] or atom.formalcharge > 0,  # is charged (plus)
                            atomicnum in [9, 17, 35, 53],  # is halogen?
                            False,  # alpha
                            False  # beta
                            )

        if self.protein:
            # Protein Residues (alpha helix and beta sheet)
            res_dtype = [('id', 'int16'),
                         ('resname', 'a3'),
                         ('N', 'float32', 3),
                         ('CA', 'float32', 3),
                         ('C', 'float32', 3),
                         ('isalpha', 'bool'),
                         ('isbeta', 'bool')
                         ]  # N, CA, C

            b = []
            for residue in self.residues:
                backbone = {}
                for atom in residue:
                    if residue.OBResidue.GetAtomProperty(atom.OBAtom, 1):
                        if atom.atomicnum == 7:
                            backbone['N'] = atom.coords
                        elif atom.atomicnum == 6:
                            if atom.type == 'C3':
                                backbone['CA'] = atom.coords
                            else:
                                backbone['C'] = atom.coords
                if len(backbone.keys()) == 3:
                    b.append((residue.idx, residue.name, backbone['N'], backbone['CA'], backbone['C'], False, False))
            res_dict = np.array(b, dtype=res_dtype)

            # detect secondary structure by phi and psi angles
            first = res_dict[:-1]
            second = res_dict[1:]
            psi = dihedral(first['N'], first['CA'], first['C'], second['N'])
            phi = dihedral(first['C'], second['N'], second['CA'], second['C'])
            # mark atoms belonging to alpha and beta
            res_mask_alpha = np.where(((phi > -145) & (phi < -35) & (psi > -70) & (psi < 50)))  # alpha
            res_dict['isalpha'][res_mask_alpha] = True
            for i in res_dict[res_mask_alpha]['id']:
                atom_dict['isalpha'][atom_dict['resid'] == i] = True

            res_mask_beta = np.where(((phi >= -180) & (phi < -40) & (psi <= 180) & (psi > 90)) | ((phi >= -180) & (phi < -70) & (psi <= -165)))  # beta
            res_dict['isbeta'][res_mask_beta] = True
            atom_dict['isbeta'][np.in1d(atom_dict['resid'], res_dict[res_mask_beta]['id'])] = True

        # Aromatic Rings
        r = []
        for ring in self.sssr:
            if ring.IsAromatic():
                path = ring._path
                atoms = atom_dict[np.in1d(atom_dict['id'], path)]
                if len(atoms):
                    atom = atoms[0]
                    coords = atoms['coords']
                    centroid = coords.mean(axis=0)
                    # get vector perpendicular to ring
                    vector = np.cross(coords - np.vstack((coords[1:], coords[:1])),
                                      np.vstack((coords[1:], coords[:1])) - np.vstack((coords[2:], coords[:2]))
                                      ).mean(axis=0) - centroid
                    r.append((centroid, vector, atom['isalpha'], atom['isbeta']))
        ring_dict = np.array(r, dtype=[('centroid', 'float32', 3),
                                       ('vector', 'float32', 3),
                                       ('isalpha', 'bool'),
                                       ('isbeta', 'bool')])

        self._atom_dict = atom_dict
        self._atom_dict.setflags(write=False)
        self._ring_dict = ring_dict
        self._ring_dict.setflags(write=False)
        if self.protein:
            self._res_dict = res_dict
            self._res_dict.setflags(write=False)

    def __getstate__(self):
        pickle_format = 'mol2'
        return {'fmt': self._source['fmt'] if self._source else pickle_format,
                'string': self._source['string'] if self._source else self.write(pickle_format),
                'data': dict(self.data.items()),
                'dicts': {'atom_dict': self._atom_dict,
                          'ring_dict': self._ring_dict,
                          'res_dict': self._res_dict,
                          }
                }

    def __setstate__(self, state):
        Molecule.__init__(self, source=state)
        self.data.update(state['data'])
        self._atom_dict = state['dicts']['atom_dict']
        self._ring_dict = state['dicts']['ring_dict']
        self._res_dict = state['dicts']['res_dict']

# Extend pybel.Molecule
pybel.Molecule = Molecule


class AtomStack(object):
    def __init__(self, OBMol):
        self.OBMol = OBMol

    def __iter__(self):
        for i in range(self.OBMol.NumAtoms()):
            yield Atom(self.OBMol.GetAtom(i + 1))

    def __len__(self):
        return self.OBMol.NumAtoms()

    def __getitem__(self, i):
        if 0 <= i < self.OBMol.NumAtoms():
            return Atom(self.OBMol.GetAtom(int(i + 1)))
        else:
            raise AttributeError("There is no atom with Idx %i" % i)


class Atom(pybel.Atom):
    @property
    def neighbors(self):
        return [Atom(a) for a in OBAtomAtomIter(self.OBAtom)]

    @property
    def residue(self):
        return Residue(self.OBAtom.GetResidue())

    @property
    def bonds(self):
        return [Bond(self.OBAtom.GetBond(n.OBAtom)) for n in self.neighbors]

pybel.Atom = Atom


class BondStack(object):
    def __init__(self, OBMol):
        self.OBMol = OBMol

    def __iter__(self):
        for i in range(self.OBMol.NumBonds()):
            yield Bond(self.OBMol.GetBond(i))

    def __len__(self):
        return self.OBMol.NumBonds()

    def __getitem__(self, i):
        if 0 <= i < self.OBMol.NumBonds():
            return Bond(self.OBMol.GetBond(i))
        else:
            raise AttributeError("There is no bond with Idx %i" % i)


class Bond(object):
    def __init__(self, OBBond):
        self.OBBond = OBBond

    @property
    def order(self):
        return self.OBBond.GetBondOrder()

    @property
    def atoms(self):
        return (Atom(self.OBBond.GetBeginAtom()), Atom(self.OBBond.GetEndAtom()))

    @property
    def isrotor(self):
        return self.OBBond.IsRotor()


class Residue(object):
    """Represent a Pybel residue.

    Required parameter:
       OBResidue -- an Open Babel OBResidue

    Attributes:
       atoms, idx, name.

    (refer to the Open Babel library documentation for more info).

    The original Open Babel atom can be accessed using the attribute:
       OBResidue
    """

    def __init__(self, OBResidue):
        self.OBResidue = OBResidue

    @property
    def atoms(self):
        return [Atom(atom) for atom in ob.OBResidueAtomIter(self.OBResidue)]

    @property
    def idx(self):
        return self.OBResidue.GetIdx()

    @property
    def name(self):
        return self.OBResidue.GetName()

    def __iter__(self):
        """Iterate over the Atoms of the Residue.

        This allows constructions such as the following:
           for atom in residue:
               print(atom)
        """
        return iter(self.atoms)


class MoleculeData(pybel.MoleculeData):
    def _data(self):
        blacklist_keys = ['OpenBabel Symmetry Classes', 'MOL Chiral Flag', 'PartialCharges']
        data = chain(self._mol.GetAllData(pybel._obconsts.PairData),
                     self._mol.GetAllData(pybel._obconsts.CommentData))
        return [x for x in data if x.GetAttribute() not in blacklist_keys]

    def to_dict(self):
        return dict((x.GetAttribute(), x.GetValue()) for x in self._data())

pybel.MoleculeData = MoleculeData


class Outputfile(pybel.Outputfile):
    def __init__(self, format, filename, overwrite=False, opt=None):
        if format == 'mol2':
            if opt:
                opt['c'] = None
            else:
                opt = {'c': None}
        return super(Outputfile, self).__init__(format, filename, overwrite=overwrite, opt=opt)


class Fingerprint(pybel.Fingerprint):
    @property
    def raw(self):
        return _unrollbits(self.fp, pybel.ob.OBFingerprint.Getbitsperint())


def _unrollbits(fp, bitsperint):
    """ Unroll unsigned int fingerprint to bool """
    ans = np.zeros(len(fp) * bitsperint)
    start = 1
    for x in fp:
        i = start
        while x > 0:
            ans[i] = x % 2
            x >>= 1
            i += 1
        start += bitsperint
    return ans

pybel.Fingerprint = Fingerprint


class Smarts(pybel.Smarts):
    def match(self, molecule):
        """ Checks if there is any match. Returns True or False """
        return self.obsmarts.HasMatch(molecule.OBMol)
