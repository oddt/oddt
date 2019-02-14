from __future__ import print_function

# All functions using f2py need to be loaded before pybel/openbabel,
# otherwise it will segfault.
# See BUG report: https://github.com/numpy/numpy/issues/1746
from scipy.optimize import fmin_l_bfgs_b

from itertools import chain
from subprocess import check_output
import warnings
from tempfile import NamedTemporaryFile

import gzip
from base64 import b64encode
from six import PY3, text_type
from sklearn.utils.deprecation import deprecated
import pybel
import openbabel as ob
from pybel import *
import numpy as np
from openbabel import OBAtomAtomIter, OBAtomBondIter, OBTypeTable

from oddt.utils import check_molecule
from oddt.toolkits.common import detect_secondary_structure, canonize_ring_path

ob.OBIterWithDepth.__next__ = ob.OBIterWithDepth.next

backend = 'ob'
image_backend = 'png'  # png or svg
image_size = (200, 200)

try:
    if get_ipython().config:
        ipython_notebook = True
    else:
        ipython_notebook = False
except NameError:
    ipython_notebook = False

try:
    __version__ = check_output(['obabel', '-V']).split()[2].decode('ascii')
except Exception as e:
    __version__ = ''
# setup typetable to translate atom types
typetable = OBTypeTable()
typetable.SetFromType('INT')
typetable.SetToType('SYB')

# setup ElementTable
elementtable = ob.OBElementTable()

# hash OB!
ob.obErrorLog.StopLogging()


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
            if line[:6] == 'ENDMDL':
                yield Molecule(source={'fmt': 'pdb', 'string': block, 'opt': opt})
                n += 1
                block = ''
        if block:  # open last molecule if any
            yield Molecule(source={'fmt': 'pdb', 'string': block, 'opt': opt})


def readfile(format, filename, opt=None, lazy=False):
    if format == 'mol2':
        if __version__ < '2.4.0':
            warnings.warn('OpenBabel 2.3.2 does not support writing data in '
                          'comments ["-xc"]. Please upgrade to OB 2.4')
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

        self._protein = protein

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
            self._OBMol = readstring(self._source['fmt'],
                                     self._source['string'],
                                     opt=self._source['opt'] if 'opt' in self._source else {}).OBMol
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
        size = size or (200, 200)
        if format == 'png':
            format = '_png2'
            opt = opt or {}
            opt['w'] = size[0]
            opt['h'] = size[1]
        # Use lazy molecule if possible
        if self._source and 'fmt' in self._source and self._source['fmt'] == format and self._source['string']:
            return self._source['string']
        # Workaround OB 2.3.2 + Py3 PNG encoding error
        elif format == '_png2' and filename is None and PY3 and __version__ < '2.4.0':
            with NamedTemporaryFile(suffix='.png') as f:
                super(Molecule, self).write(format=format, filename=f.name, overwrite=True, opt=opt)
                output = f.read()
            return output
        else:
            return super(Molecule, self).write(format=format, filename=filename, overwrite=overwrite, opt=opt)

    # Backport code implementing resudues (by me) to support older versions of OB (aka 'stable')
    @property
    def residues(self):
        return ResidueStack(self.OBMol)

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        if ipython_notebook:
            return self._repr_html_()
        else:
            return super(Molecule, self).__repr__()

    @property
    def protein(self):
        """
        A flag for identifing the protein molecules, for which `atom_dict`
        procedures may differ.
        """
        return self._protein

    @protein.setter
    def protein(self, protein):
        """atom_dict caches must be cleared due to property change"""
        self._clear_cache()
        self._protein = protein

    def addh(self, only_polar=False):
        """Add hydrogens"""
        if only_polar:
            self.OBMol.AddPolarHydrogens()
        else:
            self.OBMol.AddHydrogens()
        self._clear_cache()

    def removeh(self):
        """Remove hydrogens"""
        super(Molecule, self).removeh()
        self._clear_cache()

    def make3D(self, forcefield="mmff94", steps=50):
        """Generate 3D coordinates"""
        super(Molecule, self).make3D(forcefield=forcefield, steps=steps)
        self._clear_cache()

    def make2D(self):
        """Generate 2D coordinates for molecule"""
        pybel._operations['gen2D'].Do(self.OBMol)
        self._clear_cache()

    def calccharges(self, model='gasteiger'):
        """Calculate partial charges for a molecule. By default the Gasteiger
        charge model is used.

        Parameters
        ----------
        model : str (default="gasteiger")
            Method for generating partial charges. Supported models:
            * gasteiger
            * mmff94
            * others supported by OpenBabel (`obabel -L charges`)
        """
        if __version__ < '2.4.0':  # TODO: Get rid of this block for new OB
            if model in pybel._getpluginnames('charges'):
                m = pybel._getplugins(ob.OBChargeModel.FindType, [model])[model]
                if not m.ComputeCharges(self.OBMol):
                    raise Exception('Could not assigh partial charges for '
                                    'molecule "%s"' % self.title)
            else:
                raise ValueError('Model "%s" is not supported in OpenBabel' %
                                 model)
        else:
            super(Molecule, self).calccharges(model)
        self._clear_cache()

    # Custom ODDT properties #
    def __getattr__(self, attr):
        for desc in pybel._descdict.keys():
            if attr.lower() == desc.lower():
                return self.calcdesc([desc])[desc]
        raise AttributeError('Molecule has no such property: %s' % attr)

    def _clear_cache(self):
        """Clear all ODDT caches and dicts"""
        self._atom_dict = None
        self._res_dict = None
        self._ring_dict = None
        self._coords = None
        self._charges = None
        self._residues = None

    @property
    def num_rotors(self):
        """Number of strict rotatable """
        rot_bond = Smarts('[!$(*#*)&!D1&!$(C(F)(F)F)&'
                          '!$(C(Cl)(Cl)Cl)&'
                          '!$(C(Br)(Br)Br)&'
                          '!$(C([CH3])([CH3])[CH3])&'
                          '!$([CD3](=[N,O,S])-!@[#7,O,S!D1])&'
                          '!$([#7,O,S!D1]-!@[CD3]=[N,O,S])&'
                          '!$([CD3](=[N+])-!@[#7!D1])&'
                          '!$([#7!D1]-!@[CD3]=[N+])]-!@[!$(*#*)&'
                          '!D1&!$(C(F)(F)F)&'
                          '!$(C(Cl)(Cl)Cl)&'
                          '!$(C(Br)(Br)Br)&'
                          '!$(C([CH3])([CH3])[CH3])]')
        return len(rot_bond.findall(self))

    def _repr_svg_(self):
        if isinstance(image_size, int):
            size = (image_size, image_size)
        elif isinstance(image_size, (tuple, list)) and len(image_size) == 2:
            size = tuple(image_size)
        else:
            raise ValueError('oddt.toolkit.image_size has bad value - '
                             'it should be int or list/tuple of two ints. '
                             'Got: %s ' % image_size)
        if image_backend == 'svg':
            return self.clone.write('svg',
                                    opt={'d': None,
                                         't': None},
                                    size=size).replace('\n', '')
        else:
            return None

    def _repr_png_(self):
        if isinstance(image_size, int):
            size = (image_size, image_size)
        elif isinstance(image_size, (tuple, list)) and len(image_size) == 2:
            size = tuple(image_size)
        else:
            raise ValueError('oddt.toolkit.image_size has bad value - '
                             'it should be int or list/tuple of two ints. '
                             'Got: %s ' % image_size)
        if image_backend == 'png':
            string = self.clone.write('png',
                                      opt={'d': None,
                                           't': None},
                                      size=size)
            if PY3 and isinstance(string, text_type):
                string = string.encode('utf-8', errors='surrogateescape')
            return string
        else:
            return None

    def _repr_html_(self):
        if image_backend == 'png':
            return '<img src="data:image/png;base64,%s" alt="%s">' % (
                b64encode(self._repr_png_()).decode('ascii'),
                self.title)
        elif image_backend == 'svg':
            return self._repr_svg_()
        else:
            return None

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
        max_neighbors = 6  # max of 6 neighbors should be enough
        # Atoms
        atom_dtype = [('id', np.uint32),
                      # atom info
                      ('coords', np.float32, 3),
                      ('radius', np.float32),
                      ('charge', np.float32),
                      ('atomicnum', np.int8),
                      ('atomtype', 'U5' if PY3 else 'a5'),
                      ('hybridization', np.int8),
                      ('neighbors_id', np.int16, max_neighbors),
                      ('neighbors', np.float32, (max_neighbors, 3)),
                      # residue info
                      ('resid', np.int16),
                      ('resnum', np.int16),
                      ('resname', 'U3' if PY3 else 'a3'),
                      ('isbackbone', bool),
                      # atom properties
                      ('isacceptor', bool),
                      ('isdonor', bool),
                      ('isdonorh', bool),
                      ('ismetal', bool),
                      ('ishydrophobe', bool),
                      ('isaromatic', bool),
                      ('isminus', bool),
                      ('isplus', bool),
                      ('ishalogen', bool),
                      # secondary structure
                      ('isalpha', bool),
                      ('isbeta', bool)
                      ]

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
            neighbors = np.zeros(max_neighbors, dtype=[('id', np.int16),
                                                       ('coords', np.float32, 3),
                                                       ('atomicnum', np.int8)])
            neighbors['coords'].fill(np.nan)
            for n, nbr_atom in enumerate(atom.neighbors):
                if n >= max_neighbors:
                    warnings.warn('Error while parsing molecule "%s" '
                                  'for `atom_dict`. Atom #%i (%s) has %i '
                                  'neighbors (max_neighbors=%i). Additional '
                                  'neighbors are ignored.' % (self.title,
                                                              atom.idx0,
                                                              atomtype,
                                                              len(atom.neighbors),
                                                              max_neighbors),
                                  UserWarning)
                    break
                neighbors[n] = (nbr_atom.idx0, nbr_atom.coords, nbr_atom.atomicnum)
            assert i == atom.idx0
            atom_dict[i] = (i,
                            coords,
                            elementtable.GetVdwRad(atomicnum),
                            partialcharge,
                            atomicnum,
                            atomtype,
                            atom.OBAtom.GetHyb(),
                            neighbors['id'],
                            neighbors['coords'],
                            # residue info
                            residue.idx0 if residue else 0,
                            residue.number if residue else 0,
                            residue.name if residue else '',
                            residue.OBResidue.GetAtomProperty(atom.OBAtom, 2) if residue else False,  # is backbone
                            # atom properties
                            False,  # atom.OBAtom.IsHbondAcceptor(),
                            False,  # atom.OBAtom.IsHbondDonor(),
                            False,  # atom.OBAtom.IsHbondDonorH(),
                            atomicnum in metals,
                            atomicnum == 6 and np.in1d(neighbors['atomicnum'], [6, 1, 0]).all(),  # hydrophobe
                            atom.OBAtom.IsAromatic(),
                            atom.formalcharge < 0,  # is charged (minus)
                            atom.formalcharge > 0,  # is charged (plus)
                            atomicnum in [9, 17, 35, 53],  # is halogen?
                            False,  # alpha
                            False  # beta
                            )

        not_carbon = np.argwhere(~np.in1d(atom_dict['atomicnum'], [1, 6])).flatten()
        # Acceptors
        patt = Smarts('[$([O;H1;v2]),'
                      '$([O;H0;v2;!$(O=N-*),'
                      '$([O;-;!$(*-N=O)]),'
                      '$([o;+0])]),'
                      '$([n;+0;!X3;!$([n;H1](cc)cc),'
                      '$([$([N;H0]#[C&v4])]),'
                      '$([N&v3;H0;$(Nc)])]),'
                      '$([F;$(F-[#6]);!$(FC[F,Cl,Br,I])])]')
        matches = np.array(patt.findall(self)).flatten()
        if len(matches) > 0:
            atom_dict['isacceptor'][np.intersect1d(matches - 1, not_carbon)] = True

        # Donors
        patt = Smarts('[$([N&!H0&v3,N&!H0&+1&v4,n&H1&+0,$([$([Nv3](-C)(-C)-C)]),'
                      '$([$(n[n;H1]),'
                      '$(nc[n;H1])])]),'
                      # Guanidine can be tautormeic - e.g. Arginine
                      '$([NX3,NX2]([!O,!S])!@C(!@[NX3,NX2]([!O,!S]))!@[NX3,NX2]([!O,!S])),'
                      '$([O,S;H1;+0])]')
        matches = np.array(patt.findall(self)).flatten()
        if len(matches) > 0:
            atom_dict['isdonor'][np.intersect1d(matches - 1, not_carbon)] = True
            atom_dict['isdonorh'][[n.idx0
                                   for idx in np.argwhere(atom_dict['isdonor']).flatten()
                                   for n in self.atoms[int(idx)].neighbors
                                   if n.atomicnum == 1]] = True

        # Basic group
        patt = Smarts('[$([N;H2&+0][$([C,a]);!$([C,a](=O))]),'
                      '$([N;H1&+0]([$([C,a]);!$([C,a](=O))])[$([C,a]);!$([C,a](=O))]),'
                      '$([N;H0&+0]([C;!$(C(=O))])([C;!$(C(=O))])[C;!$(C(=O))]),'
                      '$([N,n;X2;+0])]')
        matches = np.array(patt.findall(self)).flatten()
        if len(matches) > 0:
            atom_dict['isplus'][np.intersect1d(matches - 1, not_carbon)] = True

        # Acidic group
        patt = Smarts('[$([C,S](=[O,S,P])-[O;H1])]')
        matches = np.array(patt.findall(self)).flatten()
        if len(matches) > 0:
            atom_dict['isminus'][np.intersect1d(matches - 1, not_carbon)] = True

        if self.protein:
            # Protein Residues (alpha helix and beta sheet)
            res_dtype = [('id', np.int16),
                         ('resnum', np.int16),
                         ('resname', 'U3' if PY3 else 'a3'),
                         ('N', np.float32, 3),
                         ('CA', np.float32, 3),
                         ('C', np.float32, 3),
                         ('O', np.float32, 3),
                         ('isalpha', bool),
                         ('isbeta', bool)
                         ]  # N, CA, C, O

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
                        elif atom.atomicnum == 8:
                            backbone['O'] = atom.coords
                if len(backbone.keys()) == 4:
                    b.append((residue.idx0,
                              residue.number,
                              residue.name,
                              backbone['N'],
                              backbone['CA'],
                              backbone['C'],
                              backbone['O'],
                              False,
                              False))
            res_dict = np.array(b, dtype=res_dtype)
            res_dict = detect_secondary_structure(res_dict)
            alpha_mask = np.in1d(atom_dict['resid'],
                                 res_dict[res_dict['isalpha']]['id'])
            atom_dict['isalpha'][alpha_mask] = True
            beta_mask = np.in1d(atom_dict['resid'],
                                res_dict[res_dict['isbeta']]['id'])
            atom_dict['isbeta'][beta_mask] = True

        # Aromatic Rings
        r = []
        for ring in self.sssr:
            if ring.IsAromatic():
                path = [x - 1 for x in ring._path]  # NOTE: mol.sssr is 1-based
                atoms = atom_dict[canonize_ring_path(path)]
                if len(atoms):
                    atom = atoms[0]
                    coords = atoms['coords']
                    centroid = coords.mean(axis=0)
                    # get vector perpendicular to ring
                    ring_vectors = coords - centroid
                    vector = np.cross(ring_vectors, np.roll(ring_vectors, 1)).mean(axis=0)
                    r.append((centroid,
                              vector,
                              atom['resid'],
                              atom['resnum'],
                              atom['resname'],
                              atom['isalpha'],
                              atom['isbeta']))
        ring_dict = np.array(r, dtype=[('centroid', np.float32, 3),
                                       ('vector', np.float32, 3),
                                       ('resid', np.int16),
                                       ('resnum', np.int16),
                                       ('resname', 'U3' if PY3 else 'a3'),
                                       ('isalpha', bool),
                                       ('isbeta', bool)])

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
                'data': dict(self.data.items()) if self._source is None else {},
                'protein': self.protein,
                'dicts': {'atom_dict': self._atom_dict,
                          'ring_dict': self._ring_dict,
                          'res_dict': self._res_dict,
                          }
                }

    def __setstate__(self, state):
        Molecule.__init__(self, source=state, protein=state['protein'])
        if state['data']:
            self.data.update(state['data'])
        self._atom_dict = state['dicts']['atom_dict']
        self._ring_dict = state['dicts']['ring_dict']
        self._res_dict = state['dicts']['res_dict']


# Extend pybel.Molecule
pybel.Molecule = Molecule


def diverse_conformers_generator(mol, n_conf=10, method='confab', seed=None,
                                 **kwargs):
    """Produce diverse conformers using current conformer as starting point.
    Returns a generator. Each conformer is a copy of original molecule object.

    .. versionadded:: 0.6

    Parameters
    ----------
    mol : oddt.toolkit.Molecule object
        Molecule for which generating conformers

    n_conf : int (default=10)
        Targer number of conformers

    method : string (default='confab')
        Method for generating conformers. Supported methods:
        * confab
        * ga

    seed : None or int (default=None)
        Random seed

    mutability : int (default=5)
        The inverse of probability of mutation. By default 5, which translates
        to 1/5 (20%) chance of mutation. This setting only works with genetic
        algorithm method ("ga").

    convergence : int (default=5)
        The number of generations with unchanged fitness, should the algorothm
        converge. This setting only works with genetic algorithm method ("ga").

    rmsd : float (default=0.5)
        The conformers are pruned unless their RMSD is higher than this cutoff.
        This setting only works with Confab method ("confab").

    nconf : int (default=10000)
        The number of initial conformers to generate before energy pruning.
        This setting only works with Confab method ("confab").

    energy_gap : float (default=5000.)
        Energy gap from the lowest energy conformer to the highest possible.
        This setting only works with Confab method ("confab").

    Returns
    -------
    mols : list of oddt.toolkit.Molecule objects
        Molecules with diverse conformers
    """
    if __version__ < '2.4.0':
        raise NotImplementedError('Diverse conformer generation is not '
                                  'implemented in OpenBabel before 2.4.0.')

    check_molecule(mol, force_coords=True)
    mol_clone = mol.clone
    if seed is not None:
        rand = ob.OBRandom(True)
        rand.Seed(seed)
    if method == 'ga':
        if not hasattr(ob, 'OBConformerSearch'):
            raise ValueError('OpenBabel needs to be compiled with eigen to '
                             'perform conformer search.')
        cs = ob.OBConformerSearch()
        cs.Setup(mol_clone.OBMol,
                 n_conf,  # numConformers
                 n_conf * 2,  # numChildren
                 kwargs.get('mutability', 5),  # mutability
                 kwargs.get('convergence', 5))  # convergence
        cs.Search()
        cs.GetConformers(mol_clone.OBMol)
    elif method == 'confab':
        ff = pybel._forcefields['uff']
        ff.Setup(mol_clone.OBMol)
        ff.DiverseConfGen(kwargs.get('rmsd', 0.5),  # rmsd
                          kwargs.get('nconfs', 10000),  # nconfs (initial)
                          kwargs.get('energy_gap', 5000.0),  # energy_gap
                          False)  # verbose
        ff.GetConformers(mol_clone.OBMol)
    else:
        raise ValueError('Method %s is not implemented' % method)

    out = []
    for i in range(mol_clone.OBMol.NumConformers()):
        if i >= n_conf:
            break
        mol_output_clone = mol_clone.clone
        mol_output_clone.OBMol.SetConformer(i)
        out.append(mol_output_clone)
    return out


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
    @deprecated('RDKit is 0-based and OpenBabel is 1-based. '
                'State which convention you desire and use `idx0` or `idx1`.')
    def idx(self):
        """Note that this index is 1-based as OpenBabel's internal index."""
        return self.idx1

    @property
    def idx1(self):
        """Note that this index is 1-based as OpenBabel's internal index."""
        return self.OBAtom.GetIdx()

    @property
    def idx0(self):
        """Note that this index is 0-based and OpenBabel's internal index in
        1-based. Changed to be compatible with RDKit"""
        return self.OBAtom.GetIdx() - 1

    @property
    def neighbors(self):
        return [Atom(a) for a in OBAtomAtomIter(self.OBAtom)]

    @property
    def residue(self):
        return Residue(self.OBAtom.GetResidue())

    @property
    def bonds(self):
        return [Bond(b) for b in OBAtomBondIter(self.OBAtom)]


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
        """List of Atoms in the Residue"""
        return [Atom(atom) for atom in ob.OBResidueAtomIter(self.OBResidue)]

    @property
    @deprecated('Use `idx0` instead.')
    def idx(self):
        """Internal index (0-based) of the Residue"""
        return self.OBResidue.GetIdx()

    @property
    def idx0(self):
        """Internal index (0-based) of the Residue"""
        return self.OBResidue.GetIdx()

    @property
    def number(self):
        """Residue number"""
        return self.OBResidue.GetNum()

    @property
    def chain(self):
        """Resdiue chain ID"""
        return self.OBResidue.GetChain()

    @property
    def name(self):
        """Residue name"""
        return self.OBResidue.GetName()

    def __iter__(self):
        """Iterate over the Atoms of the Residue.

        This allows constructions such as the following:
           for atom in residue:
               print(atom)
        """
        return iter(self.atoms)


class ResidueStack(object):
    def __init__(self, OBMol):
        self.OBMol = OBMol

    def __iter__(self):
        for i in range(self.OBMol.NumResidues()):
            yield Residue(self.OBMol.GetResidue(i))

    def __len__(self):
        return self.OBMol.NumResidues()

    def __getitem__(self, i):
        if 0 <= i < self.OBMol.NumResidues():
            return Residue(self.OBMol.GetResidue(i))
        else:
            raise AttributeError("There is no residue with Idx %i" % i)


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
        fp = np.zeros(len(self.fp) * ob.OBFingerprint.Getbitsperint(), dtype=int)
        np.add.at(fp, np.array(self.bits) - 1, 1)  # self.bits is 1-based
        return fp


pybel.Fingerprint = Fingerprint


class Smarts(pybel.Smarts):
    def __init__(self, smartspattern):
        """Initialise with a SMARTS pattern."""
        self.amap = None
        if isinstance(smartspattern, Molecule):
            tmp = smartspattern.write('smi', opt={'i': None,
                                                  'c': None,
                                                  'h': None})
            self.amap = np.array(smartspattern.data['SMILES Atom Order'].split(), dtype=int) - 1
            smartspattern = tmp.lstrip().split()[0]  # extract only SMILES
        super(Smarts, self).__init__(smartspattern)

    def match(self, molecule):
        """ Checks if there is any match. Returns True or False """
        return self.obsmarts.HasMatch(molecule.OBMol)

    def findall(self, molecule, unique=True):
        """Find all matches of the SMARTS pattern to a particular molecule """
        self.obsmarts.Match(molecule.OBMol)
        if unique:
            matches = list(self.obsmarts.GetUMapList())
        else:
            matches = list(self.obsmarts.GetMapList())
        if self.amap is None:
            return matches
        else:
            idx = np.argsort(self.amap)
            return [np.array(m)[idx].tolist() for m in matches]
