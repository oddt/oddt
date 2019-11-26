import numpy as np
import oddt
from oddt.spatial import distance
from oddt.docking.internal import vina_ligand, get_children, get_close_neighbors, num_rotors_pdbqt
from oddt.scoring.functions import nnscore, rfscore, ri_score


class CustomEngine(object):
    def __init__(self, rec, lig=None, scoring_func=None, box=None, box_size=1.):
        self.box_size = box_size
        self.scoring_func = scoring_func
        if rec:
            if isinstance(rec, str):
                receptor_format = rec.split('.')[-1]
                try:
                    self.receptor = next(oddt.toolkit.readfile(receptor_format, rec))
                except ValueError:
                    raise Exception('Unsupported receptor file format.')
            elif isinstance(rec, oddt.toolkit.Molecule):
                self.receptor = rec
            else:
                raise Exception('Unsupported protein format.')

            self.receptor.protein = True
            self.receptor.removeh()
            self.set_protein(self.receptor)
        if lig:
            if isinstance(lig, str):
                ligand_format = lig.split('.')[-1]
                try:
                    self.ligand = list(oddt.toolkit.readfile(ligand_format, lig))
                except ValueError:
                    raise Exception('Unsupported ligand file format.')
            elif isinstance(lig, oddt.toolkit.Molecule):
                self.ligand = lig
            else:
                raise Exception('Unsupported ligand format.')
            self.ligand.removeh()
            self.set_ligand(self.ligand)

        self.prepare_scoring_function()
        self.set_box(box)
        self.mask_inter = {}
        self.mask_intra = {}

    def set_box(self, box):
        if box is not None:
            self.box = np.array(box)
            # delete unused atoms
            r = self.rec_dict['coords']
            # X, Y, Z within box and cutoff
            mask = (self.box[0][0] - 8 <= r[:, 0]) & (r[:, 0] <= self.box[1][0] + 8)
            mask *= (self.box[0][1] - 8 <= r[:, 1]) & (r[:, 1] <= self.box[1][1] + 8)
            mask *= (self.box[0][2] - 8 <= r[:, 2]) & (r[:, 2] <= self.box[1][2] + 8)
            self.rec_dict = self.rec_dict   # [mask]
        else:
            self.box = box

    def set_protein(self, rec):
        """Prepare protein to docking."""
        if rec is None:
            self.rec_dict = None
            self.mask_inter = {}
        else:
            self.rec_dict = rec.atom_dict[rec.atom_dict['atomicnum'] != 1].copy()
            self.mask_inter = {}

    def set_ligand(self, lig):
        """Prepare ligand to docking."""
        lig_hvy_mask = (lig.atom_dict['atomicnum'] != 1)
        self.lig_dict = lig.atom_dict[lig_hvy_mask].copy()
        self.num_rotors = num_rotors_pdbqt(lig)
        self.mask_inter = {}
        self.mask_intra = {}

        # Find distant members (min 3 consecutive bonds)
        mask = np.vstack([~get_close_neighbors(lig, i, num_bonds=3)
                          for i in range(len(lig.atoms))])
        mask = mask[lig_hvy_mask[np.newaxis, :] * lig_hvy_mask[:, np.newaxis]]
        self.lig_distant_members = mask.reshape(lig_hvy_mask.sum(), lig_hvy_mask.sum())

        # prepare rotors dictionary
        self.rotors = []
        for b in lig.bonds:
            if b.isrotor:
                a2 = int(b.atoms[0].idx0)
                a3 = int(b.atoms[1].idx0)
                for n in b.atoms[0].neighbors:
                    if a3 != int(n.idx0) and n.atomicnum != 1:
                        a1 = int(n.idx0)
                        break
                for n in b.atoms[1].neighbors:
                    if a2 != int(n.idx0) and n.atomicnum != 1:
                        a4 = int(n.idx0)
                        break
                rot_mask = get_children(lig, a3, a2)[lig_hvy_mask]
                # translate atom indicies to lig_dict indicies (heavy only)
                a1 = np.argwhere(self.lig_dict['id'] == a1).flatten()[0]
                a2 = np.argwhere(self.lig_dict['id'] == a2).flatten()[0]
                a3 = np.argwhere(self.lig_dict['id'] == a3).flatten()[0]
                a4 = np.argwhere(self.lig_dict['id'] == a4).flatten()[0]
                # rotate smaller part of the molecule
                if rot_mask.sum() > len(rot_mask):
                    rot_mask = -rot_mask
                    a4, a3, a2, a1 = a1, a2, a3, a4
                self.rotors.append({'atoms': (a1, a2, a3, a4), 'mask': rot_mask})

        # Setup cached ligand coords
        self.lig = vina_ligand(self.lig_dict['coords'].copy(), len(self.rotors), self, self.box_size)

    def prepare_scoring_function(self):
        """Initiate scoring functions based on AI/ML methods."""
        sf = None
        if self.scoring_func == 'nnscore':
            sf = nnscore.load()
            sf.set_protein(self.receptor)
        elif self.scoring_func == 'rfscore':
            sf = rfscore.load()
            sf.set_protein(self.receptor)
        self.trained_scorer = sf

    def score(self, coords=None):
        """Score given coordinates."""
        if coords is None:
            coords = self.lig_dict['coords']
        self.ligand.coords = coords
        if self.trained_scorer:  # nnscore/rfscore
            return self.trained_scorer.predict([self.ligand])[0]
        elif self.scoring_func == 'ri_score':
            return ri_score(self.ligand, self.receptor)
        elif self.scoring_func == 'interaction_energy':
            return np.sum(self.score_inter(coords) + self.score_intra(coords))
        else:
            raise Exception('Unsupported scoring function.')

    def score_inter(self, coords):
        """Calculate inter-molecular energy between protein and ligand."""
        # Inter-molecular
        r = distance(self.rec_dict['coords'], coords)
        d = (r - self.rec_dict['radius'][:, np.newaxis] - self.lig_dict['radius'][np.newaxis, :])
        mask = r < 8

        inter = []
        # Gauss 1
        inter.append(np.exp(-(d[mask] / 0.5)**2).sum())
        # Gauss 2
        inter.append(np.exp(-((d[mask] - 3.) / 2.)**2).sum())
        # Repulsion
        inter.append((d[(d < 0) & mask]**2).sum())

        # Hydrophobic
        if 'hyd' not in self.mask_inter:
            self.mask_inter['hyd'] = ((self.rec_dict['ishydrophobe'] | self.rec_dict['ishalogen'])[:, np.newaxis] *
                                      (self.lig_dict['ishydrophobe'] | self.lig_dict['ishalogen'])[np.newaxis, :])
        mask_hyd = mask & self.mask_inter['hyd']
        d_hyd = d[mask_hyd]
        inter.append((d_hyd <= 0.5).sum() + (1.5 - d_hyd[(0.5 < d_hyd) & (d_hyd < 1.5)]).sum())

        # H-Bonding
        if 'da' not in self.mask_inter:
            self.mask_inter['da'] = ((self.rec_dict['isdonor'] | self.rec_dict['ismetal'])[:, np.newaxis] *
                                     self.lig_dict['isacceptor'][np.newaxis, :])
        if 'ad' not in self.mask_inter:
            self.mask_inter['ad'] = (self.rec_dict['isacceptor'][:, np.newaxis] *
                                     (self.lig_dict['isdonor'] | self.lig_dict['ismetal'])[np.newaxis, :])
        d_h = d[mask & (self.mask_inter['da'] | self.mask_inter['ad'])]
        inter.append((d_h <= -0.7).sum() + (d_h[(-0.7 < d_h) & (d_h < 0)] / -0.7).sum())

        return np.array(inter)

    def score_intra(self, coords):
        """Calculate intra-molecular energy between protein and ligand."""
        # Intra-molceular
        r = distance(coords, coords)
        d = (r - self.lig_dict['radius'][:, np.newaxis] - self.lig_dict['radius'][np.newaxis, :])

        mask = self.lig_distant_members & (r < 8)

        intra = []
        # Gauss 1
        intra.append(np.exp(-(d[mask] / 0.5)**2).sum())
        # Gauss 2
        intra.append(np.exp(-((d[mask] - 3.) / 2.)**2).sum())
        # Repulsion
        intra.append((d[(d < 0) & mask]**2).sum())

        # Hydrophobic
        if 'hyd' not in self.mask_intra:
            self.mask_intra['hyd'] = ((self.lig_dict['ishydrophobe'] | self.lig_dict['ishalogen'])[:, np.newaxis] *
                                      (self.lig_dict['ishydrophobe'] | self.lig_dict['ishalogen'])[np.newaxis, :])
        mask_hyd = mask & self.mask_intra['hyd']
        d_hyd = d[mask_hyd]
        intra.append((d_hyd <= 0.5).sum() + (1.5 - d_hyd[(0.5 < d_hyd) & (d_hyd < 1.5)]).sum())

        # H-Bonding
        if 'da' not in self.mask_intra:
            self.mask_intra['da'] = ((self.lig_dict['isdonor'] | self.lig_dict['ismetal'])[..., np.newaxis] *
                                     self.lig_dict['isacceptor'][np.newaxis, ...])
        if 'ad' not in self.mask_intra:
            self.mask_intra['ad'] = (self.lig_dict['isacceptor'][..., np.newaxis] *
                                     (self.lig_dict['isdonor'] | self.lig_dict['ismetal'])[np.newaxis, ...])
        d_h = d[mask & (self.mask_intra['da'] | self.mask_intra['ad'])]
        intra.append((d_h <= -0.7).sum() + (d_h[(-0.7 < d_h) & (d_h < 0)] / -0.7).sum())

        return np.array(intra)

