""" ODDT's internal docking/scoring engines """
import numpy as np
import math
from oddt.spatial import distance, dihedral, rotate


def get_children(molecule, mother, restricted):
    atoms = np.zeros(len(molecule.atoms), dtype=bool)
    atoms[mother - 1] = True
    d = 1  # Pass first
    prev = 0
    while d > 0:
        atoms[[n.idx - 1
               for i in np.nonzero(atoms)[0] if i != restricted - 1
               for n in molecule.atoms[i].neighbors if n.idx != restricted]] = True
        d = atoms.sum() - prev
        prev = atoms.sum()
    return atoms


def get_close_neighbors(molecule, a_idx, num_bonds=1):
    atoms = np.zeros(len(molecule.atoms), dtype=bool)
    atoms[a_idx - 1] = True
    for i in range(num_bonds):
        atoms[[n.idx - 1
               for i in np.nonzero(atoms)[0]
               for n in molecule.atoms[i].neighbors]] = True
    return atoms


def change_dihedral(coords, a1, a2, a3, a4, target_angle, rot_mask):
    sin = math.sin(target_angle)
    cos = math.cos(target_angle)
    t = 1 - cos
    v0 = coords[a2] - coords[a3]
    v = (v0) / np.linalg.norm(v0)
    rot_matrix = np.array([[t * v[0] * v[0] + cos,
                            t * v[0] * v[1] + sin * v[2],
                            t * v[0] * v[2] - sin * v[1]],
                           [t * v[0] * v[1] - sin * v[2],
                            t * v[1] * v[1] + cos,
                            t * v[1] * v[2] + sin * v[0]],
                           [t * v[0] * v[2] + sin * v[1],
                            t * v[1] * v[2] - sin * v[0],
                            t * v[2] * v[2] + cos]])

    centroid = coords[a3]
    coords = coords - centroid
    # Old and slower version
    # coords[rot_mask] = (coords[rot_mask,np.newaxis,:] * rot_matrix).sum(axis=2)
    coords[rot_mask] = np.einsum("ij,jk->ik", coords[rot_mask], rot_matrix)
    return coords + centroid


def num_rotors_pdbqt(lig):
    i = 0
    for atom in lig.atoms:
        if atom.atomicnum == 1:
            continue
        num_local_rot = sum(int(b.isrotor) for b in atom.bonds)
        if num_local_rot == 0:
            pass
        elif num_local_rot == 1:
            i += 0.5
        elif num_local_rot == 2:
            i += 1.0
        elif num_local_rot >= 3:
            i += 0.5
    return i


class vina_docking(object):
    def __init__(self, rec, lig=None, box=None, box_size=1., weights=None):
        self.box_size = box_size  # TODO: Unify box
        if rec:
            self.set_protein(rec)
        if lig:
            self.set_ligand(lig)

        self.set_box(box)
        # constants
        self.weights = weights or np.array((-0.0356, -0.00516, 0.840, -0.0351, -0.587, 0.0585))
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
            self.rec_dict = self.rec_dict#[mask]
        else:
            self.box = box

    def set_protein(self, rec):
        if rec is None:
            self.rec_dict = None
            self.mask_inter = {}
        else:
            self.rec_dict = rec.atom_dict[rec.atom_dict['atomicnum'] != 1].copy()
            self.rec_dict = self.correct_radius(self.rec_dict)
            self.mask_inter = {}

    def set_ligand(self, lig):
        lig_hvy_mask = (lig.atom_dict['atomicnum'] != 1)
        self.lig_dict = lig.atom_dict[lig_hvy_mask].copy()
        self.lig_dict = self.correct_radius(self.lig_dict)
        self.num_rotors = num_rotors_pdbqt(lig)
        self.mask_inter = {}
        self.mask_intra = {}

        # Find distant members (min 3 consecutive bonds)
        mask = np.vstack([~get_close_neighbors(lig, a1.idx, num_bonds=3) for a1 in lig])
        mask = mask[lig_hvy_mask[np.newaxis, :] * lig_hvy_mask[:, np.newaxis]]
        self.lig_distant_members = mask.reshape(lig_hvy_mask.sum(), lig_hvy_mask.sum())

        # prepare rotors dictionary
        self.rotors = []
        for b in lig.bonds:
            if b.isrotor:
                a2 = int(b.atoms[0].idx)
                a3 = int(b.atoms[1].idx)
                for n in b.atoms[0].neighbors:
                    if a3 != int(n.idx) and n.atomicnum != 1:
                        a1 = int(n.idx)
                        break
                for n in b.atoms[1].neighbors:
                    if a2 != int(n.idx) and n.atomicnum != 1:
                        a4 = int(n.idx)
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

    def set_coords(self, coords):
        self.lig_dict['coords'] = coords

    def score(self, coords=None):
        return (self.score_inter(coords) * self.weights[:5]).sum() / (1 + self.weights[5] * self.num_rotors)
        # inter = (self.score_inter(coords) * self.weights[:5]).sum()
        # total = (self.score_total(coords) * self.weights[:5]).sum()
        # return total/(1+self.weights[5]*self.num_rotors)

    def weighted_total(self, coords=None):
        return (self.score_total(coords) * self.weights[:5]).sum()

    def score_total(self, coords=None):
        return self.score_inter(coords) + self.score_intra(coords)

    def weighted_inter(self, coords=None):
        return (self.score_inter(coords) * self.weights[:5]).sum()

    def weighted_intra(self, coords=None):
        return (self.score_intra(coords) * self.weights[:5]).sum()

    def score_inter(self, coords=None):
        if coords is None:
            coords = self.lig_dict['coords']

        # Inter-molecular
        r = distance(self.rec_dict['coords'], coords)
        d = (r - self.rec_dict['radius'][:, np.newaxis] - self.lig_dict['radius'][np.newaxis, :])
        mask = r < 8

        inter = []
        # Gauss 1
        inter.append(np.exp(-(d[mask] / 0.5)**2).sum())
        # Gauss 2
        inter.append(np.exp(-((d[mask] - 3.) / 2.)**2).sum())
        # Repiulsion
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

    def score_intra(self, coords=None):
        if coords is None:
            coords = self.lig_dict['coords']
        # Intra-molceular
        r = distance(coords, coords)
        d = (r - self.lig_dict['radius'][:, np.newaxis] - self.lig_dict['radius'][np.newaxis, :])

        mask = self.lig_distant_members & (r < 8)

        intra = []
        # Gauss 1
        intra.append(np.exp(-(d[mask] / 0.5)**2).sum())
        # Gauss 2
        intra.append(np.exp(-((d[mask] - 3.) / 2.)**2).sum())
        # Repiulsion
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

    def correct_radius(self, atom_dict):
        vina_r = {6: 1.9,
                  7: 1.8,
                  8: 1.7,
                  9: 1.5,
                  15: 2.1,
                  16: 2.0,
                  17: 1.8,
                  35: 2.0,
                  53: 2.2,
                  }
        for a, r in vina_r.items():
            atom_dict['radius'][atom_dict['atomicnum'] == a] = r
        # metals - 1.2 A
        atom_dict['radius'][atom_dict['ismetal']] = 1.2
        return atom_dict


class vina_ligand(object):
    def __init__(self, c0, num_rotors, engine, box_size=1):
        self.c0 = c0.copy()
        self.x0 = np.zeros(6 + num_rotors)
        self.c1 = c0.copy()
        self.x1 = np.zeros_like(self.x0)
        self.engine = engine
        self.box_size = box_size

    def mutate(self, x2, force=False):
        delta_x0 = x2 - self.x0
        delta_x1 = x2 - self.x1
        if not force and (delta_x1 != 0).sum() <= 3:
            return self._inc_mutate(delta_x1, self.c1)
        elif not force and (delta_x0 != 0).sum() <= 3:
            return self._inc_mutate(delta_x0, self.c0)
        else:
            return self._full_mutate(x2)

    def _full_mutate(self, x):
        c = self.c0.copy()
        trans_vec = x[:3]
        rot_vec = x[3:6]
        rotors_vec = x[6:]
        c = rotate(c, *rot_vec) + trans_vec * self.box_size
        for i in range(len(rotors_vec)):
            a = self.engine.rotors[i]['atoms']
            mask = self.engine.rotors[i]['mask']
            c = change_dihedral(c, a[0], a[1], a[2], a[3], rotors_vec[i], mask)
        self.c1 = c.copy()
        self.x1 = x.copy()
        return c

    def _inc_mutate(self, x, c):
        c = c.copy()
        trans_vec = x[:3]
        rot_vec = x[3:6]
        rotors_vec = x[6:]
        if (rot_vec != 0).any():
            c = rotate(c, *rot_vec)
        if (trans_vec != 0).any():
            c += trans_vec * self.box_size
        for i in np.where(rotors_vec != 0)[0]:
            a = self.engine.rotors[i]['atoms']
            mask = self.engine.rotors[i]['mask']
            c = change_dihedral(c, a[0], a[1], a[2], a[3], rotors_vec[i], mask)
        return c
