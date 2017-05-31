""" ODDT's internal docking/scoring engines """
import numpy as np
from math import floor, ceil, sin, cos
from collections import Counter
from itertools import chain
import oddt
from oddt.spatial import distance, dihedral, rotate, angle
from oddt.property import xlogp2_atom_contrib
from pprint import pprint


def get_children(molecule, mother, restricted):
    # TODO: Fix RDKit 0-based indexes
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
    # TODO: Fix RDKit 0-based indexes
    atoms = np.zeros(len(molecule.atoms), dtype=bool)
    atoms[a_idx - 1] = True
    for i in range(num_bonds):
        atoms[[n.idx - 1
               for i in np.nonzero(atoms)[0]
               for n in molecule.atoms[i].neighbors]] = True
    return atoms


def change_dihedral(coords, a1, a2, a3, a4, target_angle, rot_mask):
    angle_sin = sin(target_angle)
    angle_cos = cos(target_angle)
    t = 1 - cos
    v0 = coords[a2] - coords[a3]
    v = (v0) / np.linalg.norm(v0)
    rot_matrix = np.array([[t * v[0] * v[0] + angle_cos,
                            t * v[0] * v[1] + angle_sin * v[2],
                            t * v[0] * v[2] - angle_sin * v[1]],
                           [t * v[0] * v[1] - angle_sin * v[2],
                            t * v[1] * v[1] + angle_cos,
                            t * v[1] * v[2] + angle_sin * v[0]],
                           [t * v[0] * v[2] + angle_sin * v[1],
                            t * v[1] * v[2] - angle_sin * v[0],
                            t * v[2] * v[2] + angle_cos]])

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


def surface_mesh(r=1., spacing=0.5):
    """
    Compute an evenly distributed mesh on the surface of sphere.

    Parameters
    ----------
    r : float
        The radius of sphere

    spacing: float
        Distance between mesh points

    Returns
    -------
    meshcoords : numpy arrays, shape = [n_points, 4]
        3d coordinates of mesh points (0-centered). 4-th dim is the singular
        surface of mesh point.
    """
    coords = []
    total_surf = 4 * np.pi * r * r
    tolerance = 1e-6
    num = 0
    for alpha in np.linspace(0., np.pi, ceil(np.pi * r / spacing)):
        r2 = r * sin(alpha)
        # the number of steps must be uneven to cover the top and bottom
        for beta in np.linspace(0., 2 * np.pi, ceil(2 * np.pi * r2 / spacing) + 1):
            # remove duplicate/overlapping dots
            if ((not abs(alpha - np.pi) < tolerance and
                 abs(beta - 2 * np.pi) < tolerance) or
                (abs(alpha - np.pi) < tolerance and
                 not abs(beta - 2 * np.pi) < tolerance)):
                continue
            coords.append((r2 * cos(beta),
                           r2 * sin(beta),
                           r * cos(alpha)))
            num += 1
    coords = np.array(coords)
    return np.hstack((coords, np.array([total_surf / num] * num).reshape(-1, 1)))


def num_rotors_xscore(lig, atom_contrib=False):
    SMARTS_EXLUSIONS = [
        '!$(C(F)(F)F)',
        '!$(C(Cl)(Cl)Cl)',
        '!$(C(Br)(Br)Br)',
        '!$(C([CD1])([D1])[CD1])',
        '!$(C(!@N)(!@[N,N+]))',
    ]
    SMARTS_SINGLE_EXCLUSIONS = [
        '!$([CD3](=[N,O,S])-!@[#7,O,S!D1])',
        '!$([#7,O,S!D1]-!@[CD3]=[N,O,S])',
        '!$([#7!D1]-!@[CD3]=[N+])',
    ]
    SMARTS = ('[!D1&%s]-&!@[!D1&%s]' % ('&'.join(SMARTS_EXLUSIONS +
                                                 SMARTS_SINGLE_EXCLUSIONS),
                                        '&'.join(SMARTS_EXLUSIONS)))
    s = oddt.toolkit.Smarts(SMARTS)
    rotor_ids = list(chain(*s.findall(lig)))

    out = np.zeros(len(lig.atoms))
    for i, count in Counter(rotor_ids).items():
        if oddt.toolkit.backend == 'ob':
            i -= 1
        if count == 1:
            out[i] += 0.5
        elif count == 2:
            out[i] += 1.0
        elif count >= 3:
            out[i] += 0.5
    return out if atom_contrib else out.sum()


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


class xscore_docking(vina_docking):
    """Internal implementation of XSCORE"""
    WATER_R = 1.4

    def set_ligand(self, lig):
        super(xscore_docking, self).set_ligand(lig)
        self.num_rotors = num_rotors_xscore(lig, atom_contrib=True)
        self.lig_xlogp2_contrib = xlogp2_atom_contrib(lig)[lig.atom_dict['atomicnum'] != 1]

        self.ligand_atom_mesh = [surface_mesh(r=self.lig_dict['radius'][i] + self.WATER_R, spacing=.5)
                                 for i in range(len(self.lig_dict))]

    def set_protein(self, rec):
        rec.protein = True
        if rec is None:
            self.rec_dict = None
            self.mask_inter = {}
        else:
            self.rec_dict = rec.atom_dict[rec.atom_dict['atomicnum'] != 1].copy()

            self.rec_xlogp2_contrib = xlogp2_atom_contrib(rec)[np.array([a.atomicnum != 1 for a in rec.atoms], dtype=bool)]
            self.rec_xlogp2_contrib = self.rec_xlogp2_contrib[self.rec_dict['resname'] != 'HOH']

            # Remove waters
            self.rec_dict = self.rec_dict[self.rec_dict['resname'] != 'HOH']

            self.rec_dict = self.correct_radius(self.rec_dict)
            self.mask_inter = {}

    def correct_radius(self, atom_dict):
        vina_r = {6: 1.8,
                  7: 1.75,
                  8: 1.65,
                  9: 1.5,
                  15: 2.0,
                  16: 2.0,
                  17: 1.75,
                  35: 1.9,
                  53: 2.05,
                  }
        for a, r in vina_r.items():
            atom_dict['radius'][atom_dict['atomicnum'] == a] = r

        # C sp2
        atom_dict['radius'][(atom_dict['atomicnum'] == 6) & (atom_dict['hybridization'] == 2)] = 1.9
        # C sp3
        atom_dict['radius'][(atom_dict['atomicnum'] == 6) & (atom_dict['hybridization'] == 3)] = 2.1
        # C aromatic
        atom_dict['radius'][(atom_dict['atomicnum'] == 6) & atom_dict['isaromatic']] = 2.0
        # carbocation
        atom_dict['radius'][atom_dict['atomtype'] == 'C.cat'] = 1.9
        # N sp3
        atom_dict['radius'][(atom_dict['atomicnum'] == 7) & (atom_dict['hybridization'] == 3)] = 1.8
        # 0 sp2
        atom_dict['radius'][(atom_dict['atomicnum'] == 8) & (atom_dict['hybridization'] == 2)] = 1.55
        # S sp3
        atom_dict['radius'][(atom_dict['atomicnum'] == 16) & (atom_dict['hybridization'] == 3)] = 2.1
        # metals - 1.2 A
        atom_dict['radius'][atom_dict['ismetal']] = 1.2
        return atom_dict

    def gen_molecule_surface_mesh(self, coords=None, probe=1.4):
        molecule_surface_mesh = []
        if coords is None:
            coords = self.lig_dict['coords'].copy()
        for i in range(len(self.lig_dict)):
            a_dict = self.lig_dict[i]
            atom_mesh = self.ligand_atom_mesh[i]

            atom_mesh[:, :3] += coords[i]
            d = distance(atom_mesh[:, :3], coords)

            mask = d > self.lig_dict['radius'] + probe
            mask[:, i] = True  # mark current atom
            mask = mask.all(axis=1)

            edge_mask = (d - self.lig_dict['radius'] - probe <
                         np.sqrt(atom_mesh[:, 3]).reshape(-1, 1))
            edge_mask[:, i] = False
            atom_mesh[:, 3][edge_mask.any(axis=1)] *= 0.5

            # print(d - self.lig_dict['radius'] - probe)
            # print(edge_mask.shape, edge_mask.sum(), (~edge_mask).sum())
            molecule_surface_mesh.append(atom_mesh[mask])
        return molecule_surface_mesh

    def score_inter(self, coords=None):
        local_lig_dict = self.lig_dict.copy()
        if coords is None:
            coords = self.lig_dict['coords']
        else:
            # BUG: Neighbors are out of order!
            local_lig_dict['coords'] = coords

        # Inter-molecular
        d = distance(self.rec_dict['coords'], coords)
        d0 = (self.rec_dict['radius'][:, np.newaxis] +
              self.lig_dict['radius'][np.newaxis, :])
        mask = d <= 8

        inter = []
        # Van def Waals
        d_vdw = d[mask]
        d_vdw0 = d0[mask]

        out = np.zeros_like(d)
        np.add.at(out, mask, -(((d_vdw0 / d_vdw) ** 8) - 2 * ((d_vdw0 / d_vdw) ** 4)))
        inter.append(out.sum())

        # pprint(list(zip(self.lig_dict['atomtype'],
        #        self.lig_dict['radius'],
        #        out.sum(0))))

        # H-Bonding
        if 'da' not in self.mask_inter:
            self.mask_inter['da'] = ((self.rec_dict['isdonor'] | self.rec_dict['ismetal'])[:, np.newaxis] *
                                     self.lig_dict['isacceptor'][np.newaxis, :])
        if 'ad' not in self.mask_inter:
            self.mask_inter['ad'] = (self.rec_dict['isacceptor'][:, np.newaxis] *
                                     (self.lig_dict['isdonor'] | self.lig_dict['ismetal'])[np.newaxis, :])
        mask_hb = d <= 15
        d_h = d * (mask_hb & (self.mask_inter['da'] | self.mask_inter['ad']))
        d_h0 = d0 * (mask_hb & (self.mask_inter['da'] | self.mask_inter['ad']))

        # the angle between donor root (DR), donor (D) and acceptor (A)
        theta1 = np.zeros_like(d)

        mask_d, mask_a = np.where(mask_hb & self.mask_inter['da'])
        A = coords[mask_a]
        D = self.rec_dict['coords'][mask_d]
        DR = self.rec_dict['neighbors'][mask_d]
        theta1_1 = angle(A[:, np.newaxis, :], D[:, np.newaxis, :], DR)
        theta1_1 = np.nanmin(theta1_1, axis=-1)
        np.add.at(theta1, mask_hb & self.mask_inter['da'], theta1_1.flatten())

        mask_a, mask_d = np.where(mask_hb & self.mask_inter['ad'])
        A = self.rec_dict['coords'][mask_a]
        D = coords[mask_d]
        DR = self.lig_dict['neighbors'][mask_d]
        theta1_2 = angle(A[:, np.newaxis, :], D[:, np.newaxis, :], DR)
        theta1_2 = np.nanmin(theta1_2, axis=-1)
        np.add.at(theta1, mask_hb & self.mask_inter['ad'], theta1_2.flatten())

        # the angle between donor (D), acceptor (A) and acceptor root (AR)
        theta2 = np.zeros_like(d)

        mask_a, mask_d = np.where(mask_hb & self.mask_inter['ad'])
        D = coords[mask_d]
        A = self.rec_dict['coords'][mask_a]
        AR = self.rec_dict['neighbors'][mask_a]
        theta2_1 = angle(D[:, np.newaxis, :], A[:, np.newaxis, :], AR)
        theta2_1 = np.nanmin(theta2_1, axis=-1)
        np.add.at(theta2, mask_hb & self.mask_inter['ad'], theta2_1.flatten())

        mask_d, mask_a = np.where(mask_hb & self.mask_inter['da'])
        D = self.rec_dict['coords'][mask_d]
        A = coords[mask_a]
        AR = self.lig_dict['neighbors'][mask_a]
        theta2_2 = angle(D[:, np.newaxis, :], A[:, np.newaxis, :], AR)
        theta2_2 = np.nanmin(theta2_2, axis=-1)
        np.add.at(theta2, mask_hb & self.mask_inter['da'], theta2_2.flatten())

        f_d = ((d_h <= d_h0 - 0.7).astype(float) +
               ((d_h0 - d_h) * ((d_h > d_h0 - 0.7) & (d_h < d_h0)) / 0.7))

        f_theta1 = np.clip(((theta1 >= 120) +
                           (theta1 * ((theta1 < 120) & (theta1 >= 60)) / 60 - 1)), 0, 1)
        f_theta2 = np.clip(((theta2 >= 120) +
                           (theta2 * ((theta2 < 120) & (theta2 >= 60)) / 60 - 1)), 0, 1)

        # print(f_d.min(), f_d.max())
        # a = (d_h <= d_h0 - 0.7).astype(float)
        # print(a.min(), a.max())
        # a = ((d_h0 - d_h) * ((d_h > d_h0 - 0.7) & (d_h < d_h0)) / 0.7)
        # print(a.min(), a.max())
        #
        # print(f_theta1.min(), f_theta1.max())
        # print(f_theta2.min(), f_theta2.max())
        #
        # print(theta1.min(), theta1.max())
        # print(theta2.min(), theta2.max())

        out = f_d * f_theta1 * f_theta2

        # pprint(list(zip(self.lig_dict['atomtype'],
        #        out.sum(0),
        #        f_d.sum(0),
        #        f_theta1.sum(0),
        #        f_theta2.sum(0),
        #        )))

        inter.append(out.sum())

        # Deformation effect
        out = self.num_rotors
        inter.append(out.sum())

        # Hydrophobic effect
        # i) Hydrophobic surface (HS)
        molecule_surface_mesh = self.gen_molecule_surface_mesh(coords)
        out = np.zeros(len(self.lig_dict))
        for i in range(len(self.lig_dict)):
            if not self.lig_dict['ishydrophobe'][i]:
                continue
            # limit distance calculation
            rec_local_dict = self.rec_dict[mask.any(axis=1)]
            mask_hs = (distance(rec_local_dict['coords'],
                                molecule_surface_mesh[i][:, :3]) <
                       rec_local_dict['radius'].reshape(-1, 1) + self.WATER_R).any(axis=0)
            out[i] = molecule_surface_mesh[i][mask_hs, 3].sum()

        # ref = [0, 0, 0, 0, 0, 10.1, 21.9, 19.3, 30.1, 13.9, 0, 0, 0, 0, 0, 0, 0,
        #        0, 0, 23.4, 1.6, 10.6, 23, 14.8, 17.9, 2.6, 0, 0, 0, 0, 0, 0, 0,
        #        0, 11.5, 1.7, 0, ]
        # pprint(list(zip(self.lig_dict['atomtype'],
        #        self.lig_dict['radius'],
        #        out,
        #        ref)))

        inter.append(out.sum())

        # ii) Hydrophobic pairs (HP)
        if 'hyd' not in self.mask_inter:
            self.mask_inter['hyd'] = ((self.rec_dict['ishydrophobe'] | self.rec_dict['ishalogen'])[:, np.newaxis] *
                                      (self.lig_dict['ishydrophobe'] | self.lig_dict['ishalogen'])[np.newaxis, :])
        mask_hyd = mask & self.mask_inter['hyd']
        d_hyd = d[mask_hyd]
        d_hyd0 = d0[mask_hyd]

        out = np.zeros_like(d, dtype=float)
        np.add.at(out, np.where(mask_hyd), d_hyd <= (d_hyd0 + 0.5))
        np.add.at(out, np.where(mask_hyd),
                  ((d_hyd0 + 2.2 - d_hyd) / (2.2 - 0.5) *
                   ((d_hyd > d_hyd0 + 0.5) & (d_hyd <= d_hyd0 + 2.2))))

        # pprint(list(zip(self.lig_dict['atomtype'],
        #        self.lig_dict['radius'],
        #        out.sum(0))))

        inter.append(out.sum())

        # iii) Hydrophobic matching (HM)
        out = np.zeros_like(d, dtype=float)
        out[d <= (d0 + 0.5)] += 1
        ix = (d > d0 + 0.5) & (d <= d0 + 2.2)
        out[ix] += ((d0 + 2.2 - d) / (2.2 - 0.5))[ix]

        hm_env = (self.rec_xlogp2_contrib[:, np.newaxis] * out).sum(axis=0)

        out = (np.clip(self.lig_xlogp2_contrib, 0, None) *
               ((hm_env > -0.5) | (self.lig_xlogp2_contrib > 0.5)))

        # pprint(list(zip(self.lig_dict['atomtype'],
        #                 self.lig_xlogp2_contrib,
        #                 hm_env,
        #                 out)))

        inter.append(out.sum())

        return np.array(inter)
