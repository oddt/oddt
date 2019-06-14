import numpy as np

__all__ = [
    'b_factor',
    'ri_score',
]


def distance(a, b):
    """Numexpr powered distance that is float32 friendly (memory efficient)"""
    a = a[..., np.newaxis, :]
    return np.sqrt(np.sum((a - b)**2, axis=2))


def b_factor(ligand, protein, correlation_type='exponential',
             tau=3., k=40, v=40):
    """ Flexibility-Rigity Index, based on 10.1021/acs.jcim.7b00226.

    Parameters
    ----------
    ligand, protein : oddt.toolkit.Molecule object
        Molecules, which are used to compute score.

    correlation_type: string (default='exponential')
        Type of kernel -> 'exponential' or 'lorenz'.

    tau: float (deault=3.)
        Adjustable parameter.

    k, v: int (deafult=40)
        If set to high values, correlation functions behave like ideal low-pass filter.

    Returns
    -------
    b_factor : float
        Flexibility-Rigity score.
    """
    # dicts with heavy atoms
    ligand_atoms = ligand.atom_dict[ligand.atom_dict['atomicnum'] > 1]
    protein_atoms = protein.atom_dict[protein.atom_dict['atomicnum'] > 1]
    complex_atoms = np.hstack((protein_atoms, ligand_atoms))

    # distance matrix for heavy atoms, casted to float32 for RAM efficiency
    distance_complex = distance(complex_atoms['coords'],
                                complex_atoms['coords']).astype(np.float32)
    # Reuse distances to save time and memory
    n_prot = len(protein_atoms)
    distance_protein = distance_complex[:n_prot, :n_prot]
    assert distance_complex.shape == (len(complex_atoms), len(complex_atoms))
    assert distance_protein.shape == (n_prot, n_prot)

    # weights, which are set to 1
    w_protein = 1.  # / (len(protein_atoms) ** 2)
    w_complex = 1.  # / ((len(ligand_atoms) + len(protein_atoms)) * len(protein_atoms))

    b_factor_protein = 1. / correlation(distance_protein,
                                        w_protein,
                                        protein_atoms['radius'],
                                        correlation_type=correlation_type,
                                        tau=tau,
                                        k=k,
                                        v=v)
    b_factor_complex = 1. / correlation(distance_complex,
                                        w_complex,
                                        complex_atoms['radius'],
                                        correlation_type=correlation_type,
                                        tau=tau,
                                        k=k,
                                        v=v)

    delta_bfactor = (b_factor_complex - b_factor_protein) / b_factor_protein

    return delta_bfactor


def ri_score(ligand, protein, correlation_type='exponential', tau=3., k=40, v=40):
    """ Rigidity index based scoring functions (RI-Score), based on
    10.1021/acs.jcim.7b00226.

    Parameters
    ----------
    ligand, protein : oddt.toolkit.Molecule object
        Molecules, which are used to compute score

    correlation_type: string (default='exponential')
        Type of kernel -> 'exponential' or 'lorenz'

    tau: float (default=3.)
        Adjustable parameter.

    k, v: int (default=40)
        If set to high values, correlation functions behave like ideal low-pass filter.

    Returns
    -------
    ri_score: float
        Rigidity score.

    """
    # dicts with heavy atoms
    ligand_atoms = ligand.atom_dict[ligand.atom_dict['atomicnum'] > 1]
    protein_atoms = protein.atom_dict[protein.atom_dict['atomicnum'] > 1]
    distance_complex = distance(protein_atoms['coords'],
                                ligand_atoms['coords']).astype(np.float32)
    assert distance_complex.shape == (len(protein_atoms), len(ligand_atoms))
    weight = 1.
    return correlation(distance_complex,
                       weight,
                       protein_atoms['radius'],
                       ligand_atoms['radius'],
                       correlation_type=correlation_type,
                       tau=tau,
                       k=k,
                       v=v)


def correlation(distance, weight, radius, radius_secondary=None,
                correlation_type='exponential', tau=3., k=40, v=40):
    """ Generalized exponential functions used in Flexibility-Rigidity Index computation.
        Parameters
        ----------
        distance:
            Distance between coords of heavy atoms.

        weight: float (default=1)
            Particle-type dependent weight.

        radius:
            Radius of protein atoms.

        radius_secondary:
            Radius of ligand atoms.

        correlation_type: string (default='exponential')
            Type of kernel -> 'exponential' or 'lorenz'

        tau: float (default=3.)
            Adjustable parameter.

        k, v: int (default=40)
            If set to high values, correlation functions behave like ideal low-pass filter.

        Returns
        -------
        correlation: float
            Correlation function score.

    """
    assert len(distance) == len(radius)

    if radius_secondary is None:
        radius_secondary = radius
    radius = radius[:, np.newaxis]
    if correlation_type == 'exponential':
        return float(np.sum(weight * np.exp(-(
            (distance / (radius * radius_secondary * tau)) ** k))))
    elif correlation_type == 'lorentz':
        return float(np.sum(weight * 1. / (1. + (
            (distance / (radius * radius_secondary * tau)) ** v))))
    else:
        raise ValueError('"{}" is unsupported correlation function type.'
                         'Use one of: ["exponential", "lorentz"]/'.format(
                             correlation_type))
