"""Common utilities for ODDT"""
import oddt


def is_molecule(obj):
    """Check whether an object is an `oddt.toolkits.{rdk,ob}.Molecule` instance.
    """
    return is_openbabel_molecule(obj) or is_rdkit_molecule(obj)


def is_openbabel_molecule(obj):
    """Check whether an object is an `oddt.toolkits.ob.Molecule` instance."""
    return (hasattr(oddt.toolkits, 'ob') and
            isinstance(obj, oddt.toolkits.ob.Molecule))


def is_rdkit_molecule(obj):
    """Check whether an object is an `oddt.toolkits.rdk.Molecule` instance."""
    return (hasattr(oddt.toolkits, 'rdk') and
            isinstance(obj, oddt.toolkits.rdk.Molecule))
