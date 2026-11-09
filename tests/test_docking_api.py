"""ODDT-prefixed engines and deprecated, identity-preserving class aliases."""

import importlib
import pickle
import warnings

import pytest

import oddt.docking as docking

MODEL_NAMES = {
    "vina": ("internal", "Vina"),
    "vinardo": ("vinardo", "Vinardo"),
    "ad4": ("ad4", "AD4"),
}
ALIASES = [
    (module, legacy, "Oddt" + prefix + new)
    for model, (source, prefix) in MODEL_NAMES.items()
    for old, new in (("scoring", "Scoring"), ("ga", "GA"), ("mc", "MC"), ("de", "DE"))
    for legacy in (model + "_" + old, prefix + new)
    for module in ("oddt.docking", "oddt.docking." + source)
] + [
    ("oddt.docking", "vina_docking", "OddtVinaDocking"),
    ("oddt.docking.internal", "vina_docking", "OddtVinaDocking"),
    ("oddt.docking", "VinaDocking", "OddtVinaDocking"),
    ("oddt.docking.internal", "VinaDocking", "OddtVinaDocking"),
    ("oddt.docking.internal", "vina_ligand", "LigandPose"),
    ("oddt.docking", "autodock_vina", "AutoDockVina"),
    ("oddt.docking.AutodockVina", "autodock_vina", "AutoDockVina"),
]
MODULES = sorted({module for module, _, _ in ALIASES})
CANONICAL = ["Oddt" + prefix + suffix for _, prefix in MODEL_NAMES.values() for suffix in ("Scoring", "GA", "MC", "DE")]


@pytest.mark.parametrize("module_name,old,new", ALIASES)
def test_legacy_alias_preserves_identity_and_warns_at_caller(module_name, old, new):
    module = importlib.import_module(module_name)
    replacement = getattr(module, new)
    with pytest.warns(DeprecationWarning, match=old + ".*" + new) as captured:
        alias = getattr(module, old)
    assert alias is replacement
    assert captured[0].filename == __file__
    assert old in dir(module)
    assert old in module.__all__


@pytest.mark.parametrize("module_name,old,new", ALIASES)
def test_legacy_from_import_remains_a_class_alias(module_name, old, new):
    namespace = {}
    with pytest.warns(DeprecationWarning, match=old + ".*" + new):
        exec("from %s import %s as imported" % (module_name, old), namespace)
    assert namespace["imported"] is getattr(importlib.import_module(module_name), new)


@pytest.mark.parametrize("module_name,old,new", ALIASES)
def test_legacy_pickle_global_resolves_to_canonical_class(module_name, old, new):
    # Protocol-zero GLOBAL uses the exact module/name stored by older pickles.
    payload = ("c%s\n%s\n." % (module_name, old)).encode("ascii")
    with pytest.warns(DeprecationWarning, match=old + ".*" + new):
        restored = pickle.loads(payload)
    assert restored is getattr(importlib.import_module(module_name), new)


@pytest.mark.parametrize("name", CANONICAL + ["OddtVinaDocking", "AutoDockVina"])
def test_canonical_class_pickle_roundtrip_does_not_warn(name):
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        cls = getattr(docking, name)
        assert pickle.loads(pickle.dumps(cls)) is cls
        if name != "AutoDockVina":
            assert cls.__name__.startswith("Oddt")
            instance = cls(None)
            assert type(pickle.loads(pickle.dumps(instance))) is cls


@pytest.mark.parametrize("module_name", MODULES)
def test_star_import_preserves_deprecated_exports(module_name):
    namespace = {}
    with pytest.warns(DeprecationWarning):
        exec("from %s import *" % module_name, namespace)
    for module, old, new in ALIASES:
        if module == module_name:
            assert namespace[old] is getattr(importlib.import_module(module_name), new)


@pytest.mark.parametrize("module_name", MODULES)
def test_unknown_module_attribute_does_not_warn(module_name):
    module = importlib.import_module(module_name)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        with pytest.raises(AttributeError, match="has no attribute 'missing_engine'"):
            getattr(module, "missing_engine")
    assert not captured


def test_default_vina_docking_remains_monte_carlo():
    assert docking.OddtVinaDocking is docking.OddtVinaMC


def test_external_vina_wrapper_keeps_unprefixed_name():
    from oddt.docking.AutodockVina import AutoDockVina

    assert docking.AutoDockVina is AutoDockVina
    assert AutoDockVina.__name__ == "AutoDockVina"
    assert "OddtAutoDockVina" not in docking.__all__
