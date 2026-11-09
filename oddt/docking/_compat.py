"""Deprecated class aliases that preserve class identity and pickle lookups."""

import warnings


def deprecated_class_alias(name, namespace, aliases):
    """Resolve a legacy module attribute and point callers at its replacement."""
    module = namespace["__name__"]
    if name not in aliases:
        raise AttributeError("module %r has no attribute %r" % (module, name))
    replacement = aliases[name]
    warnings.warn(
        "%s.%s is deprecated; use %s.%s instead. The old name will be removed in a future release."
        % (module, name, module, replacement),
        DeprecationWarning,
        stacklevel=3,
    )
    return namespace[replacement]
