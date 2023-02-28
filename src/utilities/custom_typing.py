from typing import Type


# mypy allows overriding a property with a field, but python does not, this hacks around that limitation. TODO: is there a better way to do this?
def implement_properties_as_fields(t: Type) -> Type:
    abstract_methods = t.__abstractmethods__
    t.__abstractmethods__ = frozenset(
        abstract_method
        for abstract_method in abstract_methods
        if not isinstance(getattr(t, abstract_method), property)
    )
    for abstract_method in abstract_methods:
        if isinstance(getattr(t, abstract_method), property):
            setattr(t, abstract_method, None)
    return t
