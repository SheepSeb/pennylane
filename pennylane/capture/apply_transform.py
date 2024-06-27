import jax
from jax.tree_util import tree_flatten, tree_unflatten

from .interpreters import ConvertToTape
from .switches import disable, enable


def reapply(obj):
    vals, structure = tree_flatten(obj)
    return tree_unflatten(structure, vals)


def recapture_tape(tape):
    for op in tape.operations:
        reapply(op)
    return [reapply(m) for m in tape.measurements]


def apply_transform(plxpr, transform, *targs, **tkwargs):
    def f(*args):
        disable()
        tape = ConvertToTape()(plxpr.jaxpr, plxpr.consts, *args)
        (new_tape,), _ = transform(tape, *targs, **tkwargs)
        enable()

        return recapture_tape(new_tape)

    return jax.make_jaxpr(f)
