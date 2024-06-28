import jax

import pennylane as qml

### The measure primitive ###############

measure_prim = jax.core.Primitive("measure")
measure_prim.multiple_results = True


def trivial_processing(results):
    return results


def _get_shapes_for(*measurements, shots=None, num_device_wires=0):
    if jax.config.jax_enable_x64:
        dtype_map = {
            float: jax.numpy.float64,
            int: jax.numpy.int64,
            complex: jax.numpy.complex128,
        }
    else:
        dtype_map = {
            float: jax.numpy.float32,
            int: jax.numpy.int32,
            complex: jax.numpy.complex64,
        }

    shapes = []
    if not shots:
        shots = [None]

    for s in shots:
        for m in measurements:
            shape, dtype = m.abstract_eval(shots=s, num_device_wires=num_device_wires)
            shapes.append(jax.core.ShapedArray(shape, dtype_map.get(dtype, dtype)))
    return shapes


@measure_prim.def_impl
def _(*measurements, shots, num_device_wires):
    # depends on the jax interpreter
    raise NotImplementedError("requires an interpreter to perform a measurement.")


@measure_prim.def_abstract_eval
def _(*measurements, shots, num_device_wires):
    return _get_shapes_for(*measurements, shots=shots, num_device_wires=num_device_wires)


def measure(*measurements, shots=None, num_device_wires=0):
    """Perform a measurement."""
    shots = qml.measurements.Shots(shots)
    return measure_prim.bind(measurements[0], shots=shots, num_device_wires=num_device_wires)
