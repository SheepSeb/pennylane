import jax

import pennylane as qml

### The measure primitive ###############

measure_prim = jax.core.Primitive("measure")
measure_prim.multiple_results = True


def trivial_processing(results):
    return results


@measure_prim.def_impl
def _(*measurements, shots, num_device_wires):
    # depends on the jax interpreter
    if not all(isinstance(m, qml.measurements.MidMeasureMP) for m in measurements):
        raise NotImplementedError("requires an interpreter to perform a measurement.")
    return qml.measurements.MeasurementValue(measurements, trivial_processing)


@measure_prim.def_abstract_eval
def _(*measurements, shots, num_device_wires):

    if not shots.has_partitioned_shots:
        kwargs = {"shots": shots.total_shots, "num_device_wires": num_device_wires}
        return tuple(m.abstract_eval(n_wires=m.n_wires, **kwargs) for m in measurements)
    vals = []
    for s in shots:
        v = tuple(
            m.abstract_eval(n_wires=m.n_wires, shots=s, num_device_wires=num_device_wires)
            for m in measurements
        )
        vals.extend(v)
    return vals


def measure(*measurements, shots=None, num_device_wires=0):
    """Perform a measurement."""
    shots = qml.measurements.Shots(shots)
    return measure_prim.bind(*measurements, shots=shots, num_device_wires=num_device_wires)
