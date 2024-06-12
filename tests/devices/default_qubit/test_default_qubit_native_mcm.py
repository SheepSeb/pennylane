# Copyright 2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for default qubit preprocessing."""
from functools import reduce
from typing import Iterable, Sequence

import numpy as np
import pytest

import pennylane as qml
from pennylane.devices.qubit.apply_operation import MidMeasureMP, apply_mid_measure
from pennylane.transforms.dynamic_one_shot import fill_in_value

pytestmark = pytest.mark.slow


def get_device(**kwargs):
    kwargs.setdefault("shots", None)
    kwargs.setdefault("seed", 8237945)
    return qml.device("default.qubit", **kwargs)


def validate_counts(shots, results1, results2, batch_size=None):
    """Compares two counts.

    If the results are ``Sequence``s, loop over entries.

    Fails if a key of ``results1`` is not found in ``results2``.
    Passes if counts are too low, chosen as ``100``.
    Otherwise, fails if counts differ by more than ``20`` plus 20 percent.
    """
    if isinstance(shots, Sequence):
        assert isinstance(results1, tuple)
        assert isinstance(results2, tuple)
        assert len(results1) == len(results2) == len(shots)
        for s, r1, r2 in zip(shots, results1, results2):
            validate_counts(s, r1, r2, batch_size=batch_size)
        return

    if batch_size is not None:
        assert isinstance(results1, Iterable)
        assert isinstance(results2, Iterable)
        assert len(results1) == len(results2) == batch_size
        for r1, r2 in zip(results1, results2):
            validate_counts(shots, r1, r2, batch_size=None)
        return

    for key1, val1 in results1.items():
        val2 = results2[key1]
        if abs(val1 + val2) > 100:
            assert np.allclose(val1, val2, atol=20, rtol=0.2)


def validate_samples(shots, results1, results2, batch_size=None):
    """Compares two samples.

    If the results are ``Sequence``s, loop over entries.

    Fails if the results do not have the same shape, within ``20`` entries plus 20 percent.
    This is to handle cases when post-selection yields variable shapes.
    Otherwise, fails if the sums of samples differ by more than ``20`` plus 20 percent.
    """
    if isinstance(shots, Sequence):
        assert isinstance(results1, tuple)
        assert isinstance(results2, tuple)
        assert len(results1) == len(results2) == len(shots)
        for s, r1, r2 in zip(shots, results1, results2):
            validate_samples(s, r1, r2, batch_size=batch_size)
        return

    if batch_size is not None:
        assert isinstance(results1, Iterable)
        assert isinstance(results2, Iterable)
        assert len(results1) == len(results2) == batch_size
        for r1, r2 in zip(results1, results2):
            validate_samples(shots, r1, r2, batch_size=None)
        return

    sh1, sh2 = results1.shape[0], results2.shape[0]
    assert np.allclose(sh1, sh2, atol=20, rtol=0.2)
    assert results1.ndim == results2.ndim
    if results2.ndim > 1:
        assert results1.shape[1] == results2.shape[1]
    np.allclose(qml.math.sum(results1), qml.math.sum(results2), atol=20, rtol=0.2)


def validate_expval(shots, results1, results2, batch_size=None):
    """Compares two expval, probs or var.

    If the results are ``Sequence``s, validate the average of items.

    If ``shots is None``, validate using ``np.allclose``'s default parameters.
    Otherwise, fails if the results do not match within ``0.01`` plus 20 percent.
    """
    if isinstance(shots, Sequence):
        assert isinstance(results1, tuple)
        assert isinstance(results2, tuple)
        assert len(results1) == len(results2) == len(shots)
        results1 = reduce(lambda x, y: x + y, results1) / len(results1)
        results2 = reduce(lambda x, y: x + y, results2) / len(results2)
        validate_expval(sum(shots), results1, results2, batch_size=batch_size)
        return

    if shots is None:
        assert np.allclose(results1, results2)
        return

    if batch_size is not None:
        assert len(results1) == len(results2) == batch_size
        for r1, r2 in zip(results1, results2):
            validate_expval(shots, r1, r2, batch_size=None)

    assert np.allclose(results1, results2, atol=0.01, rtol=0.2)


def validate_measurements(func, shots, results1, results2, batch_size=None):
    """Calls the correct validation function based on measurement type."""
    if func is qml.counts:
        validate_counts(shots, results1, results2, batch_size=batch_size)
        return

    if func is qml.sample:
        validate_samples(shots, results1, results2, batch_size=batch_size)
        return

    validate_expval(shots, results1, results2, batch_size=batch_size)


def test_apply_mid_measure():
    """Test that apply_mid_measure raises if applied to a batched state."""
    with pytest.raises(ValueError, match="MidMeasureMP cannot be applied to batched states."):
        _ = apply_mid_measure(
            MidMeasureMP(0), np.zeros((2, 2)), is_state_batched=True, mid_measurements={}
        )


def test_all_invalid_shots_circuit():
    """Test that circuits in which all shots mismatch with post-selection conditions return the same answer as ``defer_measurements``."""
    dev = get_device()

    @qml.qnode(dev)
    def circuit_op():
        m = qml.measure(0, postselect=1)
        qml.cond(m, qml.PauliX)(1)
        return (
            qml.expval(op=qml.PauliZ(1)),
            qml.probs(op=qml.PauliY(0) @ qml.PauliZ(1)),
            qml.var(op=qml.PauliZ(1)),
        )

    res1 = circuit_op()
    res2 = circuit_op(shots=10)
    for r1, r2 in zip(res1, res2):
        if isinstance(r1, Sequence):
            assert len(r1) == len(r2)
        assert np.all(np.isnan(r1))
        assert np.all(np.isnan(r2))

    @qml.qnode(dev)
    def circuit_mcm():
        m = qml.measure(0, postselect=1)
        qml.cond(m, qml.PauliX)(1)
        return qml.expval(op=m), qml.probs(op=m), qml.var(op=m)

    res1 = circuit_mcm()
    res2 = circuit_mcm(shots=10)
    for r1, r2 in zip(res1, res2):
        if isinstance(r1, Sequence):
            assert len(r1) == len(r2)
        assert np.all(np.isnan(r1))
        assert np.all(np.isnan(r2))


def test_unsupported_measurement():
    """Test that circuits with unsupported measurements raise the correct error."""
    dev = get_device(shots=1000)
    params = np.pi / 4 * np.ones(2)

    @qml.qnode(dev)
    def func(x, y):
        qml.RX(x, wires=0)
        m0 = qml.measure(0)
        qml.cond(m0, qml.RY)(y, wires=1)
        return qml.classical_shadow(wires=0)

    with pytest.raises(
        TypeError,
        match=f"Native mid-circuit measurement mode does not support {type(qml.classical_shadow(wires=0)).__name__}",
    ):
        func(*params)


@pytest.mark.parametrize("postselect_mode", ["hw-like", "fill-shots"])
def test_tree_traversal_postselect_mode(postselect_mode):
    """Test that invalid shots are discarded if requested"""
    shots = 100
    dev = qml.device("default.qubit", shots=shots)

    @qml.qnode(dev, mcm_method="tree-traversal", postselect_mode=postselect_mode)
    def f(x):
        qml.RX(x, 0)
        _ = qml.measure(0, postselect=1)
        return qml.sample(wires=[0, 1])

    res = f(np.pi / 2)

    if postselect_mode == "hw-like":
        assert len(res) < shots
    else:
        assert len(res) == shots
    assert np.all(res != np.iinfo(np.int32).min)


# pylint: disable=unused-argument
def obs_tape(x, y, z, reset=False, postselect=None):
    qml.RX(x, 0)
    qml.RZ(np.pi / 4, 0)
    m0 = qml.measure(0, reset=reset)
    qml.cond(m0 == 0, qml.RX)(np.pi / 4, 0)
    qml.cond(m0 == 0, qml.RZ)(np.pi / 4, 0)
    qml.cond(m0 == 1, qml.RX)(-np.pi / 4, 0)
    qml.cond(m0 == 1, qml.RZ)(-np.pi / 4, 0)
    qml.RX(y, 1)
    qml.RZ(np.pi / 4, 1)
    m1 = qml.measure(1, postselect=postselect)
    qml.cond(m1 == 0, qml.RX)(np.pi / 4, 1)
    qml.cond(m1 == 0, qml.RZ)(np.pi / 4, 1)
    qml.cond(m1 == 1, qml.RX)(-np.pi / 4, 1)
    qml.cond(m1 == 1, qml.RZ)(-np.pi / 4, 1)
    return m0, m1


@pytest.mark.parametrize("mcm_method", ["tree-traversal"])
@pytest.mark.parametrize("shots", [5500, [5500, 5501]])
@pytest.mark.parametrize("postselect", [None])
@pytest.mark.parametrize("measure_f", [qml.counts, qml.expval, qml.probs, qml.sample, qml.var])
@pytest.mark.parametrize(
    "meas_obj",
    [qml.PauliZ(0), qml.PauliY(1), [0], [0, 1], [1, 0], "mcm", "composite_mcm", "mcm_list"],
)
def test_simple_dynamic_circuit(mcm_method, shots, measure_f, postselect, meas_obj):
    """Tests that DefaultQubit handles a simple dynamic circuit with the following measurements:

        * qml.counts with obs (comp basis or not), single wire, multiple wires (ordered/unordered), MCM, f(MCM), MCM list
        * qml.expval with obs (comp basis or not), MCM, f(MCM), MCM list
        * qml.probs with obs (comp basis or not), single wire, multiple wires (ordered/unordered), MCM, f(MCM), MCM list
        * qml.sample with obs (comp basis or not), single wire, multiple wires (ordered/unordered), MCM, f(MCM), MCM list
        * qml.var with obs (comp basis or not), MCM, f(MCM), MCM list

    The above combinations should work for finite shots, shot vectors and post-selecting of either the 0 or 1 branch.
    """

    if measure_f in (qml.expval, qml.var) and (
        isinstance(meas_obj, list) or meas_obj == "mcm_list"
    ):
        pytest.skip("Can't use wires/mcm lists with var or expval")

    dev = get_device(shots=shots)
    params = [np.pi / 2.5, np.pi / 3, -np.pi / 3.5]

    def func(x, y, z):
        m0, m1 = obs_tape(x, y, z, postselect=postselect)
        mid_measure = (
            m0 if meas_obj == "mcm" else (0.5 * m0 if meas_obj == "composite_mcm" else [m0, m1])
        )
        measurement_key = "wires" if isinstance(meas_obj, list) else "op"
        measurement_value = mid_measure if isinstance(meas_obj, str) else meas_obj
        return measure_f(**{measurement_key: measurement_value})

    results0 = qml.QNode(func, dev, mcm_method=mcm_method)(*params)
    results1 = qml.QNode(func, dev, mcm_method="deferred")(*params)

    validate_measurements(measure_f, shots, results1, results0)


@pytest.mark.parametrize("mcm_method", ["tree-traversal"])
@pytest.mark.parametrize("postselect", [None])
@pytest.mark.parametrize("reset", [False, True])
def test_multiple_measurements_and_reset(mcm_method, postselect, reset):
    """Tests that DefaultQubit handles a circuit with a single mid-circuit measurement with reset
    and a conditional gate. Multiple measurements of the mid-circuit measurement value are
    performed. This function also tests `reset` parametrizing over the parameter."""
    shots = 5000
    dev = get_device(shots=shots)
    params = [np.pi / 2.5, np.pi / 3, -np.pi / 3.5]
    obs = qml.PauliY(1)

    def func(x, y, z):
        mcms = obs_tape(x, y, z, reset=reset, postselect=postselect)
        return (
            qml.counts(op=obs),
            qml.expval(op=mcms[0]),
            qml.probs(op=obs),
            qml.sample(op=mcms[0]),
            qml.var(op=obs),
        )

    results0 = qml.QNode(func, dev, mcm_method=mcm_method)(*params)
    results1 = qml.QNode(func, dev, mcm_method="deferred")(*params)

    for measure_f, r1, r0 in zip(
        [qml.counts, qml.expval, qml.probs, qml.sample, qml.var], results1, results0
    ):
        validate_measurements(measure_f, shots, r1, r0)


@pytest.mark.parametrize("mcm_method", ["tree-traversal"])
@pytest.mark.parametrize(
    "mcm_f",
    [
        lambda x: x * -1,
        lambda x: x * 1,
        lambda x: x * 2,
        lambda x: 1 - x,
        lambda x: x + 1,
        lambda x: x & 3,
        "mix",
        "list",
    ],
)
@pytest.mark.parametrize("measure_f", [qml.counts, qml.expval, qml.probs, qml.sample, qml.var])
def test_composite_mcms(mcm_method, mcm_f, measure_f):
    """Tests that DefaultQubit handles a circuit with a composite mid-circuit measurement and a
    conditional gate. A single measurement of a composite mid-circuit measurement is performed
    at the end."""

    if measure_f in (qml.expval, qml.var) and (mcm_f in ("list", "mix")):
        pytest.skip(
            "expval/var does not support measuring sequences of measurements or observables."
        )

    if measure_f == qml.probs and mcm_f == "mix":
        pytest.skip(
            "Cannot use qml.probs() when measuring multiple mid-circuit measurements collected using arithmetic operators."
        )

    shots = 3000

    dev = get_device(shots=shots)
    param = np.pi / 3

    def func(x):
        qml.RX(x, 0)
        m0 = qml.measure(0)
        qml.RX(0.5 * x, 1)
        m1 = qml.measure(1)
        qml.cond((m0 + m1) == 2, qml.RY)(2.0 * x, 0)
        m2 = qml.measure(0)
        obs = (
            (m0 - 2 * m1) * m2 + 7
            if mcm_f == "mix"
            else ([m0, m1, m2] if mcm_f == "list" else mcm_f(m2))
        )
        return measure_f(op=obs)

    results0 = qml.QNode(func, dev, mcm_method=mcm_method)(param)
    results1 = qml.QNode(func, dev, mcm_method="deferred")(param)

    validate_measurements(measure_f, shots, results1, results0)


@pytest.mark.parametrize("mcm_method", ["tree-traversal"])
@pytest.mark.parametrize(
    "mcm_f",
    [
        lambda x, y: x + y,
        lambda x, y: x - 7 * y,
        lambda x, y: x & y,
        lambda x, y: x == y,
        lambda x, y: 4.0 * x + 2.0 * y,
    ],
)
def test_counts_return_type(mcm_method, mcm_f):
    """Tests that DefaultQubit returns the same keys for ``qml.counts`` measurements with ``dynamic_one_shot`` and ``defer_measurements``."""
    shots = 500

    dev = get_device(shots=shots)
    param = np.pi / 3

    def func(x):
        qml.RX(x, 0)
        m0 = qml.measure(0)
        qml.RX(0.5 * x, 1)
        m1 = qml.measure(1)
        qml.cond((m0 + m1) == 2, qml.RY)(2.0 * x, 0)
        return qml.counts(op=mcm_f(m0, m1))

    results0 = qml.QNode(func, dev, mcm_method=mcm_method)(param)
    results1 = qml.QNode(func, dev, mcm_method="deferred")(param)

    for r1, r0 in zip(results1.keys(), results0.keys()):
        assert r1 == r0


@pytest.mark.parametrize("shots", [5000])
@pytest.mark.parametrize("postselect", [None])
@pytest.mark.parametrize("reset", [False, True])
@pytest.mark.parametrize("measure_f", [qml.expval])
def composite_mcm_gradient_measure_obs(shots, postselect, reset, measure_f):
    """Tests that DefaultQubit can differentiate a circuit with a composite mid-circuit
    measurement and a conditional gate. A single measurement of a common observable is
    performed at the end."""

    dev = get_device(shots=shots)
    param = qml.numpy.array([np.pi / 3, np.pi / 6])
    obs = qml.PauliZ(0) @ qml.PauliZ(1)

    @qml.qnode(dev, diff_method="parameter-shift")
    def func(x, y):
        qml.RX(x, 0)
        m0 = qml.measure(0)
        qml.RX(y, 1)
        m1 = qml.measure(1, reset=reset, postselect=postselect)
        qml.cond((m0 + m1) == 2, qml.RY)(2 * np.pi / 3, 0)
        qml.cond((m0 + m1) > 0, qml.RY)(2 * np.pi / 3, 1)
        return measure_f(op=obs)

    func1 = func
    func2 = qml.defer_measurements(func)

    results1 = func1(*param)
    results2 = func2(*param)

    validate_measurements(measure_f, shots, results1, results2)

    grad1 = qml.grad(func)(*param)
    grad2 = qml.grad(func2)(*param)

    assert np.allclose(grad1, grad2, atol=0.01, rtol=0.3)


@pytest.mark.parametrize("shots", [5000, [5000, 5001]])
@pytest.mark.parametrize("postselect", [None])
@pytest.mark.parametrize("measure_fn", [qml.expval, qml.sample, qml.probs, qml.counts])
def test_broadcasting_qnode(shots, postselect, measure_fn):
    """Test that executing qnodes with broadcasting works as expected"""
    if measure_fn is qml.sample and postselect is not None:
        pytest.skip("Postselection with samples doesn't work with broadcasting")

    dev = get_device(shots=shots)
    param = [[np.pi / 3, np.pi / 4], [np.pi / 6, 2 * np.pi / 3]]
    obs = qml.PauliZ(0) @ qml.PauliZ(1)

    @qml.qnode(dev)
    def func(x, y):
        obs_tape(x, y, None, postselect=postselect)
        return measure_fn(op=obs)

    func1 = func
    func2 = qml.defer_measurements(func)

    results1 = func1(*param)
    results2 = func2(*param)

    validate_measurements(measure_fn, shots, results1, results2, batch_size=2)

    if measure_fn is qml.sample and postselect is None:
        for i in range(2):  # batch_size
            if isinstance(shots, list):
                for s, r1, r2 in zip(shots, results1, results2):
                    assert len(r1[i]) == len(r2[i]) == s
            else:
                assert len(results1[i]) == len(results2[i]) == shots


def test_sample_with_broadcasting_and_postselection_error():
    """Test that an error is raised if returning qml.sample if postselecting with broadcasting"""
    tape = qml.tape.QuantumScript(
        [qml.RX([0.1, 0.2], 0), MidMeasureMP(0, postselect=1)], [qml.sample(wires=0)], shots=10
    )
    with pytest.raises(ValueError, match="Returning qml.sample is not supported when"):
        qml.transforms.dynamic_one_shot(tape)

    dev = get_device(shots=10)

    @qml.qnode(dev)
    def circuit():
        qml.RX([0.1, 0.2], 0)
        qml.measure(0, postselect=1)
        return qml.sample(wires=0)

    with pytest.raises(ValueError, match="Returning qml.sample is not supported when"):
        _ = circuit()


# pylint: disable=not-an-iterable
@pytest.mark.jax
@pytest.mark.parametrize("shots", [100, [100, 101]])
@pytest.mark.parametrize("postselect", [None])
def test_sample_with_prng_key(shots, postselect):
    """Test that setting a PRNGKey gives the expected behaviour. With separate calls
    to DefaultQubit.execute, the same results are expected when using a PRNGKey"""
    # pylint: disable=import-outside-toplevel
    from jax.random import PRNGKey

    dev = get_device(shots=shots, seed=PRNGKey(678))
    param = [np.pi / 4, np.pi / 3]
    obs = qml.PauliZ(0) @ qml.PauliZ(1)

    @qml.qnode(dev)
    def func(x, y):
        obs_tape(x, y, None, postselect=postselect)
        return qml.sample(op=obs)

    func1 = func
    func2 = qml.defer_measurements(func)

    results1 = func1(*param)
    results2 = func2(*param)

    validate_measurements(qml.sample, shots, results1, results2, batch_size=None)

    evals = obs.eigvals()
    for eig in evals:
        # When comparing with the results from a circuit with deferred measurements
        # we're not always expected to have the functions used to sample are different
        if isinstance(shots, list):
            for r in results1:
                assert not np.all(np.isclose(r, eig))
        else:
            assert not np.all(np.isclose(results1, eig))

    results3 = func1(*param)
    # Same result expected with multiple executions
    if isinstance(shots, list):
        for r1, r3 in zip(results1, results3):
            assert np.allclose(r1, r3)
    else:
        assert np.allclose(results1, results3)


# pylint: disable=import-outside-toplevel, not-an-iterable
@pytest.mark.jax
@pytest.mark.parametrize("diff_method", [None, "best"])
@pytest.mark.parametrize("postselect", [None, 1])
@pytest.mark.parametrize("reset", [False, True])
def test_jax_jit(diff_method, postselect, reset):
    """Tests that DefaultQubit handles a circuit with a single mid-circuit measurement and a
    conditional gate. A single measurement of a common observable is performed at the end."""
    import jax

    shots = 10

    dev = get_device(shots=shots, seed=jax.random.PRNGKey(678))
    params = [np.pi / 2.5, np.pi / 3, -np.pi / 3.5]
    obs = qml.PauliY(0)

    @qml.qnode(dev, diff_method=diff_method)
    def func(x, y, z):
        m0, m1 = obs_tape(x, y, z, reset=reset, postselect=postselect)
        return (
            qml.probs(wires=[1]),
            qml.probs(wires=[0, 1]),
            qml.sample(wires=[1]),
            qml.sample(wires=[0, 1]),
            qml.expval(obs),
            qml.probs(obs),
            qml.sample(obs),
            qml.var(obs),
            qml.expval(op=m0 + 2 * m1),
            qml.probs(op=m0),
            qml.sample(op=m0 + 2 * m1),
            qml.var(op=m0 + 2 * m1),
            qml.probs(op=[m0, m1]),
        )

    func1 = func
    results1 = func1(*params)

    jaxpr = str(jax.make_jaxpr(func)(*params))
    if diff_method == "best":
        assert "pure_callback" in jaxpr
        pytest.xfail("QNode with diff_method='best' cannot be compiled with jax.jit.")
    else:
        assert "pure_callback" not in jaxpr

    func2 = jax.jit(func)
    results2 = func2(*params)

    measures = [
        qml.probs,
        qml.probs,
        qml.sample,
        qml.sample,
        qml.expval,
        qml.probs,
        qml.sample,
        qml.var,
        qml.expval,
        qml.probs,
        qml.sample,
        qml.var,
        qml.probs,
    ]
    for measure_f, r1, r2 in zip(measures, results1, results2):
        r1, r2 = np.array(r1).ravel(), np.array(r2).ravel()
        if measure_f == qml.sample:
            r2 = r2[r2 != fill_in_value]
        np.allclose(r1, r2)


@pytest.mark.torch
@pytest.mark.parametrize("postselect", [None, 1])
@pytest.mark.parametrize("diff_method", [None, "best"])
@pytest.mark.parametrize("measure_f", [qml.probs, qml.sample, qml.expval, qml.var])
@pytest.mark.parametrize("meas_obj", [qml.PauliZ(1), [0, 1], "composite_mcm", "mcm_list"])
def test_torch_integration(postselect, diff_method, measure_f, meas_obj):
    """Test that native MCM circuits are executed correctly with Torch"""
    if measure_f in (qml.expval, qml.var) and (
        isinstance(meas_obj, list) or meas_obj == "mcm_list"
    ):
        pytest.skip("Can't use wires/mcm lists with var or expval")

    import torch

    shots = 7000
    dev = get_device(shots=shots, seed=123456789)
    param = torch.tensor(np.pi / 3, dtype=torch.float64)

    @qml.qnode(dev, diff_method=diff_method)
    def func(x):
        qml.RX(x, 0)
        m0 = qml.measure(0)
        qml.RX(0.5 * x, 1)
        m1 = qml.measure(1, postselect=postselect)
        qml.cond((m0 + m1) == 2, qml.RY)(2.0 * x, 0)
        m2 = qml.measure(0)

        mid_measure = 0.5 * m2 if meas_obj == "composite_mcm" else [m1, m2]
        measurement_key = "wires" if isinstance(meas_obj, list) else "op"
        measurement_value = mid_measure if isinstance(meas_obj, str) else meas_obj
        return measure_f(**{measurement_key: measurement_value})

    func1 = func
    func2 = qml.defer_measurements(func)

    results1 = func1(param)
    results2 = func2(param)

    validate_measurements(measure_f, shots, results1, results2)
