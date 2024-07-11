# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Riemannian gradient optimizer"""
import warnings

import numpy as np
from scipy.sparse.linalg import expm

import pennylane as qml
from pennylane import transform
from pennylane.queuing import QueuingManager
from pennylane.tape import QuantumTape, TapeBatch
from pennylane.typing import PostprocessingFn


@transform
def append_time_evolution(
    tape: QuantumTape, riemannian_gradient, t, n, exact=False
) -> tuple[TapeBatch, PostprocessingFn]:
    r"""Append an approximate time evolution, corresponding to a Riemannian
    gradient on the Lie group, to an existing circuit.

    We want to implement the time evolution generated by an operator of the form

    .. math::

        \text{grad} f(U) = sum_i c_i O_i,

    where :math:`O_i` are Pauli words and :math:`c_t \in \mathbb{R}`.
    If ``exact`` is ``False``, we Trotterize this operator and apply the unitary

    .. math::

        U' = \prod_{n=1}^{N_{Trot.}} \left(\prod_i \exp{-it / N_{Trot.} O_i}\right),

    which is then appended to the current circuit.

    If ``exact`` is ``True``, we calculate the exact time evolution for the Riemannian gradient
     by way of the matrix exponential.

    .. math:

        U' = \exp{-it \text{grad} f(U)}

    and append this unitary.

    Args:
        tape (QuantumTape or QNode or Callable): circuit to transform.
        riemannian_gradient (.Hamiltonian): Hamiltonian object representing the Riemannian gradient.
        t (float): time evolution parameter.
        n (int): number of Trotter steps.

    Returns:
        qnode (QNode) or quantum function (Callable) or tuple[List[QuantumTape], function]: The transformed circuit as described in :func:`qml.transform <pennylane.transform>`.


    """
    new_operations = tape.operations
    if exact:
        with QueuingManager.stop_recording():
            new_operations.append(
                qml.QubitUnitary(
                    expm(-1j * t * riemannian_gradient.sparse_matrix(tape.wires).toarray()),
                    wires=range(len(riemannian_gradient.wires)),
                )
            )
    else:
        with QueuingManager.stop_recording():
            new_operations.append(qml.templates.ApproxTimeEvolution(riemannian_gradient, t, n))

    new_tape = type(tape)(new_operations, tape.measurements, shots=tape.shots)

    def null_postprocessing(results):
        """A postprocesing function returned by a transform that only converts the batch of results
        into a result for a single ``QuantumTape``.
        """
        return results[0]  # pragma: no cover

    return [new_tape], null_postprocessing


def algebra_commutator(tape, observables, lie_algebra_basis_names, nqubits):
    """Calculate the Riemannian gradient in the Lie algebra with the parameter shift rule
    (see :meth:`RiemannianGradientOptimizer.get_omegas`).

    Args:
        tape (.QuantumTape or .QNode): input circuit
        observables (list[.Observable]): list of observables to be measured. Can be grouped.
        lie_algebra_basis_names (list[str]): List of strings corresponding to valid Pauli words.
        nqubits (int): the number of qubits.

    Returns:
        function or tuple[list[QuantumTape], function]:

        - If the input is a QNode, an object representing the Riemannian gradient function
          of the QNode that can be executed with the same arguments as the QNode to obtain
          the Lie algebra commutator.

        - If the input is a tape, a tuple containing a
          list of generated tapes, together with a post-processing
          function to be applied to the results of the evaluated tapes
          in order to obtain the Lie algebra commutator.
    """
    tapes_plus_total = []
    tapes_min_total = []
    for obs in observables:
        for o in obs:
            # create a list of tapes for the plus and minus shifted circuits
            queues_plus = [qml.queuing.AnnotatedQueue() for _ in lie_algebra_basis_names]
            queues_min = [qml.queuing.AnnotatedQueue() for _ in lie_algebra_basis_names]

            # loop through all operations on the input tape
            for op in tape.operations:
                for t in queues_plus + queues_min:
                    with t:
                        qml.apply(op)
            for i, t in enumerate(queues_plus):
                with t:
                    qml.PauliRot(
                        np.pi / 2,
                        lie_algebra_basis_names[i],
                        wires=list(range(nqubits)),
                    )
                    qml.expval(o)
            for i, t in enumerate(queues_min):
                with t:
                    qml.PauliRot(
                        -np.pi / 2,
                        lie_algebra_basis_names[i],
                        wires=list(range(nqubits)),
                    )
                    qml.expval(o)
            tapes_plus_total.extend(
                [
                    qml.tape.QuantumScript(*qml.queuing.process_queue(q))
                    for q, p in zip(queues_plus, lie_algebra_basis_names)
                ]
            )
            tapes_min_total.extend(
                [
                    qml.tape.QuantumScript(*qml.queuing.process_queue(q))
                    for q, p in zip(queues_min, lie_algebra_basis_names)
                ]
            )
    return tapes_plus_total + tapes_min_total


class RiemannianGradientOptimizer:
    r"""Riemannian gradient optimizer.

    Riemannian gradient descent algorithms can be used to optimize a function directly on a Lie group
    as opposed to on an Euclidean parameter space. Consider the function
    :math:`f(U) = \text{Tr}(U \rho_0 U^\dagger H)`
    for a given Hamiltonian :math:`H`, unitary :math:`U\in \text{SU}(2^N)` and initial state
    :math:`\rho_0`. One can show that this function is minimized by the flow equation

    .. math::

        \dot{U} = \text{grad}f(U)

    where :math:`\text{grad}` is the Riemannian gradient operator on :math:`\text{SU}(2^N)`.
    By discretizing the flow above, we see that a step of this optimizer
    iterates the Riemannian gradient flow on :math:`\text{SU}(2^N)` as

    .. math::

        U^{(t+1)} = \exp\left\{\epsilon\: \text{grad}f(U^{(t)}) U^{(t)}\right\},


    where :math:`\epsilon` is a user-defined hyperparameter corresponding to the step size.

    The Riemannian gradient in the Lie algebra is given by

    .. math::

        \text{grad}f(U^{(t)}) = -\left[U \rho U^\dagger, H\right] .


    Hence we see that subsequent steps of this optimizer will append the unitary generated by the Riemannian
    gradient and grow the circuit.

    The exact Riemannian gradient flow on :math:`\text{SU}(2^N)` has desirable optimization properties
    that can guarantee convergence to global minima under mild assumptions. However, this comes
    at a cost. Since :math:`\text{dim}(\text{SU}(2^N)) = 4^N-1`, we need an exponential number
    of parameters to calculate the gradient. This will not be problematic for small systems (:math:`N<5`),
    but will quickly get out of control as the number of qubits increases.

    To resolve this issue, we can restrict the Riemannian gradient to a subspace of the Lie algebra and calculate an
    approximate Riemannian gradient flow. The choice of restriction will affect the optimization behaviour
    and quality of the final solution.

    For more information on Riemannian gradient flows on Lie groups see
    `T. Schulte-Herbrueggen et. al. (2008) <https://arxiv.org/abs/0802.4195>`_
    and the application to quantum circuits
    `Wiersema and Killoran (2022) <https://arxiv.org/abs/2202.06976>`_.

    Args:
        circuit (.QNode): a user defined circuit that does not take any arguments and returns
            the expectation value of a ``qml.Hamiltonian``.
        stepsize (float): the user-defined hyperparameter :math:`\epsilon`.
        restriction (.Hamiltonian): Restrict the Lie algebra to a corresponding subspace of
            the full Lie algebra. This restriction should be passed in the form of a
            ``qml.Hamiltonian`` that consists only of Pauli words.
        exact (bool): Flag that indicates wether we approximate the Riemannian gradient with a
            Trotterization or calculate the exact evolution via a matrix exponential. The latter is
            not hardware friendly and can only be done in simulation.

    **Examples**

    Define a Hamiltonian cost function to minimize:

    >>> coeffs = [-1., -1., -1.]
    >>> observables = [qml.X(0), qml.Z(1), qml.Y(0) @ qml.X(1)]
    >>> hamiltonian = qml.Hamiltonian(coeffs, observables)

    Create an initial state and return the expectation value of the Hamiltonian:

    >>> @qml.qnode(qml.device("default.qubit", wires=2))
    ... def quant_fun():
    ...     qml.RX(0.1, wires=[0])
    ...     qml.RY(0.5, wires=[1])
    ...     qml.CNOT(wires=[0,1])
    ...     qml.RY(0.6, wires=[0])
    ...     return qml.expval(hamiltonian)

    Instantiate the optimizer with the initial circuit and the cost function and set the stepsize
    accordingly:

    >>> opt = qml.RiemannianGradientOptimizer(circuit=quant_fun, stepsize=0.1)

    Applying 5 steps gets us close the ground state of :math:`E\approx-2.23`:

    >>> for step in range(6):
    ...    circuit, cost = opt.step_and_cost()
    ...    print(f"Step {step} - cost {cost}")
    Step 0 - cost -1.3351865007304005
    Step 1 - cost -1.9937887238935206
    Step 2 - cost -2.1524234485729834
    Step 3 - cost -2.1955105378898487
    Step 4 - cost -2.2137628169764256
    Step 5 - cost -2.2234364822091575

    The optimized circuit is returned at each step, and can be used as any other QNode:

    >>> circuit()
    -2.2283086057521713

    """

    # pylint: disable=too-many-arguments
    # pylint: disable=too-many-instance-attributes
    def __init__(self, circuit, stepsize=0.01, restriction=None, exact=False, trottersteps=1):
        if not isinstance(circuit, qml.QNode):
            raise TypeError(f"circuit must be a QNode, received {type(circuit)}")

        self.circuit = circuit
        self.circuit.construct([], {})
        self.hamiltonian = circuit.func().obs
        if not isinstance(self.hamiltonian, (qml.ops.Hamiltonian, qml.ops.LinearCombination)):
            raise TypeError(
                f"circuit must return the expectation value of a Hamiltonian,"
                f"received {type(circuit.func().obs)}"
            )
        self.nqubits = len(circuit.device.wires)

        if self.nqubits > 4:
            warnings.warn(
                "The exact Riemannian gradient is exponentially expensive in the number of qubits, "
                f"optimizing a {self.nqubits} qubit circuit may be slow.",
                UserWarning,
            )
        if restriction is not None and not isinstance(
            restriction, (qml.ops.Hamiltonian, qml.ops.LinearCombination)
        ):
            raise TypeError(f"restriction must be a Hamiltonian, received {type(restriction)}")
        (
            self.lie_algebra_basis_ops,
            self.lie_algebra_basis_names,
        ) = self.get_su_n_operators(restriction)
        self.exact = exact
        self.trottersteps = trottersteps
        self.coeffs, self.observables = self.hamiltonian.terms()
        self.stepsize = stepsize

    def step(self):
        r"""Update the circuit with one step of the optimizer.

        Returns:
           float: the optimized circuit and the objective function output prior
           to the step.
        """
        return self.step_and_cost()[0]

    def step_and_cost(self):
        r"""Update the circuit with one step of the optimizer and return the corresponding
        objective function value prior to the step.

        Returns:
            tuple[.QNode, float]: the optimized circuit and the objective function output prior
            to the step.
        """
        # pylint: disable=not-callable

        cost = self.circuit()
        omegas = self.get_omegas()
        non_zero_omegas = -omegas[omegas != 0]

        nonzero_idx = np.nonzero(omegas)[0]
        non_zero_lie_algebra_elements = [self.lie_algebra_basis_names[i] for i in nonzero_idx]

        lie_gradient = qml.Hamiltonian(
            non_zero_omegas,
            [qml.pauli.string_to_pauli_word(ps) for ps in non_zero_lie_algebra_elements],
        )
        new_circuit = append_time_evolution(
            self.circuit.func, lie_gradient, self.stepsize, self.trottersteps, self.exact
        )

        # we can set diff_method=None because the gradient of the QNode is computed
        # directly in this optimizer
        self.circuit = qml.QNode(new_circuit, self.circuit.device, diff_method=None)
        return self.circuit, cost

    def get_su_n_operators(self, restriction):
        r"""Get the SU(N) operators. The dimension of the group is :math:`N^2-1`.

        Args:
            restriction (.Hamiltonian): Restrict the Riemannian gradient to a subspace.

        Returns:
            tuple[list[array[complex]], list[str]]: list of :math:`N^2 \times N^2` NumPy complex arrays and corresponding Pauli words.
        """

        operators = []
        names = []
        # construct the corresponding pennylane observables
        wire_map = dict(zip(range(self.nqubits), range(self.nqubits)))
        if restriction is None:
            for ps in qml.pauli.pauli_group(self.nqubits):
                operators.append(ps)
                names.append(qml.pauli.pauli_word_to_string(ps, wire_map=wire_map))
        else:
            for ps in set(restriction.ops):
                operators.append(ps)
                names.append(qml.pauli.pauli_word_to_string(ps, wire_map=wire_map))

        return operators, names

    def get_omegas(self):
        r"""Measure the coefficients of the Riemannian gradient with respect to a Pauli word basis.
        We want to calculate the components of the Riemannian gradient in the Lie algebra
        with respect to a Pauli word basis. For a Hamiltonian of the form :math:`H = \sum_i c_i O_i`,
        where :math:`c_i\in\mathbb{R}`, this can be achieved by calculating

        .. math::

            \omega_{i,j} = \text{Tr}(c_i[\rho, O_i] P_j)

        where :math:`P_j` is a Pauli word in the set of Pauli monomials on :math:`N` qubits.

        Via the parameter shift rule, the commutator can be calculated as

        .. math::

            [\rho, O_i] = \frac{1}{2}(V(\pi/2) \rho V^\dagger(\pi/2) - V(-\pi/2) \rho V^\dagger(-\pi/2))

        where :math:`V` is the unitary generated by the Pauli word :math:`V(\theta) = \exp\{-i\theta P_j\}`.

        Returns:
            array: array of omegas for each direction in the Lie algebra.
        """

        obs_groupings, _ = qml.pauli.group_observables(self.observables, self.coeffs)
        # get all circuits we need to calculate the coefficients
        circuits = algebra_commutator(
            self.circuit.qtape,
            obs_groupings,
            self.lie_algebra_basis_names,
            self.nqubits,
        )

        if isinstance(self.circuit.device, qml.devices.Device):
            program, config = self.circuit.device.preprocess()

            circuits = qml.execute(
                circuits,
                self.circuit.device,
                transform_program=program,
                config=config,
                gradient_fn=None,
            )
        else:
            circuits = qml.execute(
                circuits, self.circuit.device, gradient_fn=None
            )  # pragma: no cover

        program, _ = self.circuit.device.preprocess()

        circuits_plus = np.array(circuits[: len(circuits) // 2]).reshape(
            len(self.coeffs), len(self.lie_algebra_basis_names)
        )
        circuits_min = np.array(circuits[len(circuits) // 2 :]).reshape(
            len(self.coeffs), len(self.lie_algebra_basis_names)
        )

        # For each observable O_i in the Hamiltonian, we have to calculate all Lie coefficients
        omegas = circuits_plus - circuits_min

        return np.dot(self.coeffs, omegas)
