# Copyright 2018-2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Defines a LegacyDeviceFacade class for converting legacy devices to the
new interface.
"""
# pylint: disable=not-callable
from contextlib import contextmanager
from dataclasses import replace

import pennylane as qml
from pennylane.measurements import Shots
from pennylane.transforms.core.transform_program import TransformProgram

from .default_qubit import adjoint_observables, adjoint_ops
from .device_api import Device
from .execution_config import DefaultExecutionConfig
from .modifiers import single_tape_support
from .preprocess import (
    decompose,
    no_sampling,
    validate_adjoint_trainable_params,
    validate_measurements,
    validate_observables,
)


@contextmanager
def _set_shots(device, shots):
    """Context manager to temporarily change the shots
    of a device.

    This context manager can be used in two ways.

    As a standard context manager:

    >>> dev = qml.device("default.qubit.legacy", wires=2, shots=None)
    >>> with _set_shots(dev, shots=100):
    ...     print(dev.shots)
    100
    >>> print(dev.shots)
    None

    Or as a decorator that acts on a function that uses the device:

    >>> _set_shots(dev, shots=100)(lambda: dev.shots)()
    100
    """
    # note, this function duplicates qml.workflow.set_shots
    # duplicated here to avoid circular dependency issues
    # qml.workflow.set_shots can be independently deprecated soon
    # this version of the function is private to LegacyDeviceFacade
    shots = qml.measurements.Shots(shots)
    shots = shots.shot_vector if shots.has_partitioned_shots else shots.total_shots
    if shots == device.shots:
        yield
        return

    original_shots = device.shots
    original_shot_vector = device._shot_vector  # pylint: disable=protected-access

    try:
        device.shots = shots
        yield
    finally:
        device.shots = original_shots
        device._shot_vector = original_shot_vector  # pylint: disable=protected-access


def null_postprocessing(results):
    """A postprocessing function with null behavior."""
    return results[0]


@qml.transform
def legacy_device_expand_fn(tape, device):
    """Turn the ``expand_fn`` from the legacy device interface into a transform."""
    new_tape = _set_shots(device, tape.shots)(device.expand_fn)(tape)
    return (new_tape,), null_postprocessing


@qml.transform
def legacy_device_batch_transform(tape, device):
    """Turn the ``batch_transform`` from the legacy device interface into a transform."""
    return _set_shots(device, tape.shots)(device.batch_transform)(tape)


def _add_adjoint_transforms(program: TransformProgram, name="adjoint"):
    """Add the adjoint specific transforms to the transform program."""
    program.add_transform(no_sampling, name=name)
    program.add_transform(
        decompose,
        stopping_condition=adjoint_ops,
        name=name,
    )
    program.add_transform(validate_observables, adjoint_observables, name=name)
    program.add_transform(
        validate_measurements,
        name=name,
    )
    program.add_transform(qml.transforms.broadcast_expand)
    program.add_transform(validate_adjoint_trainable_params)


@single_tape_support
class LegacyDeviceFacade(Device):
    """
    A Facade that converts a device from the old ``qml.Device`` interface into the new interface.

    Args:
        device (qml.device.LegacyDevice): a device that follows the legacy device interface.

    >>> from pennylane.devices import DefaultMixed, LegacyDeviceFacade
    >>> legacy_dev = DefaultMixed(wires=2)
    >>> new_dev = LegacyDeviceFacade(dev)
    >>> new_dev.preprocess()
    (TransformProgram(legacy_device_batch_transform, legacy_device_expand_fn, defer_measurements),
    ExecutionConfig(grad_on_execution=None, use_device_gradient=None, use_device_jacobian_product=None,
    gradient_method=None, gradient_keyword_arguments={}, device_options={}, interface=None,
    derivative_order=1, mcm_config=MCMConfig(mcm_method=None, postselect_mode=None)))
    >>> new_dev.shots
    Shots(total_shots=None, shot_vector=())
    >>> tape = qml.tape.QuantumScript([], [qml.sample(wires=0)], shots=5)
    >>> new_dev.execute(tape)
    array([0., 0., 0., 0., 0.])

    """

    # pylint: disable=super-init-not-called
    def __init__(self, device: "qml.devices.LegacyDevice"):
        if not isinstance(device, qml.devices.LegacyDevice):
            raise ValueError(
                "The LegacyDeviceFacade only accepts a device of type qml.devices.LegacyDevice."
            )
        self._device = device

    @property
    def tracker(self):
        """A :class:`~.Tracker` that can store information about device executions, shots, batches,
        intermediate results, or any additional device dependent information.
        """
        return self._device.tracker

    @tracker.setter
    def tracker(self, new_tracker):
        self._device.tracker = new_tracker

    @property
    def name(self) -> str:
        return self._device.short_name

    def __repr__(self):
        return f"<LegacyDeviceFacade: {repr(self._device)}>"

    def __getattr__(self, name):
        return getattr(self._device, name)

    @property
    def target_device(self) -> "qml._device.Device":
        """The device wrapped by the facade."""
        return self._device

    @property
    def wires(self) -> qml.wires.Wires:
        return self._device.wires

    # pylint: disable=protected-access
    @property
    def shots(self) -> Shots:
        if self._device._shot_vector:
            return Shots(self._device._raw_shot_sequence)
        return Shots(self._device.shots)

    @property
    def _debugger(self):
        return self._device._debugger

    @_debugger.setter
    def _debugger(self, new_debugger):
        self._device._debugger = new_debugger

    def preprocess(self, execution_config=DefaultExecutionConfig):
        execution_config = self._setup_execution_config(execution_config)
        program = qml.transforms.core.TransformProgram()
        # note: need to wrap these methods with a set_shots modifier
        program.add_transform(legacy_device_batch_transform, device=self._device)
        program.add_transform(legacy_device_expand_fn, device=self._device)
        if execution_config.gradient_method == "adjoint":
            _add_adjoint_transforms(program, name=f"{self.name} + adjoint")

        if not self._device.capabilities().get("supports_mid_measure", False):
            program.add_transform(qml.defer_measurements, device=self)

        return program, execution_config

    def _setup_backprop_config(self, execution_config):
        tape = qml.tape.QuantumScript()
        if not self._validate_backprop_method(tape):
            raise qml.DeviceError("device does not support backprop.")
        if execution_config.use_device_gradient is None:
            return replace(execution_config, use_device_gradient=True)
        return execution_config

    def _setup_adjoint_config(self, execution_config):
        tape = qml.tape.QuantumScript([], [])
        if not self._validate_adjoint_method(tape):
            raise qml.DeviceError("device does not support device derivatives")
        updated_values = {
            "gradient_keyword_arguments": {"use_device_state": True, "method": "adjoint_jacobian"}
        }

        if execution_config.use_device_gradient is None:
            updated_values["use_device_gradient"] = True
        if execution_config.grad_on_execution is None:
            updated_values["grad_on_execution"] = True
        return replace(execution_config, **updated_values)

    def _setup_device_config(self, execution_config):
        tape = qml.tape.QuantumScript([], [])
        if not self._validate_device_method(tape):
            raise qml.DeviceError("device does not support device derivatives")
        updated_values = {}
        if execution_config.use_device_gradient is None:
            updated_values["use_device_gradient"] = True
        if execution_config.grad_on_execution is None:
            updated_values["grad_on_execution"] = True
        return replace(execution_config, **updated_values)

    # pylint: disable=too-many-return-statements
    def _setup_execution_config(self, execution_config):
        if execution_config.gradient_method == "best":
            tape = qml.tape.QuantumScript([], [])
            if self._validate_backprop_method(tape):
                config = replace(execution_config, gradient_method="backprop")
                return self._setup_backprop_config(config)
            if self._validate_adjoint_method(tape):
                config = replace(execution_config, gradient_method="adjoint")
                return self._setup_adjoint_config(config)
            if self._validate_device_method(tape):
                config = replace(execution_config, gradient_method="device")
                return self._setup_execution_config(config)

        if execution_config.gradient_method == "backprop":
            return self._setup_backprop_config(execution_config)
        if execution_config.gradient_method == "adjoint":
            return self._setup_adjoint_config(execution_config)
        if execution_config.gradient_method == "device":
            return self._setup_device_config(execution_config)

        return execution_config

    def supports_derivatives(self, execution_config=None, circuit=None) -> bool:
        circuit = qml.tape.QuantumScript([], [], shots=self.shots) if circuit is None else circuit

        if execution_config is None or execution_config.gradient_method == "best":
            validation_methods = (
                self._validate_backprop_method,
                self._validate_adjoint_method,
                self._validate_device_method,
            )
            return any(validate(circuit) for validate in validation_methods)

        if execution_config.gradient_method == "backprop":
            return self._validate_backprop_method(circuit)
        if execution_config.gradient_method == "adjoint":
            return self._validate_adjoint_method(circuit)
        if execution_config.gradient_method == "device":
            return self._validate_device_method(circuit)
        return False

    # pylint: disable=protected-access
    def _create_temp_device(self, batch):
        """Create a temporary device for use in a backprop execution."""
        params = []
        for t in batch:
            params.extend(t.get_parameters(trainable_only=False))
        interface = qml.math.get_interface(*params)
        if interface == "numpy":
            return self._device

        mapped_interface = qml.workflow.execution.INTERFACE_MAP.get(interface, interface)

        backprop_interface = self._device.capabilities().get("passthru_interface", None)
        if mapped_interface == backprop_interface:
            return self._device

        backprop_devices = self._device.capabilities().get("passthru_devices", None)

        if backprop_devices is None:
            raise qml.DeviceError(f"Device {self} does not support backpropagation.")

        if backprop_devices[mapped_interface] == self._device.short_name:
            return self._device

        # create new backprop device
        expand_fn = self._device.expand_fn
        batch_transform = self._device.batch_transform
        if hasattr(self._device, "_debugger"):
            debugger = self._device._debugger
        else:
            debugger = "No debugger"
        tracker = self._device.tracker

        backprop_devices = self._device.capabilities().get("passthru_devices", None)

        new_device = qml.device(
            backprop_devices[mapped_interface], wires=self._device.wires, shots=self._device.shots
        )
        new_device.expand_fn = expand_fn
        new_device.batch_transform = batch_transform
        if debugger != "No debugger":
            new_device._debugger = debugger
        new_device.tracker = tracker

        return new_device

    # pylint: disable=protected-access
    def _update_original_device(self, temp_device):
        """After performing an execution with a backprop device, update the state of the original device."""
        # Update for state vector simulators that have the _pre_rotated_state attribute
        if hasattr(self._device, "_pre_rotated_state"):
            self._device._pre_rotated_state = temp_device._pre_rotated_state

        # Update for state vector simulators that have the _state attribute
        if hasattr(self._device, "_state"):
            self._device._state = temp_device._state

    def _validate_backprop_method(self, tape):
        print("in _valdiate_backprop_method")
        if tape.shots:
            print("has shots")
            return False
        params = tape.get_parameters(trainable_only=False)
        interface = qml.math.get_interface(*params)

        if tape and any(isinstance(m.obs, qml.SparseHamiltonian) for m in tape.measurements):
            print("sparse hamiltonian")
            return False
        if interface == "numpy":
            interface = None
        mapped_interface = qml.workflow.execution.INTERFACE_MAP.get(interface, interface)

        # determine if the device supports backpropagation
        backprop_interface = self._device.capabilities().get("passthru_interface", None)
        print("im here")
        if backprop_interface is not None:
            # device supports backpropagation natively
            print("backprop_interface", backprop_interface)
            print("mapped_interface: ", mapped_interface)
            return mapped_interface in [backprop_interface, "Numpy"]
        # determine if the device has any child devices that support backpropagation
        backprop_devices = self._device.capabilities().get("passthru_devices", None)

        if backprop_devices is None:
            return False
        return mapped_interface in backprop_devices or mapped_interface == "Numpy"

    def _validate_adjoint_method(self, tape):
        # The conditions below provide a minimal set of requirements that we can likely improve upon in
        # future, or alternatively summarize within a single device capability. Moreover, we also
        # need to inspect the circuit measurements to ensure only expectation values are taken. This
        # cannot be done here since we don't yet know the composition of the circuit.

        required_attrs = ["_apply_operation", "_apply_unitary", "adjoint_jacobian"]
        supported_device = all(hasattr(self._device, attr) for attr in required_attrs)
        supported_device = supported_device and self._device.capabilities().get("returns_state")

        if not supported_device:
            return False
        return not bool(tape.shots)

    def _validate_device_method(self, _):
        # determine if the device provides its own jacobian method
        return self._device.capabilities().get("provides_jacobian", False)

    def execute(self, circuits, execution_config=DefaultExecutionConfig):
        dev = (
            self._create_temp_device(circuits)
            if execution_config.gradient_method == "backprop"
            else self._device
        )

        first_shot = circuits[0].shots
        if all(t.shots == first_shot for t in circuits):
            results = _set_shots(dev, first_shot)(dev.batch_execute)(circuits)
        else:
            results = tuple(_set_shots(dev, t.shots)(dev.batch_execute)((t,))[0] for t in circuits)

        if dev is not self._device:
            self._update_original_device(dev)

        return results

    def execute_and_compute_derivatives(self, circuits, execution_config=DefaultExecutionConfig):
        first_shot = circuits[0].shots
        if all(t.shots == first_shot for t in circuits):
            return _set_shots(self._device, first_shot)(self._device.execute_and_gradients)(
                circuits, **execution_config.gradient_keyword_arguments
            )
        batched_res = tuple(
            self.execute_and_compute_derivatives((c,), execution_config) for c in circuits
        )
        return tuple(zip(*batched_res))

    def compute_derivatives(self, circuits, execution_config=DefaultExecutionConfig):
        first_shot = circuits[0].shots
        if all(t.shots == first_shot for t in circuits):
            return _set_shots(self._device, first_shot)(self._device.gradients)(
                circuits, **execution_config.gradient_keyword_arguments
            )
        return tuple(self.compute_derivatives((c,), execution_config) for c in circuits)
