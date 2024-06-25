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
"""
Contains the general execute function, for executing tapes on devices with auto-
differentiation support.
"""

# pylint: disable=import-outside-toplevel,too-many-branches,not-callable,unexpected-keyword-arg
# pylint: disable=unused-argument,unnecessary-lambda-assignment,inconsistent-return-statements
# pylint: disable=invalid-unary-operand-type,isinstance-second-argument-not-valid-type
# pylint: disable=too-many-arguments,too-many-statements,function-redefined,too-many-function-args

import inspect
import logging
import warnings
from functools import partial
from typing import Callable, MutableMapping, Optional, Sequence, Tuple, Union

from cachetools import Cache, LRUCache

import pennylane as qml
from pennylane.tape import QuantumTape
from pennylane.transforms import transform
from pennylane.typing import ResultBatch

from .jacobian_products import (
    DeviceDerivatives,
    DeviceJacobianProducts,
    LightningVJPs,
    TransformJacobianProducts,
)
from .cache_transform import cache_transform

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

jpc_interfaces = {
    "autograd",
    "numpy",
    "torch",
    "pytorch",
    "jax",
    "jax-python",
    "jax-jit",
    "tf",
    "tensorflow",
}

INTERFACE_MAP = {
    None: "Numpy",
    "auto": "auto",
    "autograd": "autograd",
    "numpy": "autograd",
    "scipy": "numpy",
    "jax": "jax",
    "jax-jit": "jax",
    "jax-python": "jax",
    "JAX": "jax",
    "torch": "torch",
    "pytorch": "torch",
    "tf": "tf",
    "tensorflow": "tf",
    "tensorflow-autograph": "tf",
    "tf-autograph": "tf",
}
"""dict[str, str]: maps an allowed interface specification to its canonical name."""

#: list[str]: allowed interface strings
SUPPORTED_INTERFACES = list(INTERFACE_MAP)
"""list[str]: allowed interface strings"""



def _use_tensorflow_autograph():
    import tensorflow as tf

    return not tf.executing_eagerly()


def _get_ml_boundary_execute(
    interface: str, grad_on_execution: bool, device_vjp: bool = False, differentiable=False
) -> Callable:
    """Imports and returns the function that binds derivatives of the required ml framework.

    Args:
        interface (str): The designated ml framework.

        grad_on_execution (bool): whether or not the device derivatives are taken upon execution
    Returns:
        Callable

    Raises:
        pennylane.QuantumFunctionError if the required package is not installed.

    """
    mapped_interface = INTERFACE_MAP[interface]
    try:
        if mapped_interface == "autograd":
            from .interfaces.autograd import autograd_execute as ml_boundary

        elif mapped_interface == "tf":
            if "autograph" in interface:
                from .interfaces.tensorflow_autograph import execute as ml_boundary

                ml_boundary = partial(ml_boundary, grad_on_execution=grad_on_execution)

            else:
                from .interfaces.tensorflow import tf_execute as full_ml_boundary

                ml_boundary = partial(full_ml_boundary, differentiable=differentiable)

        elif mapped_interface == "torch":
            from .interfaces.torch import execute as ml_boundary

        elif interface == "jax-jit" and not differentiable:
            if device_vjp:
                from .interfaces.jax_jit import jax_jit_vjp_execute as ml_boundary
            else:
                from .interfaces.jax_jit import jax_jit_jvp_execute as ml_boundary
        else:  # interface in {"jax", "jax-python", "JAX"}:
            if device_vjp:
                from .interfaces.jax_jit import jax_jit_vjp_execute as ml_boundary
            else:
                from .interfaces.jax import jax_jvp_execute as ml_boundary

    except ImportError as e:  # pragma: no-cover
        raise qml.QuantumFunctionError(
            f"{mapped_interface} not found. Please install the latest "
            f"version of {mapped_interface} to enable the '{mapped_interface}' interface."
        ) from e
    return ml_boundary


def _make_inner_execute(device, cache, execution_config=None, numpy_only=True) -> Callable:
    """Construct the function that will execute the tapes inside the ml framework registration
    for the 1st order derivatives.

    Steps in between the ml framework execution and the device are:
    - device expansion (old device)
    - conversion to numpy
    - caching

    For higher order derivatives, the "inner execute" will be another ml framework execute.
    """

    def inner_execute(tapes: Sequence[QuantumTape], **_) -> ResultBatch:
        """Execution that occurs within a machine learning framework boundary.

        Closure Variables:
            numpy_only (bool): whether or not to convert the data to numpy or leave as is
            device (qml.devices.Device)
            cache (None | MutableMapping): The cache to use. If ``None``, caching will not occur.
        """
        transform_program = qml.transforms.core.TransformProgram()

        if numpy_only:
            transform_program.add_transform(qml.transforms.convert_to_numpy_parameters)

        if cache is not None:
            transform_program.add_transform(cache_transform, cache=cache)

        transformed_tapes, transform_post_processing = transform_program(tapes)

        if transformed_tapes:
            results = device.execute(transformed_tapes, execution_config=execution_config)
        else:
            results = ()

        return transform_post_processing(results)

    return inner_execute



def execute(
    tapes: Sequence[QuantumTape],
    device: "qml.devices.Device",
    gradient_fn: Optional[Union[Callable, str]] = None,
    interface: Optional[str] = "auto",
    transform_program=None,
    config=None,
    grad_on_execution="best",
    gradient_kwargs=None,
    cache: Union[None, bool, dict, Cache] = True,
    cachesize=10000,
    max_diff=1,
    override_shots: int = False,  # doesnt do anything anymore
    expand_fn="device",  # type: ignore # doesnt do anything anymore
    max_expansion=10,  # doesnt do anythiny anymore
    device_batch_transform=True,  # doesnt do anything anymore
    device_vjp=False,
) -> ResultBatch:
    """New function to execute a batch of tapes on a device in an autodifferentiable-compatible manner. More cases will be added,
    during the project. The current version is supporting forward execution for NumPy and does not support shot vectors.

    Args:
        tapes (Sequence[.QuantumTape]): batch of tapes to execute
        device (pennylane.Device): Device to use to execute the batch of tapes.
            If the device does not provide a ``batch_execute`` method,
            by default the tapes will be executed in serial.
        gradient_fn (None or callable): The gradient transform function to use
            for backward passes. If "device", the device will be queried directly
            for the gradient (if supported).
        interface (str): The interface that will be used for classical autodifferentiation.
            This affects the types of parameters that can exist on the input tapes.
            Available options include ``autograd``, ``torch``, ``tf``, ``jax`` and ``auto``.
        transform_program(.TransformProgram): A transform program to be applied to the initial tape.
        config (qml.devices.ExecutionConfig): A datastructure describing the parameters needed to fully describe the execution.
        grad_on_execution (bool, str): Whether the gradients should be computed on the execution or not. Only applies
            if the device is queried for the gradient; gradient transform
            functions available in ``qml.gradients`` are only supported on the backward
            pass. The 'best' option chooses automatically between the two options and is default.
        gradient_kwargs (dict): dictionary of keyword arguments to pass when
            determining the gradients of tapes
        cache (None, bool, dict, Cache): Whether to cache evaluations. This can result in
            a significant reduction in quantum evaluations during gradient computations.
        cachesize (int): the size of the cache
        max_diff (int): If ``gradient_fn`` is a gradient transform, this option specifies
            the maximum number of derivatives to support. Increasing this value allows
            for higher order derivatives to be extracted, at the cost of additional
            (classical) computational overhead during the backwards pass.
        override_shots (int): The number of shots to use for the execution. If ``False``, then the
            number of shots on the device is used.
        expand_fn (str, function): Tape expansion function to be called prior to device execution.
            Must have signature of the form ``expand_fn(tape, max_expansion)``, and return a
            single :class:`~.QuantumTape`. If not provided, by default :meth:`Device.expand_fn`
            is called.
        max_expansion (int): The number of times the internal circuit should be expanded when
            executed on a device. Expansion occurs when an operation or measurement is not
            supported, and results in a gate decomposition. If any operations in the decomposition
            remain unsupported by the device, another expansion occurs.
        device_batch_transform (bool): Whether to apply any batch transforms defined by the device
            (within :meth:`Device.batch_transform`) to each tape to be executed. The default behaviour
            of the device batch transform is to expand out Hamiltonian measurements into
            constituent terms if not supported on the device.
        device_vjp=False (Optional[bool]): whether or not to use the device provided jacobian
            product if it is available.

    Returns:
        list[tensor_like[float]]: A nested list of tape results. Each element in
        the returned list corresponds in order to the provided tapes.

    **Example**

    Consider the following cost function:

    .. code-block:: python

        dev = qml.device("lightning.qubit", wires=2)

        def cost_fn(params, x):
            ops1 = [qml.RX(params[0], wires=0), qml.RY(params[1], wires=0)]
            measurements1 = [qml.expval(qml.Z(0))]
            tape1 = qml.tape.QuantumTape(ops1, measurements1)

            ops2 = [
                qml.RX(params[2], wires=0),
                qml.RY(x[0], wires=1),
                qml.CNOT(wires=(0,1))
            ]
            measurements2 = [qml.probs(wires=0)]
            tape2 = qml.tape.QuantumTape(ops2, measurements2)

            tapes = [tape1, tape2]

            # execute both tapes in a batch on the given device
            res = qml.execute(tapes, dev, gradient_fn=qml.gradients.param_shift, max_diff=2)

            return res[0] + res[1][0] - res[1][1]

    In this cost function, two **independent** quantum tapes are being
    constructed; one returning an expectation value, the other probabilities.
    We then batch execute the two tapes, and reduce the results to obtain
    a scalar.

    Let's execute this cost function while tracking the gradient:

    >>> params = np.array([0.1, 0.2, 0.3], requires_grad=True)
    >>> x = np.array([0.5], requires_grad=True)
    >>> cost_fn(params, x)
    1.93050682

    Since the ``execute`` function is differentiable, we can
    also compute the gradient:

    >>> qml.grad(cost_fn)(params, x)
    (array([-0.0978434 , -0.19767681, -0.29552021]), array([5.37764278e-17]))

    Finally, we can also compute any nth-order derivative. Let's compute the Jacobian
    of the gradient (that is, the Hessian):

    >>> x.requires_grad = False
    >>> qml.jacobian(qml.grad(cost_fn))(params, x)
    array([[-0.97517033,  0.01983384,  0.        ],
           [ 0.01983384, -0.97517033,  0.        ],
           [ 0.        ,  0.        , -0.95533649]])
    """

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            """Entry with args=(tapes=%s, device=%s, gradient_fn=%s, interface=%s, grad_on_execution=%s, gradient_kwargs=%s, cache=%s, cachesize=%s, max_diff=%s, override_shots=%s, expand_fn=%s, max_expansion=%s, device_batch_transform=%s) called by=%s""",
            tapes,
            repr(device),
            (
                gradient_fn
                if not (logger.isEnabledFor(qml.logging.TRACE) and inspect.isfunction(gradient_fn))
                else "\n" + inspect.getsource(gradient_fn) + "\n"
            ),
            interface,
            grad_on_execution,
            gradient_kwargs,
            cache,
            cachesize,
            max_diff,
            override_shots,
            (
                expand_fn
                if not (logger.isEnabledFor(qml.logging.TRACE) and inspect.isfunction(expand_fn))
                else "\n" + inspect.getsource(expand_fn) + "\n"
            ),
            max_expansion,
            device_batch_transform,
            "::L".join(str(i) for i in inspect.getouterframes(inspect.currentframe(), 2)[1][1:3]),
        )

    ### Specifying and preprocessing variables ####

    if interface == "auto":
        params = []
        for tape in tapes:
            params.extend(tape.get_parameters(trainable_only=False))
        interface = qml.math.get_interface(*params)
    if INTERFACE_MAP.get(interface, "") == "tf" and _use_tensorflow_autograph():
        interface = "tf-autograph"
        raise NotImplementedError
    if interface == "jax":
        try:  # pragma: no-cover
            from .interfaces.jax import get_jax_interface_name
        except ImportError as e:  # pragma: no-cover
            raise qml.QuantumFunctionError(  # pragma: no-cover
                "jax not found. Please install the latest "  # pragma: no-cover
                "version of jax to enable the 'jax' interface."  # pragma: no-cover
            ) from e  # pragma: no-cover

        interface = get_jax_interface_name(tapes)
        # Only need to calculate derivatives with jax when we know it will be executed later.
        if interface in {"jax", "jax-jit"}:
            grad_on_execution = grad_on_execution if isinstance(gradient_fn, Callable) else False

    if (
        device_vjp
        and isinstance(device, qml.devices.LegacyDeviceFacade)
        and "lightning" not in getattr(device, "short_name", "").lower()
    ):
        raise qml.QuantumFunctionError(
            "device provided jacobian products are not compatible with the old device interface."
        )

    gradient_kwargs = gradient_kwargs or {}
    config = config or _get_execution_config(
        gradient_fn, grad_on_execution, interface, device, device_vjp
    )

    if transform_program is None:
        transform_program = device.preprocess(config)[0]

    # If caching is desired but an explicit cache is not provided, use an ``LRUCache``.
    if cache is True:
        cache = LRUCache(maxsize=cachesize)
        setattr(cache, "_persistent_cache", False)

    # Ensure that ``cache`` is not a Boolean to simplify downstream code.
    elif cache is False:
        cache = None

    # changing this set of conditions causes a bunch of tests to break.
    no_interface_boundary_required = interface is None or config.gradient_method in {
        None,
        "backprop",
    }
    device_supports_interface_data = no_interface_boundary_required and (
        interface is None
        or config.gradient_method == "backprop"
        or getattr(device, "short_name", "") == "default.mixed"
    )

    inner_execute = _make_inner_execute(
        device,
        cache,
        config,
        numpy_only=not device_supports_interface_data,
    )

    execute_fn = inner_execute
    #### Executing the configured setup #####
    tapes, post_processing = transform_program(tapes)

    if transform_program.is_informative:
        return post_processing(tapes)

    # Exiting early if we do not need to deal with an interface boundary
    if no_interface_boundary_required:
        results = inner_execute(tapes)
        return post_processing(results)

    if (
        device_vjp
        and getattr(device, "short_name", "") in ("lightning.gpu", "lightning.kokkos")
        and interface in jpc_interfaces
    ):
        if INTERFACE_MAP[interface] == "jax" and "use_device_state" in gradient_kwargs:
            gradient_kwargs["use_device_state"] = False
        jpc = LightningVJPs(device, gradient_kwargs=gradient_kwargs)

    elif config.use_device_jacobian_product and interface in jpc_interfaces:
        jpc = DeviceJacobianProducts(device, config)

    elif config.use_device_gradient:
        jpc = DeviceDerivatives(device, config)

        execute_fn = (
            jpc.execute_and_cache_jacobian if config.grad_on_execution else inner_execute
        )

    elif config.grad_on_execution is True:
        # In "forward" mode, gradients are automatically handled
        # within execute_and_gradients, so providing a gradient_fn
        # in this case would have ambiguous behaviour.
        raise ValueError("Gradient transforms cannot be used with grad_on_execution=True")
    elif interface in jpc_interfaces:
        # See autograd.py submodule docstring for explanation for ``cache_full_jacobian``
        cache_full_jacobian = (interface == "autograd") and not cache

        # we can have higher order derivatives when the `inner_execute` used to take
        # transform gradients is itself differentiable
        # To make the inner execute itself differentiable, we make it an interface boundary with
        # its own jacobian product class
        # this mechanism unpacks the currently existing recursion
        jpc = TransformJacobianProducts(
            execute_fn, gradient_fn, gradient_kwargs, cache_full_jacobian
        )
        for i in range(1, max_diff):
            differentiable = i > 1
            ml_boundary_execute = _get_ml_boundary_execute(
                interface, config.grad_on_execution, differentiable=differentiable
            )
            execute_fn = partial(
                ml_boundary_execute,
                execute_fn=execute_fn,
                jpc=jpc,
                device=device,
            )
            jpc = TransformJacobianProducts(execute_fn, gradient_fn, gradient_kwargs)

            if interface == "jax-jit":
                # no need to use pure callbacks around execute_fn or the jpc when taking
                # higher order derivatives
                interface = "jax"

    # trainable parameters can only be set on the first pass for jax
    # not higher order passes for higher order derivatives
    if interface in {"jax", "jax-python", "jax-jit"}:
        for tape in tapes:
            params = tape.get_parameters(trainable_only=False)
            tape.trainable_params = qml.math.get_trainable_indices(params)

    ml_boundary_execute = _get_ml_boundary_execute(
        interface,
        config.grad_on_execution,
        config.use_device_jacobian_product,
        differentiable=max_diff > 1,
    )
    results = ml_boundary_execute(tapes, execute_fn, jpc, device=device)

    return post_processing(results)


def _get_execution_config(gradient_fn, grad_on_execution, interface, device, device_vjp):
    """Helper function to get the execution config."""
    if gradient_fn is None:
        _gradient_method = None
    elif isinstance(gradient_fn, str):
        _gradient_method = gradient_fn
    else:
        _gradient_method = "gradient-transform"
    config = qml.devices.ExecutionConfig(
        interface=interface,
        gradient_method=_gradient_method,
        grad_on_execution=None if grad_on_execution == "best" else grad_on_execution,
        use_device_jacobian_product=device_vjp,
    )
    return device.preprocess(config)[1]
