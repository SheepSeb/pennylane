from pennylane_lightning.lightning_qubit._state_vector import (
    LightningMeasurements,
    LightningStateVector,
)

from .interpreters import PlxprInterpreter


class LightningInterpreter(PlxprInterpreter):

    def __init__(self, num_wires):
        self._num_wires = num_wires

    def setup(self):
        self._state = LightningStateVector(self._num_wires)

    def interpret_operation(self, op):
        self._state._apply_lightning([op])

    def interpret_measurement(self, m):
        return LightningMeasurements(self._state).measurement(m)
