from pennylane.capture.interpreters import PlxprInterpreter
from pennylane.measurements.mid_measure import MidMeasureMP

from .apply_operation import apply_operation
from .initialize_state import create_initial_state
from .measure import measure
from .sampling import measure_with_samples


class DefaultQubitInterpreter(PlxprInterpreter):

    _state = None

    def __init__(self, num_wires, shots):
        self.num_wires = num_wires
        self.shots = shots

    def setup(self):
        self._state = create_initial_state(range(self.num_wires))
        self._mcms = {}

    def cleanup(self):
        self._state = None

    def interpret_operation(self, op):
        self._state = apply_operation(op, self._state)

    def interpret_measurement(self, m):
        if isinstance(m, MidMeasureMP):
            mcms = {}
            self._state = apply_operation(m, self._state, mid_measurements=mcms)
            return mcms[m]

        if self.shots:
            return measure_with_samples([m], self._state, self.shots)
        return measure(m, self._state)
