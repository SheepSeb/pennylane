from pennylane.capture.interpreters import PlxprInterpreter

from .apply_operation import apply_operation
from .initialize_state import create_initial_state
from .measure import measure


class DefaultQubitInterpreter(PlxprInterpreter):

    _state = None

    def __init__(self, num_wires):
        self.num_wires = num_wires

    def setup(self):
        self._state = create_initial_state(range(self.num_wires))

    def cleanup(self):
        self._state = None

    def interpret_operation(self, op):
        self._state = apply_operation(op, self._state)

    def interpret_measurement(self, m):
        return measure(m, self._state)
