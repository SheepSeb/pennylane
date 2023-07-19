# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

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
Tests for the :mod:`pennylane.data.base.typing_util` functions.
"""

import typing

import pytest

import pennylane as qml
from pennylane.data.base.typing_util import get_type, get_type_str
from pennylane.qchem import Molecule


@pytest.mark.parametrize(
    "type_, expect",
    [
        (list, "list"),
        (typing.List, "list"),
        (Molecule, "pennylane.qchem.molecule.Molecule"),
        ("nonsense", "nonsense"),
        (typing.List[int], "list[int]"),
        (typing.List[typing.Tuple[int, "str"]], "list[tuple[int, str]]"),
        (typing.Optional[int], "Union[int, None]"),
        (typing.Union[int, "str", Molecule], "Union[int, str, pennylane.qchem.molecule.Molecule]"),
        (str, "str"),
        (typing.Type[str], "type[str]"),
    ],
)
def test_get_type_str(type_, expect):
    """Test that ``get_type_str()`` returns the expected value for various
    typing forms."""
    assert get_type_str(type_) == expect


@pytest.mark.parametrize(
    "obj, expect",
    [
        (list, list),
        ([1, 2], list),
        (typing.List, list),
        (typing.List[int], list),
        (qml.RX, qml.RX),
        (qml.RX(1, [1]), qml.RX),
    ],
)
def test_get_type(obj, expect):
    """Test that ``get_type()`` returns the expected value for various objects
    and types."""
    assert get_type(obj) is expect
