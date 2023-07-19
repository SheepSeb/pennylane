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
"""Contains DatasetAttribute definition for ``pennylane.qchem.Molecule``."""

from typing import Tuple, Type

from pennylane.data.base.attribute import DatasetAttribute
from pennylane.data.base.hdf5 import HDF5Group
from pennylane.data.base.mapper import AttributeTypeMapper
from pennylane.qchem import Molecule


class DatasetMolecule(DatasetAttribute[HDF5Group, Molecule, Molecule]):
    """Attribute type for ``pennylane.qchem.Molecule``."""

    type_id = "molecule"

    @classmethod
    def consumes_types(cls) -> Tuple[Type[Molecule]]:
        return (Molecule,)

    def hdf5_to_value(self, bind: HDF5Group) -> Molecule:
        mapper = AttributeTypeMapper(bind)

        return Molecule(
            symbols=mapper["symbols"].get_value(),
            coordinates=mapper["coordinates"].get_value(),
            charge=mapper["charge"].get_value(),
            mult=mapper["mult"].get_value(),
            basis_name=mapper["basis_name"].get_value(),
            l=mapper["l"].get_value(),
            alpha=mapper["alpha"].get_value(),
            coeff=mapper["coeff"].get_value(),
        )

    def value_to_hdf5(self, bind_parent: HDF5Group, key: str, value: Molecule) -> HDF5Group:
        bind = bind_parent.create_group(key)

        bind = AttributeTypeMapper(bind)

        bind["symbols"] = value.symbols
        bind["coordinates"] = value.coordinates
        bind["charge"] = value.charge
        bind["mult"] = value.mult
        bind["basis_name"] = value.basis_name
        bind["l"] = value.l
        bind["alpha"] = value.alpha
        bind["coeff"] = value.coeff

        return bind.bind
