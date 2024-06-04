# Copyright 2018-2020 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This module contains functions to read the structure of molecules, build a Hartree-Fock state,
build an active space and generate single and double excitations.
"""
# pylint: disable=too-many-locals
import os
import re
from shutil import copyfile

import numpy as np

# Bohr-Angstrom correlation coefficient (https://physics.nist.gov/cgi-bin/cuu/Value?bohrrada0)
bohr_angs = 0.529177210903


def read_structure(filepath, outpath="."):
    r"""Read the structure of the polyatomic system from a file and returns
    a list with the symbols of the atoms in the molecule and a 1D array
    with their positions :math:`[x_1, y_1, z_1, x_2, y_2, z_2, \dots]` in
    atomic units (Bohr radius = 1).

    The atomic coordinates in the file must be in Angstroms.
    The `xyz <https://en.wikipedia.org/wiki/XYZ_file_format>`_ format is supported. Additionally,
    the new file ``structure.xyz``, containing the input geometry, is created in a directory with
    path given by ``outpath``.

    Args:
        filepath (str): name of the molecular structure file in the working directory
            or the absolute path to the file if it is located in a different folder
        outpath (str): path to the output directory

    Returns:
        tuple[list, array]: symbols of the atoms in the molecule and a 1D array with their
        positions in atomic units.

    **Example**

    >>> symbols, coordinates = read_structure('h2.xyz')
    >>> print(symbols, coordinates)
    ['H', 'H'] [0.    0.   -0.66140414    0.    0.    0.66140414]
    """
    file_in = filepath.strip()
    file_out = os.path.join(outpath, "structure.xyz")

    copyfile(file_in, file_out)

    symbols = []
    coordinates = []
    with open(file_out, encoding="utf-8") as f:
        for line in f.readlines()[2:]:
            symbol, x, y, z = line.split()
            symbols.append(symbol)
            coordinates.append(float(x))
            coordinates.append(float(y))
            coordinates.append(float(z))

    return symbols, np.array(coordinates) / bohr_angs


def active_space(electrons, orbitals, mult=1, active_electrons=None, active_orbitals=None):
    r"""Build the active space for a given number of active electrons and active orbitals.

    Post-Hartree-Fock (HF) electron correlation methods expand the many-body wave function
    as a linear combination of Slater determinants, commonly referred to as configurations.
    This configurations are generated by exciting electrons from the occupied to the
    unoccupied HF orbitals as sketched in the figure below. Since the number of configurations
    increases combinatorially with the number of electrons and orbitals this expansion can be
    truncated by defining an active space.

    The active space is created by classifying the HF orbitals as core, active and
    external orbitals:

    - Core orbitals are always occupied by two electrons
    - Active orbitals can be occupied by zero, one, or two electrons
    - The external orbitals are never occupied

    |

    .. figure:: ../../_static/qchem/sketch_active_space.png
        :align: center
        :width: 50%

    |

    .. note::
        The number of active *spin*-orbitals ``2*active_orbitals`` determines the number of
        qubits required to perform the quantum simulations of the electronic structure
        of the many-electron system.

    Args:
        electrons (int): total number of electrons
        orbitals (int): total number of orbitals
        mult (int): Spin multiplicity :math:`\mathrm{mult}=N_\mathrm{unpaired} + 1` for
            :math:`N_\mathrm{unpaired}` unpaired electrons occupying the HF orbitals.
            Possible values for ``mult`` are :math:`1, 2, 3, \ldots`. If not specified,
            a closed-shell HF state is assumed.
        active_electrons (int): Number of active electrons. If not specified, all electrons
            are treated as active.
        active_orbitals (int): Number of active orbitals. If not specified, all orbitals
            are treated as active.

    Returns:
        tuple: lists of indices for core and active orbitals

    **Example**

    >>> electrons = 4
    >>> orbitals = 4
    >>> core, active = active_space(electrons, orbitals, active_electrons=2, active_orbitals=2)
    >>> print(core) # core orbitals
    [0]
    >>> print(active) # active orbitals
    [1, 2]
    """
    # pylint: disable=too-many-branches

    if active_electrons is None:
        ncore_orbs = 0
        core = []
    else:
        if active_electrons <= 0:
            raise ValueError(
                f"The number of active electrons ({active_electrons}) " f"has to be greater than 0."
            )

        if active_electrons > electrons:
            raise ValueError(
                f"The number of active electrons ({active_electrons}) "
                f"can not be greater than the total "
                f"number of electrons ({electrons})."
            )

        if active_electrons < mult - 1:
            raise ValueError(
                f"For a reference state with multiplicity {mult}, "
                f"the number of active electrons ({active_electrons}) should be "
                f"greater than or equal to {mult - 1}."
            )

        if mult % 2 == 1:
            if active_electrons % 2 != 0:
                raise ValueError(
                    f"For a reference state with multiplicity {mult}, "
                    f"the number of active electrons ({active_electrons}) should be even."
                )
        else:
            if active_electrons % 2 != 1:
                raise ValueError(
                    f"For a reference state with multiplicity {mult}, "
                    f"the number of active electrons ({active_electrons}) should be odd."
                )

        ncore_orbs = (electrons - active_electrons) // 2
        core = list(range(ncore_orbs))

    if active_orbitals is None:
        active = list(range(ncore_orbs, orbitals))
    else:
        if active_orbitals <= 0:
            raise ValueError(
                f"The number of active orbitals ({active_orbitals}) " f"has to be greater than 0."
            )

        if ncore_orbs + active_orbitals > orbitals:
            raise ValueError(
                f"The number of core ({ncore_orbs}) + active orbitals ({active_orbitals}) cannot "
                f"be greater than the total number of orbitals ({orbitals})"
            )

        homo = (electrons + mult - 1) / 2
        if ncore_orbs + active_orbitals <= homo:
            raise ValueError(
                f"For n_active_orbitals={active_orbitals}, there are no virtual orbitals "
                f"in the active space."
            )

        active = list(range(ncore_orbs, ncore_orbs + active_orbitals))

    return core, active


def excitations(electrons, orbitals, delta_sz=0):
    r"""Generate single and double excitations from a Hartree-Fock reference state.

    Single and double excitations can be generated by acting with the operators
    :math:`\hat T_1` and :math:`\hat T_2` on the Hartree-Fock reference state:

    .. math::

        && \hat{T}_1 = \sum_{r \in \mathrm{occ} \\ p \in \mathrm{unocc}}
        \hat{c}_p^\dagger \hat{c}_r \\
        && \hat{T}_2 = \sum_{r>s \in \mathrm{occ} \\ p>q \in
        \mathrm{unocc}} \hat{c}_p^\dagger \hat{c}_q^\dagger \hat{c}_r \hat{c}_s.


    In the equations above the indices :math:`r, s` and :math:`p, q` run over the
    occupied (occ) and unoccupied (unocc) *spin* orbitals and :math:`\hat c` and
    :math:`\hat c^\dagger` are the electron annihilation and creation operators,
    respectively.

    |

    .. figure:: ../../_static/qchem/sd_excitations.png
        :align: center
        :width: 80%

    |

    Args:
        electrons (int): Number of electrons. If an active space is defined, this
            is the number of active electrons.
        orbitals (int): Number of *spin* orbitals. If an active space is defined,
            this is the number of active spin-orbitals.
        delta_sz (int): Specifies the selection rules ``sz[p] - sz[r] = delta_sz`` and
            ``sz[p] + sz[p] - sz[r] - sz[s] = delta_sz`` for the spin-projection ``sz`` of
            the orbitals involved in the single and double excitations, respectively.
            ``delta_sz`` can take the values :math:`0`, :math:`\pm 1` and :math:`\pm 2`.

    Returns:
        tuple(list, list): lists with the indices of the spin orbitals involved in the
        single and double excitations

    **Example**

    >>> electrons = 2
    >>> orbitals = 4
    >>> singles, doubles = excitations(electrons, orbitals)
    >>> print(singles)
    [[0, 2], [1, 3]]
    >>> print(doubles)
    [[0, 1, 2, 3]]
    """

    if not electrons > 0:
        raise ValueError(
            f"The number of active electrons has to be greater than 0 \n"
            f"Got n_electrons = {electrons}"
        )

    if orbitals <= electrons:
        raise ValueError(
            f"The number of active spin-orbitals ({orbitals}) "
            f"has to be greater than the number of active electrons ({electrons})."
        )

    if delta_sz not in (0, 1, -1, 2, -2):
        raise ValueError(
            f"Expected values for 'delta_sz' are 0, +/- 1 and +/- 2 but got ({delta_sz})."
        )

    # define the spin projection 'sz' of the single-particle states
    sz = np.array([0.5 if (i % 2 == 0) else -0.5 for i in range(orbitals)])

    singles = [
        [r, p]
        for r in range(electrons)
        for p in range(electrons, orbitals)
        if sz[p] - sz[r] == delta_sz
    ]

    doubles = [
        [s, r, q, p]
        for s in range(electrons - 1)
        for r in range(s + 1, electrons)
        for q in range(electrons, orbitals - 1)
        for p in range(q + 1, orbitals)
        if (sz[p] + sz[q] - sz[r] - sz[s]) == delta_sz
    ]

    return singles, doubles


def _beta_matrix(orbitals):
    r"""Generate the conversion matrix to transform the occupation number basis to the Bravyi-Kitaev basis.

    Args:
        orbitals (int): Number of spin orbitals. If an active space is defined,
            this is the number of active spin orbitals.

    Returns:
        (array): The transformation matrix
    """

    bin_range = int(np.ceil(np.log2(orbitals)))

    beta = np.array([[1]])

    for i in range(bin_range):
        beta = np.kron(np.eye(2), beta)
        beta[-1, : 2**i] = 1

    return beta[:orbitals, :orbitals]


def hf_state(electrons, orbitals, basis="occupation_number"):
    r"""Generate the Hartree-Fock statevector with respect to a chosen basis.

    The many-particle wave function in the Hartree-Fock (HF) approximation is a `Slater determinant
    <https://en.wikipedia.org/wiki/Slater_determinant>`_. In Fock space, a Slater determinant
    for :math:`N` electrons is represented by the occupation-number vector:

    .. math::

        \vert {\bf n} \rangle = \vert n_1, n_2, \dots, n_\mathrm{orbs} \rangle,
        n_i = \left\lbrace \begin{array}{ll} 1 & i \leq N \\ 0 & i > N \end{array} \right.,

    where :math:`n_i` indicates the occupation of the :math:`i`-th orbital.

    The Hartree-Fock state can also be generated in the parity basis, where each qubit stores the parity of
    the spin orbital, and in the Bravyi-Kitaev basis, where a qubit :math:`j` stores the occupation state of orbital
    :math:`j` if :math:`j` is even and stores partial sum of the occupation state of a set of orbitals of indices
    less than :math:`j` if :math:`j` is odd [`Tranter et al. Int. J. Quantum Chem. 115, 1431 (2015)
    <https://doi.org/10.1002/qua.24969>`_].

    Args:
        electrons (int): Number of electrons. If an active space is defined, this
            is the number of active electrons.
        orbitals (int): Number of *spin* orbitals. If an active space is defined,
            this is the number of active spin-orbitals.
        basis (string): Basis in which the HF state is represented. Options are ``occupation_number``, ``parity`` and ``bravyi_kitaev``.

    Returns:
        array: NumPy array containing the vector :math:`\vert {\bf n} \rangle`

    **Example**

    >>> state = hf_state(2, 6)
    >>> print(state)
    [1 1 0 0 0 0]

    >>> state = hf_state(2, 6, basis="parity")
    >>> print(state)
    [1 0 0 0 0 0]

    >>> state = hf_state(2, 6, basis="bravyi_kitaev")
    >>> print(state)
    [1 0 0 0 0 0]

    """

    if electrons <= 0:
        raise ValueError(
            f"The number of active electrons has to be larger than zero; "
            f"got 'electrons' = {electrons}"
        )

    if electrons > orbitals:
        raise ValueError(
            f"The number of active orbitals cannot be smaller than the number of active electrons;"
            f" got 'orbitals'={orbitals} < 'electrons'={electrons}"
        )

    state = np.where(np.arange(orbitals) < electrons, 1, 0)

    if basis == "parity":
        pi_matrix = np.tril(np.ones((orbitals, orbitals)))
        return (np.matmul(pi_matrix, state) % 2).astype(int)

    if basis == "bravyi_kitaev":
        beta_matrix = _beta_matrix(orbitals)
        return (np.matmul(beta_matrix, state) % 2).astype(int)

    return state


def excitations_to_wires(singles, doubles, wires=None):
    r"""Map the indices representing the single and double excitations
    generated with the function :func:`~.excitations` to the wires that
    the Unitary Coupled-Cluster (UCCSD) template will act on.

    Args:
        singles (list[list[int]]): list with the indices ``r``, ``p`` of the two qubits
            representing the single excitation
            :math:`\vert r, p \rangle = \hat{c}_p^\dagger \hat{c}_r \vert \mathrm{HF}\rangle`
        doubles (list[list[int]]): list with the indices ``s``, ``r``, ``q``, ``p`` of the four
            qubits representing the double excitation
            :math:`\vert s, r, q, p \rangle = \hat{c}_p^\dagger \hat{c}_q^\dagger
            \hat{c}_r \hat{c}_s \vert \mathrm{HF}\rangle`
        wires (Iterable[Any]): Wires of the quantum device. If None, will use consecutive wires.

    The indices :math:`r, s` and :math:`p, q` in these lists correspond, respectively, to the
    occupied and virtual orbitals involved in the generated single and double excitations.

    Returns:
        tuple[list[list[Any]], list[list[list[Any]]]]: lists with the sequence of wires,
        resulting from the single and double excitations, that the Unitary Coupled-Cluster
        (UCCSD) template will act on.

    **Example**

    >>> singles = [[0, 2], [1, 3]]
    >>> doubles = [[0, 1, 2, 3]]
    >>> singles_wires, doubles_wires = excitations_to_wires(singles, doubles)
    >>> print(singles_wires)
    [[0, 1, 2], [1, 2, 3]]
    >>> print(doubles_wires)
    [[[0, 1], [2, 3]]]

    >>> wires=['a0', 'b1', 'c2', 'd3']
    >>> singles_wires, doubles_wires = excitations_to_wires(singles, doubles, wires=wires)
    >>> print(singles_wires)
    [['a0', 'b1', 'c2'], ['b1', 'c2', 'd3']]
    >>> print(doubles_wires)
    [[['a0', 'b1'], ['c2', 'd3']]]
    """

    if (not singles) and (not doubles):
        raise ValueError(
            f"'singles' and 'doubles' lists can not be both empty; "
            f"got singles = {singles}, doubles = {doubles}"
        )

    expected_shape = (2,)
    for single_ in singles:
        if np.array(single_).shape != expected_shape:
            raise ValueError(
                f"Expected entries of 'singles' to be of shape (2,); got {np.array(single_).shape}"
            )

    expected_shape = (4,)
    for double_ in doubles:
        if np.array(double_).shape != expected_shape:
            raise ValueError(
                f"Expected entries of 'doubles' to be of shape (4,); got {np.array(double_).shape}"
            )

    max_idx = 0
    if singles:
        max_idx = np.max(singles)
    if doubles:
        max_idx = max(np.max(doubles), max_idx)

    if wires is None:
        wires = range(max_idx + 1)
    elif len(wires) != max_idx + 1:
        raise ValueError(f"Expected number of wires is {max_idx + 1}; got {len(wires)}")

    singles_wires = []
    for r, p in singles:
        s_wires = [wires[i] for i in range(r, p + 1)]
        singles_wires.append(s_wires)

    doubles_wires = []
    for s, r, q, p in doubles:
        d1_wires = [wires[i] for i in range(s, r + 1)]
        d2_wires = [wires[i] for i in range(q, p + 1)]
        doubles_wires.append([d1_wires, d2_wires])

    return singles_wires, doubles_wires


def mol_data(identifier, identifier_type="name"):
    r"""Obtain symbols and geometry of a compound from the PubChem Database.

    The `PubChem <https://pubchem.ncbi.nlm.nih.gov>`__ database is one of the largest public
    repositories for information on chemical substances from which symbols and geometry can be
    retrieved for a compound by its name, SMILES, InChI, InChIKey, or PubChem Compound ID (CID) to
    build a molecule object for Hartree-Fock calculations. The retrieved atomic coordinates will be
    converted to `atomic units <https://en.wikipedia.org/wiki/Bohr_radius>`__ for consistency.

    Args:
        identifier (str or int): compound's identifier as required by the PubChem database
        identifier_type (str): type of the provided identifier - name, CAS, CID, SMILES, InChI, InChIKey

    Returns:
        Tuple(list[str], array[float]): symbols and geometry (in Bohr radius) of the compound

    **Example**

    >>> mol_data("BeH2")
    (['Be', 'H', 'H'],
    tensor([[ 4.79404621,  0.29290755,  0.        ],
            [ 3.77945225, -0.29290755,  0.        ],
            [ 5.80882913, -0.29290755,  0.        ]], requires_grad=True))

    >>> mol_data(223, "CID")
    (['N', 'H', 'H', 'H', 'H'],
    tensor([[ 0.        ,  0.        ,  0.        ],
            [ 1.82264085,  0.52836742,  0.40402345],
            [ 0.01417295, -1.67429735, -0.98038991],
            [-0.98927163, -0.22714508,  1.65369933],
            [-0.84773114,  1.373075  , -1.07733286]], requires_grad=True))

    .. details::

        ``mol_data`` can also be used with other chemical identifiers - CAS, SMILES, InChI, InChIKey:

        >>> mol_data("74-82-8", "CAS")
        (['C', 'H', 'H', 'H', 'H'],
        tensor([[ 0.        ,  0.        ,  0.        ],
                [ 1.04709725,  1.51102501,  0.93824902],
                [ 1.29124986, -1.53710323, -0.47923455],
                [-1.47058487, -0.70581271,  1.26460472],
                [-0.86795121,  0.7320799 , -1.7236192 ]], requires_grad=True))

        >>> mol_data("[C]", "SMILES")
        (['C', 'H', 'H', 'H', 'H'],
        tensor([[ 0.        ,  0.        ,  0.        ],
                [ 1.04709725,  1.51102501,  0.93824902],
                [ 1.29124986, -1.53710323, -0.47923455],
                [-1.47058487, -0.70581271,  1.26460472],
                [-0.86795121,  0.7320799 , -1.7236192 ]], requires_grad=True))

        >>> mol_data("InChI=1S/CH4/h1H4", "InChI")
        (['C', 'H', 'H', 'H', 'H'],
        tensor([[ 0.        ,  0.        ,  0.        ],
                [ 1.04709725,  1.51102501,  0.93824902],
                [ 1.29124986, -1.53710323, -0.47923455],
                [-1.47058487, -0.70581271,  1.26460472],
                [-0.86795121,  0.7320799 , -1.7236192 ]], requires_grad=True))

        >>> mol_data("VNWKTOKETHGBQD-UHFFFAOYSA-N", "InChIKey")
        (['C', 'H', 'H', 'H', 'H'],
        tensor([[ 0.        ,  0.        ,  0.        ],
                [ 1.04709725,  1.51102501,  0.93824902],
                [ 1.29124986, -1.53710323, -0.47923455],
                [-1.47058487, -0.70581271,  1.26460472],
                [-0.86795121,  0.7320799 , -1.7236192 ]], requires_grad=True))

    """

    try:
        # pylint: disable=import-outside-toplevel, unused-import, multiple-imports
        import pubchempy as pcp
    except ImportError as Error:
        raise ImportError(
            "This feature requires pubchempy.\nIt can be installed with: pip install pubchempy."
        ) from Error

    # https://gist.github.com/lsauer/1312860/264ae813c2bd2c27a769d261c8c6b38da34e22fb#file-smiles_inchi_annotated-js
    identifier_patterns = {
        "name": re.compile(r"^[a-zA-Z0-9_+-]+$"),
        "cas": re.compile(r"^\d{1,7}\-\d{2}\-\d$"),
        "smiles": re.compile(
            r"^(?!InChI=)(?!\d{1,7}\-\d{2}\-\d)(?![A-Z]{14}\-[A-Z]{10}(\-[A-Z])?)[^J][a-zA-Z0-9@+\-\[\]\(\)\\\/%=#$]{1,}$"
        ),
        "inchi": re.compile(
            r"^InChI\=1S?\/[A-Za-z0-9\.]+(\+[0-9]+)?(\/[cnpqbtmsih][A-Za-z0-9\-\+\(\)\,\/\?\;\.]+)*$"
        ),
        "inchikey": re.compile(r"^[A-Z]{14}\-[A-Z]{10}(\-[A-Z])?"),
    }
    if identifier_type.lower() == "cid":
        cid = int(identifier)
    else:
        if identifier_type.lower() not in identifier_patterns:
            raise ValueError(
                "Specified identifier type is not supported. Supported type are: name, CAS, SMILES, InChI, InChIKey."
            )
        try:
            if identifier_patterns[identifier_type.lower()].search(identifier):
                if identifier_type.lower() == "cas":
                    identifier_type = "name"
                cid = pcp.get_cids(identifier, namespace=identifier_type.lower())[0]
            else:
                raise ValueError(
                    f"Specified identifier doesn't seem to match type: {identifier_type}."
                )
        except (IndexError, pcp.NotFoundError) as exc:
            raise ValueError("Specified molecule does not exist in the PubChem Database.") from exc

    try:
        pcp_molecule = pcp.Compound.from_cid(cid, record_type="3d")
    except pcp.NotFoundError:
        pcp_molecule = pcp.Compound.from_cid(cid, record_type="2d")
    except ValueError as exc:
        raise ValueError("Provided CID (or Identifier) is None.") from exc

    data_mol = pcp_molecule.to_dict(properties=["atoms"])
    symbols = [atom["element"] for atom in data_mol["atoms"]]
    geometry = (
        np.array([[atom["x"], atom["y"], atom.get("z", 0.0)] for atom in data_mol["atoms"]])
        / bohr_angs
    )

    return symbols, geometry
