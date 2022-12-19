import ase
from ase import units
import numpy as np
from pytest import fixture

from ttm import TTM
from ttm.ase import TTMCalculator


@fixture()
def hexamer() -> (np.ndarray, float, np.ndarray):
    """Get the positions of a hexamer cage and the expected energies"""

    positions = np.array([[0.87715956, 1.70409266, 0.47858616],
                          [1.70199937, 1.19722607, 0.29220265],
                          [1.16181787, 2.60891668, 0.65546661],
                          [-0.82054445, 0.61636011, -1.63123430],
                          [-0.26682713, 1.17302621, -1.05309865],
                          [-0.36885969, -0.24447946, -1.58195952],
                          [-0.64071556, -0.48854013, 1.64104190],
                          [-0.19915392, 0.37247798, 1.52856380],
                          [-1.54433112, -0.32408724, 1.29835600],
                          [0.57630762, -1.69154092, -0.42866929],
                          [0.43417462, -2.64420081, -0.48595869],
                          [0.08911917, -1.38946090, 0.38589939],
                          [2.81159896, -0.10395953, -0.18689424],
                          [3.46294039, -0.45173380, 0.43482057],
                          [2.15849520, -0.82915775, -0.30211113],
                          [-2.88801599, -0.06633690, 0.05761427],
                          [-2.29024563, 0.26724676, -0.65069518],
                          [-3.66545029, 0.50362404, 0.03495814]])
    energy = -44.00990
    forces = np.array([[-17.171631, -8.501426, -1.129756],
                       [15.737880, -5.080882, -0.557946],
                       [4.148115, 11.227446, 2.146692],
                       [-14.176072, 2.584080, -5.868841],
                       [7.047068, 4.271805, 4.834894],
                       [6.021064, -7.268982, 0.103362],
                       [9.904202, -16.366005, 1.193872],
                       [0.472663, 10.252552, 0.000899],
                       [-9.915398, 6.602982, -1.776479],
                       [11.645372, 12.264258, -15.884264],
                       [-1.032711, -12.792524, -0.344699],
                       [-12.280451, 4.581854, 19.976125],
                       [-3.474920, 15.323084, -8.571923],
                       [6.292993, -4.201467, 7.860706],
                       [-4.973061, -12.842316, -0.098187],
                       [3.244097, -13.576127, 8.849190],
                       [7.953128, 6.482030, -11.249730],
                       [-9.442338, 7.039638, 0.516085]])
    return positions, energy, forces


def test_hexamer(hexamer):
    ttm21f = TTM()
    cage_hexamer, known_eng, known_forces = hexamer
    energy, gradients = ttm21f.evaluate(cage_hexamer)

    assert np.isclose(energy, -44.00990)
    assert np.isclose(-gradients, known_forces, atol=1e-6).sum() == 18 * 3


def test_ase(hexamer):
    """Make sure the ase interface works"""

    cage_hexamer, known_eng, known_forces = hexamer
    atoms = ase.Atoms(positions=cage_hexamer, symbols='OHH' * 6)

    calc = TTMCalculator()
    energy = calc.get_potential_energy(atoms)
    assert np.isclose(energy, -1.908450378)  # Converted using https://www.colby.edu/chemistry/PChem/Hartree.html

    forces = calc.get_forces(atoms)
    assert np.isclose(forces, known_forces * units.kcal / units.mol, atol=1e-6).all()
