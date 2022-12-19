"""Interface to the TTM code

Author: Joseph Heindel
"""

import sys
import numpy as np


class TTM:
    def __init__(self, model=21):
        """Evaluates the energy and gradients of the TTM family of potentials.

        Args:
            model (int, optional): The TTM model which will be used. Options are 2, 21, and 3. Defaults to 21.
        """
        try:
            from ttm.flib import ttm_from_f2py
        except ImportError:
            raise ValueError("Could not load the ttm module. Make sure the ttm library can be linked against "
                             "and the f2py module can be imported from this directory.")

        self.pot_function = ttm_from_f2py
        self.model = model
        possible_models = [2, 21, 3]
        if self.model not in possible_models:
            raise ValueError("The possible TTM versions are 2, 21, or 3. Please choose one of these.")

    def evaluate(self, coords):
        """Takes xyz coordinates of water molecules in O H H, O H H order and re-orders to OOHHHH order
        then transposes to fortran column-ordered matrix and calls the TTM potential from an f2py module.


        Args:
            coords (ndarray3d): xyz coordinates of a system which can be evaluated by this potential.
        Returns:
            energy (float): energy of the system in hartree
            forces (ndarray3d): forces of the system in hartree / bohr
        """
        # Sadly, we need to re-order the geometry to TTM format which is all oxygens first.
        coords = self.ttm_ordering(coords)
        gradients, energy = self.pot_function(self.model, np.asarray(coords).T)
        return energy, (-self.normal_water_ordering(gradients.T))

    @staticmethod
    def ttm_ordering(coords):
        """Sorts an array of coordinates in OHHOHH format to OOHHHH format.

        Args:
            coords (ndarray3d): numpy array of coordinates

        Returns:
            ndarray3d: numpy array of coordinate sorted according to the order TTM wants.
        """
        atom_order = []
        for i in range(0, coords.shape[0], 3):
            atom_order.append(i)
        for i in range(0, coords.shape[0], 3):
            atom_order.append(i + 1)
            atom_order.append(i + 2)
        return coords[atom_order, :]

    @staticmethod
    def normal_water_ordering(coords):
        """Sorts an array of coordinates in OOHHHH format to OHHOHH format.

        Args:
            coords (ndarray3d): numpy array of coordinates

        Returns:
            numpy array of coordinate sorted in the normal way for water.
        """
        atom_order = []
        Nw = int(coords.shape[0] / 3)
        for i in range(0, Nw, 1):
            atom_order.append(i)
            atom_order.append(Nw + 2 * i)
            atom_order.append(Nw + 2 * i + 1)
        return coords[atom_order, :]
