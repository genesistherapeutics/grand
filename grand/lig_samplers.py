"""
Description
-----------
This module is written to execute GCMC moves with any small molecules/fragments in OpenMM, via a series of Sampler objects.
Codes were adopted and modified from grand

Kibum Park kibum@genesistherapeutics.ai
"""

import numpy as np
import mdtraj
import os
import logging
import parmed
import math
from copy import deepcopy
from simtk import unit
from simtk import openmm
from openmmtools.integrators import NonequilibriumLangevinIntegrator

import grand.lig_utils as lu

class BaseGCMCSampler(object):
    """
    Base class for carrying out GCMC moves in OpenMM.
    All other Sampler objects are derived from this
    """
    def __init__(self, system, topology, temperature, ghost, log='gcmc.log', overwrite=False):
        # Create logging object
        if os.path.isfile(log):
            if overwrite:
                os.remove(log)
            else:
                raise Exception("File {} already exists, not overwriting...".format(log))
        
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        file_handler = logging.FileHandler(log)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s: %(message)s'))
        self.logger.addHandler(file_handler)

        # Set simulation variables
        self.system = system
        self.topology = topology
        self.positions = None
        self.context = None
        self.kT = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA * temperature
        self.simulation_box = np.zeros(3) * unit.nanometer

        self.logger.info(f'kT = {self.kT.in_units_of(unit.kilocalorie_per_mole)}')

        # Find NonbondedForce - needs to be updated to switch waters on/off
        for f in range(system.getNumForces()):
            force = system.getForce(f)
            if force.__class__.__name__ == "NonbondedForce":
                self.nonbonded_force = force
            # Flag an error if not simulating at constant volume
            elif "Barostat" in force.__class__.__name__:
                self.raiseError("GCMC must be used at constant volume - {} cannot be used!".format(force.__class__.__name__))
        
        # Set GCMC-specific variables
        self.N = 0  # Initialise N as zero
        self.Ns = []  # Store all observed values of N
        self.n_moves = 0
        self.n_accepted = 0
        self.acceptance_probabilities = []  # Store acceptance probabilities

        # Get parameters for the ligand model
        self.lig_params = self.get_ligand_parameters(self.nonbonded_force)


    def adjust_specific_ligand(self, atoms, params, mode):
        for atom_id in atoms:
            if mode == 'on':
                charge, sigma, eps = params[atom_id]
            elif mode == 'off':
                charge = 0.
                sigma = 0.
                eps = 0.
            else:
                error_msg = 'Mode should be either on or off'
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            self.nonbonded_force.setParticleParameters(atom_id, charge, sigma, eps)
        self.nonbonded_force.updateParametersInContext(self.context)

    def insert(atoms, insert_point, random_rotate=True):
        if random_rotate:
            
