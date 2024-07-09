"""
Description
-----------
This module is written to execute GCMC moves with any small molecules/fragments in OpenMM, via a series of Sampler objects.
Codes were adopted and modified from grand

Kibum Park kibum@genesistherapeutics.ai
"""

import numpy as np
from collections import defaultdict
import mdtraj
import os
import logging
import parmed
import math
from copy import deepcopy
import openmm.unit as unit
import openmm
from openmmtools.integrators import NonequilibriumLangevinIntegrator

from grand.utils import random_rotation_matrix
import grand.lig_utils as lu

class BaseGCMCSampler(object):
    """
    Base class for carrying out GCMC moves in OpenMM.
    All other Sampler objects are derived from this
    """
    def __init__(self, system, topology, temperature, ligands, log='gcmc.log', overwrite=False):
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
        self.lig_res_ids = ligands
        self.ghost_lig_res_ids = []
        self.real_lig_res_ids = []
        self.lig_atom_ids = defaultdict(list)
        for resid, residue in enumerate(topology.residues()):
            if resid in self.lig_res_ids:
                for atom in residue.atoms():
                    self.lig_atom_ids[resid].append(atom.index)
        self.lig_params = defaultdict(list)
        for lig_res_id in self.lig_atom_ids.keys():
            for atom_id in self.lig_atom_ids[lig_res_id]:
                charge, simga, eps = self.nonbonded_force.getParticleParameters(atom_id)
                self.lig_params[atom_id] = [charge, simga, eps]
                self.nonbonded_force.setParticleParameters(atom_id, 0., 0., 0.,)
    
    def initialize(self,B, positions,integrator,ghosts=None):
        self.B = B
        self.positions = positions
        self.simulation = openmm.app.Simulation(
            self.topology,
            self.system,    
            integrator,
            openmm.Platform.getPlatformByName("CUDA"),  # faster if running in vacuum
        )
        self.context = self.simulation.context
        self.context.setPositions(self.positions)
        self.energy = self.context.getState(getEnergy=True).getPotentialEnergy()
        if ghosts:
            self.ghost_lig_res_ids = ghosts
        else:
            self.ghost_lig_res_ids = self.lig_res_ids
    
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

    def insert(self, atoms, insert_point=None, random_rotate=True):
        R = random_rotation_matrix()
        new_positions = deepcopy(self.positions)
        for i, index in enumerate(atoms):
            #  Translate coordinates to an origin defined by the oxygen atom, and normalise
            atom_position = self.positions[index] - self.positions[atoms[0]]
            # Rotate about the oxygen position
            if i != 0:
                vec_length = np.linalg.norm(atom_position)
                atom_position = atom_position / vec_length
                if random_rotate:
                    # Rotate coordinates & restore length
                    atom_position = vec_length * np.dot(R, atom_position)
            if insert_point is not None:
                # Translate to insertion point
                new_positions[index] = atom_position + insert_point
            else:
                new_positions[index] = atom_position
        self.adjust_specific_ligand(atoms,self.lig_params,mode='on')
        return new_positions
    
    def delete(self, atoms):
        self.adjust_specific_ligand(atoms,self.lig_params,mode='off')

    def move(self):
        if np.random.randint(2) == 1:
            # Insert
            res_id = np.random.choice(self.ghost_lig_res_ids)
            atoms = self.lig_atom_ids[res_id]
            insert_point = (np.random.rand(3) * self.simulation_box)
            new_positions = self.insert(atoms, insert_point)
            self.context.setPositions(new_positions)
            new_energy = self.context.getState(getEnergy=True).getPotentialEnergy()
            acc_prob = math.exp(self.B) * math.exp(-(new_energy - self.energy) / self.kT) / (self.N + 1)
            if acc_prob < np.random.rand() or np.isnan(acc_prob):
                self.adjust_specific_ligand(atoms,self.lig_params,mode='off')
                self.context.setPositions(self.positions)
            else:
                # Update some variables if move is accepted
                self.positions = deepcopy(new_positions)
                self.N += 1
                self.n_accepted += 1
                self.real_lig_res_ids.append(str(res_id))
                self.ghost_lig_res_ids.remove(res_id)
                # Update energy
                self.energy = new_energy
        elif len(self.real_lig_res_ids) != 0:
            # Delete
            res_id = np.random.choice(self.real_lig_res_ids)
            atoms = self.lig_atom_ids[res_id]
            self.delete(atoms)
            new_energy = self.context.getState(getEnergy=True).getPotentialEnergy()
            acc_prob = self.N * math.exp(-self.B) * math.exp(-(new_energy - self.energy) / self.kT)
            if acc_prob < np.random.rand() or np.isnan(acc_prob):
                self.adjust_specific_ligand(atoms,self.lig_params,mode='on')
                self.context.setPositions(self.positions)
            else:
                # Update some variables if move is accepted
                self.positions = deepcopy(new_positions)
                self.N += 1
                self.n_accepted += 1
                self.ghost_lig_res_ids.append(str(res_id))
                self.real_lig_res_ids.remove(res_id)
                # Update energy
                self.energy = new_energy


