"""
Description
-----------
This module is written to execute NCMC moves with any small molecules/fragments in OpenMM, via a series of Sampler objects.
Codes were adopted and modified from grand

Kibum Park kibum@genesistherapeutics.ai
"""

import numpy as np
from collections import defaultdict
import mdtraj
import os
import logging
import pathlib
import parmed
import math
from copy import deepcopy
import openmm.unit as unit
import openmm
from openmmtools.integrators import NonequilibriumLangevinIntegrator, PeriodicNonequilibriumIntegrator, AlchemicalNonequilibriumLangevinIntegrator, BAOABIntegrator
from openmmtools.constants import ONE_4PI_EPS0
from openmmtools import alchemy
import time
import tqdm

from grand.utils import random_rotation_matrix

class BaseNCMCSampler(object):
    """
    Base class for carrying out NCMC moves in OpenMM.
    All other Sampler objects are derived from this
    """
    def __init__(self, reference_system, topology, positions, alchemical_atoms, alchemical_functions, reporter=None,
                 nsteps_neq=100, nsteps_eq=1000, temperature=298*unit.kelvin, periodic=False, voronoi_vertices=[], frag_atoms=[]):        
        # Set up alchemy
        self.alchemical_factory = alchemy.AbsoluteAlchemicalFactory(consistent_exceptions=True)
        self.alchemical_region = alchemy.AlchemicalRegion(alchemical_atoms=alchemical_atoms)
        self.alchemical_system = self.alchemical_factory.create_alchemical_system(reference_system, self.alchemical_region)
        self.alchemical_state = alchemy.AlchemicalState.from_system(self.alchemical_system)
        self.alchemical_functions = alchemical_functions
        self.alchemical_atoms = alchemical_atoms
        self.nsteps_neq = nsteps_neq
        self.nsteps_eq = nsteps_eq
        self.insert_points_list = voronoi_vertices
        self.frag_atoms = frag_atoms
        self.n_accepted = 0
        self.n_trial = 0

        # Set up logger - will be implemented later
        self._logger = logging.getLogger()
        self._logger.setLevel(logging.INFO)
        self.save_freq_eq = 10000000
        self.save_freq_neq = 4000000000

        # Set up integrator
        self.temperature = temperature
        self.kT = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA * temperature
        if periodic:
            self.nc_integrator = PeriodicNonequilibriumIntegrator(
                temperature=self.temperature, collision_rate=1.0/unit.picoseconds, timestep=1.0*unit.femtoseconds,
                alchemical_functions=self.alchemical_functions, splitting="V R H O R V",
                nsteps_eq=self.nsteps_eq, nsteps_neq=self.nsteps_neq
            )
            self.nsteps_per_period = 2*self.nsteps_eq + 2*self.nsteps_neq
        else:
            self.nc_integrator = AlchemicalNonequilibriumLangevinIntegrator(
                temperature=self.temperature, collision_rate=1.0/unit.picoseconds, timestep=1.0*unit.femtoseconds,
                alchemical_functions=self.alchemical_functions, splitting="V R H O R V",
                nsteps_neq=self.nsteps_neq
            )
            self.nsteps_per_period = self.nsteps_neq
        self.md_integrator = BAOABIntegrator(298*unit.kelvin, 1.0/unit.picosecond, 0.0005*unit.picosecond, measure_heat=True, measure_shadow_work=True)
        self.compound_integrator = openmm.CompoundIntegrator()
        self.compound_integrator.addIntegrator(self.nc_integrator)
        self.compound_integrator.addIntegrator(self.md_integrator)

        # Set up simulation and reporter
        platform = openmm.Platform.getPlatformByName("CUDA")
        self.simulation = openmm.app.Simulation(
            topology,
            self.alchemical_system,
            self.compound_integrator,
            platform,
        )
        if reporter:
            self.reporter = reporter
            self.simulation.reporters.append(self.reporter)

        self.positions = positions
        self.context = self.simulation.context
        self.context.setPositions(self.positions)
        self.context.setVelocitiesToTemperature(self.temperature)

        # Minimize
        self.simulation.minimizeEnergy()

    def insert_fragment(self, atoms, insert_point=None, random_rotate=True):
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
        return new_positions

    def ncmc_move(self, nsteps):
        for step in tqdm.tqdm(range(nsteps)):
            protocol_work = 0
            insert_point = self.insert_points_list[np.random.choice(len(self.insert_points_list))] * unit.nanometer
            #insert_point = openmm.Vec3(0.4961, 1.4597, 2.3242) * openmm.unit.nanometer
            new_positions = self.insert_fragment(self.frag_atoms, insert_point)
            self.context.setPositions(new_positions)
            self.reporter.report(self.simulation,self.context.getState(getPositions=True,getEnergy=True))
            self.compound_integrator.setCurrentIntegrator(1)
            # Minimize energy
            #self.simulation.minimizeEnergy()

            # NC steps without MD
            self.simulation.step(self.nsteps_neq)
            
            # NC steps with MD
            """
            for neq_step in range(self.nsteps_neq):
                # print(self.nc_integrator.getGlobalVariableByName('lambda'))
                self.compound_integrator.setCurrentIntegrator(0)
                self.compound_integrator.step(1)
                self.compound_integrator.setCurrentIntegrator(1)
                # Run MD
                self.compound_integrator.step(self.nsteps_eq)
                # Run Minimization
                #self.simulation.minimizeEnergy(maxIterations=100)
            """
            log_acc_prob = - (self.nc_integrator.protocol_work)/self.kT
            # Reject
            if log_acc_prob < np.log(np.random.rand()) or np.isnan(log_acc_prob):
                self.context.setPositions(self.positions)
            # Accept
            else:
                self.positions = deepcopy(self.context.getState(getPositions=True,getEnergy=True).getPositions())
                self.context.setPositions(self.positions)
                self.reporter.report(self.simulation,self.context.getState(getPositions=True,getEnergy=True))
            self.nc_integrator.reset()

    def test_insert(self):
        insert_point = self.insert_points_list[np.random.choice(len(self.insert_points_list))] * unit.nanometer
        new_positions = self.insert_fragment(self.alchemical_atoms, insert_point)
        self.context.setPositions(new_positions)
        self.reporter.report(self.simulation,self.context.getState(getPositions=True,getEnergy=True))