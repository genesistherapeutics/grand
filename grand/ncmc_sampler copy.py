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
import grand.integrators as ig

from grand.utils import random_rotation_matrix

class BaseNCMCSampler(object):
    """
    Base class for carrying out NCMC moves in OpenMM.
    All other Sampler objects are derived from this
    """
    def __init__(self, reference_system, topology, positions, alchemical_atoms, alchemical_functions, reporters=[],
                 nsteps_neq=100, nsteps_eq=1000, temperature=298*unit.kelvin, insert_points_list=[], frag_atoms=[]):        
        # Set up alchemy
        self.alchemical_factory = alchemy.AbsoluteAlchemicalFactory(consistent_exceptions=True)
        self.alchemical_region = alchemy.AlchemicalRegion(alchemical_atoms=alchemical_atoms)
        self.alchemical_system = self.alchemical_factory.create_alchemical_system(reference_system, self.alchemical_region)
        self.alchemical_state = alchemy.AlchemicalState.from_system(self.alchemical_system)
        self.alchemical_atoms = alchemical_atoms
        self.nsteps_neq = nsteps_neq
        self.nsteps_eq = nsteps_eq
        self.insert_points_list = insert_points_list
        self.frag_atoms = frag_atoms
        self.n_accepted = 0
        self.n_trial = 0

        # Set up logger - will be implemented later
        self._logger = logging.getLogger()
        self._logger.setLevel(logging.INFO)
        #self.save_freq_eq = 10000000
        self.save_freq = 10000

        # Set up integrator
        self.temperature = temperature
        self.kT = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA * temperature
        self.md_integrator = BAOABIntegrator(self.temperature, 1.0/unit.picosecond, 0.002*unit.picosecond, measure_heat=True, measure_shadow_work=True)
        x = 'lambda'
        DEL_ALCHEMICAL_FUNCTIONS = {
                             'lambda_sterics': f"1-min(1.0, 2.0*{x})",
                             'lambda_electrostatics': f"1-max(0.0, 2.0*({x}-0.5))",
                             }
        INS_ALCHEMICAL_FUNCTIONS = {
                             'lambda_electrostatics': f"min(1.0, 2.0*{x})",
                             'lambda_sterics': f"max(0.0, 2.0*({x}-0.5))",
                             }
        self.delete_integrator = AlchemicalNonequilibriumLangevinIntegrator(
                temperature=self.temperature, collision_rate=.1/unit.picoseconds, timestep=0.1*unit.femtoseconds,
                alchemical_functions=DEL_ALCHEMICAL_FUNCTIONS, splitting="O V R H R V O",
                nsteps_neq=self.nsteps_neq, measure_heat=True, measure_shadow_work=True
            )
        self.insert_integrator = AlchemicalNonequilibriumLangevinIntegrator(
                temperature=self.temperature, collision_rate=.1/unit.picoseconds, timestep=0.1*unit.femtoseconds,
                alchemical_functions=INS_ALCHEMICAL_FUNCTIONS, splitting="O V R H R V O",
                nsteps_neq=self.nsteps_neq*10, measure_heat=True, measure_shadow_work=True
            )
        self.cyclic_integrator = openmm.CompoundIntegrator()
        self.cyclic_integrator.addIntegrator(self.delete_integrator)
        self.cyclic_integrator.addIntegrator(self.insert_integrator)
        self.cyclic_integrator.addIntegrator(self.md_integrator)

        self.delete_integrator.addGlobalVariable('pe', 0)
        self.delete_integrator.addComputeGlobal('pe', 'energy')
        self.md_integrator.addGlobalVariable('pe', 0)
        self.md_integrator.addComputeGlobal('pe', 'energy')

        # Set up simulation and reporter
        platform = openmm.Platform.getPlatformByName("CUDA")
        platform.setPropertyDefaultValue('Precision', 'mixed')
        platform.setPropertyDefaultValue('DeterministicForces', 'true')
        self.topology = topology
        self.simulation = openmm.app.Simulation(
            self.topology,
            self.alchemical_system,
            self.cyclic_integrator,
            platform,
        )
        if len(reporters) > 0:
            for reporter in reporters:
                self.simulation.reporters.append(reporter)
            self.reporter = self.simulation.reporters

        self.context = self.simulation.context
        self.context.setPeriodicBoxVectors(*self.alchemical_system.getDefaultPeriodicBoxVectors())
        self.context.setPositions(positions)
        self.context.setVelocitiesToTemperature(temperature)

        # Compute ligand center of mass
        lig_pos = []
        self.lig_mass = []
        for residue in self.topology.residues():
            if residue.name == "UNK":
                for atom in residue.atoms():
                    print(atom)
                    lig_pos.append(positions[atom.index].value_in_unit(openmm.unit.nanometer))
                    self.lig_mass.append(atom.element._mass)

        self.center_of_mass = parmed.geometry.center_of_mass(np.array(lig_pos), np.array(self.lig_mass)) * openmm.unit.nanometer

        # Minimize
        self.simulation.minimizeEnergy()
        self.cyclic_integrator.setCurrentIntegrator(2)
        self.cyclic_integrator.step(self.nsteps_eq)

        # Store minimized coords
        self.positions = deepcopy(self.context.getState(getPositions=True).getPositions())
        self.velocities = self.context.getState(getVelocities=True).getVelocities(asNumpy=True)

    def insert_fragment(self, atoms, insert_point=None, random_rotate=True):
        R = random_rotation_matrix()
        new_positions = deepcopy(self.context.getState(getPositions=True).getPositions())
        self.center_of_mass = parmed.geometry.center_of_mass(np.array(new_positions.value_in_unit(openmm.unit.nanometer))[atoms], np.array(self.lig_mass)) * openmm.unit.nanometer
        for i, index in enumerate(atoms):
            #  Translate coordinates to an origin defined by the oxygen atom, and normalise
            atom_position = self.positions[index] - self.center_of_mass #self.positions[atoms[0]]
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
    
    def insert_fragment2(self, atoms, insert_point=None, random_rotate=True):
        
        new_positions = deepcopy(self.context.getState(getPositions=True).getPositions())

        lig_positions = np.array(new_positions.value_in_unit(openmm.unit.nanometer))[atoms]*openmm.unit.nanometer
        self.center_of_mass = parmed.geometry.center_of_mass(lig_positions, np.array(self.lig_mass)) * openmm.unit.nanometer
        reduced_pos = lig_positions - self.center_of_mass

        # Define random rotational move on the ligand
        rand_quat = mdtraj.utils.uniform_quaternion(size=None, random_state=None)
        rand_rotation_matrix = mdtraj.utils.rotation_matrix_from_quaternion(rand_quat)
        #multiply lig coordinates by rot matrix and add back COM translation from origin
        if insert_point:
            rot_move = np.dot(reduced_pos, rand_rotation_matrix) * openmm.unit.nanometer + insert_point
        else:
            rot_move = np.dot(reduced_pos, rand_rotation_matrix) * openmm.unit.nanometer + self.center_of_mass
        # Update ligand positions in nc_sim
        for index, atomidx in enumerate(atoms):
            new_positions[atomidx] = rot_move[index]
        return new_positions

    def ncmc_move(self, nsteps):
        for step in tqdm.tqdm(range(nsteps)):
            # Deletion
            self.cyclic_integrator.setCurrentIntegrator(0)
            self.delete_integrator.reset()
            self.cyclic_integrator.step(self.nsteps_neq)
            #self.reporter.report(self.simulation,self.context.getState(getPositions=True,getEnergy=True))
            # Insertion
            insert_point = (self.insert_points_list[np.random.choice(len(self.insert_points_list))] ) * unit.nanometer
            #+ np.random.normal(loc=0.,scale=.3,size=3)
            new_positions = self.insert_fragment2(self.frag_atoms, insert_point)
            self.context.setPositions(new_positions)
            self.context.setVelocitiesToTemperature(self.temperature)
            self.reporter[0].report(self.simulation,self.context.getState(getPositions=True,getEnergy=True))
            #openmm.app.PDBFile.writeFile(self.topology,self.context.getState(getPositions=True,getEnergy=True).getPositions(),f'rotated_{step}.pdb')
            #self.reporter[0].report(self.simulation,self.context.getState(getPositions=True,getEnergy=True))
            self.cyclic_integrator.setCurrentIntegrator(1)
            self.insert_integrator.reset()
            self.cyclic_integrator.step(self.nsteps_neq)
            # Relaxing side chain
            self.cyclic_integrator.setCurrentIntegrator(2)
            self.context.setVelocitiesToTemperature(self.temperature)
            #self.reporter.report(self.simulation,self.context.getState(getPositions=True,getEnergy=True))
            oldE = self.md_integrator.getGlobalVariableByName('pe') * openmm.unit.kilojoules_per_mole
            #self.cyclic_integrator.step(self.nsteps_eq)
            newE = self.md_integrator.getGlobalVariableByName('pe') * openmm.unit.kilojoules_per_mole
            #self.reporter.report(self.simulation,self.context.getState(getPositions=True,getEnergy=True))
            #for neq in range(self.nsteps_neq):
                #self.cyclic_integrator.step(self.nsteps_neq)
                #print(self.context.getParameter('lambda_sterics'))
                #self.cyclic_integrator.step(1)
                #self.reporter.report(self.simulation,self.context.getState(getPositions=True,getEnergy=True))
            #log_acc_prob = - (self.delete_integrator.protocol_work + self.insert_integrator.protocol_work + newE - oldE)/self.kT
            log_acc_prob = - (self.delete_integrator.total_work + self.insert_integrator.total_work + newE - oldE)/self.kT
            # Reject
            if log_acc_prob < np.log(np.random.rand()) or np.isnan(log_acc_prob):
                print(self.delete_integrator.protocol_work,self.insert_integrator.protocol_work,self.insert_integrator.total_work, self.insert_integrator.heat, newE-oldE, 'rejected')
                self.context.setPositions(self.positions)
                self.context.setVelocities(-self.velocities)
                #self.context.setVelocitiesToTemperature(self.temperature)

                self.cyclic_integrator.setCurrentIntegrator(2)
                self.cyclic_integrator.step(self.nsteps_eq)

                #self.reporter[0].report(self.simulation,self.context.getState(getPositions=True,getEnergy=True))
                self.reporter[1].report(self.simulation,self.context.getState(getPositions=True,getEnergy=True))
                
            # Accept
            else:
                print(self.delete_integrator.protocol_work,self.insert_integrator.protocol_work,self.insert_integrator.total_work, self.insert_integrator.heat, newE-oldE, 'accepted')
                self.n_accepted += 1
                
                self.cyclic_integrator.setCurrentIntegrator(2)
                self.cyclic_integrator.step(self.nsteps_eq)

                self.positions = deepcopy(self.context.getState(getPositions=True).getPositions())
                #openmm.app.PDBFile.writeFile(self.topology,self.positions,f'snap_{self.n_accepted}.pdb')
                self.context.setVelocitiesToTemperature(self.temperature)
                #self.context.setPositions(self.positions)
                #self.reporter[0].report(self.simulation,self.context.getState(getPositions=True,getEnergy=True))
                self.reporter[1].report(self.simulation,self.context.getState(getPositions=True,getEnergy=True))

            #self.reporter[0].report(self.simulation,self.context.getState(getPositions=True,getEnergy=True))
    