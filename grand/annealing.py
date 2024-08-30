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

class BaseGCNCMCSampler(object):
    """
    Base class for carrying out NCMC moves in OpenMM.
    All other Sampler objects are derived from this
    """
    def __init__(self, reference_system, topology, positions, alchemical_atoms, reporters=[], ligands=[],
                 nsteps_neq=100, nsteps_eq=1000, temperature=298*unit.kelvin, insert_points_list=[], frag_atoms=[]):
        # Set up alchemy
        self.alchemical_factory = alchemy.AbsoluteAlchemicalFactory(consistent_exceptions=True)
        self.alchemical_region = alchemy.AlchemicalRegion(alchemical_atoms=alchemical_atoms)
        self.reference_system = reference_system
        self.alchemical_system = self.alchemical_factory.create_alchemical_system(reference_system, self.alchemical_region)
        self.alchemical_state = alchemy.AlchemicalState.from_system(self.alchemical_system)
        self.alchemical_atoms = alchemical_atoms
        self.nsteps_neq = nsteps_neq
        self.nsteps_eq = nsteps_eq
        self.insert_points_list = insert_points_list
        self.frag_atoms = frag_atoms
        self.n_accepted = 0
        self.n_trial = 0

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
                alchemical_functions=DEL_ALCHEMICAL_FUNCTIONS, splitting="R V O H O V R",
                nsteps_neq=self.nsteps_neq, measure_heat=True, measure_shadow_work=True
            )
        self.insert_integrator = AlchemicalNonequilibriumLangevinIntegrator(
                temperature=self.temperature, collision_rate=.1/unit.picoseconds, timestep=0.1*unit.femtoseconds,
                alchemical_functions=INS_ALCHEMICAL_FUNCTIONS, splitting="R V O H O V R",
                nsteps_neq=self.nsteps_neq, measure_heat=True, measure_shadow_work=True
            )
        self.compound_integrator = openmm.CompoundIntegrator()
        self.compound_integrator.addIntegrator(self.delete_integrator)
        self.compound_integrator.addIntegrator(self.insert_integrator)
        self.compound_integrator.addIntegrator(self.md_integrator)

        self.md_integrator.addGlobalVariable('pe', 0)
        self.md_integrator.addComputeGlobal('pe', 'energy')

        # Set up simulation and reporter
        self.platform = openmm.Platform.getPlatformByName("CUDA")
        self.platform.setPropertyDefaultValue('Precision', 'mixed')
        self.platform.setPropertyDefaultValue('DeterministicForces', 'true')
        self.topology = topology
        self.simulation = openmm.app.Simulation(
            self.topology,
            self.alchemical_system,
            self.compound_integrator,
            self.platform,
        )
        if len(reporters) > 0:
            for reporter in reporters:
                self.simulation.reporters.append(reporter)
            self.reporter = self.simulation.reporters

        self.context = self.simulation.context
        self.context.setPeriodicBoxVectors(*self.alchemical_system.getDefaultPeriodicBoxVectors())
        self.context.setPositions(positions)
        self.context.setVelocitiesToTemperature(temperature)

        # Set interaction group
        self.ghosts = []
        self.reals = []
        self.lig_res_ids = []
        self.real_ligs = []
        self.ghost_ligs = []
        self.lig_atom_ids = defaultdict(list)
        for resid, residue in enumerate(self.topology.residues()):
            if residue.name == 'UNK':
                self.lig_res_ids.append(resid)
                for atom in residue.atoms():
                    self.lig_atom_ids[resid].append(atom.index)
            elif residue.name == 'HOH':
                for atom in residue.atoms():
                    self.reals.append(atom.index)
            else:
                for atom in residue.atoms():
                    self.reals.append(atom.index)
        for lig_res in self.lig_res_ids:
            if np.random.random() < 0.5:
                self.ghosts.extend(self.lig_atom_ids[lig_res])
                self.ghost_ligs.append(lig_res)
            else:
                self.reals.extend(self.lig_atom_ids[lig_res])
                self.real_ligs.append(lig_res)

        i = 1
        self.custom_nb_forces = []
        self.interaction_groups = []
        for force in self.alchemical_system.getForces():
            force.setForceGroup(i)
            i += 1
            #if force.__class__.__name__ == "NonbondedForce":
            #    for j in range(force.getNumParticles()):
            #        assert (force.getParticleParameters(j)[0] == 0) and (force.getParticleParameters(j)[2] == 0)
            if force.__class__.__name__ == "CustomNonbondedForce":
                self.custom_nb_forces.append(force)    
                self.interaction_groups.append(force.addInteractionGroup(self.reals,self.reals))


        # Minimize
        self.simulation.minimizeEnergy()
        self.compound_integrator.setCurrentIntegrator(2)
        self.compound_integrator.step(self.nsteps_eq)

        # Store minimized coords
        self.positions = deepcopy(self.context.getState(getPositions=True).getPositions())
        self.velocities = self.context.getState(getVelocities=True).getVelocities(asNumpy=True)

    def insert_fragment(self, atoms, insert_point=None, random_rotate=True):
        R = random_rotation_matrix()
        new_positions = deepcopy(self.context.getState(getPositions=True).getPositions())
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

    def gcncmc_move(self, nsteps, B):
        for step in tqdm.tqdm(range(nsteps)):
            if np.random.random() < 0.5:
                # Deletion
                res_id = np.random.choice(self.real_ligs)
                frag_atoms = self.lig_atom_ids[res_id]
                self.real_ligs.remove(res_id)
                self.ghost_ligs.append(res_id)
                for atom_id in self.lig_atom_ids[res_id]:
                    self.reals.remove(atom_id)
                for idx, nb in enumerate(self.custom_nb_forces):
                    nb.setInteractionGroupParameters(self.interaction_groups[idx],self.reals,self.reals)
                self.compound_integrator.setCurrentIntegrator(0)
                self.delete_integrator.reset()
                self.compound_integrator.step(self.nsteps_eq)
                log_acc_prob = -(self.delete_integrator.protocol_work)/self.kT - B + len(self.real_ligs) - 1

                # Reject
                if log_acc_prob < np.log(np.random.rand()) or np.isnan(log_acc_prob):
                    print(self.delete_integrator.protocol_work, self.delete_integrator.total_work, 'deletion rejected')
                    self.context.setPositions(self.positions)
                    self.context.setVelocities(-self.velocities)

                    self.real_ligs.append(res_id)
                    self.ghost_ligs.remove(res_id)
                    self.reals.extend(self.lig_atom_ids[res_id])

                    self.reporter[0].report(self.simulation,self.context.getState(getPositions=True,getEnergy=True))
                    self.reporter[1].report(self.simulation,self.context.getState(getPositions=True,getEnergy=True))
                    
                # Accept
                else:
                    print(self.delete_integrator.protocol_work, self.delete_integrator.total_work, 'deletion accepted')
                    self.n_accepted += 1

                    self.positions = deepcopy(self.context.getState(getPositions=True).getPositions())
                    self.context.setVelocitiesToTemperature(self.temperature)
                    
                    self.reporter[0].report(self.simulation,self.context.getState(getPositions=True,getEnergy=True))
                    self.reporter[1].report(self.simulation,self.context.getState(getPositions=True,getEnergy=True))

            else:
                # Insertion
                res_id = np.random.choice(self.ghost_ligs)
                frag_atoms = self.lig_atom_ids[res_id]
                self.ghost_ligs.remove(res_id)
                self.real_ligs.append(res_id)
                self.reals.extend(self.lig_atom_ids[res_id])
                for idx, nb in enumerate(self.custom_nb_forces):
                    nb.setInteractionGroupParameters(self.interaction_groups[idx],self.reals,self.reals)
                insert_point = (self.insert_points_list[np.random.choice(len(self.insert_points_list))] + np.random.normal(loc=0.,scale=.3,size=3)* unit.nanometer) 
                new_positions = self.insert_fragment(frag_atoms, insert_point)
                self.context.setPositions(new_positions)
                self.context.setVelocitiesToTemperature(self.temperature)
                self.compound_integrator.setCurrentIntegrator(1)
                self.insert_integrator.reset()
                self.compound_integrator.step(self.nsteps_neq)
                log_acc_prob = -(self.insert_integrator.protocol_work)/self.kT + B - len(self.real_ligs)

                # Reject
                if log_acc_prob < np.log(np.random.rand()) or np.isnan(log_acc_prob):
                    print(self.insert_integrator.protocol_work, self.insert_integrator.total_work, 'insertion rejected')
                    self.context.setPositions(self.positions)
                    self.context.setVelocities(-self.velocities)

                    self.ghost_ligs.append(res_id)
                    self.real_ligs.remove(res_id)
                    for atom_id in self.lig_atom_ids[res_id]:
                        self.reals.remove(atom_id)

                    self.reporter[0].report(self.simulation,self.context.getState(getPositions=True,getEnergy=True))
                    self.reporter[1].report(self.simulation,self.context.getState(getPositions=True,getEnergy=True))
                    
                # Accept
                else:
                    print(self.insert_integrator.protocol_work, self.insert_integrator.total_work, 'insertion accepted')
                    self.n_accepted += 1

                    self.positions = deepcopy(self.context.getState(getPositions=True).getPositions())
                    self.context.setVelocitiesToTemperature(self.temperature)
                    
                    self.reporter[0].report(self.simulation,self.context.getState(getPositions=True,getEnergy=True))
                    self.reporter[1].report(self.simulation,self.context.getState(getPositions=True,getEnergy=True))
            
            self.compound_integrator.setCurrentIntegrator(2)
            self.compound_integrator.step(self.nsteps_eq)