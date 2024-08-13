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
from openmmtools.constants import ONE_4PI_EPS0

from grand.utils import random_rotation_matrix
import grand.lig_utils as lu

class BaseGCMCSampler(object):
    """
    Base class for carrying out GCMC moves in OpenMM.
    All other Sampler objects are derived from this
    """
    def __init__(self, forcefield, topology, temperature, ligands=[], custom_force=True, log='gcmc.log', overwrite=False,
                 B=-6., insert_prob=0.5, positions=None,integrator=None,reporter=None, ghosts=None,
                 water_resn='HOH'):
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

        # Set OpenMM system variables
        self.forcefield = forcefield
        self.topology = topology
        self.system = self.forcefield.createSystem(self.topology)
        self.positions = positions
        i = 1
        for force in self.system.getForces():
            force.setForceGroup(i)
            i += 1

        # Set GCMC related variables
        self.B = B
        self.temperature = temperature
        self.insert_prob = insert_prob
        self.kT = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA * temperature
        self.logger.info(f'kT = {self.kT.in_units_of(unit.kilocalorie_per_mole)}')

        # Find NonbondedForce - needs to be updated to switch waters on/off
        for f in range(self.system.getNumForces()):
            force = self.system.getForce(f)
            if force.__class__.__name__ == "NonbondedForce":
                self.nonbonded_force = force
                self.nonbonded_force_index = f
            # Flag an error if not simulating at constant volume
            elif "Barostat" in force.__class__.__name__:
                self.raiseError("GCMC must be used at constant volume - {} cannot be used!".format(force.__class__.__name__))
        
        # Set GCMC-specific variables
        self.N = 0  # Initialise N as zero
        self.Ns = []  # Store all observed values of N
        self.n_moves = 0
        self.n_accepted = 0
        self.acceptance_probabilities = []  # Store acceptance probabilities
        
        # Store ligand/water/protein related variables
        self.lig_res_ids = ligands
        if ghosts:
            self.ghost_lig_res_ids = ghosts
        else:
            self.ghost_lig_res_ids = self.lig_res_ids
        self.real_lig_res_ids = []
        self.lig_atom_ids = defaultdict(list)
        self.wat_atom_ids = defaultdict(list)
        self.prot_positions = []
        for resid, residue in enumerate(topology.residues()):
            if resid in self.lig_res_ids:
                for atom in residue.atoms():
                    self.lig_atom_ids[resid].append(atom.index)
            elif residue.name == water_resn:
                self.water_res_ids.append(resid)
                for atom in residue.atoms():
                    self.wat_atom_ids[resid].append(atom.index)
            else:
                for atom in residue.atoms():
                    self.prot_positions.append(self.positions[atom.index])
        self.prot_positions = unit.Quantity(self.prot_positions)

        # Customize force so that ligand-ligand interaction can be ignored
        if custom_force:
            self.customize_forces()

        # Set OpenMM simulation variables
        self.simulation = openmm.app.Simulation(
            self.topology,
            self.system,    
            integrator,
            openmm.Platform.getPlatformByName("CUDA"),  # faster if running in vacuum
        )
        self.context = self.simulation.context
        self.context.setPositions(self.positions)
        if reporter:
            self.reporter = reporter
            self.simulation.reporters.append(self.reporter)


        # Get forcefield parameters per atom and store it as a dict
        self.lig_params = defaultdict(list)
        for lig_res_id in self.lig_atom_ids.keys():
            for atom_id in self.lig_atom_ids[lig_res_id]:
                if custom_force:
                    soluteFlag, charge, sigma, eps, l = self.nonbonded_force.getParticleParameters(atom_id)
                    self.lig_params[atom_id] = [charge, sigma, eps]
                    self.nonbonded_force.setParticleParameters(atom_id, [1., 0., sigma, 0., 0.])
                else:
                    charge, sigma, eps = self.nonbonded_force.getParticleParameters(atom_id)
                    self.lig_params[atom_id] = [charge, sigma, eps]
                    self.nonbonded_force.setParticleParameters(atom_id, 0., sigma, 0.)
        self.nonbonded_force.updateParametersInContext(self.context)

        # Compute energy and forces
        for f in range(self.system.getNumForces()):
            print(self.system.getForce(f).__class__.__name__)
            force_group = self.system.getForce(f).getForceGroup()
            print(force_group)
            print(self.context.getState(getEnergy=True,groups={force_group}).getPotentialEnergy())
        self.energy = self.context.getState(getEnergy=True).getPotentialEnergy()
        print(f'Initial Energy: {self.energy}')


        # Define simualtion box/GCMC box
        # This was set to be a box around a protein with a padding of n angstrom
        self.simulation_box = np.zeros(3) * unit.nanometer
        self.min_dimension = self.prot_positions.min() - np.array([0.3,0.3,0.3]) * unit.nanometer
        self.max_dimension = self.prot_positions.max() + np.array([0.3,0.3,0.3]) * unit.nanometer

    def customize_forces(self):
        """
        Create a CustomNonbondedForce to handle ligand-ligand interactions
        For custom steric (LJ potential), soft core potential will be used
        For custom coulomb force, cufoff periodic with reaction field will be used
        These custom forces are applied to turn off ligand-ligand interactions
        """
        #if self.nonbonded_force.getNonbondedMethod() != openmm.NonbondedForce.CutoffPeriodic:
        #    self.raiseError("Currently only supporting CutoffPeriodic for long range electrostatics")
        #else:
        eps_solvent = self.nonbonded_force.getReactionFieldDielectric()
        cutoff = self.nonbonded_force.getCutoffDistance()
        krf = (1/ (cutoff**3)) * (eps_solvent - 1) / (2*eps_solvent + 1)
        crf = (1/ cutoff) * (3* eps_solvent) / (2*eps_solvent + 1)

        energy_expression  = "select(condition, 0, 1)*all;"
        energy_expression += "condition = soluteFlag1*soluteFlag2;" #solute must have flag int(1)
        energy_expression += "all=(lambda^soft_a) * 4 * epsilon * x * (x-1.0) + ONE_4PI_EPS0*chargeprod*(1/r + krf*r*r - crf);"
        energy_expression += "x = (sigma/reff)^6;"  # Define x as sigma/r(effective)
        energy_expression += "reff = sigma*((soft_alpha*(1.0-lambda)^soft_b + (r/sigma)^soft_c))^(1/soft_c);" # Calculate effective distance
        energy_expression += "lambda = lambda1*lambda2;"
        energy_expression += "epsilon = epsilon1*epsilon2;"
        energy_expression += "sigma = 0.5*(sigma1+sigma2);"
        energy_expression += "ONE_4PI_EPS0 = {:f};".format(ONE_4PI_EPS0)  # already in OpenMM units
        energy_expression += "chargeprod = charge1*charge2;"
        energy_expression += "krf = {:f};".format(krf.value_in_unit(unit.nanometer**-3))
        energy_expression += "crf = {:f};".format(crf.value_in_unit(unit.nanometer**-1))
        custom_nonbonded_force = openmm.CustomNonbondedForce(energy_expression)
        custom_nonbonded_force.addPerParticleParameter('soluteFlag')
        custom_nonbonded_force.addPerParticleParameter('charge')
        custom_nonbonded_force.addPerParticleParameter('sigma')
        custom_nonbonded_force.addPerParticleParameter('epsilon')
        custom_nonbonded_force.addPerParticleParameter('lambda')
        # Configure force
        custom_nonbonded_force.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
        custom_nonbonded_force.setUseSwitchingFunction(self.nonbonded_force.getUseSwitchingFunction())
        custom_nonbonded_force.setCutoffDistance(self.nonbonded_force.getCutoffDistance())
        custom_nonbonded_force.setSwitchingDistance(self.nonbonded_force.getSwitchingDistance())
        self.nonbonded_force.setUseDispersionCorrection(False)
        custom_nonbonded_force.setUseLongRangeCorrection(self.nonbonded_force.getUseDispersionCorrection())
        # Set softcore parameters
        custom_nonbonded_force.addGlobalParameter('soft_alpha', 0.5)
        custom_nonbonded_force.addGlobalParameter('soft_a', 1)
        custom_nonbonded_force.addGlobalParameter('soft_b', 1)
        custom_nonbonded_force.addGlobalParameter('soft_c', 6)

        lig_atom_ids = []
        for resid, residue in enumerate(self.topology.residues()):
            if resid in self.lig_res_ids:
                for atom in residue.atoms():
                    lig_atom_ids.append(atom.index)

        # Copy all steric interactions into the custom force, and remove them from the original force
        for atom_idx in range(self.nonbonded_force.getNumParticles()):
            # Get atom parameters
            [charge, sigma, epsilon] = self.nonbonded_force.getParticleParameters(atom_idx)
            # Make sure that sigma is not equal to zero
            if np.isclose(sigma._value, 0.0):
                sigma = 1.0 * unit.angstrom
            # Add particle to the custom force (with lambda=1 for now)
            if atom_idx in lig_atom_ids:
                custom_nonbonded_force.addParticle([1.0, charge, sigma, epsilon, 1.0])
            else:
                custom_nonbonded_force.addParticle([0.0, charge, sigma, epsilon, 1.0])

        # Copy over all exceptions into the new force as custom bonded forces
        # Exceptions between non-ligand atoms will be excluded here, and handled by the NonbondedForce
        # If exceptions (other than ignored interactions) are found involving ligand atoms, we have a problem
        energy_expression = "4*epsilon*((sigma/r)^12 - (sigma/r)^6) + ONE_4PI_EPS0*chargeprod*(1/r);"
        energy_expression += "ONE_4PI_EPS0 = {:f};".format(ONE_4PI_EPS0)
        custom_bond_force = openmm.CustomBondForce(energy_expression)
        custom_bond_force.addPerBondParameter('chargeprod')
        custom_bond_force.addPerBondParameter('sigma')
        custom_bond_force.addPerBondParameter('epsilon')
        
        for exception_idx in range(self.nonbonded_force.getNumExceptions()):
            [i, j, chargeprod, sigma, epsilon] = self.nonbonded_force.getExceptionParameters(exception_idx)
            # Add this to the list of exclusions
            custom_nonbonded_force.addExclusion(i, j)
            if i in lig_atom_ids and j in lig_atom_ids:
                pass
            else:
                custom_bond_force.addBond(i, j, [chargeprod, sigma, epsilon])

        # Update system
        new_index = self.system.addForce(custom_nonbonded_force)
        self.system.addForce(custom_bond_force)
        self.system.removeForce(self.nonbonded_force_index)
        self.nonbonded_force = custom_nonbonded_force
        self.nonbonded_force_index = new_index

        return None


    def adjust_specific_ligand(self, atoms, params, mode, l=1.0):
        for atom_id in atoms:
            if mode == 'on':
                charge, sigma, eps = params[atom_id]
            elif mode == 'off':
                charge, sigma, eps = params[atom_id]
                charge = 0.
                eps = 0.
            else:
                error_msg = 'Mode should be either on or off'
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            self.nonbonded_force.setParticleParameters(atom_id, [1., charge, sigma, eps, l])
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
        return new_positions
    
    def delete(self, atoms):
        self.adjust_specific_ligand(atoms,self.lig_params,mode='off')

    def move(self):
        if np.random.rand() < self.insert_prob:
            # Insert
            res_id = np.random.choice(self.ghost_lig_res_ids)
            atoms = self.lig_atom_ids[res_id]
            #for atom_id in atoms:
            #    charge, sigma, eps = self.lig_params[atom_id]
            #    self.nonbonded_force.setParticleParameters(atom_id, charge, sigma, eps)
            #insert_point = (np.random.rand(3) * self.simulation_box)
            insert_point = (np.random.rand(3) * (self.max_dimension - self.min_dimension).value_in_unit(unit.nanometer)) * unit.nanometer + self.min_dimension
            new_positions = self.insert(atoms, insert_point)
            self.context.setPositions(new_positions)
            self.adjust_specific_ligand(atoms,self.lig_params,mode='on')
            self.nonbonded_force.updateParametersInContext(self.context)
            #print(self.context.getState(getEnergy=True).getPotentialEnergy())
            #self.context.setVelocitiesToTemperature(self.temperature)
            try:
                #self.simulation.step(1000)
                self.simulation.minimizeEnergy()
                #self.simulation.minimizeEnergy(maxIterations=100)
                new_energy = self.context.getState(getEnergy=True).getPotentialEnergy()
                #new_velocities = self.context.getState(getVelocities=True).getVelocities()
                #self.reporter.report(self.simulation,self.context.getState(getPositions=True,getEnergy=True))
                #try:
                #    acc_prob = math.exp(self.B -(new_energy - self.energy) / self.kT) / (self.N + 1)
                #except OverflowError:
                #    acc_prob = float('inf')
                log_acc_prob = self.B - (new_energy - self.energy)/self.kT - self.N - 1
                if log_acc_prob < np.log(np.random.rand()) or np.isnan(log_acc_prob):
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
                    #self.velocities = new_velocities
                    self.reporter.report(self.simulation,self.context.getState(getPositions=True,getEnergy=True))
            except:
                print('Explosion')
                self.context.setPositions(self.positions)
                self.adjust_specific_ligand(atoms,self.lig_params,mode='off')
                pass
            
        else:
            # Delete
            res_id = np.random.choice(self.real_lig_res_ids)
            atoms = self.lig_atom_ids[res_id]
            self.delete(atoms)
            #self.simulation.step(10)
            self.simulation.minimizeEnergy(maxIterations=10)
            new_energy = self.context.getState(getEnergy=True).getPotentialEnergy()
            #acc_prob = self.N * math.exp(-self.B) * math.exp(-(new_energy - self.energy) / self.kT)
            log_acc_prob = - self.B - (new_energy - self.energy)/self.kT + self.N 
            if log_acc_prob < np.log(np.random.rand()) or np.isnan(log_acc_prob):
                self.adjust_specific_ligand(atoms,self.lig_params,mode='on')
                self.context.setPositions(self.positions)
            else:
                # Update some variables if move is accepted
                # self.positions = deepcopy(new_positions)
                self.N -= 1
                self.n_accepted += 1
                self.ghost_lig_res_ids.append(str(res_id))
                self.real_lig_res_ids.remove(res_id)
                # Update energy
                self.energy = new_energy
                self.reporter.report(self.simulation,self.context.getState(getPositions=True,getEnergy=True))

    def raiseError(self, error_msg):
        """
        Make it nice and easy to report an error in a consisent way - also easier to manage error handling in future

        Parameters
        ----------
        error_msg : str
            Message describing the error
        """
        # Write to the log file
        self.logger.error(error_msg)
        # Raise an Exception
        raise Exception(error_msg)

        return None

class VoronoiGCMCSampler(BaseGCMCSampler):
    def __init__(self,system, topology, temperature, ligands=[], custom_force=True, voronoi_vertices=[], log='gcmc.log', overwrite=False,
                 B=-6., insert_prob=0.5, positions=None,integrator=None,reporter=None, ghosts=None,
                 water_resn='HOH'):
        BaseGCMCSampler.__init__(self, system, topology, temperature, ligands=ligands, custom_force=custom_force, log=log, overwrite=overwrite,
                 B=B, insert_prob=insert_prob, positions=positions,integrator=integrator,reporter=reporter, ghosts=ghosts,
                 water_resn=water_resn)
        self.insert_points_list = voronoi_vertices

    def move(self):
        if np.random.rand() < self.insert_prob:
            # Insert
            res_id = np.random.choice(self.ghost_lig_res_ids)
            atoms = self.lig_atom_ids[res_id]
            insert_point =  self.insert_points_list[np.random.choice(len(self.insert_points_list))]* unit.nanometer
            new_positions = self.insert(atoms, insert_point)
            self.context.setPositions(new_positions)
            self.adjust_specific_ligand(atoms,self.lig_params,mode='on')
            self.nonbonded_force.updateParametersInContext(self.context)
            try:
                self.simulation.minimizeEnergy()
                new_energy = self.context.getState(getEnergy=True).getPotentialEnergy()
                log_acc_prob = self.B - (new_energy - self.energy)/self.kT - self.N - 1
                if log_acc_prob < np.log(np.random.rand()) or np.isnan(log_acc_prob):
                    self.adjust_specific_ligand(atoms,self.lig_params,mode='off')
                    self.context.setPositions(self.positions)
                else:
                    self.positions = deepcopy(new_positions)
                    self.N += 1
                    self.n_accepted += 1
                    self.real_lig_res_ids.append(str(res_id))
                    self.ghost_lig_res_ids.remove(res_id)
                    self.energy = new_energy
                    self.reporter.report(self.simulation,self.context.getState(getPositions=True,getEnergy=True))
            except:
                print('Explosion')
                self.context.setPositions(self.positions)
                self.adjust_specific_ligand(atoms,self.lig_params,mode='off')
                pass
            
        else:
            # Delete
            res_id = np.random.choice(self.real_lig_res_ids)
            atoms = self.lig_atom_ids[res_id]
            self.delete(atoms)
            self.simulation.minimizeEnergy(maxIterations=10)
            new_energy = self.context.getState(getEnergy=True).getPotentialEnergy()
            log_acc_prob = - self.B - (new_energy - self.energy)/self.kT + self.N 
            if log_acc_prob < np.log(np.random.rand()) or np.isnan(log_acc_prob):
                self.adjust_specific_ligand(atoms,self.lig_params,mode='on')
                self.context.setPositions(self.positions)
            else:
                self.N -= 1
                self.n_accepted += 1
                self.ghost_lig_res_ids.append(str(res_id))
                self.real_lig_res_ids.remove(res_id)
                self.energy = new_energy
                self.reporter.report(self.simulation,self.context.getState(getPositions=True,getEnergy=True))

class GCNCMCSampler(BaseGCMCSampler):
    def __init__(self,system, topology, temperature, ligands=[], lambdas=None, n_pert_steps=1, n_prop_steps_per_pert=1, log='gcmc.log', overwrite=False,
                 B=-6., insert_prob=0.5, positions=None,integrator=None,reporter=None, ghosts=None,
                 water_resn='HOH',time_step=2*openmm.unit.femtoseconds):
        
        # Set compound integrator
        self.compound_integrator = openmm.CompoundIntegrator()
        self.nc_integrator = NonequilibriumLangevinIntegrator(temperature=temperature,
                                                              collision_rate=1.0/unit.picosecond,
                                                              timestep=time_step, splitting="V R O R V")
        self.compound_integrator.addIntegrator(integrator)
        self.compound_integrator.addIntegrator(self.nc_integrator)
        self.compound_integrator.setCurrentIntegrator(0)
        
        BaseGCMCSampler.__init__(self, system, topology, temperature, ligands=ligands, log=log, overwrite=overwrite,
                 B=B, insert_prob=insert_prob, positions=positions,integrator=self.compound_integrator,reporter=reporter, ghosts=ghosts,
                 water_resn=water_resn)

        # Set NCMC related variables
        self.time_step = time_step.in_units_of(openmm.unit.picosecond)
        self.n_pert_steps = n_pert_steps 
        self.n_prop_steps_per_pert = n_prop_steps_per_pert
        if lambdas is not None:
            assert np.isclose(lambdas[0], 0.0) and np.isclose(lambdas[-1], 1.0), "Lambda series must start at 0 and end at 1"
            self.lambdas = lambdas
            self.n_pert_steps = len(self.lambdas) - 1
        else:
            self.n_pert_steps = n_pert_steps
            self.lambdas = np.linspace(0.0, 1.0, self.n_pert_steps + 1)

        

    def adjust_lambda(self, atoms, l):
        for atom_id in atoms:
            charge, sigma, eps = self.lig_params[atom_id]
            self.nonbonded_force.setParticleParameters(atom_id, [1., charge, sigma, eps, l])
        self.nonbonded_force.updateParametersInContext(self.context)
        

    def move(self):
        protocol_work = 0.0 * openmm.unit.kilocalories_per_mole
        explosion = False
        self.compound_integrator.setCurrentIntegrator(1)

        if np.random.rand() < self.insert_prob:
            # Insert
            res_id = np.random.choice(self.ghost_lig_res_ids)
            atoms = self.lig_atom_ids[res_id]
            #insert_point = (np.random.rand(3) * (self.max_dimension - self.min_dimension).value_in_unit(unit.nanometer)) * unit.nanometer + self.min_dimension
            insert_point = np.array([2.7880, 0.6749, 0.3701]) * openmm.unit.nanometer
            new_positions = self.insert(atoms, insert_point)
            self.context.setPositions(new_positions)
            self.adjust_specific_ligand(atoms,self.lig_params,mode='on')
            self.nonbonded_force.updateParametersInContext(self.context)
            for n in range(self.n_pert_steps):
                energy_initial = self.context.getState(getEnergy=True).getPotentialEnergy()
                l = self.lambdas[n]
                self.adjust_lambda(atoms,l)
                #self.simulation.step(self.n_prop_steps_per_pert)
                state = self.context.getState(getEnergy=True)
                energy_final = self.context.getState(getEnergy=True).getPotentialEnergy()
                protocol_work += energy_final - energy_initial
                try:
                    self.nc_integrator.step(self.n_prop_steps_per_pert)
                except:
                    print("Caught explosion!")
                    explosion = True
                    #self.n_explosions += 1
                    break
            new_energy = energy_final
            log_acc_prob = self.B - protocol_work/self.kT - self.N
            if log_acc_prob < np.log(np.random.rand()) or np.isnan(log_acc_prob) or explosion:
                self.adjust_specific_ligand(atoms,self.lig_params,mode='off')
                self.context.setPositions(self.positions)
            else:
                # Update some variables if move is accepted
                self.positions = deepcopy(new_positions)
                self.N += 1
                self.n_accepted += 1
                self.real_lig_res_ids.append(str(res_id))
                self.ghost_lig_res_ids.remove(res_id)
                #self.velocities = new_velocities
                self.reporter.report(self.simulation,self.context.getState(getPositions=True,getEnergy=True))
        else:
            # Delete
            res_id = np.random.choice(self.real_lig_res_ids)
            atoms = self.lig_atom_ids[res_id]
            self.delete(atoms)
            #self.simulation.step(10)
            #self.simulation.minimizeEnergy(maxIterations=10)
            new_energy = self.context.getState(getEnergy=True).getPotentialEnergy()
            #acc_prob = self.N * math.exp(-self.B) * math.exp(-(new_energy - self.energy) / self.kT)
            log_acc_prob = - self.B - (new_energy - self.energy)/self.kT + self.N 
            if log_acc_prob < np.log(np.random.rand()) or np.isnan(log_acc_prob):
                self.adjust_specific_ligand(atoms,self.lig_params,mode='on')
                self.context.setPositions(self.positions)
            else:
                # Update some variables if move is accepted
                # self.positions = deepcopy(new_positions)
                self.N -= 1
                self.n_accepted += 1
                self.ghost_lig_res_ids.append(str(res_id))
                self.real_lig_res_ids.remove(res_id)
                # Update energy
                self.energy = new_energy
                self.reporter.report(self.simulation,self.context.getState(getPositions=True,getEnergy=True))
            
class MDSimulator():
    def __init__(self, sampler, reporter=None, integrator=None, explicit=True, ligand_resn='UNK', water_resn='HOH', custom_force=True, save_pdb=None):
        # Delete ghost fragments from the system
        modeller = openmm.app.Modeller(sampler.topology, sampler.positions)
        toDelete = [r for r in sampler.topology.residues() if r.index in sampler.ghost_lig_res_ids]
        modeller.delete(toDelete)
        print('Adding solvent')
        # Add water if simulation is in explicit solvent
        if explicit:
            modeller.addSolvent(forcefield=sampler.forcefield,padding=1*openmm.unit.nanometer)
        # Store topology and positions
        self.topology = modeller.topology
        self.positions = modeller.positions
        if save_pdb:
            openmm.app.PDBFile.writeFile(self.topology, self.positions, save_pdb)


        print('Setting parameters')
        # Store ligand/water/protein related variables
        self.lig_res_ids = []
        self.lig_atom_ids = defaultdict(list)
        self.wat_atom_ids = defaultdict(list)
        self.prot_positions = []
        for resid, residue in enumerate(self.topology.residues()):
            if residue.name == ligand_resn:
                self.lig_res_ids.append(resid)
                for atom in residue.atoms():
                    self.lig_atom_ids[resid].append(atom.index)
            elif residue.name == water_resn:
                for atom in residue.atoms():
                    self.wat_atom_ids[resid].append(atom.index)
            else:
                for atom in residue.atoms():
                    self.prot_positions.append(self.positions[atom.index])
        self.prot_positions = unit.Quantity(self.prot_positions)
        
        # Setup OpenMM
        self.system = sampler.forcefield.createSystem(self.topology,nonbondedMethod=openmm.app.PME,
                        nonbondedCutoff=1.0*openmm.unit.nanometers, constraints=openmm.app.HBonds, rigidWater=True,
                        ewaldErrorTolerance=0.0005)
        i = 1
        for force in self.system.getForces():
            force.setForceGroup(i)
            i += 1
        
        # Find NonbondedForce - needs to be updated to switch waters on/off
        for f in range(self.system.getNumForces()):
            force = self.system.getForce(f)
            if force.__class__.__name__ == "NonbondedForce":
                self.nonbonded_force = force
                self.nonbonded_force_index = f

        # Customize force so that ligand-ligand interaction can be ignored
        if custom_force:
            self.customize_forces()
       
        # Set OpenMM simulation
        self.simulation = openmm.app.Simulation(
            self.topology,
            self.system,    
            integrator,
            openmm.Platform.getPlatformByName("CUDA"),  # faster if running in vacuum
        )
        if reporter:
            self.reporter = reporter
            self.simulation.reporters.append(self.reporter)
        self.context = self.simulation.context
        self.context.setPositions(self.positions)



        # Compute energy and forces
        for f in range(self.system.getNumForces()):
            print(self.system.getForce(f).__class__.__name__)
            force_group = self.system.getForce(f).getForceGroup()
            print(force_group)
            print(self.context.getState(getEnergy=True,groups={force_group}).getPotentialEnergy())
        self.energy = self.context.getState(getEnergy=True).getPotentialEnergy()
        print(f'Initial Energy: {self.energy}')
        
        print('Minimizing')
        # Minimize the system
        self.simulation.minimizeEnergy()

    def customize_forces(self):
        """
        Create a CustomNonbondedForce to handle ligand-ligand interactions
        For custom steric (LJ potential), soft core potential will be used
        For custom coulomb force, cufoff periodic with reaction field will be used
        These custom forces are applied to turn off ligand-ligand interactions
        """
        #if self.nonbonded_force.getNonbondedMethod() != openmm.NonbondedForce.CutoffPeriodic:
        #    self.raiseError("Currently only supporting CutoffPeriodic for long range electrostatics")
        #else:
        eps_solvent = self.nonbonded_force.getReactionFieldDielectric()
        cutoff = self.nonbonded_force.getCutoffDistance()
        krf = (1/ (cutoff**3)) * (eps_solvent - 1) / (2*eps_solvent + 1)
        crf = (1/ cutoff) * (3* eps_solvent) / (2*eps_solvent + 1)

        energy_expression  = "select(condition, 0, 1)*all;"
        energy_expression += "condition = soluteFlag1*soluteFlag2;" #solute must have flag int(1)
        energy_expression += "all=(lambda^soft_a) * 4 * epsilon * x * (x-1.0) + ONE_4PI_EPS0*chargeprod*(1/r + krf*r*r - crf);"
        energy_expression += "x = (sigma/reff)^6;"  # Define x as sigma/r(effective)
        energy_expression += "reff = sigma*((soft_alpha*(1.0-lambda)^soft_b + (r/sigma)^soft_c))^(1/soft_c);" # Calculate effective distance
        energy_expression += "lambda = lambda1*lambda2;"
        energy_expression += "epsilon = epsilon1*epsilon2;"
        energy_expression += "sigma = 0.5*(sigma1+sigma2);"
        energy_expression += "ONE_4PI_EPS0 = {:f};".format(ONE_4PI_EPS0)  # already in OpenMM units
        energy_expression += "chargeprod = charge1*charge2;"
        energy_expression += "krf = {:f};".format(krf.value_in_unit(unit.nanometer**-3))
        energy_expression += "crf = {:f};".format(crf.value_in_unit(unit.nanometer**-1))
        custom_nonbonded_force = openmm.CustomNonbondedForce(energy_expression)
        custom_nonbonded_force.addPerParticleParameter('soluteFlag')
        custom_nonbonded_force.addPerParticleParameter('charge')
        custom_nonbonded_force.addPerParticleParameter('sigma')
        custom_nonbonded_force.addPerParticleParameter('epsilon')
        custom_nonbonded_force.addPerParticleParameter('lambda')
        # Configure force
        custom_nonbonded_force.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
        custom_nonbonded_force.setUseSwitchingFunction(self.nonbonded_force.getUseSwitchingFunction())
        custom_nonbonded_force.setCutoffDistance(self.nonbonded_force.getCutoffDistance())
        custom_nonbonded_force.setSwitchingDistance(self.nonbonded_force.getSwitchingDistance())
        self.nonbonded_force.setUseDispersionCorrection(False)
        custom_nonbonded_force.setUseLongRangeCorrection(self.nonbonded_force.getUseDispersionCorrection())
        # Set softcore parameters
        custom_nonbonded_force.addGlobalParameter('soft_alpha', 0.5)
        custom_nonbonded_force.addGlobalParameter('soft_a', 1)
        custom_nonbonded_force.addGlobalParameter('soft_b', 1)
        custom_nonbonded_force.addGlobalParameter('soft_c', 6)

        lig_atom_ids = []
        for resid, residue in enumerate(self.topology.residues()):
            if resid in self.lig_res_ids:
                for atom in residue.atoms():
                    lig_atom_ids.append(atom.index)

        # Copy all steric interactions into the custom force, and remove them from the original force
        for atom_idx in range(self.nonbonded_force.getNumParticles()):
            # Get atom parameters
            [charge, sigma, epsilon] = self.nonbonded_force.getParticleParameters(atom_idx)
            # Make sure that sigma is not equal to zero
            if np.isclose(sigma._value, 0.0):
                sigma = 1.0 * unit.angstrom
            # Add particle to the custom force (with lambda=1 for now)
            if atom_idx in lig_atom_ids:
                custom_nonbonded_force.addParticle([1.0, charge, sigma, epsilon, 1.0])
            else:
                custom_nonbonded_force.addParticle([0.0, charge, sigma, epsilon, 1.0])

        # Copy over all exceptions into the new force as custom bonded forces
        # Exceptions between non-ligand atoms will be excluded here, and handled by the NonbondedForce
        # If exceptions (other than ignored interactions) are found involving ligand atoms, we have a problem
        energy_expression = "4*epsilon*((sigma/r)^12 - (sigma/r)^6) + ONE_4PI_EPS0*chargeprod*(1/r);"
        energy_expression += "ONE_4PI_EPS0 = {:f};".format(ONE_4PI_EPS0)
        custom_bond_force = openmm.CustomBondForce(energy_expression)
        custom_bond_force.addPerBondParameter('chargeprod')
        custom_bond_force.addPerBondParameter('sigma')
        custom_bond_force.addPerBondParameter('epsilon')
        
        for exception_idx in range(self.nonbonded_force.getNumExceptions()):
            [i, j, chargeprod, sigma, epsilon] = self.nonbonded_force.getExceptionParameters(exception_idx)
            # Add this to the list of exclusions
            custom_nonbonded_force.addExclusion(i, j)
            if i in lig_atom_ids and j in lig_atom_ids:
                pass
            else:
                custom_bond_force.addBond(i, j, [chargeprod, sigma, epsilon])

        # Update system
        new_index = self.system.addForce(custom_nonbonded_force)
        self.system.addForce(custom_bond_force)
        self.system.removeForce(self.nonbonded_force_index)
        self.nonbonded_force = custom_nonbonded_force
        self.nonbonded_force_index = new_index

        return None

    #def remove_duplicates

    def run_npt(self,n,P=1,T=298):
        self.system.addForce(openmm.MonteCarloBarostat(P*openmm.unit.bar, T*openmm.unit.kelvin))
        self.context.reinitialize(preserveState=True)
        self.simulation.step(n)