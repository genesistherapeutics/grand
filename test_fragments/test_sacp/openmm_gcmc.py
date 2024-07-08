import openff.toolkit
import openmm.app
import os
import numpy as np

from collections import defaultdict
from copy import deepcopy
import lig_utils as lu
import math
import tqdm

import time
#import lig_samplers as ls

# TODO
# wrap insertion into a function and move it under sampler
# deletion steps

# ISSUES
# - Performance
# Somehow setPositions/simulation.context.getState seems to be the bottleneck
# - ligand conformation is fixed
# - Protein conformaiton is also fixed
# Added MD steps in between GCMC - it now takes 40 seconds per step (without it 6s/step)
# Add 1nm padding around protein and have 4nm*4nm*4nm simulation box
# Put phenols at the single point where it does not interact with the protein 

def main():
    # OpenFF to OpenMM - Ligand
    mol = openff.toolkit.Molecule.from_smiles("c1ccccc1O")
    mol.generate_conformers(n_conformers=1000)
    topology = mol.to_topology().to_openmm()
    positions = mol.conformers[0].to_openmm()
    positions = positions + np.array([2,2,2])*openmm.unit.nanometer # translate to avoid clash with protein

    # Load protein
    prot = openmm.app.PDBFile('1uao.pdb')
    prot.topology.setUnitCellDimensions([4,4,4])
    
    # Define ligand position
    

    # Merge prot and lig
    if os.path.exists('gcmc-ghosts.pdb'):
        pdb = openmm.app.PDBFile('gcmc-ghosts.pdb')
        sys_top = pdb.topology
        sys_pos = pdb.positions
    else:
        sys_top, sys_pos, ghosts = lu.add_ghosts(prot.topology, prot.positions, topology, positions, n=1000)

    ff = openmm.app.ForceField('amber14-all.xml', 'amber14/tip3pfb.xml','phenol_openff.xml')
    
    sys = ff.createSystem(sys_top)
    sys.addForce(openmm.CMMotionRemover())  # remove COM motion to avoid drift

    integrator = openmm.LangevinIntegrator(
        300.0 * openmm.unit.kelvin, 1.0 / openmm.unit.picosecond, 1.0 * openmm.unit.femtosecond
    )

    sim = openmm.app.Simulation(
        sys_top,
        sys,
        integrator,
        openmm.Platform.getPlatformByName("CUDA"),  # faster if running in vacuum
    )
    sim.reporters.append(openmm.app.DCDReporter("phenol_sacp.dcd", reportInterval=100))
    dcd_reporter = openmm.app.DCDReporter("phenol_sacp.dcd", reportInterval=100)
    #sim.reporters.append(openmm.app.PDBReporter("proposal_tracking.pdb",reportInterval=100)
    sim.context.setPositions(sys_pos)

    for f in range(sys.getNumForces()):
        force = sys.getForce(f)
        if force.__class__.__name__ == "NonbondedForce":
            nonbonded_force = force

    # Get a list of all fragments and non-fragment atom IDs
    frag_res_ids = []
    for resid, residue in enumerate(sys_top.residues()):
        if residue.name == "UNK":
            frag_res_ids.append(resid)

    frag_atom_ids = defaultdict(list)
    for resid, residue in enumerate(sys_top.residues()):
        if resid in frag_res_ids:
            for atom in residue.atoms():
                frag_atom_ids[resid].append(atom.index)

    # Turn off interactions of ghost molecules
    # Do this by setting epsilon and charge to zero
    atom_param_dict = defaultdict(list)
    for frag_res_id in frag_atom_ids.keys():
        for atom_id in frag_atom_ids[frag_res_id]:
            charge, simga, eps = nonbonded_force.getParticleParameters(atom_id)
            atom_param_dict[atom_id] = [charge, simga, eps]
            nonbonded_force.setParticleParameters(atom_id, 0., 0., 0.,)
    nonbonded_force.updateParametersInContext(sim.context)

    n_step = 10
    B = 100
    ghost_frag_res_ids = frag_res_ids
    real_frag_res_ids = []
    # Only insertion
    positions = deepcopy(sys_pos)
    energy = sim.context.getState(getEnergy=True).getPotentialEnergy()
    N = 0
    n_accepted = 0
    kT = openmm.unit.BOLTZMANN_CONSTANT_kB * openmm.unit.AVOGADRO_CONSTANT_NA * 298 * openmm.unit.kelvin
    f = open('ghosts_in_sys.txt','w')
    for step in tqdm.tqdm(range(n_step)):
        res_id = np.random.choice(ghost_frag_res_ids)
        for atom_id in frag_atom_ids[res_id]:
            charge, sigma, eps = atom_param_dict[atom_id]
            nonbonded_force.setParticleParameters(atom_id, charge, sigma, eps)
        nonbonded_force.updateParametersInContext(sim.context)
        ghost_frag_res_ids.remove(res_id)
        
        rot_start = time.time()
        # Randomly rotate selected residue
        insert_point = (np.random.rand(3) * 4) * openmm.unit.nanometers
        new_positions = lu.rotate_molecule(sys_pos,frag_atom_ids[res_id],insert_point)
        sim.context.setPositions(new_positions)
        print(time.time()-rot_start)
        #sim.step(1)
        mc_start = time.time()
        #new_state = sim.context.getState(getPositions=True,getEnergy=True)
        #new_positions = sim.context.getState(getPositions=True).getPositions()
        new_energy = sim.context.getState(getEnergy=True).getPotentialEnergy()
        print(time.time()-mc_start)
        #pdb_reporter.report(sim,sim.context.getState(getPositions=True,getEnergy=True))
        dcd_reporter.report(sim,sim.context.getState(getPositions=True,getEnergy=True))
        # Decide
        acc_prob = math.exp(B) * math.exp(-(new_energy - energy) / kT) / (N + 1)
        if acc_prob < np.random.rand() or np.isnan(acc_prob):
            # Need to revert the changes made if the move is to be rejected
            # Switch off nonbonded interactions involving this water
            for atom_id in frag_atom_ids[res_id]:
                nonbonded_force.setParticleParameters(atom_id, 0., 0., 0.)
            nonbonded_force.updateParametersInContext(sim.context)
            sim.context.setPositions(positions)
            ghost_frag_res_ids.append(res_id)
        else:
            # Update some variables if move is accepted
            positions = deepcopy(new_positions)
            N += 1
            n_accepted += 1
            real_frag_res_ids.append(str(res_id))
            # Update energy
            energy = new_energy
        #openmm.app.PDBFile.writeFile(sim.topology,positions,open(f'output/GCMC_test_{step}.pdb','w'))
        print(time.time()-mc_start)
        f.write(",".join(real_frag_res_ids)+"\n")
    print(n_accepted)
    
    # Deletion
    n_accepted=0
    for step in tqdm.tqdm(range(n_step)):
        res_id = np.random.choice(real_frag_res_ids)
        for atom_id in frag_atom_ids[res_id]:
            nonbonded_force.setParticleParameters(atom_id, 0., 0., 0.)
        nonbonded_force.updateParametersInContext(sim.context)
        real_frag_res_ids.remove(res_id)

        #sim.step(1)
        mc_start = time.time()
        #new_state = sim.context.getState(getPositions=True,getEnergy=True)
        #new_positions = sim.context.getState(getPositions=True).getPositions()
        new_energy = sim.context.getState(getEnergy=True).getPotentialEnergy()
        print(time.time()-mc_start)
        #pdb_reporter.report(sim,sim.context.getState(getPositions=True,getEnergy=True))
        dcd_reporter.report(sim,sim.context.getState(getPositions=True,getEnergy=True))
        # Decide
        acc_prob = N * math.exp(-B) * math.exp(-(new_energy - energy) / kT)
        print('prob:',acc_prob, n_accepted)
        if acc_prob < np.random.rand() or np.isnan(acc_prob):
            # Need to revert the changes made if the move is to be rejected
            # Switch off nonbonded interactions involving this water
            for atom_id in frag_atom_ids[res_id]:
                charge, sigma, eps = atom_param_dict[atom_id]
                nonbonded_force.setParticleParameters(atom_id, charge, sigma, eps)
            nonbonded_force.updateParametersInContext(sim.context)
            sim.context.setPositions(positions)
            real_frag_res_ids.append(res_id)
        else:
            # Update some variables if move is accepted
            positions = deepcopy(new_positions)
            N += 1
            n_accepted += 1
            ghost_frag_res_ids.append(str(res_id))
            # Update energy
            energy = new_energy
        #openmm.app.PDBFile.writeFile(sim.topology,positions,open(f'output/GCMC_test_{step}.pdb','w'))
        print(time.time()-mc_start)
        f.write(",".join(real_frag_res_ids)+"\n")
    print(n_accepted)    


    #sim.step(1000)


if __name__ == '__main__':
    main()