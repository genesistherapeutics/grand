import os
import numpy as np
import mdtraj
import parmed
import openmm
from openmm import unit
from openmm import app
from openmmtools.constants import ONE_4PI_EPS0
from copy import deepcopy
from scipy.cluster import hierarchy
import warnings

import tqdm
import multiprocessing

# TODO: Need to rewirte description once everything is done
# TODO: Possible parallelization using multiprocessing
def add_ghosts(prot_top, prot_pos, lig_top, lig_pos, n=10, output='gcmc-ghosts.pdb'): 
    """
    Function to add water molecules to a topology, as extras for GCMC
    This is to avoid changing the number of particles throughout a simulation
    Instead, we can just switch between 'ghost' and 'real' waters...

    Notes
    -----
    Ghosts currently all added to a new chain
    Residue numbering continues from the existing PDB numbering

    Parameters
    ----------
    topology : simtk.openmm.app.Topology
        Topology of the initial system
    positions : simtk.unit.Quantity
        Atomic coordinates of the initial system
    n : int
        Number of waters to add to the system
    output : str
        Name of the PDB file to write containing the updated system
        This will be useful for visualising the results obtained.

    Returns
    -------
    modeller.topology : simtk.openmm.app.Topology
        Topology of the system after modification
    modeller.positions : simtk.unit.Quantity
        Atomic positions of the system after modification
    ghosts : list
        List of the residue numbers (counting from 0) of the ghost
        waters added to the system.
    """
    # Create a Modeller instance of the system
    modeller = app.Modeller(topology=prot_top, positions=prot_pos)

    # Read the chain IDs
    chain_ids = []
    for chain in modeller.topology.chains():
        chain_ids.append(chain.id)

    # Read in simulation box size
    box_vectors = prot_top.getPeriodicBoxVectors()
    box_size = np.array([box_vectors[0][0]._value,
                         box_vectors[1][1]._value,
                         box_vectors[2][2]._value]) * unit.nanometer

    # Add multiple copies of the same water, then write out a pdb (for visualisation)
    ghosts = []        
    new_centres = np.zeros((n,3)) * unit.nanometer  + np.random.rand(n,3) * unit.nanometer #* 10 + box_size
    for idx in tqdm.tqdm(range(n)):
        # Need to translate the water to a random point in the simulation box]
        new_positions = deepcopy(lig_pos)
        for i in range(len(lig_pos)):
            new_positions[i] = lig_pos[i] + new_centres[idx] - lig_pos[0]

        # Add the water to the model and include the resid in a list
        modeller.add(addTopology=lig_top, addPositions=new_positions)
        ghosts.append(modeller.topology._numResidues - 1)

    # Take the ghost chain as the one after the last chain (alphabetically)
    new_chain = chr(((ord(chain_ids[-1]) - 64) % 26) + 65)

    # Renumber all ghost waters and assign them to the new chain
    for resid, residue in enumerate(modeller.topology.residues()):
        if resid in ghosts:
            residue.id = str(((resid) % 9999) + 1)
            residue.chain.id = 'A'

    # Write the new topology and positions to a PDB file
    if output is not None:
        openmm.app.PDBFile.writeFile(topology=modeller.topology, positions=modeller.positions, file=output, keepIds=True)

    return modeller.topology, modeller.positions, ghosts

def random_rotation_matrix():
    """
    Generate a random axis and angle for rotation of the water coordinates (using the
    method used for this in the ProtoMS source code (www.protoms.org), and then return
    a rotation matrix produced from these

    Returns
    -------
    rot_matrix : numpy.ndarray
        Rotation matrix generated
    """
    # First generate a random axis about which the rotation will occur
    rand1 = rand2 = 2.0

    while (rand1**2 + rand2**2) >= 1.0:
        rand1 = np.random.rand()
        rand2 = np.random.rand()
    rand_h = 2 * np.sqrt(1.0 - (rand1**2 + rand2**2))
    axis = np.array([rand1 * rand_h, rand2 * rand_h, 1 - 2*(rand1**2 + rand2**2)])
    axis /= np.linalg.norm(axis)

    # Get a random angle
    theta = np.pi * (2*np.random.rand() - 1.0)

    # Simplify products & generate matrix
    x, y, z = axis[0], axis[1], axis[2]
    x2, y2, z2 = axis[0]*axis[0], axis[1]*axis[1], axis[2]*axis[2]
    xy, xz, yz = axis[0]*axis[1], axis[0]*axis[2], axis[1]*axis[2]
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    rot_matrix = np.array([[cos_theta + x2*(1-cos_theta),   xy*(1-cos_theta) - z*sin_theta, xz*(1-cos_theta) + y*sin_theta],
                           [xy*(1-cos_theta) + z*sin_theta, cos_theta + y2*(1-cos_theta),   yz*(1-cos_theta) - x*sin_theta],
                           [xz*(1-cos_theta) - y*sin_theta, yz*(1-cos_theta) + x*sin_theta, cos_theta + z2*(1-cos_theta)  ]])

    return rot_matrix

def rotate_molecule(positions, atom_indices, insert_point=None):
    R = random_rotation_matrix()
    new_positions = deepcopy(positions)
    for i, index in enumerate(atom_indices):
        #  Translate coordinates to an origin defined by the oxygen atom, and normalise
        atom_position = positions[index] - positions[atom_indices[0]]
        # Rotate about the oxygen position
        if i != 0:
            vec_length = np.linalg.norm(atom_position)
            atom_position = atom_position / vec_length
            # Rotate coordinates & restore length
            atom_position = vec_length * np.dot(R, atom_position) #* unit.nanometer
        if insert_point is not None:
            # Translate to insertion point
            new_positions[index] = atom_position + insert_point
        else:
            new_positions[index] = atom_position
    return new_positions

#TODO: Double-check custom nb is correct
#TODO: add appropriate global parameters
def custom_nonbonded_force(alpha_ewald, cutoff_distance):
    #CustomNonbondedForce with LJ and Coulomb terms
    energy_expression  = "select(condition,0,1)*all;"
    energy_expression += "condition = soluteFlag1*soluteFlag2;" #solute must have flag int(1)
    energy_expression += "all=4*epsilon*((sigma/r)^12 - (sigma/r)^6) + ONE_4PI_EPS0*chargeprod*erfc(alpha_ewald*r)/r;"
    energy_expression += "epsilon = epsilon1*epsilon2;"
    energy_expression += "sigma = 0.5*(sigma1+sigma2);"
    energy_expression += "ONE_4PI_EPS0 = {:f};".format(ONE_4PI_EPS0)  # already in OpenMM units
    energy_expression += "chargeprod = charge1*charge2;"
    energy_expression += "alpha_ewald = {:f};".format(alpha_ewald.value_in_unit_system(unit.md_unit_system))
    custom_nonbonded_force = openmm.CustomNonbondedForce(energy_expression)
    custom_nonbonded_force.addPerParticleParameter('soluteFlag')
    custom_nonbonded_force.addPerParticleParameter('charge')
    custom_nonbonded_force.addPerParticleParameter('sigma')
    custom_nonbonded_force.addPerParticleParameter('epsilon')
    # Configure force
    custom_nonbonded_force.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
    custom_nonbonded_force.setCutoffDistance(1*unit.nanometer)
    
    custom_nonbonded_force.setUseLongRangeCorrection(False)
    custom_nonbonded_force.setUseSwitchingFunction(True)
    custom_nonbonded_force.setSwitchingDistance(cutoff_distance - 1.0*unit.angstroms)

    #Now must add a bond force to handle exceptions.
    #(exceptions are altered interactions - but not completely
    #removed! Also PME is not required now.
    energy_expression  = "scalingFactor*all;"
    energy_expression += "all=4*epsilon*((sigma/r)^12 - (sigma/r)^6) + ONE_4PI_EPS0*chargeprod/r;"
    energy_expression += "ONE_4PI_EPS0 = {:f};".format(ONE_4PI_EPS0)  # already in OpenMM units
    custom_bond_force = openmm.CustomBondForce(energy_expression)
    custom_bond_force.addGlobalParameter('scalingFactor', 1)
    custom_bond_force.addEnergyParameterDerivative('scalingFactor')
    custom_bond_force.addPerBondParameter('chargeprod')
    custom_bond_force.addPerBondParameter('sigma')
    custom_bond_force.addPerBondParameter('epsilon')

    return custom_nonbonded_force, custom_bond_force
