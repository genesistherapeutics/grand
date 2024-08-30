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
import Bio.PDB

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

    atom_ids = []
    for resid, residue in enumerate(lig_top.residues()):
        for atom in residue.atoms():
            atom_ids.append(atom.index)

    ca_coords = []
    
    for atom in prot_top.atoms():
        if atom.name == 'CA':
            ca_coords.append(prot_pos[atom.index])
    # Read in simulation box size
    box_vectors = prot_top.getPeriodicBoxVectors()
    #box_size = np.array([box_vectors[0][0]._value,
    #                     box_vectors[1][1]._value,
    #                     box_vectors[2][2]._value]) * unit.nanometer

    # Add multiple copies of the same water, then write out a pdb (for visualisation)
    ghosts = []
    offsets = np.random.random((n,3))
    offsets /= np.linalg.norm(offsets, axis=1,keepdims=True)
    offsets *= np.random.random((n,1)) * np.sqrt(0.5) * unit.nanometer
    for idx in tqdm.tqdm(range(n)):
        new_center = ca_coords[np.random.choice(len(ca_coords))] + offsets[idx]
        new_positions = rotate_molecule(lig_pos,atom_ids,insert_point=new_center)
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
    """
    Rotate and translate the given molecule.
    """
    R = random_rotation_matrix()
    new_positions = deepcopy(positions)
    #positions = positions.value_in_unit(unit.nanometer)
    for i, index in enumerate(atom_indices):
        #  Translate coordinates to an origin defined by the oxygen atom, and normalise
        atom_position = positions[index] - positions[atom_indices[0]]
        # Rotate about the oxygen position
        if i != 0:
            vec_length = np.linalg.norm(atom_position)
            atom_position = atom_position / vec_length
            # Rotate coordinates & restore length
            atom_position = vec_length * np.dot(R, atom_position) * atom_position.unit
        if insert_point is not None:
            # Translate to insertion point
            new_positions[index] = atom_position + insert_point
        else:
            new_positions[index] = atom_position
    return new_positions