import openff.toolkit
import openmm.app
import openmm.unit
import os

def main():
    # OpenFF to OpenMM - Ligand
    mol = openff.toolkit.Molecule.from_smiles("c1ccccc1O")
    mol.generate_conformers(n_conformers=1)
    topology = mol.to_topology().to_openmm()
    positions = mol.conformers[0].to_openmm()

    ff = openmm.app.ForceField('amber14-all.xml', 'amber14/tip3pfb.xml','phenol_openff.xml')

    topology.setUnitCellDimensions([4,4,4])
    modeller = openmm.app.Modeller(topology, positions)
    modeller.addSolvent(forcefield=ff)
    openmm.app.PDBFile.writeFile(modeller.topology, modeller.positions, open('phenol_water.pdb','w'))

    sys = ff.createSystem(modeller.topology)
    sys.addForce(openmm.CMMotionRemover())  # remove COM motion to avoid drift

    integrator = openmm.LangevinIntegrator(
        300.0 * openmm.unit.kelvin, 1.0 / openmm.unit.picosecond, 1.0 * openmm.unit.femtosecond
    )

    sim = openmm.app.Simulation(
        modeller.topology,
        sys,
        integrator,
        openmm.Platform.getPlatformByName("Reference"),  # faster if running in vacuum
    )
    #sim.reporters.append(openmm.app.PDBReporter("phenol_water-traj.pdb", reportInterval=100))
    sim.reporters.append(openmm.app.DCDReporter("phenol_water-traj.dcd",reportInterval=100))
    sim.context.setPositions(modeller.positions)

    sim.step(1000)


if __name__ == '__main__':
    main()