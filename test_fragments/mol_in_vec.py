import openff.toolkit
import openmm.app


def main():
    mol = openff.toolkit.Molecule.from_smiles("c1ccccc1O")
    mol.generate_conformers(n_conformers=1)

    # mol = openff.toolkit.Molecule.from_file("phenol.sdf", "SDF")

    # ff = openff.toolkit.ForceField("openff-2.0.0.offxml")
    ff = openff.toolkit.ForceField("phenol_openff.xml")

    sys = ff.create_openmm_system(mol.to_topology())
    sys.addForce(openmm.CMMotionRemover())  # remove COM motion to avoid drift

    integrator = openmm.LangevinIntegrator(
        300.0 * openmm.unit.kelvin, 1.0 / openmm.unit.picosecond, 1.0 * openmm.unit.femtosecond
    )

    sim = openmm.app.Simulation(
        mol.to_topology().to_openmm(),
        sys,
        integrator,
        openmm.Platform.getPlatformByName("Reference"),  # faster if running in vacuum
    )
    sim.reporters.append(openmm.app.PDBReporter("phenol-traj.pdb", reportInterval=100))
    sim.context.setPositions(mol.conformers[0].to_openmm())

    sim.step(10000)


if __name__ == '__main__':
    main()