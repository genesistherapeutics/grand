import openff.toolkit
import openmmforcefields.generators


def main():
    mol = openff.toolkit.Molecule.from_smiles("c1ccccc1O")

    generator = openmmforcefields.generators.SMIRNOFFTemplateGenerator(
        molecules=[mol], forcefield="openff-2.0.0.offxml"
    )

    xml_str = generator.generate_residue_template(mol)
    print(xml_str)


if __name__ == '__main__':
    main()