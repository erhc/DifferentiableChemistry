from neuralogic.core import R, Template, V
from neuralogic.dataset import FileDataset
from file_datasets.DatasetInfo import DatasetInfo
import os

def PTCFMTemplate(param_size):
    dataset = FileDataset(examples_file="./datasets/ptcfm_examples.txt", queries_file="./datasets/ptcfm_queries.txt")

    template = Template()

    atom_types = [f"atom_{i}" for i in range(18)]
    key_atoms = ["atom_1", "atom_4", "atom_3", "atom_6"]
    bond_types = ["b_1", "b_2", "b_3", "b_0"]

    template.add_rules([
        (R.bond_embed(V.B)[param_size,] <= R.get(bond)(V.B)) for bond in bond_types
    ])

    template.add_rules([
        (R.atom_embed(V.A)[param_size,] <= R.get(atom)(V.A)) for atom in atom_types
    ])

    return template, dataset, DatasetInfo("atom_embed", "bond_embed", "bond", atom_types, key_atoms, bond_types,
                                          "b_1", "b_1", "b_0",  ["b_0", "b_1", "b_2"], ["b_3"],
                                          "atom_2", "atom_3", "h", "atom_4", "atom_6", ["atom_9", "atom_5", "atom_7", "atom_13"])

