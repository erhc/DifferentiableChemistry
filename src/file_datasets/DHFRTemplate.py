from neuralogic.core import R, Template, V
from neuralogic.dataset import FileDataset
from file_datasets.DatasetInfo import DatasetInfo

def DHFRTemplate(param_size):
    dataset = FileDataset(examples_file=os.path.abspath("./datasets/dhfr_examples.txt"), queries_file="./datasets/dhfr_queries.txt")

    template = Template()

    atom_types = [f"atom_{i}" for i in range(7)]
    key_atoms = ["atom_0", "atom_5", "atom_3"]
    bond_types = ["b_4", "b_2", "b_3", "b_0", "b_1"]

    template.add_rules([
        (R.bond_embed(V.B)[param_size,] <= R.get(bond)(V.B)) for bond in bond_types
    ])

    template.add_rules([
        (R.atom_embed(V.A)[param_size,] <= R.get(atom)(V.A)) for atom in atom_types
    ])

    return template, dataset, DatasetInfo("atom_embed", "bond_embed", "bond", atom_types, key_atoms, bond_types,
                                          "b_2", "b_3", "b_4", ["b_2", "b_3", "b_4"], ["b_0"],
                                          "atom_1", "atom_3", "h", "atom_0", "atom_5", ["atom_2", "atom_4", "atom_6"])
