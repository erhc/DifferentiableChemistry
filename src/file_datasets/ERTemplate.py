import os

from chemdiff.file_datasets.DatasetInfo import DatasetInfo
from neuralogic.core import R, Template, V
from neuralogic.dataset import FileDataset


def ERTemplate(param_size):
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset = FileDataset(
        examples_file=os.path.join(current_dir, "datasets/er_examples.txt"),
        queries_file=os.path.join(current_dir, "datasets/er_queries.txt")
    )

    template = Template()

    atom_types = [f"atom_{i}" for i in range(10)]
    key_atoms = ["atom_1", "atom_2", "atom_4", "atom_9"]
    bond_types = ["b_4", "b_2", "b_3", "b_0", "b_1"]

    template.add_rules(
        [(R.bond_embed(V.B)[param_size,] <= R.get(bond)(V.B)) for bond in bond_types]
    )

    template.add_rules(
        [(R.atom_embed(V.A)[param_size,] <= R.get(atom)(V.A)) for atom in atom_types]
    )

    return (
        template,
        dataset,
        DatasetInfo(
            "atom_embed",
            "bond_embed",
            "bond",
            atom_types,
            key_atoms,
            bond_types,
            "b_2",
            "b_3",
            "b_4",
            ["b_2", "b_3", "b_4"],
            ["b_0"],
            "atom_0",
            "atom_1",
            "h",
            "atom_2",
            "atom_4",
            ["atom_3", "atom_5", "atom_6", "atom_8"],
        ),
    )
