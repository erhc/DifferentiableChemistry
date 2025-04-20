import os

from chemdiff.file_datasets.DatasetInfo import DatasetInfo
from neuralogic.core import R, Template, V
from neuralogic.dataset import FileDataset


def PTCMMTemplate(param_size):
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset = FileDataset(
        examples_file=os.path.join(current_dir, "datasets/ptcmm_examples.txt"),
        queries_file=os.path.join(current_dir, "datasets/ptcmm_queries.txt")
    )

    template = Template()

    atom_types = [f"atom_{i}" for i in range(20)]
    key_atoms = ["atom_1", "atom_2", "atom_3", "atom_7"]
    bond_types = ["b_1", "b_2", "b_3", "b_0"]

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
            "b_1",
            "b_0",
            ["b_0", "b_1", "b_2"],
            ["b_3"],
            "atom_5",
            "atom_2",
            "h",
            "atom_3",
            "atom_7",
            ["atom_9", "atom_6", "atom_8", "atom_15"],
        ),
    )
