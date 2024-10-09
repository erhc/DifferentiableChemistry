import os

from chemdiff.file_datasets.DatasetInfo import DatasetInfo
from neuralogic.core import R, Template, V
from neuralogic.dataset import FileDataset


def CustomDataset(examples, queries, param_size):
    """
    Create a custom dataset for training.

    Args:
        examples (str): The path to the examples file.
        queries (str): The path to the queries file.
        param_size (int): The size of the parameter.

    Returns:
        tuple: A tuple containing the following elements:
            - template: The template object.
            - dataset: The dataset object.
            - dataset_info: The dataset information object.
    """
    dataset = FileDataset(
        examples_file=os.path.abspath(examples),
        queries_file=os.path.abspath(queries),
    )

    template = Template()

    atom_types = ["c", "o", "br", "i", "f", "h", "n", "cl"]
    key_atoms = ["o", "s", "n"]
    bond_types = ["b_1", "b_2", "b_3", "b_4"]

    template.add_rules(
        [(R.atom_embed(V.A)[param_size,] <= R.get(atom)(V.A)) for atom in atom_types]
    )

    template.add_rules(
        [(R.bond_embed(V.B)[param_size,] <= R.get(bond)(V.B)) for bond in bond_types]
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
            "b_1",
            "b_2",
            "b_3",
            ["b_1", "b_2", "b_3"],
            ["b_4", "b_5", "b_6", "b_7"],
            "c",
            "o",
            "h",
            "n",
            "s",
            ["f", "cl", "br", "i"],
        ),
    )
