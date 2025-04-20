from chemdiff.file_datasets.DatasetInfo import DatasetInfo
from neuralogic.core import R, Template, V
from neuralogic.utils.data import Mutagenesis


# TODO: try the MUTAG from TUD datasets (different bond types and implicit hydrogens)
def MutagenesisTemplate(param_size):
    _, dataset = Mutagenesis()

    template = Template()

    atom_types = ["c", "o", "br", "i", "f", "h", "n", "cl"]
    key_atoms = ["o", "s", "n"]
    bond_types = ["b_1", "b_2", "b_3", "b_4", "b_5", "b_7"]

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
