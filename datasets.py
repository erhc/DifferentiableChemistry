from neuralogic.core import R, Template, V
from neuralogic.utils.data import Mutagenesis
from torch_geometric.datasets import TUDataset
from neuralogic.dataset import TensorDataset, Data, FileDataset

MUTAGENESIS = "mutagen"
PTC = "ptc"
PTCFR = "ptc_fr"
PTCMM = "ptc_mm"
PTCFM = "ptc_fm"
COX = "cox"
DHFR = "dhfr"
ER = "er"

class DatasetInfo:
    def __init__(self, node_embed, edge_embed, connection, atom_types, key_atom_type,
                 bond_types, single_bond, double_bond, triple_bond, aliphatic_bond, aromatic_bonds,
                 carbon, oxygen, hydrogen, nitrogen, sulfur,
                 halogens):
        self.node_embed = node_embed
        self.edge_embed = edge_embed
        self.connection = connection
        self.atom_types = atom_types
        self.bond_types = bond_types
        self.key_atom_type = key_atom_type
        self.single_bond = single_bond
        self.double_bond = double_bond
        self.carbon = carbon
        self.aliphatic_bond = aliphatic_bond
        self.halogens = halogens
        self.triple_bond = triple_bond
        self.aromatic_bonds = aromatic_bonds
        self.oxygen = oxygen
        self.hydrogen = hydrogen
        self.nitrogen = nitrogen
        self.sulfur = sulfur


def get_dataset(dataset, param_size):
    if dataset == MUTAGENESIS:
        return MutagenesisTemplate(param_size)
    elif dataset == PTC:
        return PTCTemplate(param_size)
    elif dataset == PTCFR:
        return PTCFRTemplate(param_size)
    elif dataset == PTCFM:
        return PTCFMTemplate(param_size)
    elif dataset == PTCMM:
        return PTCMMTemplate(param_size)
    elif dataset == COX:
        return COXTemplate(param_size)
    elif dataset == DHFR:
        return DHFRTemplate(param_size)
    elif dataset == ER:
        return ERTemplate(param_size)
    else:
        raise Exception("Invalid dataset name")


def MutagenesisTemplate(param_size):
    _, dataset = Mutagenesis()

    template = Template()

    atom_types = ["c", "o", "br", "i", "f", "h", "n", "cl"]
    key_atoms = ["o", "s", "n"]
    bond_types = ["b_1", "b_2", "b_3", "b_4", "b_5", "b_7"]

    template.add_rules([
        (R.atom_embed(V.A)[param_size,] <= R.get(atom)(V.A)) for atom in atom_types
    ])

    template.add_rules([
        (R.bond_embed(V.B)[param_size,] <= R.get(bond)(V.B)) for bond in bond_types
    ])

    return template, dataset, DatasetInfo("atom_embed", "bond_embed", "bond", atom_types, key_atoms, bond_types,
                                          "b_1", "b_2", "b_3",  ["b_1", "b_2", "b_3"], ["b_4", "b_5", "b_6", "b_7"],
                                          "c", "o", "h", "n", "s", ["f", "cl", "br", "i"])


def PTCTemplate(param_size):
    dataset = FileDataset(examples_file="datasets/ptc_examples.txt", queries_file="datasets/ptc_queries.txt")

    template = Template()

    atom_types = [f"atom_{i}" for i in range(18)]
    key_atoms = ["atom_1", "atom_2", "atom_3", "atom_7"]
    bond_types = ["b_1", "b_2", "b_3", "b_0"]

    template.add_rules([
        (R.bond_embed(V.B)[param_size,] <= R.get(bond)(V.B)) for bond in bond_types
    ])

    template.add_rules([
        (R.atom_embed(V.A)[param_size,] <= R.get(atom)(V.A)) for atom in atom_types
    ])

    return template, dataset, DatasetInfo("atom_embed", "bond_embed", "bond", atom_types, key_atoms, bond_types,
                                          "b_2", "b_1", "b_0",  ["b_0", "b_1", "b_2"], ["b_3"],
                                          "atom_5", "atom_2", "h", "atom_3", "atom_7", ["atom_9", "atom_6", "atom_8", "atom_13"])

def PTCFRTemplate(param_size):
    dataset = FileDataset(examples_file="datasets/ptcfr_examples.txt", queries_file="datasets/ptcfr_queries.txt")

    template = Template()

    atom_types = [f"atom_{i}" for i in range(19)]
    key_atoms = ["atom_1", "atom_2", "atom_3", "atom_7"]
    bond_types = ["b_1", "b_2", "b_3", "b_0"]

    template.add_rules([
        (R.bond_embed(V.B)[param_size,] <= R.get(bond)(V.B)) for bond in bond_types
    ])

    template.add_rules([
        (R.atom_embed(V.A)[param_size,] <= R.get(atom)(V.A)) for atom in atom_types
    ])

    return template, dataset, DatasetInfo("atom_embed", "bond_embed", "bond", atom_types, key_atoms, bond_types,
                                          "b_2", "b_1", "b_0",  ["b_0", "b_1", "b_2"], ["b_3"],
                                          "atom_5", "atom_2", "h", "atom_3", "atom_7", ["atom_9", "atom_6", "atom_8", "atom_14"])

def PTCFMTemplate(param_size):
    dataset = FileDataset(examples_file="datasets/ptcfm_examples.txt", queries_file="datasets/ptcfm_queries.txt")

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

def PTCMMTemplate(param_size):
    dataset = FileDataset(examples_file="datasets/ptcmm_examples.txt", queries_file="datasets/ptcmm_queries.txt")

    template = Template()

    atom_types = [f"atom_{i}" for i in range(20)]
    key_atoms = ["atom_1", "atom_2", "atom_3", "atom_7"]
    bond_types = ["b_1", "b_2", "b_3", "b_0"]

    template.add_rules([
        (R.bond_embed(V.B)[param_size,] <= R.get(bond)(V.B)) for bond in bond_types
    ])

    template.add_rules([
        (R.atom_embed(V.A)[param_size,] <= R.get(atom)(V.A)) for atom in atom_types
    ])

    return template, dataset, DatasetInfo("atom_embed", "bond_embed", "bond", atom_types, key_atoms, bond_types,
                                          "b_2", "b_1", "b_0",  ["b_0", "b_1", "b_2"], ["b_3"],
                                          "atom_5", "atom_2", "h", "atom_3", "atom_7", ["atom_9", "atom_6", "atom_8", "atom_15"])


def COXTemplate(param_size):
    dataset = FileDataset(examples_file="datasets/cox_examples.txt", queries_file="datasets/cox_queries.txt")

    template = Template()

    atom_types = [f"atom_{i}" for i in range(7)]
    key_atoms = ["atom_1", "atom_4", "atom_3"]
    bond_types = ["b_4", "b_2", "b_3", "b_0", "b_1"]

    template.add_rules([
        (R.bond_embed(V.B)[param_size,] <= R.get(bond)(V.B)) for bond in bond_types
    ])

    template.add_rules([
        (R.atom_embed(V.A)[param_size,] <= R.get(atom)(V.A)) for atom in atom_types
    ])

    return template, dataset, DatasetInfo("atom_embed", "bond_embed", "bond", atom_types, key_atoms, bond_types,
                                          "b_2", "b_3", "b_4", ["b_2", "b_3", "b_4"], ["b_0"],
                                          "atom_0", "atom_4", "h", "atom_1", "atom_3", ["atom_2", "atom_5", "atom_6"])
def DHFRTemplate(param_size):
    dataset = FileDataset(examples_file="datasets/dhfr_examples.txt", queries_file="datasets/dhfr_queries.txt")

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

def ERTemplate(param_size):
    dataset = FileDataset(examples_file="datasets/er_examples.txt", queries_file="datasets/er_queries.txt")

    template = Template()

    atom_types = [f"atom_{i}" for i in range(10)]
    key_atoms = ["atom_1", "atom_2", "atom_4", "atom_9"]
    bond_types = ["b_4", "b_2", "b_3", "b_0", "b_1"]

    template.add_rules([
        (R.bond_embed(V.B)[param_size,] <= R.get(bond)(V.B)) for bond in bond_types
    ])

    template.add_rules([
        (R.atom_embed(V.A)[param_size,] <= R.get(atom)(V.A)) for atom in atom_types
    ])

    return template, dataset, DatasetInfo("atom_embed", "bond_embed", "bond", atom_types, key_atoms, bond_types,
                                          "b_2", "b_3", "b_4", ["b_2", "b_3", "b_4"], ["b_0"],
                                          "atom_0", "atom_1", "h", "atom_2", "atom_4", ["atom_3", "atom_5", "atom_6", "atom_8"])