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