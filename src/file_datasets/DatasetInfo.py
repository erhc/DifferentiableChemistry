class DatasetInfo:
    def __init__(
        self,
        node_embed,
        edge_embed,
        connection,
        atom_types,
        key_atom_type,
        bond_types,
        single_bond,
        double_bond,
        triple_bond,
        aliphatic_bond,
        aromatic_bonds,
        carbon,
        oxygen,
        hydrogen,
        nitrogen,
        sulfur,
        halogens,
    ):
        """
        Initializes a DatasetInfo object.

        Args:
            node_embed (str): The node embedding predicate.
            edge_embed (str): The edge embedding predicate.
            connection (str): The connection type predicate.
            atom_types (list): List of atom types predicates.
            key_atom_type (str): The key atom type predicate.
            bond_types (list): List of bond types predicates.
            single_bond (str): The single bond type predicate.
            double_bond (str): The double bond type predicate.
            triple_bond (str): The triple bond type predicate.
            aliphatic_bond (str): The aliphatic bond type predicate.
            aromatic_bonds (str): The aromatic bond type predicate.
            carbon (str): The carbon element predicate.
            oxygen (str): The oxygen element predicate.
            hydrogen (str): The hydrogen element predicate.
            nitrogen (str): The nitrogen element predicate.
            sulfur (str): The sulfur element predicate.
            halogens (str): The halogens element predicates.
        """
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
