from neuralogic.core import R, V, Aggregation


def get_subgraphs(layer_name: str, node_embed: str, edge_embed: str, connection: str, param_size: int,
                  max_cycle_size: int = 10, max_depth: int = 5,
                  single_bond=None, double_bond=None, carbon=None, atom_types=None, aliphatic_bond=None,
                  cycles=False, paths=False, y_shape=False, nbhoods=False, circular=False, collective=False):
    template = []

    template += [R.predict[1, param_size] <= R.get(f"{layer_name}_subgraph_pattern")]

    param_size = (param_size, param_size)

    if cycles or circular or collective:
        template += get_cycles(layer_name, node_embed, edge_embed, connection, param_size, max_cycle_size)
    if paths or collective:
        template += get_path(layer_name, edge_embed, connection, param_size, max_depth)
    if y_shape:
        template += get_y_shape(layer_name, node_embed, edge_embed, connection, param_size, double_bond)
    if nbhoods:
        for t in atom_types:
            template += [R.get(f"{layer_name}_key_atoms") <= R.get(t)]
        template += get_nbhoods(layer_name, node_embed, edge_embed, connection, param_size, carbon,
                                f"{layer_name}_key_atoms")
    if circular:
        template += get_circular(layer_name, node_embed, edge_embed, connection, param_size, carbon, single_bond,
                                 double_bond)
    if collective:
        for t in aliphatic_bond:
            template += [R.get(f"{layer_name}_aliphatic_bond") <= R.get(t)]
        template += get_collective(layer_name, node_embed, edge_embed, connection, param_size, carbon,
                                   f"{layer_name}_aliphatic_bond",
                                   max_depth)

    template += [R.get(f"{layer_name}_subgraph_pattern")(V.X) <= R.get(f"{layer_name}_pattern")(V.X)[param_size]]

    template += [R.get(f"{layer_name}_subgraph_pattern") <= R.get(f"{layer_name}_subgraph_pattern")(V.X)[param_size]]

    return template


def get_cycles(layer_name: str, node_embed: str, edge_embed: str, connection: str, param_size: tuple,
               max_cycle_size: int):
    def get_cycle(n):
        body = [R.get(connection)(f"X{i}", f"X{(i + 1) % n}", f"B{i}") for i in range(n)]
        body.extend(R.get(node_embed)(f"X{i}")[param_size] for i in range(n))
        body.extend(R.get(edge_embed)(f"B{i}")[param_size] for i in range(n))
        body.append(R.special.alldiff(...))

        return [(R.get(f"{layer_name}_cycle")([f"X{i}" for i in range(n)]) <= body)]

    def connect_cycle(n):
        return [(R.get(f"{layer_name}_cycle")([f"X{i}" for i in range(n - 1)]) <= R.get(f"{layer_name}_cycle")(
            [f"X{i}" for i in range(n)]))]

    template = []

    for i in range(3, max_cycle_size):
        template += get_cycle(i)
        template += connect_cycle(i)

    template += [R.get(f"{layer_name}_pattern")(V.X) <= R.get(f"{layer_name}_cycle")(V.X)[param_size]]

    return template


def get_path(layer_name, edge_embed, connection, param_size, max_depth):
    template = []
    template += [R.get(f"{layer_name}_path")(V.X, V.Y, 0) <= (R.get(edge_embed)(V.B)[param_size],
                                                              R.get(connection)(V.X, V.Y, V.B))]
    template += [R.get(f"{layer_name}_path")(V.X, V.Y, V.T) <= (
        R.get(edge_embed)(V.B)[param_size],
        R.get(f"{layer_name}_path")(V.Z, V.Y, V.T1)[param_size],
        R.get(connection)(V.X, V.Z, V.B), R.special.next(V.T1, V.T))]

    template += [(R.get(f"{layer_name}_path")(V.X, V.Y) <= (R.get(f"{layer_name}_path")(V.X, V.Y, max_depth)))]

    for i in range(max_depth):
        template += [(R._next(i, i + 1))]

    template += [R.get(f"{layer_name}_pattern")(V.X) <= R.get(f"{layer_name}_path")(V.X)[param_size]]

    return template


def get_y_shape(layer_name, node_embed, edge_embed, connection, param_size, double_bond):
    template = []
    template += [R.get(f"{layer_name}_double_bond_subgraph")(V.C) <= (R.get(node_embed)(V.F)[param_size],
                                                                      R.get(edge_embed)(V.B)[param_size],
                                                                      R.get(connection)(V.C, V.F, V.B),
                                                                      R.get(double_bond)(V.B))]

    template += [R.get(f"{layer_name}_y_subgraph")(V.X1, V.X2, V.X3, V.X4) <= (
        R.get(connection)(V.X1, V.X2, V.B1),
        R.get(connection)(V.X1, V.X3, V.B2),
        R.get(connection)(V.X1, V.X4, V.B3),
        R.get(edge_embed)(V.B1),
        R.get(edge_embed)(V.B2),
        R.get(edge_embed)(V.B3),
        R.get(node_embed)(V.X1),
        R.get(node_embed)(V.X2),
        R.get(node_embed)(V.X3),
        R.get(node_embed)(V.X4),
        R.special.alldiff(...))]

    template += [R.get(f"{layer_name}_y_bond")(V.X1, V.X2, V.X3, V.X4) <= (
        R.get(f"{layer_name}_y_subgraph")(V.X1, V.X2, V.X3, V.X4),
        R.get(connection)(V.X1, V.X2, V.B1), R.get(double_bond)(V.B1),
        R.special.alldiff(...))]

    template += [
        R.get(f"{layer_name}_y_group")(V.X1, V.X2, V.X3) <= (R.get(f"{layer_name}_y_bond")(V.Y1, V.Y2, V.X1, V.X2),
                                                             R.get(f"{layer_name}_y_bond")(V.Z1, V.Z2, V.X2, V.X3),
                                                             R.special.alldiff(...))]

    template += [
        (R.get(f"{layer_name}_y_bond_patterns")(V.X) <= R.get(f"{layer_name}_double_bond_subgraph")(V.X)[param_size]),
        (R.get(f"{layer_name}_y_bond_patterns")(V.X) <= R.get(f"{layer_name}_y_subgraph")(V.X1, V.X2, V.X3, V.X4)[
            param_size]),
        (R.get(f"{layer_name}_y_bond_patterns")(V.X2) <= R.get(f"{layer_name}_y_subgraph")(V.X1, V.X2, V.X3, V.X4)[
            param_size]),
        (R.get(f"{layer_name}_y_bond_patterns")(V.X3) <= R.get(f"{layer_name}_y_subgraph")(V.X1, V.X2, V.X3, V.X4)[
            param_size]),
        (R.get(f"{layer_name}_y_bond_patterns")(V.X) <= R.get(f"{layer_name}_y_bond")(V.X1, V.X2, V.X3, V.X4)[
            param_size]),
        (R.get(f"{layer_name}_y_bond_patterns")(V.X2) <= R.get(f"{layer_name}_y_bond")(V.X1, V.X2, V.X3, V.X4)[
            param_size]),
        (R.get(f"{layer_name}_y_bond_patterns")(V.X3) <= R.get(f"{layer_name}_y_bond")(V.X1, V.X2, V.X3, V.X4)[
            param_size]),
        (R.get(f"{layer_name}_y_bond_patterns")(V.X) <= R.get(f"{layer_name}_y_group")(V.X1, V.X2, V.X3)[param_size]),
        (R.get(f"{layer_name}_y_bond_patterns")(V.X2) <= R.get(f"{layer_name}_y_group")(V.X1, V.X2, V.X3)[param_size]),
        (R.get(f"{layer_name}_pattern")(V.X) <= R.get(f"{layer_name}_y_bond_patterns")(V.X)[param_size])]

    return template


def get_nbhoods(layer_name, node_embed, edge_embed, connection, param_size, carbon, atom_type):
    template = []

    template += [R.get(f"{layer_name}_four_nbhood")(V.X, V.X1, V.X2, V.X3, V.X4) <= (
        R.get(connection)(V.X, V.X1, V.B1), R.get(connection)(V.X, V.X2, V.B2),
        R.get(connection)(V.X, V.X3, V.B3), R.get(connection)(V.X, V.X4, V.B4),
        R.get(edge_embed)(V.B1), R.get(edge_embed)(V.B2),
        R.get(edge_embed)(V.B3), R.get(edge_embed)(V.B4),
        R.get(node_embed)(V.X1), R.get(node_embed)(V.X2),
        R.get(node_embed)(V.X3), R.get(node_embed)(V.X4),
        R.get(node_embed)(V.X),
        R.special.alldiff(...))]

    template += [R.get(f"{layer_name}_three_nbhood")(V.X, V.X1, V.X2, V.X3) <= (
        R.get(connection)(V.X, V.X1, V.B1), R.get(connection)(V.X, V.X2, V.B2),
        R.get(connection)(V.X, V.X3, V.B3),
        R.get(edge_embed)(V.B1), R.get(edge_embed)(V.B2), R.get(edge_embed)(V.B3),
        R.get(node_embed)(V.X1), R.get(node_embed)(V.X2), R.get(node_embed)(V.X3),
        R.get(node_embed)(V.X),
        R.special.alldiff(...))]

    template += [R.get(f"{layer_name}_chiral_center")(V.C) <= (
        R.get(carbon)(V.C),
        R.get(atom_type)(V.X1)[param_size], R.get(atom_type)(V.X2)[param_size], R.get(atom_type)(V.X3)[param_size],
        R.get(atom_type)(V.X4)[param_size],
        R.get(connection)(V.C, V.X1, V.B1), R.get(connection)(V.C, V.X2, V.B2), R.get(connection)(V.C, V.X3, V.B3),
        R.get(connection)(V.C, V.X4, V.B4),
        R.get(edge_embed)(V.B1)[param_size], R.get(edge_embed)(V.B2)[param_size], R.get(edge_embed)(V.B3)[param_size],
        R.get(edge_embed)(V.B4)[param_size],
        R.get(node_embed)(V.X1)[param_size], R.get(node_embed)(V.X2)[param_size], R.get(node_embed)(V.X3)[param_size],
        R.get(node_embed)(V.X4)[param_size],
        R.special.alldiff(...))]

    template += [R.get(f"{layer_name}_nbhood")(V.X) <= R.get(f"{layer_name}_four_nbhood")(V.X, V.X1, V.X2, V.X3, V.X4)[
        param_size]]
    template += [R.get(f"{layer_name}_nbhood")(V.X1) <= R.get(f"{layer_name}_four_nbhood")(V.X, V.X1, V.X2, V.X3, V.X4)[
        param_size]]
    template += [R.get(f"{layer_name}_nbhood")(V.X) <= R.get(f"{layer_name}_three_nbhood")(V.X, V.X1, V.X2, V.X3)[
        param_size]]
    template += [R.get(f"{layer_name}_nbhood")(V.X1) <= R.get(f"{layer_name}_three_nbhood")(V.X, V.X1, V.X2, V.X3)[
        param_size]]
    template += [R.get(f"{layer_name}_nbhood")(V.X) <= R.get(f"{layer_name}_chiral_center")(V.X)[param_size]]
    template += [R.get(f"{layer_name}_pattern")(V.X) <= R.get(f"{layer_name}_nbhood")(V.X)[param_size]]

    return template


def get_circular(layer_name, node_embed, edge_embed, connection, param_size, carbon, single_bond, double_bond):
    template = []

    template += [R.get(f"{layer_name}_n_c")(V.X) <= (R.get(carbon)(V.X))]
    template += [R.get(f"{layer_name}_heterocycle")(V.X) <= (
        R.get(f"{layer_name}_cycle")(V.X, V.C)[param_size], R.get(carbon)(V.C),
        ~R.get(f"{layer_name}_n_c")(V.X))]

    template += [R.get(f"{layer_name}_brick")(V.X) <= (
        R.get(node_embed)(V.Y1)[param_size], R.get(edge_embed)(V.B1)[param_size], R.get(connection)(V.X, V.Y1, V.B1),
        R.get(single_bond)(V.B1),
        R.get(node_embed)(V.Y2)[param_size], R.get(edge_embed)(V.B2)[param_size], R.get(connection)(V.Y1, V.Y2, V.B2),
        R.get(double_bond)(V.B1),
        R.get(node_embed)(V.Y3)[param_size], R.get(edge_embed)(V.B3)[param_size], R.get(connection)(V.Y2, V.Y3, V.B3),
        R.get(single_bond)(V.B1),
        R.get(node_embed)(V.X)[param_size], R.get(edge_embed)(V.B4)[param_size], R.get(connection)(V.Y3, V.X, V.B4),
        R.get(double_bond)(V.B1),
        R.special.alldiff(...))]

    template += [R.get(f"{layer_name}_circular")(V.X) <= R.get(f"{layer_name}_brick")(V.X)[param_size]]
    template += [R.get(f"{layer_name}_circular")(V.X) <= R.get(f"{layer_name}_heterocycle")(V.X)[param_size]]
    template += [R.get(f"{layer_name}_pattern")(V.X) <= R.get(f"{layer_name}_circular")(V.X)[param_size]]

    return template


def get_collective(layer_name, node_embed, edge_embed, connection, param_size, carbon, aliphatic_bond, max_depth):
    template = []

    template += [R.get(f"{layer_name}_n_cycle")(V.X, V.Y) <= R.get(f"{layer_name}_cycle")(V.X, V.Y)]
    template += [R.get(f"{layer_name}_bridge")(V.X) <= (
        R.get(connection)(V.X, V.Y, V.B1), R.get(connection)(V.X, V.Z, V.B2),
        ~R.get(f"{layer_name}_n_cycle")(V.X, V.X1),
        ~R.get(f"{layer_name}_n_cycle")(V.Y, V.Z),
        R.get(f"{layer_name}_cycle")(V.Y, V.Y1)[param_size],
        R.get(f"{layer_name}_cycle")(V.Z, V.Z1)[param_size],
        R.get(edge_embed)(V.B1)[param_size], R.get(edge_embed)(V.B2)[param_size],
        R.get(node_embed)(V.X)[param_size],
        R.special.alldiff(V.X, V.Y, V.Z))]

    template += [R.get(f"{layer_name}_shared_atom")(V.X) <= (
        R.get(connection)(V.X, V.Y, V.B1), R.get(connection)(V.X, V.Z, V.B2),
        R.get(f"{layer_name}_cycle")(V.X, V.Y)[param_size],
        R.get(f"{layer_name}_cycle")(V.X, V.Z)[param_size],
        ~R.get(f"{layer_name}_n_cycle")(V.Y, V.Z),
        R.get(edge_embed)(V.B1)[param_size], R.get(edge_embed)(V.B2)[param_size],
        R.get(node_embed)(V.X)[param_size],
        R.special.alldiff(...))]

    template += [R.get(f"{layer_name}_aliphatic_chain")(V.X, V.Y) <= R.get(f"{layer_name}_aliphatic_chain")(V.X, V.Y,
                                                                                                            max_depth)]
    template += [R.get(f"{layer_name}_aliphatic_chain")(V.X, V.Y, 0) <= (R.get(connection)(V.X, V.Z, V.B),
                                                                         R.get(carbon)(V.X), R.get(carbon)(V.Y),
                                                                         R.get(aliphatic_bond)(V.B),
                                                                         R.get(edge_embed)(V.B)[param_size])]

    template += [R.get(f"{layer_name}_aliphatic_chain")(V.X, V.Y, V.T) <= (
        R.get(connection)(V.X, V.Z, V.B), R.get(carbon)(V.X),
        R.get(edge_embed)(V.B)[param_size],
        R.special.next(V.T1, V.T),
        R.get(f"{layer_name}_aliphatic_chain")(V.Z, V.Y, V.T1),
        R.get(aliphatic_bond)(V.B)[param_size])]

    template += [
        R.get(f"{layer_name}_collective_pattern")(V.X) <= R.get(f"{layer_name}_aliphatic_chain")(V.X)[param_size]]
    template += [R.get(f"{layer_name}_collective_pattern")(V.X) <= R.get(f"{layer_name}_shared_atom")(V.X)[param_size]]
    template += [R.get(f"{layer_name}_collective_pattern")(V.X) <= R.get(f"{layer_name}_bridge")(V.X)[param_size]]
    template += [R.get(f"{layer_name}_pattern")(V.X) <= R.get(f"{layer_name}_collective_pattern")(V.X)[param_size]]

    template += [R.get(f"{layer_name}_subgraph_pattern")(V.X) <= (R.get(f"{layer_name}_pattern")(V.X)[param_size],
                                                                  R.get(f"{layer_name}_pattern")(V.Y)[param_size],
                                                                  R.get(f"{layer_name}_path")(V.X, V.Y))]
    return template


def get_chem_rules(layer_name: str, node_embed: str, edge_embed: str, connection: str, param_size: int,
                   halogens: list,
                   single_bond=None, double_bond=None, triple_bond=None, aromatic_bonds=None,
                   carbon=None, hydrogen=None, oxygen=None, nitrogen=None, sulfur=None,
                   path=None,
                   hydrocarbons=False, nitro=False, sulfuric=False, oxy=False, relaxations=False):
    template = []

    template += [R.predict[1, param_size] <= R.get(f"{layer_name}_chem_rules")]

    for a in halogens:
        template += [(R.get(f"{layer_name}_halogen")(V.X)[param_size,] <= R.get(a)(V.X))]
    if relaxations:
        for b in [single_bond, double_bond, triple_bond]:
            template += [(R.get(f"{layer_name}_aliphatic_bond")(V.B)[param_size,] <= R.get(b)(V.B))]

        for b in aromatic_bonds:
            template += [(R.get(f"{layer_name}_aromatic_bond")(V.B)[param_size,] <= R.get(b)(V.B))]

        for a in [oxygen, nitrogen, sulfur]:
            template += [(R.get(f"{layer_name}_key_atom")(V.A)[param_size,] <= R.get(a)(V.A))]
            template += [(R.get(f"{layer_name}_noncarbon")(V.A)[param_size,] <= R.get(a)(V.A))]

        template += [(R.get(f"{layer_name}_key_atom")(V.A)[param_size,] <= R.get(carbon)(V.A))]

    param_size = (param_size, param_size)

    template += get_general(layer_name, node_embed, edge_embed, connection, param_size,
                            single_bond, double_bond, triple_bond, f"{layer_name}_aromatic_bond", hydrogen, carbon, oxygen)

    template += [R.get(f"{layer_name}_functional_group") <= R.get(f"{layer_name}_functional_group")(V.X)]
    template += [R.get(f"{layer_name}_functional_group")(V.X)[param_size] <= R.get(f"{layer_name}_general_groups")(V.X)]
    if path:
        template += [R.get(f"{layer_name}_connected_groups")(V.X, V.Y) <= (
                     R.get(f"{layer_name}_functional_group")(V.X)[param_size],
                     R.get(f"{layer_name}_functional_group")(V.Y)[param_size],
                     R.get(path)(V.X, V.Y))]
        if relaxations:
            template += [R.get(f"{layer_name}_connected_groups")(V.X, V.Y) <= (
                         R.get(f"{layer_name}_relaxed_functional_group")(V.X)[param_size],
                         R.get(f"{layer_name}_relaxed_functional_group")(V.Y)[param_size], R.get(path)(V.X, V.Y))]
        template += [R.get(f"{layer_name}_chem_rules")[param_size] <= R.get(f"{layer_name}_connected_groups")(V.X, V.Y)]

    if hydrocarbons:
        template += get_hydrocarbons(layer_name, param_size, carbon)
        template += [R.get(f"{layer_name}_functional_group")(V.X)[param_size] <= R.get(f"{layer_name}_hydrocarbon_groups")(V.X)]
    if oxy:
        template += get_oxy(layer_name, param_size, carbon, oxygen, hydrogen)
        template += [R.get(f"{layer_name}_functional_group")(V.X)[param_size] <= R.get(f"{layer_name}_oxygen_groups")(V.X)]
    if nitro:
        template += get_nitro(layer_name, param_size, carbon, oxygen, hydrogen, nitrogen)
        template += [R.get(f"{layer_name}_functional_group")(V.X)[param_size] <= R.get(f"{layer_name}_nitrogen_groups")(V.X)]
    if sulfuric:
        template += get_sulfuric(layer_name, param_size, carbon, oxygen, hydrogen, sulfur)
        template += [R.get(f"{layer_name}_functional_group")(V.X)[param_size] <= R.get(f"{layer_name}_sulfuric_groups")(V.X)]
    if relaxations:
        template += get_relaxations(layer_name, param_size, connection, carbon)
        template += [
            R.get(f"{layer_name}_relaxed_functional_group") <= R.get(f"{layer_name}_relaxed_functional_group")(V.X)]
        template += [R.get(f"{layer_name}_chem_rules")[param_size] <= R.get(f"{layer_name}_relaxed_functional_group")]

    template += [R.get(f"{layer_name}_chem_rules")[param_size] <= R.get(f"{layer_name}_functional_group")]

    return template


def get_general(layer_name, node_embed, edge_embed, connection, param_size,
                single_bond, double_bond, triple_bond, aromatic_bond,
                hydrogen, carbon, oxygen):
    template = []
    template += [R.get(f"{layer_name}_bond_message")(V.X, V.Y, V.B) <= (
        R.get(node_embed)(V.X)[param_size], R.get(node_embed)(V.Y)[param_size], R.get(edge_embed)(V.B)[param_size])]

    template += [
        R.hidden.get(f"{layer_name}_single_bonded")(V.X, V.Y) <= (R.hidden.get(f"{layer_name}_single_bonded")(V.X, V.Y, V.B))]
    template += [R.hidden.get(f"{layer_name}_single_bonded")(V.X, V.Y, V.B) <= (
    R.get(connection)(V.X, V.Y, V.B), R.get(single_bond)(V.B))]

    template += [
        R.hidden.get(f"{layer_name}_double_bonded")(V.X, V.Y) <= (R.hidden.get(f"{layer_name}_double_bonded")(V.X, V.Y, V.B))]
    template += [R.hidden.get(f"{layer_name}_double_bonded")(V.X, V.Y, V.B) <= (
    R.get(connection)(V.X, V.Y, V.B), R.get(double_bond)(V.B))]

    template += [
        R.hidden.get(f"{layer_name}_triple_bonded")(V.X, V.Y) <= (R.hidden.get(f"{layer_name}_triple_bonded")(V.Y, V.X, V.B))]
    template += [R.hidden.get(f"{layer_name}_triple_bonded")(V.X, V.Y, V.B) <= (
    R.get(connection)(V.Y, V.X, V.B), R.get(triple_bond)(V.B))]

    template += [
        R.get(f"{layer_name}_aromatic_bonded")(V.X, V.Y) <= (R.get(f"{layer_name}_aromatic_bonded")(V.X, V.Y, V.B))]
    template += [R.get(f"{layer_name}_aromatic_bonded")(V.X, V.Y, V.B) <= (
    R.get(connection)(V.X, V.Y, V.B), R.get(aromatic_bond)(V.B))]

    template += [R.get(f"{layer_name}_saturated")(V.X) <= (R.get(carbon)(V.X),
                                                           R.hidden.get(f"{layer_name}_single_bonded")(V.X, V.Y1),
                                                           R.hidden.get(f"{layer_name}_single_bonded")(V.X, V.Y2),
                                                           R.hidden.get(f"{layer_name}_single_bonded")(V.X, V.Y3),
                                                           R.hidden.get(f"{layer_name}_single_bonded")(V.X, V.Y4),
                                                           R.special.alldiff(...))]
    #template += [(R.get(f"{layer_name}_saturated") <= (R.get(f"{layer_name}_saturated")(V.X))) | [Aggregation.MIN]]

    template += [R.get(f"{layer_name}_halogen_group")(V.Y) <= (
    R.get(f"{layer_name}_halogen")(V.X), R.hidden.get(f"{layer_name}_single_bonded")(V.X, V.Y, V.B),
    R.get(f"{layer_name}_bond_message")(V.X, V.Y, V.B))]

    template += [R.get(f"{layer_name}_hydroxyl")(V.O) <= (R.get(oxygen)(V.O), R.get(hydrogen)(V.H),
                                                          R.hidden.get(f"{layer_name}_single_bonded")(V.O, V.H, V.B),
                                                          R.get(f"{layer_name}_bond_message")(V.O, V.H, V.B))]

    template += [R.get(f"{layer_name}_carbonyl_group")(V.C, V.O) <= (
        R.get(carbon)(V.C), R.get(oxygen)(V.O), R.hidden.get(f"{layer_name}_double_bonded")(V.O, V.C, V.B),
        R.get(f"{layer_name}_bond_message")(V.O, V.C, V.B))]
    template += [
        R.get(f"{layer_name}_carbonyl_group")(V.C, V.O, V.R1, V.R2) <= (R.get(f"{layer_name}_carbonyl_group")(V.C, V.O),
                                                                        R.hidden.get(f"{layer_name}_single_bonded")(V.C, V.R1,
                                                                                                             V.B1),
                                                                        R.hidden.get(f"{layer_name}_single_bonded")(V.C, V.R2,
                                                                                                             V.B2),
                                                                        R.get(f"{layer_name}_bond_message")(V.C, V.R1,
                                                                                                            V.B1),
                                                                        R.get(f"{layer_name}_bond_message")(V.C, V.R2,
                                                                                                            V.B2),
                                                                        R.special.alldiff(...))]

    template += [R.get(f"{layer_name}_general_groups")(V.X) <= R.get(f"{layer_name}_hydroxyl")(V.X)[param_size]]
    template += [
        R.get(f"{layer_name}_general_groups")(V.X) <= R.get(f"{layer_name}_carbonyl_group")(V.X, V.Y)[param_size]]
    template += [
        R.get(f"{layer_name}_general_groups")(V.Y) <= R.get(f"{layer_name}_carbonyl_group")(V.X, V.Y)[param_size]]
    template += [
        R.get(f"{layer_name}_general_groups")(V.X) <= R.get(f"{layer_name}_halogen_group")(V.X)[param_size]]

    return template


def get_hydrocarbons(layer_name, param_size, carbon):
    template = []
    template += [R.get(f"{layer_name}_benzene_ring")(V.A) <= R.get(f"{layer_name}_benzene_ring")(V.A, V.B)]
    template += [
        R.get(f"{layer_name}_benzene_ring")(V.A, V.B) <= R.get(f"{layer_name}_benzene_ring")(V.A, V.B, V.C, V.D, V.E,
                                                                                             V.F)]

    template += [R.get(f"{layer_name}_benzene_ring")(V.A, V.B, V.C, V.D, V.E, V.F) <= (
    R.get(f"{layer_name}_aromatic_bonded")(V.A, V.B, V.B1),
    R.get(f"{layer_name}_aromatic_bonded")(V.B, V.C, V.B2),
    R.get(f"{layer_name}_aromatic_bonded")(V.C, V.D, V.B3),
    R.get(f"{layer_name}_aromatic_bonded")(V.D, V.E, V.B4),
    R.get(f"{layer_name}_aromatic_bonded")(V.E, V.F, V.B5),
    R.get(f"{layer_name}_aromatic_bonded")(V.F, V.A, V.B6),
    R.get(carbon)(V.A), R.get(carbon)(V.B), R.get(carbon)(V.C),
    R.get(carbon)(V.D), R.get(carbon)(V.E), R.get(carbon)(V.F),
    R.get(f"{layer_name}_bond_message")(V.A, V.B, V.B1),
    R.get(f"{layer_name}_bond_message")(V.B, V.C, V.B2),
    R.get(f"{layer_name}_bond_message")(V.C, V.D, V.B3),
    R.get(f"{layer_name}_bond_message")(V.D, V.E, V.B4),
    R.get(f"{layer_name}_bond_message")(V.E, V.F, V.B5),
    R.get(f"{layer_name}_bond_message")(V.F, V.A, V.B6),

    R.special.alldiff(...))]

    template += [R.get(f"{layer_name}_alkene_bond")(V.C1, V.C2) <= (
        R.get(carbon)(V.C1), R.get(carbon)(V.C2), R.hidden.get(f"{layer_name}_double_bonded")(V.C1, V.C2, V.B),
        R.get(f"{layer_name}_bond_message")(V.C1, V.C2, V.B))]
    template += [R.get(f"{layer_name}_alkyne_bond")(V.C1, V.C2) <= (
        R.get(carbon)(V.C1), R.get(carbon)(V.C2), R.hidden.get(f"{layer_name}_triple_bonded")(V.C1, V.C2, V.B),
        R.get(f"{layer_name}_bond_message")(V.C1, V.C2, V.B))]

    template += [
        R.get(f"{layer_name}_hydrocarbon_groups")(V.C1) <= R.get(f"{layer_name}_alkene_bond")(V.C1, V.C2)[param_size]]
    template += [
        R.get(f"{layer_name}_hydrocarbon_groups")(V.C1) <= R.get(f"{layer_name}_alkyne_bond")(V.C1, V.C2)[param_size]]
    template += [R.get(f"{layer_name}_hydrocarbon_groups")(V.A) <= R.get(f"{layer_name}_benzene_ring")(V.A)[param_size]]

    return template


def get_oxy(layer_name, param_size, carbon, oxygen, hydrogen):
    template = []

    template += [R.get(f"{layer_name}_alcoholic")(V.C) <= (R.get(f"{layer_name}_saturated")(V.C),
                                                           R.get(f"{layer_name}_hydroxyl")(V.O),
                                                           R.hidden.get(f"{layer_name}_single_bonded")(V.C, V.O, V.B1),
                                                           R.get(f"{layer_name}_bond_message")(V.C, V.O, V.B1))]

    template += [R.get(f"{layer_name}_ketone")(V.C) <= (R.get(f"{layer_name}_carbonyl_group")(V.C, V.O, V.R1, V.R2),
                                                        R.get(carbon)(V.R1), R.get(carbon)(V.R2))]
    template += [R.get(f"{layer_name}_aldehyde")(V.C) <= (R.get(f"{layer_name}_carbonyl_group")(V.C, V.O, V.R, V.H),
                                                          R.get(carbon)(V.R), R.get(hydrogen)(V.H))]

    template += [R.get(f"{layer_name}_acyl_halide")(V.C) <= (R.get(f"{layer_name}_carbonyl_group")(V.C, V.O, V.R, V.X),
                                                             R.get(carbon)(V.R), R.get(f"{layer_name}_halogen")(V.X))]

    template += [
        R.get(f"{layer_name}_carboxylic_acid")(V.C) <= (R.get(f"{layer_name}_carbonyl_group")(V.C, V.O, V.R, V.O1),
                                                        R.get(carbon)(V.R), R.get(f"{layer_name}_hydroxyl")(V.O1))]

    template += [R.get(f"{layer_name}_oxy_groups")(V.X) <= R.get(f"{layer_name}_alcoholic")(V.X)[param_size]]
    template += [R.get(f"{layer_name}_oxy_groups")(V.X) <= R.get(f"{layer_name}_ketone")(V.X)[param_size]]
    template += [R.get(f"{layer_name}_oxy_groups")(V.X) <= R.get(f"{layer_name}_aldehyde")(V.X)[param_size]]
    template += [R.get(f"{layer_name}_oxy_groups")(V.X) <= R.get(f"{layer_name}_acyl_halide")(V.X)[param_size]]
    template += [R.get(f"{layer_name}_oxy_groups")(V.X) <= R.get(f"{layer_name}_carboxylic_acid")(V.X)[param_size]]

    template += [R.get(f"{layer_name}_carboxylic_acid_anhydride")(V.C1, V.C2) <= (
    R.get(f"{layer_name}_carbonyl_group")(V.C1, V.X1, V.O12, V.Y1),
    R.get(oxygen)(V.O12),
    R.get(f"{layer_name}_carbonyl_group")(V.C2, V.X2, V.O12, V.Y1),
    R.special.alldiff(V.C1, V.C2))]

    template += [R.get(f"{layer_name}_ester")(V.X) <= R.get(f"{layer_name}_ester")(V.X, V.Y)]
    template += [R.get(f"{layer_name}_ester")(V.Y) <= R.get(f"{layer_name}_ester")(V.X, V.Y)]
    template += [R.get(f"{layer_name}_ester")(V.R1, V.R2) <= (
    R.get(f"{layer_name}_carbonyl_group")(V.C, V.X, V.R1, V.O),
    R.get(carbon)(V.R1), R.get(oxygen)(V.O),
    R.get(carbon)(V.R2), R.hidden.get(f"{layer_name}_single_bonded")(V.O, V.R2, V.B),
    R.get(f"{layer_name}_bond_message")(V.O, V.R2, V.B))]

    template += [
        R.get(f"{layer_name}_carbonate_ester")(V.X) <= R.get(f"{layer_name}_carbonate_ester")(V.X, V.Y)]
    template += [
        R.get(f"{layer_name}_carbonate_ester")(V.Y) <= R.get(f"{layer_name}_carbonate_ester")(V.X, V.Y)]
    template += [R.get(f"{layer_name}_carbonate_ester")(V.R1, V.R2) <= (
    R.get(f"{layer_name}_carbonyl_group")(V.C, V.X, V.O1, V.O2),
    R.get(oxygen)(V.O1), R.get(oxygen)(V.O2),
    R.get(carbon)(V.R1), R.hidden.get(f"{layer_name}_single_bonded")(V.R1, V.O1, V.B1),
    R.get(f"{layer_name}_bond_message")(V.O1, V.R1, V.B1),
    R.get(carbon)(V.R2), R.hidden.get(f"{layer_name}_single_bonded")(V.R2, V.O2, V.B2),
    R.get(f"{layer_name}_bond_message")(V.O2, V.R2, V.B2))]

    template += [R.get(f"{layer_name}_ether")(V.X) <= R.get(f"{layer_name}_ether")(V.X, V.Y)]
    template += [R.get(f"{layer_name}_ether")(V.Y) <= R.get(f"{layer_name}_ether")(V.X, V.Y)]
    template += [R.get(f"{layer_name}_ether")(V.C, V.R) <= (R.get(carbon)(V.C), R.get(oxygen)(V.O), R.get(carbon)(V.R),
                                                            R.hidden.get(f"{layer_name}_single_bonded")(V.C, V.O, V.B1),
                                                            R.hidden.get(f"{layer_name}_single_bonded")(V.O, V.R, V.B2),
                                                            R.get(f"{layer_name}_bond_message")(V.C, V.O, V.B1),
                                                            R.get(f"{layer_name}_bond_message")(V.R, V.O, V.B2),
                                                            R.special.alldiff(...))]
    template += [
        R.get(f"{layer_name}_oxy_groups")(V.X) <= R.get(f"{layer_name}_carboxylic_acid_anhydride")(V.X, V.Y)[
            param_size]]
    template += [R.get(f"{layer_name}_oxy_groups")(V.X) <= R.get(f"{layer_name}_ester")(V.X)[param_size]]
    template += [R.get(f"{layer_name}_oxy_groups")(V.X) <= R.get(f"{layer_name}_carbonate_ester")(V.X)[param_size]]
    template += [R.get(f"{layer_name}_oxy_groups")(V.X) <= R.get(f"{layer_name}_ether")(V.X)[param_size]]

    return template


def get_nitro(layer_name, param_size, carbon, oxygen, hydrogen, nitrogen):
    template = []

    template += [R.get(f"{layer_name}_n_carbonyl")(V.C, V.O) <= (R.get(f"{layer_name}_carbonyl_group")(V.C, V.O))]

    template += [R.get(f"{layer_name}_amine")(V.N) <= (
    ~R.get(f"{layer_name}_n_carbonyl")(V.C, V.O), R.get(f"{layer_name}_amino_group")(V.N, V.C, V.R1,
                                                                                     V.R2))]
    template += [
        R.get(f"{layer_name}_amino_group")(V.N, V.R1, V.R2, V.R3) <= (R.get(carbon)(V.R1), R.get(nitrogen)(V.N),
                                                                      R.hidden.get(f"{layer_name}_single_bonded")(V.N, V.R1,
                                                                                                           V.B1),
                                                                      R.hidden.get(f"{layer_name}_single_bonded")(V.N, V.R2,
                                                                                                           V.B2),
                                                                      R.hidden.get(f"{layer_name}_single_bonded")(V.N, V.R3,
                                                                                                           V.B3),
                                                                      R.get(f"{layer_name}_bond_message")(V.N, V.R1,
                                                                                                          V.B1),
                                                                      R.get(f"{layer_name}_bond_message")(V.N, V.R2,
                                                                                                          V.B2),
                                                                      R.get(f"{layer_name}_bond_message")(V.N, V.R3,
                                                                                                          V.B3),
                                                                      R.special.alldiff(...))]


    template += [R.get(f"{layer_name}_quat_ammonion")(V.N) <= (
    R.get(nitrogen)(V.N), R.get(carbon)(V.C), R.get(f"{layer_name}_amino_group")(V.N, V.R1, V.R2, V.R3),
    R.hidden.get(f"{layer_name}_single_bonded")(V.N, V.C, V.B),
    R.get(f"{layer_name}_bond_message")(V.N, V.C, V.B),
    R.special.alldiff(...))]

    template += [R.get(f"{layer_name}_amide")(V.R) <= (R.get(f"{layer_name}_amide")(V.R, V.R1, V.R2))]
    template += [R.get(f"{layer_name}_amide")(V.R1) <= (R.get(f"{layer_name}_amide")(V.R, V.R1, V.R2))]

    template += [R.get(f"{layer_name}_amide")(V.R, V.R1, V.R2) <= (
    R.get(f"{layer_name}_carbonyl_group")(V.C, V.O, V.R, V.N),
    R.get(f"{layer_name}_amino_group")(V.N, V.C, V.R1, V.R2), R.special.alldiff(...))]


    template += [R.get(f"{layer_name}_imine")(V.R) <= R.get(f"{layer_name}_imine")(V.R, V.R1, V.R2)]
    template += [R.get(f"{layer_name}_imine")(V.R1) <= R.get(f"{layer_name}_imine")(V.R, V.R1, V.R2)]
    template += [R.get(f"{layer_name}_imine")(V.R, V.R1, V.R2) <= (R.get(carbon)(V.C), R.get(nitrogen)(V.N),
                                                                             R.hidden.get(f"{layer_name}_double_bonded")(V.C,
                                                                                                                  V.N,
                                                                                                                  V.B),
                                                                             R.hidden.get(f"{layer_name}_single_bonded")(V.C,
                                                                                                                  V.R1,
                                                                                                                  V.B1),
                                                                             R.hidden.get(f"{layer_name}_single_bonded")(V.C,
                                                                                                                  V.R2,
                                                                                                                  V.B2),
                                                                             R.hidden.get(f"{layer_name}_single_bonded")(V.N,
                                                                                                                  V.R,
                                                                                                                  V.B3),
                                                                             R.get(f"{layer_name}_bond_message")(V.N,
                                                                                                                 V.C,
                                                                                                                 V.B),
                                                                             R.get(f"{layer_name}_bond_message")(V.C,
                                                                                                                 V.R1,
                                                                                                                 V.B1),
                                                                             R.get(f"{layer_name}_bond_message")(V.C,
                                                                                                                 V.R2,
                                                                                                                 V.B2),
                                                                             R.get(f"{layer_name}_bond_message")(V.N,
                                                                                                                 V.R,
                                                                                                                 V.B3),
                                                                             R.special.alldiff(...))]

    template += [R.get(f"{layer_name}_imide")(V.R) <= R.get(f"{layer_name}_imide")(V.R, V.R1, V.R2)]
    template += [R.get(f"{layer_name}_imide")(V.R1) <= R.get(f"{layer_name}_imide")(V.R, V.R1, V.R2)]
    template += [R.get(f"{layer_name}_imide")(V.R2) <= R.get(f"{layer_name}_imide")(V.R, V.R1, V.R2)]
    template += [R.get(f"{layer_name}_imide")(V.R, V.R1, V.R2) <= (
    R.get(carbon)(V.C1), R.get(nitrogen)(V.N), R.get(carbon)(V.C2),
    R.get(f"{layer_name}_carbonyl_group")(V.C1, V.O1, V.R1, V.N),
    R.get(f"{layer_name}_carbonyl_group")(V.C2, V.O2, V.R2, V.N),
    R.hidden.get(f"{layer_name}_single_bonded")(V.N, V.R, V.B),
    R.get(f"{layer_name}_bond_message")(V.N, V.R, V.B),
    R.special.alldiff(...))]

    template += [R.get(f"{layer_name}_azide")(V.C) <= (
    R.get(carbon)(V.C1), R.get(nitrogen)(V.N1), R.get(nitrogen)(V.N2), R.get(nitrogen)(V.N3),
    R.hidden.get(f"{layer_name}_single_bonded")(V.C, V.N1, V.B1),
    R.hidden.get(f"{layer_name}_double_bonded")(V.N1, V.N2, V.B2),
    R.hidden.get(f"{layer_name}_double_bonded")(V.N2, V.N3, V.B3),
    R.get(f"{layer_name}_bond_message")(V.C, V.N, V.B1),
    R.get(f"{layer_name}_bond_message")(V.N1, V.N2, V.B2),
    R.get(f"{layer_name}_bond_message")(V.N2, V.N3, V.B3),
    R.special.alldiff(...))]

    template += [R.get(f"{layer_name}_azo")(V.C1, V.C2) <= (
    R.get(carbon)(V.C1), R.get(nitrogen)(V.N1), R.get(nitrogen)(V.N2), R.get(carbon)(V.C2),
    R.hidden.get(f"{layer_name}_single_bonded")(V.C, V.N1, V.B1),
    R.hidden.get(f"{layer_name}_double_bonded")(V.N1, V.N2, V.B2),
    R.hidden.get(f"{layer_name}_single_bonded")(V.N2, V.C2, V.B3),
    R.get(f"{layer_name}_bond_message")(V.C, V.N, V.B1),
    R.get(f"{layer_name}_bond_message")(V.N1, V.N2, V.B2),
    R.get(f"{layer_name}_bond_message")(V.N2, V.C2, V.B3),
    R.special.alldiff(...))]

    template += [R.get(f"{layer_name}_cyanate")(V.R) <= (
    R.get(carbon)(V.C), R.get(nitrogen)(V.N), R.get(oxygen)(V.O), R.get(carbon)(V.R),
    R.hidden.get(f"{layer_name}_triple_bonded")(V.C, V.N, V.B1),
    R.hidden.get(f"{layer_name}_single_bonded")(V.C, V.O, V.B2),
    R.hidden.get(f"{layer_name}_single_bonded")(V.O, V.R, V.B3),
    R.get(f"{layer_name}_bond_message")(V.C, V.N, V.B1),
    R.get(f"{layer_name}_bond_message")(V.C, V.O, V.B2),
    R.get(f"{layer_name}_bond_message")(V.O, V.R, V.B3),
    R.special.alldiff(...))]

    template += [R.get(f"{layer_name}_isocyanate")(V.R) <= (
    R.get(carbon)(V.C), R.get(nitrogen)(V.N), R.get(oxygen)(V.O), R.get(carbon)(V.R),
    R.hidden.get(f"{layer_name}_double_bonded")(V.C, V.N, V.B1),
    R.hidden.get(f"{layer_name}_double_bonded")(V.C, V.O, V.B2),
    R.hidden.get(f"{layer_name}_single_bonded")(V.N, V.R, V.B3),
    R.get(f"{layer_name}_bond_message")(V.C, V.N, V.B1),
    R.get(f"{layer_name}_bond_message")(V.C, V.O, V.B2),
    R.get(f"{layer_name}_bond_message")(V.N, V.R, V.B3),
    R.special.alldiff(...))]

    template += [R.get(f"{layer_name}_nitrogen_groups")(V.X) <= R.get(f"{layer_name}_amine")(V.X)[param_size]]
    template += [R.get(f"{layer_name}_nitrogen_groups")(V.X) <= R.get(f"{layer_name}_quat_ammonion")(V.X)[param_size]]
    template += [R.get(f"{layer_name}_nitrogen_groups")(V.X) <= R.get(f"{layer_name}_amide")(V.X)[param_size]]
    template += [R.get(f"{layer_name}_nitrogen_groups")(V.X) <= R.get(f"{layer_name}_imine")(V.X)[param_size]]
    template += [R.get(f"{layer_name}_nitrogen_groups")(V.X) <= R.get(f"{layer_name}_azide")(V.X)[param_size]]
    template += [R.get(f"{layer_name}_nitrogen_groups")(V.X) <= R.get(f"{layer_name}_imide")(V.X)[param_size]]
    template += [R.get(f"{layer_name}_nitrogen_groups")(V.X) <= R.get(f"{layer_name}_azo")(V.X, V.Y)[param_size]]
    template += [R.get(f"{layer_name}_nitrogen_groups")(V.X) <= R.get(f"{layer_name}_isocyanate")(V.X)[param_size]]
    template += [R.get(f"{layer_name}_nitrogen_groups")(V.X) <= R.get(f"{layer_name}_cyanate")(V.X)[param_size]]

    template += [
        R.get(f"{layer_name}_nitro_group")(V.R) <= (R.get(nitrogen)(V.N), R.get(oxygen)(V.O1), R.get(oxygen)(V.O2),
                                                    R.hidden.get(f"{layer_name}_single_bonded")(V.R, V.N, V.B1),
                                                    R.hidden.get(f"{layer_name}_double_bonded")(V.N, V.O1, V.B2),
                                                    R.hidden.get(f"{layer_name}_single_bonded")(V.N, V.O2, V.B3),
                                                    R.get(f"{layer_name}_bond_message")(V.R, V.N, V.B1),
                                                    R.get(f"{layer_name}_bond_message")(V.N, V.O1, V.B2),
                                                    R.get(f"{layer_name}_bond_message")(V.N, V.O2, V.B3),
                                                    R.special.alldiff(...))]

    template += [R.get(f"{layer_name}_nitro")(V.C) <= (R.get(carbon)(V.C), R.get(f"{layer_name}_nitro_group")(V.C))]

    template += [R.get(f"{layer_name}_nitrate")(V.C) <= (
    R.get(carbon)(V.C), R.get(oxygen)(V.O), R.get(f"{layer_name}_nitro_group")(V.O),
    R.hidden.get(f"{layer_name}_single_bonded")(V.C, V.O, V.B),
    R.get(f"{layer_name}_bond_message")(V.C, V.O, V.B),
    R.special.alldiff(...))]

    template += [R.get(f"{layer_name}_carbamate")(V.R) <= R.get(f"{layer_name}_carbamate")(V.R, V.R1, V.R2)]
    template += [R.get(f"{layer_name}_carbamate")(V.R1) <= R.get(f"{layer_name}_carbamate")(V.R, V.R1, V.R2)]
    template += [R.get(f"{layer_name}_carbamate")(V.R2) <= R.get(f"{layer_name}_carbamate")(V.R, V.R1, V.R2)]
    template += [R.get(f"{layer_name}_carbamate")(V.C1, V.C2, V.C3) <= (
    R.get(f"{layer_name}_amide")(V.O1, V.C2, V.C3),
    R.get(oxygen)(V.O1),
    R.hidden.get(f"{layer_name}_single_bonded")(V.O1, V.C1, V.B1),
    R.get(f"{layer_name}_bond_message")(V.O1, V.C1, V.B1),
    R.special.alldiff(...))]

    template += [R.get(f"{layer_name}_aziridine")(V.C1, V.C2) <= (
    R.get(carbon)(V.C1), R.get(carbon)(V.C1), R.get(nitrogen)(V.N), R.get(hydrogen)(V.H),
    R.hidden.get(f"{layer_name}_single_bonded")(V.C1, V.C2, V.B1),
    R.hidden.get(f"{layer_name}_single_bonded")(V.N, V.C1, V.B2),
    R.hidden.get(f"{layer_name}_single_bonded")(V.N, V.C2, V.B3),
    R.hidden.get(f"{layer_name}_single_bonded")(V.N, V.H, V.B4),
    R.get(f"{layer_name}_bond_message")(V.C1, V.C2, V.B1),
    R.get(f"{layer_name}_bond_message")(V.N, V.C1, V.B2),
    R.get(f"{layer_name}_bond_message")(V.N, V.C2, V.B3),
    R.get(f"{layer_name}_bond_message")(V.N, V.H, V.B4),
    R.special.alldiff(...))]

    template += [R.get(f"{layer_name}_nitrogen_groups")(V.X) <= R.get(f"{layer_name}_nitro")(V.X)[param_size]]
    template += [R.get(f"{layer_name}_nitrogen_groups")(V.X) <= R.get(f"{layer_name}_nitrate")(V.X)[param_size]]
    template += [
        R.get(f"{layer_name}_nitrogen_groups")(V.X) <= R.get(f"{layer_name}_carbamate")(V.X)[param_size]]
    template += [R.get(f"{layer_name}_nitrogen_groups")(V.X) <= R.get(f"{layer_name}_aziridine")(V.X, V.Y)[param_size]]

    return template


def get_sulfuric(layer_name, param_size, carbon, oxygen, hydrogen, sulfur):
    template = []
    template += [R.get(f"{layer_name}_thiocyanate")(V.R) <= (
    R.get(carbon)(V.C), R.get(sulfur)(V.S), R.get(oxygen)(V.O), R.get(carbon)(V.R),
    R.hidden.get(f"{layer_name}_triple_bonded")(V.C, V.S, V.B1),
    R.hidden.get(f"{layer_name}_single_bonded")(V.C, V.O, V.B2),
    R.hidden.get(f"{layer_name}_single_bonded")(V.O, V.R, V.B3),
    R.get(f"{layer_name}_bond_message")(V.C, V.S, V.B1),
    R.get(f"{layer_name}_bond_message")(V.C, V.O, V.B2),
    R.get(f"{layer_name}_bond_message")(V.O, V.R, V.B3),
    R.special.alldiff(...))]

    template += [R.get(f"{layer_name}_isothiocyanate")(V.R) <= (
    R.get(carbon)(V.C), R.get(sulfur)(V.S), R.get(oxygen)(V.O), R.get(carbon)(V.R),
    R.hidden.get(f"{layer_name}_double_bonded")(V.C, V.S, V.B1),
    R.hidden.get(f"{layer_name}_double_bonded")(V.C, V.O, V.B2),
    R.hidden.get(f"{layer_name}_single_bonded")(V.S, V.R, V.B3),
    R.get(f"{layer_name}_bond_message")(V.C, V.S, V.B1),
    R.get(f"{layer_name}_bond_message")(V.C, V.O, V.B2),
    R.get(f"{layer_name}_bond_message")(V.S, V.R, V.B3),
    R.special.alldiff(...))]

    template += [
        R.get(f"{layer_name}_sulfide")(V.C1, V.C2) <= (R.get(carbon)(V.C1), R.get(sulfur)(V.S), R.get(carbon)(V.C2),
                                                       R.hidden.get(f"{layer_name}_single_bonded")(V.C1, V.S, V.B1),
                                                       R.hidden.get(f"{layer_name}_single_bonded")(V.S, V.C2, V.B2),
                                                       R.get(f"{layer_name}_bond_message")(V.C1, V.S, V.B1),
                                                       R.get(f"{layer_name}_bond_message")(V.S, V.C2, V.B2),
                                                       R.special.alldiff(...))]

    template += [R.get(f"{layer_name}_disulfide")(V.C1, V.C2) <= (
    R.get(carbon)(V.C1), R.get(sulfur)(V.S1), R.get(sulfur)(V.S2), R.get(carbon)(V.C2),
    R.hidden.get(f"{layer_name}_single_bonded")(V.C1, V.S1, V.B1),
    R.hidden.get(f"{layer_name}_single_bonded")(V.S2, V.S2, V.B12),
    R.hidden.get(f"{layer_name}_single_bonded")(V.S2, V.C2, V.B2),
    R.get(f"{layer_name}_bond_message")(V.C1, V.S1, V.B1),
    R.get(f"{layer_name}_bond_message")(V.S2, V.C2, V.B2),
    R.get(f"{layer_name}_bond_message")(V.S1, V.S2, V.B12),
    R.special.alldiff(...))]

    template += [R.get(f"{layer_name}_thiol")(V.C) <= (R.get(carbon)(V.C), R.get(sulfur)(V.S), R.get(hydrogen)(V.H),
                                                       R.hidden.get(f"{layer_name}_single_bonded")(V.C, V.S, V.B1),
                                                       R.get(f"{layer_name}_single_bonded")(V.S, V.H, V.B2),
                                                       R.get(f"{layer_name}_bond_message")(V.C, V.S, V.B1),
                                                       R.get(f"{layer_name}_bond_message")(V.S, V.H, V.B2),
                                                       R.special.alldiff(...))]

    template += [R.get(f"{layer_name}_sulfuric_groups")(V.X) <= R.get(f"{layer_name}_isothiocyanate")(V.X)[param_size]]
    template += [R.get(f"{layer_name}_sulfuric_groups")(V.X) <= R.get(f"{layer_name}_thiocyanate")(V.X)[param_size]]
    template += [R.get(f"{layer_name}_sulfuric_groups")(V.X) <= R.get(f"{layer_name}_thiol")(V.X)[param_size]]
    template += [R.get(f"{layer_name}_sulfuric_groups")(V.X) <= R.get(f"{layer_name}_sulfide")(V.X, V.Y)[param_size]]
    template += [R.get(f"{layer_name}_sulfuric_groups")(V.X) <= R.get(f"{layer_name}_disulfide")(V.X, V.Y)[param_size]]

    return template


def get_relaxations(layer_name, param_size, connection, carbon):
    template = []

    template += [R.get(f"{layer_name}_relaxed_aliphatic_bonded")(V.X, V.Y) <= (
        R.get(f"{layer_name}_relaxed_aliphatic_bonded")(V.X, V.Y, V.B))]
    template += [R.get(f"{layer_name}_relaxed_aliphatic_bonded")(V.X, V.Y, V.B) <= (
        R.get(connection)(V.X, V.Y, V.B), R.get(f"{layer_name}_aliphatic_bond")(V.B),
        R.get(f"{layer_name}_bond_message")(V.X, V.Y, V.B)[param_size])]

    template += [R.get(f"{layer_name}_relaxed_aromatic_bonded")(V.X, V.Y) <= (
        R.get(f"{layer_name}_relaxed_aromatic_bonded")(V.X, V.Y, V.B))]
    template += [R.get(f"{layer_name}_relaxed_aromatic_bonded")(V.X, V.Y, V.B) <= (
        R.get(connection)(V.X, V.Y, V.B), R.get(f"{layer_name}_aromatic_bond")(V.B),
        R.get(f"{layer_name}_bond_message")(V.X, V.Y, V.B)[param_size])]

    template += [R.get(f"{layer_name}_relaxed_carbonyl_group")(V.X, V.Y) <= (
    R.get(f"{layer_name}_key_atom")(V.X)[param_size], R.get(f"{layer_name}_key_atom")(V.Y)[param_size],
    R.get(f"{layer_name}_relaxed_aliphatic_bonded")(V.X, V.Y))]

    template += [R.get(f"{layer_name}_relaxed_carbonyl_group")(V.C, V.O, V.R1, V.R2) <= (
    R.get(f"{layer_name}_relaxed_carbonyl_group")(V.C, V.O),
    R.get(f"{layer_name}_relaxed_aliphatic_bonded")(V.C, V.R1),
    R.get(f"{layer_name}_relaxed_aliphatic_bonded")(V.C, V.R2),
    R.special.alldiff(...))]

    template += [
        R.get(f"{layer_name}_relaxed_benzene_ring")(V.A) <= R.get(f"{layer_name}_relaxed_benzene_ring")(V.A, V.B)]
    template += [
        R.get(f"{layer_name}_relaxed_benzene_ring")(V.A, V.B) <= R.get(f"{layer_name}_relaxed_benzene_ring")(V.A, V.B,
                                                                                                             V.C, V.D,
                                                                                                             V.E,
                                                                                                             V.F)]

    template += [R.get(f"{layer_name}_relaxed_benzene_ring")(V.A, V.B, V.C, V.D, V.E, V.F) <= (
    R.get(f"{layer_name}_relaxed_aromatic_bonded")(V.A, V.B),
    R.get(f"{layer_name}_relaxed_aromatic_bonded")(V.B, V.C),
    R.get(f"{layer_name}_relaxed_aromatic_bonded")(V.C, V.D),
    R.get(f"{layer_name}_relaxed_aromatic_bonded")(V.D, V.E),
    R.get(f"{layer_name}_relaxed_aromatic_bonded")(V.E, V.F),
    R.get(f"{layer_name}_relaxed_aromatic_bonded")(V.F, V.A),

    R.special.alldiff(...))]

    template += [
        R.get(f"{layer_name}_relaxed_functional_group")(V.X) <= R.get(f"{layer_name}_relaxed_carbonyl_group")(V.X, V.Y)[
            param_size]]
    template += [
        R.get(f"{layer_name}_relaxed_functional_group")(V.Y) <= R.get(f"{layer_name}_relaxed_carbonyl_group")(V.X, V.Y)[
            param_size]]
    template += [R.get(f"{layer_name}_relaxed_functional_group")(V.X) <=
                 R.get(f"{layer_name}_relaxed_aliphatic_bonded")(V.X, V.Y)[param_size]]
    template += [R.get(f"{layer_name}_relaxed_functional_group")(V.X) <=
                 R.get(f"{layer_name}_relaxed_aromatic_bonded")(V.X, V.Y)[param_size]]
    template += [
        R.get(f"{layer_name}_relaxed_functional_group")(V.X) <= R.get(f"{layer_name}_relaxed_benzene_ring")(V.X)[
            param_size]]

    template += [R.get(f"{layer_name}_potential_group")(V.C) <= (
    R.get(f"{layer_name}_relaxed_aliphatic_bonded")(V.C, V.X)[param_size],
    R.get(f"{layer_name}_noncarbon")(V.X)[param_size], R.get(carbon)(V.C))]

    template += [R.get(f"{layer_name}_carbonyl_derivatives")(V.C) <= (
    R.get(f"{layer_name}_relaxed_carbonyl_group")(V.C, V.O, V.R, V.O1),
    R.get(f"{layer_name}_noncarbon")(V.X)[param_size])]
    template += [R.get(f"{layer_name}_carbonyl_derivatives")(V.X) <= (
    R.get(f"{layer_name}_relaxed_carbonyl_group")(V.C, V.O, V.R, V.X),
    R.get(f"{layer_name}_noncarbon")(V.X)[param_size])]
    template += [R.get(f"{layer_name}_carbonyl_derivatives")(V.R) <= (
    R.get(f"{layer_name}_relaxed_carbonyl_group")(V.C, V.O, V.R, V.X),
    R.get(f"{layer_name}_noncarbon")(V.X)[param_size])]

    template += [
        R.get(f"{layer_name}_relaxed_functional_group")(V.X) <= R.get(f"{layer_name}_potential_group")(V.X)[param_size]]
    template += [
        R.get(f"{layer_name}_relaxed_functional_group")(V.X) <= R.get(f"{layer_name}_carbonyl_derivatives")(V.X)[
            param_size]]

    return template

