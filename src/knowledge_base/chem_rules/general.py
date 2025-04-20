from neuralogic.core import R, V


def get_general(
    layer_name: str,
    node_embed: str,
    edge_embed: str,
    connection: str,
    param_size: tuple,
    single_bond: str,
    double_bond: str,
    triple_bond: str,
    aromatic_bond: str,
    hydrogen: str,
    carbon: str,
    oxygen: str,
):

    template = []

    # Aggregating bond messages
    template += [
        R.get(f"{layer_name}_bond_message")(V.X, V.Y, V.B)
        <= (
            R.get(node_embed)(V.X)[param_size],
            R.get(node_embed)(V.Y)[param_size],
            R.get(edge_embed)(V.B)[param_size],
        )
    ]

    # Defining the predicates when two atoms are single/double/... bonded to each other
    template += [
        R.hidden.get(f"{layer_name}_single_bonded")(V.X, V.Y)
        <= (R.hidden.get(f"{layer_name}_single_bonded")(V.X, V.Y, V.B))
    ]
    template += [
        R.hidden.get(f"{layer_name}_single_bonded")(V.X, V.Y, V.B)
        <= (R.get(connection)(V.X, V.Y, V.B), R.get(single_bond)(V.B))
    ]

    template += [
        R.hidden.get(f"{layer_name}_double_bonded")(V.X, V.Y)
        <= (R.hidden.get(f"{layer_name}_double_bonded")(V.X, V.Y, V.B))
    ]
    template += [
        R.hidden.get(f"{layer_name}_double_bonded")(V.X, V.Y, V.B)
        <= (R.get(connection)(V.X, V.Y, V.B), R.get(double_bond)(V.B))
    ]

    template += [
        R.hidden.get(f"{layer_name}_triple_bonded")(V.X, V.Y)
        <= (R.hidden.get(f"{layer_name}_triple_bonded")(V.Y, V.X, V.B))
    ]
    template += [
        R.hidden.get(f"{layer_name}_triple_bonded")(V.X, V.Y, V.B)
        <= (R.get(connection)(V.Y, V.X, V.B), R.get(triple_bond)(V.B))
    ]

    template += [
        R.get(f"{layer_name}_aromatic_bonded")(V.X, V.Y)
        <= (R.get(f"{layer_name}_aromatic_bonded")(V.X, V.Y, V.B))
    ]
    template += [
        R.get(f"{layer_name}_aromatic_bonded")(V.X, V.Y, V.B)
        <= (R.get(connection)(V.X, V.Y, V.B), R.get(aromatic_bond)(V.B))
    ]

    # Defining saturated carbons
    # TODO: this won't work for datasets with implicit hydrogens
    template += [
        R.get(f"{layer_name}_saturated")(V.X)
        <= (
            R.get(carbon)(V.X),
            R.hidden.get(f"{layer_name}_single_bonded")(V.X, V.Y1),
            R.hidden.get(f"{layer_name}_single_bonded")(V.X, V.Y2),
            R.hidden.get(f"{layer_name}_single_bonded")(V.X, V.Y3),
            R.hidden.get(f"{layer_name}_single_bonded")(V.X, V.Y4),
            R.special.alldiff(...),
        )
    ]

    # Defining a halogen group (R-X)
    template += [
        R.get(f"{layer_name}_halogen_group")(V.R)
        <= (
            R.get(f"{layer_name}_halogen")(V.X),
            R.hidden.get(f"{layer_name}_single_bonded")(V.X, V.R, V.B),
            R.get(f"{layer_name}_bond_message")(V.X, V.R, V.B),
        )
    ]

    # Defining hydroxyl group (O-H)
    # TODO: this won't work for datasets with implicit hydrogens
    template += [
        R.get(f"{layer_name}_hydroxyl")(V.O)
        <= (
            R.get(oxygen)(V.O),
            R.get(hydrogen)(V.H),
            R.hidden.get(f"{layer_name}_single_bonded")(V.O, V.H, V.B),
            R.get(f"{layer_name}_bond_message")(V.O, V.H, V.B),
        )
    ]

    # Defining carbonyl group (R1-C(=O)-R2)
    # TODO: this won't work for datasets with implicit hydrogens
    template += [
        R.get(f"{layer_name}_carbonyl_group")(V.C, V.O)
        <= (
            R.get(carbon)(V.C),
            R.get(oxygen)(V.O),
            R.hidden.get(f"{layer_name}_double_bonded")(V.O, V.C, V.B),
            R.get(f"{layer_name}_bond_message")(V.O, V.C, V.B),
        )
    ]
    template += [
        R.get(f"{layer_name}_carbonyl_group")(V.C, V.O, V.R1, V.R2)
        <= (
            R.get(f"{layer_name}_carbonyl_group")(V.C, V.O),
            R.hidden.get(f"{layer_name}_single_bonded")(V.C, V.R1, V.B1),
            R.hidden.get(f"{layer_name}_single_bonded")(V.C, V.R2, V.B2),
            R.get(f"{layer_name}_bond_message")(V.C, V.R1, V.B1),
            R.get(f"{layer_name}_bond_message")(V.C, V.R2, V.B2),
            R.special.alldiff(...),
        )
    ]
    template += [
        R.get(f"{layer_name}_carbonyl_group")(V.C)
        <= (R.get(f"{layer_name}_carbonyl_group")(V.C, V.O))
    ]

    # Aggregating general patterns
    template += [
        R.get(f"{layer_name}_general_groups")(V.X)
        <= R.get(f"{layer_name}_hydroxyl")(V.X)[param_size]
    ]
    template += [
        R.get(f"{layer_name}_general_groups")(V.X)
        <= R.get(f"{layer_name}_carbonyl_group")(V.X)[param_size]
    ]
    template += [
        R.get(f"{layer_name}_general_groups")(V.X)
        <= R.get(f"{layer_name}_halogen_group")(V.X)[param_size]
    ]

    return template
