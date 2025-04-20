from neuralogic.core import R, V


def get_collective(
    layer_name: str,
    node_embed: str,
    edge_embed: str,
    connection: str,
    param_size: tuple,
    carbon: str,
    aliphatic_bond: str,
    max_depth: int,
):
    template = []

    # Defining when two atoms are NOT in a same cycle
    template += [
        R.get(f"{layer_name}_n_cycle")(V.X, V.Y)
        <= R.get(f"{layer_name}_cycle")(V.X, V.Y)
    ]

    # Bridge atom between two cycles
    template += [
        R.get(f"{layer_name}_bridge")(V.X)
        <= (
            R.get(connection)(V.X, V.Y, V.B1),
            R.get(connection)(V.X, V.Z, V.B2),
            # ~R.get(f"{layer_name}_n_cycle")(V.X, V.X1),
            ~R.get(f"{layer_name}_n_cycle")(V.Y, V.Z),
            R.get(f"{layer_name}_cycle")(V.Y, V.Y1)[param_size],
            R.get(f"{layer_name}_cycle")(V.Z, V.Z1)[param_size],
            R.get(edge_embed)(V.B1)[param_size],
            R.get(edge_embed)(V.B2)[param_size],
            R.get(node_embed)(V.X)[param_size],
            R.special.alldiff(...),
        )
    ]

    # Shared atom between two cycles
    template += [
        R.get(f"{layer_name}_shared_atom")(V.X)
        <= (
            R.get(connection)(V.X, V.Y, V.B1),
            R.get(connection)(V.X, V.Z, V.B2),
            R.get(f"{layer_name}_cycle")(V.X, V.Y)[param_size],
            R.get(f"{layer_name}_cycle")(V.X, V.Z)[param_size],
            ~R.get(f"{layer_name}_n_cycle")(V.Y, V.Z),
            R.get(edge_embed)(V.B1)[param_size],
            R.get(edge_embed)(V.B2)[param_size],
            R.get(node_embed)(V.X)[param_size],
            R.special.alldiff(...),
        )
    ]

    # Chain of carbons connected by a single bond
    template += [
        R.get(f"{layer_name}_aliphatic_chain")(V.X, V.Y)
        <= R.get(f"{layer_name}_aliphatic_chain")(V.X, V.Y, max_depth)
    ]
    template += [
        R.get(f"{layer_name}_aliphatic_chain")(V.X)
        <= R.get(f"{layer_name}_aliphatic_chain")(V.X, V.Y)
    ]
    template += [
        R.get(f"{layer_name}_aliphatic_chain")(V.X, V.Y, 0)
        <= (
            R.get(connection)(V.X, V.Z, V.B),
            R.get(carbon)(V.X),
            R.get(carbon)(V.Y),
            R.get(aliphatic_bond)(V.B),
            R.get(edge_embed)(V.B)[param_size],
            R.get(node_embed)(V.Y)[param_size],
        )
    ]

    template += [
        R.get(f"{layer_name}_aliphatic_chain")(V.X, V.Y, V.T)
        <= (
            R.get(carbon)(V.X),
            R.special.next(V.T1, V.T),
            R.get(connection)(V.X, V.Z, V.B),
            R.get(f"{layer_name}_aliphatic_chain")(V.Z, V.Y, V.T1)[param_size],
            R.get(aliphatic_bond)(V.B)[param_size],
            R.get(edge_embed)(V.B)[param_size],
            R.get(node_embed)(V.X)[param_size],
        )
    ]

    template += [
        R.get(f"{layer_name}_collective_pattern")(V.X)
        <= R.get(f"{layer_name}_aliphatic_chain")(V.X)[param_size]
    ]
    template += [
        R.get(f"{layer_name}_collective_pattern")(V.X)
        <= R.get(f"{layer_name}_shared_atom")(V.X)[param_size]
    ]
    template += [
        R.get(f"{layer_name}_collective_pattern")(V.X)
        <= R.get(f"{layer_name}_bridge")(V.X)[param_size]
    ]
    template += [
        R.get(f"{layer_name}_pattern")(V.X)
        <= R.get(f"{layer_name}_collective_pattern")(V.X)[param_size]
    ]

    template += [
        R.get(f"{layer_name}_subgraph_pattern")(V.X)
        <= (
            R.get(f"{layer_name}_pattern")(V.X)[param_size],
            R.get(f"{layer_name}_pattern")(V.Y)[param_size],
            R.get(f"{layer_name}_path")(V.X, V.Y),
        )
    ]
    return template
