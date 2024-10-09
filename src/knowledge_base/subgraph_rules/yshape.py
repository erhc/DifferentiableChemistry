from neuralogic.core import R, V


def get_y_shape(
    layer_name: str,
    node_embed: str,
    edge_embed: str,
    connection: str,
    param_size: tuple,
    double_bond: str,
):
    template = []

    # Aggregating messages in a double bond
    template += [
        R.get(f"{layer_name}_double_bond_subgraph")(V.X)
        <= (
            R.get(connection)(V.X, V.Y, V.B),
            R.get(double_bond)(V.B),
            R.get(node_embed)(V.Y)[param_size],
            R.get(edge_embed)(V.B)[param_size],
        )
    ]

    # Simple 3 neighborhood
    template += [
        R.get(f"{layer_name}_y_subgraph")(V.X1, V.X2, V.X3, V.X4)
        <= (
            R.get(connection)(V.X1, V.X2, V.B1),
            R.get(connection)(V.X1, V.X3, V.B2),
            R.get(connection)(V.X1, V.X4, V.B3),
            R.get(edge_embed)(V.B1)[param_size],
            R.get(edge_embed)(V.B2)[param_size],
            R.get(edge_embed)(V.B3)[param_size],
            R.get(node_embed)(V.X1)[param_size],
            R.get(node_embed)(V.X2)[param_size],
            R.get(node_embed)(V.X3)[param_size],
            R.get(node_embed)(V.X4)[param_size],
            R.special.alldiff(...),
        )
    ]

    # Y subgraph with a double bond
    template += [
        R.get(f"{layer_name}_y_bond")(V.X1, V.X2, V.X3, V.X4)
        <= (
            R.get(connection)(V.X1, V.X2, V.B1),
            R.get(double_bond)(V.B1),
            R.get(f"{layer_name}_y_subgraph")(V.X1, V.X2, V.X3, V.X4),
            R.special.alldiff(...),
        )
    ]

    # Two Y double bond subgraphs connected with X1 (X1-Y1(=Y2)-X2-Z1(=Z2)-X3)
    template += [
        R.get(f"{layer_name}_y_group")(V.X1, V.X2, V.X3)
        <= (
            R.get(f"{layer_name}_y_bond")(V.Y1, V.Y2, V.X1, V.X2),
            R.get(f"{layer_name}_y_bond")(V.Z1, V.Z2, V.X2, V.X3),
            R.special.alldiff(...),
        )
    ]

    # Collecting all Y patterns
    template += [
        (
            R.get(f"{layer_name}_y_bond_patterns")(V.X)
            <= R.get(f"{layer_name}_double_bond_subgraph")(V.X)[param_size]
        ),
        (
            R.get(f"{layer_name}_y_bond_patterns")(V.X1)
            <= R.get(f"{layer_name}_y_bond")(V.X1, V.X2, V.X3, V.X4)[param_size]
        ),
        (
            R.get(f"{layer_name}_y_bond_patterns")(V.X2)
            <= R.get(f"{layer_name}_y_group")(V.X1, V.X2, V.X3)[param_size]
        ),
        (
            R.get(f"{layer_name}_pattern")(V.X)
            <= R.get(f"{layer_name}_y_bond_patterns")(V.X)[param_size]
        ),
    ]

    return template
