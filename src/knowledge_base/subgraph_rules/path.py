from neuralogic.core import R, V


def get_path(
    layer_name: str,
    node_embed: str,
    edge_embed: str,
    connection: str,
    param_size: tuple,
    max_depth: int,
):
    template = []

    # Defining constants for keeping track
    for i in range(max_depth):
        template += [(R._next(i, i + 1))]

    # Base case
    template += [
        R.get(f"{layer_name}_path")(V.X, V.Y, 0)
        <= (
            R.get(connection)(V.X, V.Y, V.B),
            R.get(edge_embed)(V.B)[param_size],
            R.get(node_embed)(V.Y)[param_size],
        )
    ]
    # Recursive calls
    template += [
        R.get(f"{layer_name}_path")(V.X, V.Y, V.T)
        <= (
            R.special.next(V.T1, V.T),
            R.get(connection)(V.X, V.Z, V.B),
            R.get(f"{layer_name}_path")(V.Z, V.Y, V.T1)[param_size],
            R.get(edge_embed)(V.B)[param_size],
            R.get(node_embed)(V.X)[param_size],
        )
    ]

    # If there is a path from X to Y less than or equal to max_depth
    template += [
        (
            R.get(f"{layer_name}_path")(V.X, V.Y)
            <= (R.get(f"{layer_name}_path")(V.X, V.Y, max_depth))
        )
    ]

    # Aggregating for X
    template += [
        R.get(f"{layer_name}_pattern")(V.X)
        <= R.get(f"{layer_name}_path")(V.X, V.Y)[param_size]
    ]

    return template
