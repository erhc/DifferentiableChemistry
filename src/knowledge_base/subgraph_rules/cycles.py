from neuralogic.core import R, V


def get_cycles(
    layer_name: str,
    node_embed: str,
    edge_embed: str,
    connection: str,
    param_size: tuple,
    max_cycle_size: int,
    min_cycle_size: int = 3,
):

    def get_cycle(cycle_size):
        # Cycles are paths from a node to itself, with every node on the path being unique
        # cannot use path predicate here, because all the edges are undirected
        body = [
            R.get(connection)(f"X{i}", f"X{(i + 1) % cycle_size}", f"B{i}")
            for i in range(cycle_size)
        ]
        body.extend(R.get(node_embed)(f"X{i}")[param_size] for i in range(cycle_size))
        body.extend(R.get(edge_embed)(f"B{i}")[param_size] for i in range(cycle_size))
        body.append(
            R.special.alldiff(f"X{i}" for i in range(cycle_size))
        )  # X0....Xmax are different
        body.append(
            R.special._in((V.X,) + tuple(f"X{i}" for i in range(1, cycle_size)))
        )  # X and X0 are in the cycle

        return [R.get(f"{layer_name}_cycle")(V.X, V.X0) <= body]

    template = []

    # Generate cycles of varying sizes
    for i in range(min_cycle_size, max_cycle_size):
        template.extend(get_cycle(i))

    # Aggregating to subgraph patterns
    template.append(
        R.get(f"{layer_name}_cycle")(V.X)
        <= R.get(f"{layer_name}_cycle")(V.X, V.X0)[param_size]
    )
    template.append(
        R.get(f"{layer_name}_pattern")(V.X)
        <= R.get(f"{layer_name}_cycle")(V.X)[param_size]
    )

    return template
