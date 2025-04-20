from neuralogic.core import R, V


def get_nbhoods(
    layer_name: str,
    node_embed: str,
    edge_embed: str,
    connection: str,
    param_size: tuple,
    carbon: str,
    atom_type: str,
    nbh_min_size: int = 3,
    nbh_max_size: int = 5,
):
    template = []
    nbhoods = [f"{layer_name}_chiral_center"]

    # n-node neighborhoods
    for n in range(nbh_min_size, nbh_max_size + 1):
        connections = [(V.X, f"X{i}", f"B{i}") for i in range(n)]
        node_embeddings = [R.get(node_embed)(f"X{i}")[param_size] for i in range(n)]
        edge_embeddings = [R.get(edge_embed)(f"B{i}")[param_size] for i in range(n)]
        nbhood_body = (
            [R.get(connection)(*conn) for conn in connections]
            + node_embeddings
            + edge_embeddings
            + [R.special.alldiff(...)]
        )
        template.append(R.get(f"{layer_name}_{n}_nbhood")(V.X) <= nbhood_body)
        nbhoods += [f"{layer_name}_{n}_nbhood"]

    # Chiral center is a carbon atom surrounded by
    chiral_connections = [(V.C, f"X{i}", f"B{i}") for i in range(4)]
    chiral_edge_embeddings = [R.get(edge_embed)(f"B{i}")[param_size] for i in range(4)]
    chiral_node_embeddings = [
        R.get(atom_type)(f"X{i}")[param_size] for i in range(4)
    ] + [R.get(node_embed)(f"X{i}")[param_size] for i in range(4)]
    chiral_center_body = (
        [R.get(carbon)(V.C)]
        + [R.get(connection)(*conn) for conn in chiral_connections]
        + chiral_edge_embeddings
        + chiral_node_embeddings
        + [R.special.alldiff(...)]
    )
    template.append(R.get(f"{layer_name}_chiral_center")(V.C) <= chiral_center_body)

    # Neighborhood pattern aggregation
    for nbhood in nbhoods:
        template.append(
            R.get(f"{layer_name}_nbhood")(V.X) <= R.get(nbhood)(V.X)[param_size]
        )

    template.append(
        R.get(f"{layer_name}_pattern")(V.X)
        <= R.get(f"{layer_name}_nbhood")(V.X)[param_size]
    )

    return template
