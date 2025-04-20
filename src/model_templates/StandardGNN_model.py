from neuralogic.core import R, V


def StandardGNN_model(
    model_name: str,
    layers: int,
    node_embed: str,
    edge_embed: str,
    connection: str,
    param_size: int,
    output_layer_name: str = "predict",
    **kwargs,
):

    # gnn_k(X) <=  gnn_k-1(X), gnn_k-1(Y), connection(X, Y, B), edge_embed(B)
    def get_gnn(layer_name: str, node_embed: str, param_size: tuple):
        return [
            (
                R.get(layer_name)(V.X)
                <= (
                    R.get(node_embed)(V.X)[param_size],
                    R.get(node_embed)(V.Y)[param_size],
                    R.get(connection)(
                        V.X, V.Y, V.B
                    ),  # should be first to ground faster?
                    R.get(edge_embed)(V.B),
                )
            )
        ]  # why not parametrized?

    # Match gnn_0 to input node embeddings
    template = [(R.get(f"{model_name}_gnn_0")(V.X) <= R.get(node_embed)(V.X))]

    if param_size == 1:
        template += [(R.get(output_layer_name)[param_size,] <= R.get(f"{model_name}_gnn")(V.X))]
        param_size = (param_size,)
    else:
        template += [(R.get(output_layer_name)[1, param_size] <= R.get(f"{model_name}_gnn")(V.X))]
        param_size = (param_size, param_size)

    # Build layers
    for i in range(layers):
        template += get_gnn(
            f"{model_name}_gnn_{i + 1}",
            f"{model_name}_gnn_{i}",
            param_size,
        )

    # Write output layer and prediction
    template += [
        (R.get(f"{model_name}_gnn")(V.X) <= R.get(f"{model_name}_gnn_{layers}")(V.X))
    ]

    return template