from neuralogic.core import R, V


def RGCN_model(
    model_name: str,
    layers: int,
    node_embed: str,
    edge_embed: str,
    connection: str,
    param_size: int,
    output_layer_name: str = "predict",
    **kwargs,
):

    # Extracting additional information
    edge_types = kwargs.get("edge_types", None)

    # rgcn_k(X) <=  rgcn_k-1(X), rgcn_k-1(Y), connection(X, Y, B), edge_embed(B), edge_type(B) for all edge types
    def get_rgcn(layer_name: str, node_embed: str, param_size: tuple):
        return [
            (
                R.get(layer_name)(V.X)
                <= (
                    R.get(node_embed)(V.X)[param_size],
                    R.get(node_embed)(V.Y)[param_size],
                    R.get(connection)(V.X, V.Y, V.B),
                    # R.get(edge_embed)(V.B), # maybe doesnt make sense to have this, as the information is encoded below
                    R.get(t)(V.B),
                )
            )
            for t in edge_types
        ]

    # Match rgcn_0 to input node embeddings
    template = [(R.get(f"{model_name}_rgcn_0")(V.X) <= R.get(node_embed)(V.X))]

    if param_size == 1:
        template += [(R.get(output_layer_name)[param_size,] <= R.get(f"{model_name}_rgcn")(V.X))]
        param_size = (param_size,)
    else:
        template += [(R.get(output_layer_name)[1, param_size] <= R.get(f"{model_name}_rgcn")(V.X))]
        param_size = (param_size, param_size)

    # Build layers
    for i in range(layers):
        template += get_rgcn(
            f"{model_name}_rgcn_{i + 1}",
            f"{model_name}_rgcn_{i}",
            param_size,
        )

    # Write output layer and prediction
    template += [
        (R.get(f"{model_name}_rgcn")(V.X) <= R.get(f"{model_name}_rgcn_{layers}")(V.X))
    ]

    return template
