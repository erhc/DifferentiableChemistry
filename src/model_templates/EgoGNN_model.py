from neuralogic.core import R, V


def EgoGNN_model(
    model_name: str,
    layers: int,
    node_embed: str,
    edge_embed: str,
    connection: str,
    param_size: int,
    output_layer_name: str = "predict",
    **kwargs,
):

    def get_ego(layer_name: str, node_embed: str, param_size: tuple):
        template = []
        template += [
            R.get(layer_name + "_multigraph")(V.X)
            <= (
                R.get(connection)(V.X, V.Y, V.B),
                R.get(edge_embed)(V.B)[param_size],
                R.get(node_embed)(V.Y)[param_size],
            )
        ]

        template += [
            R.get(layer_name)(V.X)
            <= (
                R.get(connection)(V.X, V.Y, V.B),
                R.get(layer_name + "_multigraph")(V.Y)[param_size],
            )
        ]
        return template

    # Match ego_0 to input node embeddings
    template = [(R.get(f"{model_name}_ego_0")(V.X) <= (R.get(node_embed)(V.X)))]

    if param_size == 1:
        template += [(R.get(output_layer_name)[param_size,] <= R.get(f"{model_name}_ego")(V.X))]
        param_size = (param_size,)
    else:
        template += [(R.get(output_layer_name)[1, param_size] <= R.get(f"{model_name}_ego")(V.X))]
        param_size = (param_size, param_size)

    # Build layers
    for i in range(layers):
        template += get_ego(
            f"{model_name}_ego_{i + 1}",
            f"{model_name}_ego_{i}",
            param_size,
        )

    # Write output layer and prediction
    template += [
        (R.get(f"{model_name}_ego")(V.X) <= R.get(f"{model_name}_ego_{layers}")(V.X))
    ]

    return template
