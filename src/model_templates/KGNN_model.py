from neuralogic.core import R, V


def KGNN_model(
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
    local = kwargs.get("local", True)
    max_depth = kwargs.get("max_depth", 1)

    # Creating kGNN sets up to max depth
    def get_k_set(layer_name: str, prev_layer: str, param_size: tuple):
        # Defining the input and aggregating the output
        template = [(R.get(f"{layer_name}_0")(V.X) <= (R.get(f"{prev_layer}")(V.X)))]
        template += [
            (
                R.get(f"{layer_name}")(V.X)
                <= (R.get(f"{layer_name}_{max_depth}")(V.X, V.Y))
            )
        ]

        # Constructing kGNN from k-1GNN
        for k in range(max_depth):
            if k == 0:
                body = [
                    R.get(f"{layer_name}_{k}")(V.X)[param_size],
                    R.get(f"{layer_name}_{k}")(V.Y)[param_size],
                    R.special.alldiff(...),
                ]
            else:
                body = [
                    R.get(f"{layer_name}_{k}")(V.X, V.Z)[param_size],
                    R.get(f"{layer_name}_{k}")(V.Z, V.Y)[param_size],
                    R.special.alldiff(...),
                ]

            if local:
                body += [
                    R.get(connection)(V.X, V.Y, V.B),
                    R.get(edge_embed)(V.B)[param_size],
                ]

            template += [(R.get(f"{layer_name}_{k+1}")(V.X, V.Y) <= body)]

        return template

    # Match kGNN_0 to input node embeddings
    template = [(R.get(f"{model_name}_kgnn_0")(V.X) <= (R.get(node_embed)(V.X)))]

    if param_size == 1:
        template += [(R.get(output_layer_name)[param_size,] <= R.get(f"{model_name}_kgnn")(V.X))]
        param_size = (param_size,)
    else:
        template += [(R.get(output_layer_name)[1, param_size] <= R.get(f"{model_name}_kgnn")(V.X))]
        param_size = (param_size, param_size)

    # Build layers
    for i in range(layers):
        template += get_k_set(
            f"{model_name}_kgnn_{i + 1}",
            f"{model_name}_kgnn_{i}",
            param_size,
        )

    # Write output layer and prediction
    template += [
        (R.get(f"{model_name}_kgnn")(V.X) <= R.get(f"{model_name}_kgnn_{layers}")(V.X))
    ]

    return template