from neuralogic.core import R, V


def SGN_model(
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
    max_depth = kwargs.get('max_depth', 3)

    # Creating SGN sets up to max depth
    def get_sgn(layer_name: str, node_embed: str, param_size: tuple):
        template = []

        # First order SGN aggregates node embeddings sharing an edge, ensuring higher orders are connected
        template += [
            R.get(f"{layer_name}_order_1")(V.X, V.Y)
            <= (
                R.get(connection)(V.X, V.Y, V.B),
                R.get(edge_embed)(V.B)[param_size],
                R.get(node_embed)(V.X)[param_size],
                R.get(node_embed)(V.Y)[param_size],
            )
        ]

        # Constructing orders
        for i in range(2, max_depth + 1):

            # head = R.get(f"{layer_name}_order_{i}")(f"X{j}" for j in range(i+1))

            # body = [R.get(f"{layer_name}_order_{i-1}")(f"X{j}" for j in range(i))[param_size],
            #         R.get(f"{layer_name}_order_{i-1}")(f"X{j+1}" for j in range(i))[param_size],
            #         R.special.alldiff(...)]

            # template += [head <= body]
            template += [
                R.get(f"{layer_name}_order_{i}")(V.X, V.Y)
                <= (
                    R.get(f"{layer_name}_order_{i-1}")(V.X, V.Y)[param_size],
                    R.get(f"{layer_name}_order_{i-1}")(V.Y, V.Z)[param_size],
                )
            ]

        # Extracting Subgraph messages to nodes
        # template += [R.get(layer_name)(V.X0) <= (R.get(f"{layer_name}_order_{max_depth}")(f"X{j}" for j in range(max_depth+1)))]
        # template += [R.get(layer_name)(f"X{k}") <= (R.get(f"{layer_name}_order_{max_depth}")(f"X{j}" for j in range(max_depth+1))) for k in range(max_depth+1)]
        template += [
            R.get(layer_name)(V.X)
            <= (R.get(f"{layer_name}_order_{max_depth}")(V.X, V.Y)),
            R.get(layer_name)(V.Y)
            <= (R.get(f"{layer_name}_order_{max_depth}")(V.X, V.Y)),
        ]
        return template

    template = []

    # Match sgn_0 to input node embeddings
    template = [(R.get(f"{model_name}_sgn_0")(V.X) <= (R.get(node_embed)(V.X)))]

    if param_size == 1:
        template += [(R.get(output_layer_name)[param_size, ] <= R.get(f"{model_name}_sgn")(V.X))]
        param_size = (param_size,)
    else:
        template += [(R.get(output_layer_name)[1, param_size] <= R.get(f"{model_name}_sgn")(V.X))]
        param_size = (param_size, param_size)

    # Build layers
    for i in range(layers):
        template += get_sgn(
            f"{model_name}_sgn_{i + 1}",
            f"{model_name}_sgn_{i}",
            param_size,
        )

    # Write output layer and prediction
    template += [
        (R.get(f"{model_name}_sgn")(V.X) <= R.get(f"{model_name}_sgn_{layers}")(V.X))
    ]
    

    return template
