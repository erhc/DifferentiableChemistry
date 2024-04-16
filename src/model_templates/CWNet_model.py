from neuralogic.core import R, V

def CWNet_model(model_name: str, layers: int, node_embed: str, edge_embed: str, connection: str, param_size: int, **kwargs):
    # Extracting additional information
    max_ring_size = kwargs.get('max_ring_size', 7)

    # Aggregating bond features
    def bond_features(layer_name: str, prev_layer: str, param_size: tuple):
        template = []
        # atoms aggregate to bonds, bonds to rings
        template += [R.get(layer_name + "_edge")(V.B) <= (R.get(connection)(V.X, V.Y, V.B),
                                                                   R.get(f"{prev_layer}_node")(V.X)[param_size],
                                                                   R.get(f"{prev_layer}_node")(V.Y)[param_size])]

        # bonds in same cycle
        def get_bond_cycle(n):
            body = [R.get(connection)(f"X{i}", f"X{(i + 1) % n}", f"B{i}") for i in range(n)]
            body.extend(R.get(f"{prev_layer}_edge")(f"B{i}")[param_size] for i in range(n))
            body.append(R.special.alldiff(...))

            return [R.get(layer_name + "_edge")(V.B0) <= body]

        for i in range(3, max_ring_size):
            template += get_bond_cycle(i)

        return template

    # Aggregating node features
    def node_features(layer_name: str, prev_layer: str, param_size: tuple):
        template = []

        # atoms sharing a bond share messages, bonds in the same ring
        template += [R.get(layer_name + "_node")(V.X) <= (R.get(connection)(V.X, V.Y, V.B),
                                                                 R.get(f"{prev_layer}_node")(V.Y)[param_size],
                                                                 R.get(f"{prev_layer}_edge")(V.B)[param_size])]
        return template

    # Constructing a layer of CW net, aggregating node and edge features to layer output
    def get_cw(layer_name: str, prev_layer: str, param_size: tuple):
        template = []
        template += bond_features(layer_name, prev_layer, param_size)
        template += node_features(layer_name, prev_layer, param_size)

        template += [R.get(layer_name)(V.X) <= (R.get(layer_name + "_node")(V.X)[param_size])]
        template += [R.get(layer_name)(V.X) <= (R.get(layer_name + "_edge")(V.X)[param_size])]

        return template

    # Match cw_0 to input node and edge embeddings
    template = [(R.get(f"{model_name}_cw_0_node")(V.X) <= (R.get(node_embed)(V.X)))]
    template += [(R.get(f"{model_name}_cw_0_edge")(V.X) <= (R.get(edge_embed)(V.X)))]
    
    # Build layers
    for i in range(layers):
        template += get_cw(f"{model_name}_cw_{i + 1}", f"{model_name}_cw_{i}", (param_size, param_size))

    # Write output layer and prediction
    template += [(R.get(f"{model_name}_cw")(V.X) <= R.get(f"{model_name}_cw_{layers}")(V.X))]
    template += [(R.predict[1, param_size] <= R.get(f"{model_name}_cw")(V.X))]

    return template
