from neuralogic.core import R, V

def StandardGNN_model(model_name: str, layers: int, node_embed: str, edge_embed: str, connection: str, param_size: int, **kwargs):
    
    # GNN_k(X) <=  GNN_k-1(X), GNN_k-1(Y), connection(X, Y, B), edge_embed(B)
    def get_gnn(layer_name: str, node_embed: str, param_size: tuple):
        return [(R.get(layer_name)(V.X) <= (R.get(node_embed)(V.X)[param_size],
                                            R.get(node_embed)(V.Y)[param_size],
                                            R.get(connection)(V.X, V.Y, V.B), #should be first to ground faster?
                                            R.get(edge_embed)(V.B)))] # why not parametrized?

    # Match GNN_0 to input node embeddings
    template = [(R.get(f"{model_name}_GNN_0")(V.X) <= R.get(node_embed)(V.X))]

    # Build layers
    for i in range(layers):
        template += get_gnn(f"{model_name}_GNN_{i + 1}", f"{model_name}_GNN_{i}", (param_size, param_size))

    # Write output layer and prediction
    template += [(R.get(f"{model_name}_GNN")(V.X) <= R.get(f"{model_name}_GNN_{layers}")(V.X))]
    template += [(R.predict[1, param_size] <= R.get(f"{model_name}_GNN")(V.X))]

    return template