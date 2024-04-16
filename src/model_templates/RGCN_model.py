from neuralogic.core import R, V

def RGCN_model(model_name: str, layers: int, node_embed: str, edge_embed: str, connection: str, param_size: int, **kwargs):
    
    # Extracting additional information
    edge_types = kwargs.get("edge_types", None)
    
    # RGCN_k(X) <=  RGCN_k-1(X), RGCN_k-1(Y), connection(X, Y, B), edge_embed(B), edge_type(B) for all edge types
    def get_rgcn(layer_name: str, node_embed: str, param_size: tuple):
        return [(R.get(layer_name)(V.X) <= (R.get(node_embed)(V.X)[param_size],
                                           R.get(node_embed)(V.Y)[param_size],
                                           R.get(connection)(V.X, V.Y, V.B),
                                           # R.get(edge_embed)(V.B), # maybe doesnt make sense to have this, as the information is encoded below
                                           R.get(t)(V.B)[param_size])) for t in edge_types] # why this is parametrized and not edge embedding?
    # Match RGCN_0 to input node embeddings
    template = [(R.get(f"{model_name}_RGCN_0")(V.X) <= R.get(node_embed)(V.X))]
    
    # Build layers
    for i in range(layers):
        template += get_rgcn(f"{model_name}_RGCN_{i + 1}", f"{model_name}_RGCN_{i}", (param_size, param_size))

    # Write output layer and prediction
    template += [(R.get(f"{model_name}_RGCN")(V.X) <= R.get(f"{model_name}_RGCN_{layers}")(V.X))]
    template += [(R.predict[1, param_size] <= R.get(f"{model_name}_RGCN")(V.X))]

    return template