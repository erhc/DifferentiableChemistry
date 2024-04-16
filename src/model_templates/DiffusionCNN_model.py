from neuralogic.core import R, V, Aggregation, Transformation

def DiffusionCNN_model(model_name: str, layers: int, node_embed: str, edge_embed: str, connection: str, param_size: int, **kwargs):
    
    # Extracting additional information
    max_depth = kwargs.get('max_depth', 1)

    # Defining a path between nodes to a max depth
    def get_path(layer_name: str, param_size: tuple):
        template = []
        template += [(R.get(layer_name)(V.X, V.Y, 0) <= (R.get(edge_embed)(V.B)[param_size], 
                                                        R.get(connection)(V.X, V.Y, V.B)))]
        
        template += [(R.get(layer_name)(V.X, V.Y, V.T) <= (R.get(edge_embed)(V.B)[param_size],
                                                          R.get(layer_name)(V.Z, V.Y, V.T1)[param_size],
                                                          R.get(connection)(V.X, V.Z, V.B),
                                                          R.special.next(V.T1, V.T)))]
        
        # Defining constants for keeping track
        for i in range(max_depth):
            template += [(R._next(i, i + 1))]

        template += [(R.get(layer_name)(V.X, V.Y) <= (R.get(layer_name)(V.X, V.Y, max_depth)))]

        return template

    # Creating a Diffusion CNN layer
    def get_diffusion(layer_name: str, node_embed: str, param_size: tuple):
        template = []
        template += [(R.get(layer_name + "_Z")(V.X) <= (R.get(f"{model_name}_diff_path")(V.X, V.Y), 
                                                        R.get(node_embed)(V.Y)[param_size])) | [Aggregation.SUM]]
        template += [(R.get(layer_name + "_Z")(V.X) <= R.get(node_embed)(V.X)[param_size])]
        template += [(R.get(layer_name)(V.X) <= (R.get(layer_name + "_Z")(V.X))) | [Transformation.SIGMOID, Aggregation.SUM]]

        return template

    # Match diff_0 to input node embeddings and create the paths
    template = [(R.get(f"{model_name}_diff_0")(V.X) <= (R.get(node_embed)(V.X)))]
    template += get_path(f"{model_name}_diff_path", (param_size, param_size))

    # Build layers
    for i in range(layers):
        template += get_diffusion(f"{model_name}_diff_{i + 1}", f"{model_name}_diff_{i}", (param_size, param_size))

    # Write output layer and prediction
    template += [(R.get(f"{model_name}_diff")(V.X) <= R.get(f"{model_name}_diff_{layers}")(V.X))]
    template += [(R.predict[1, param_size] <= R.get(f"{model_name}_diff")(V.X))]

    return template