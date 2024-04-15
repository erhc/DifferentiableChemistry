from neuralogic.core import R, V

def get_circular(layer_name: str, node_embed: str, edge_embed: str, connection: str, param_size: tuple, carbon: str, single_bond: str, double_bond: str):
    template = []

    # Defining Carbon negation helper predicate and heterocycles
    template.append(R.get(f"{layer_name}_n_c")(V.X) <= R.get(carbon)(V.X))
    template.append(R.get(f"{layer_name}_heterocycle")(V.X) <= (R.get(carbon)(V.C),  
                                                                ~R.hidden.get(f"{layer_name}_n_c")(V.X),
                                                                R.get(f"{layer_name}_cycle")(V.X, V.C)[param_size]))

    # Defining "brick" substructure (X-Y1=Y2-Y3=X)
    template += [R.get(f"{layer_name}_brick")(V.X) <= (
        R.get(connection)(V.X, V.Y1, V.B1),
        R.get(connection)(V.Y1, V.Y2, V.B2),
        R.get(connection)(V.Y2, V.Y3, V.B3),
        R.get(connection)(V.Y3, V.X, V.B4),

        R.get(single_bond)(V.B1),
        R.get(double_bond)(V.B2),
        R.get(single_bond)(V.B3),
        R.get(double_bond)(V.B4),

        R.get(node_embed)(V.Y1)[param_size],
        R.get(node_embed)(V.Y2)[param_size],
        R.get(node_embed)(V.Y3)[param_size],
        R.get(node_embed)(V.X)[param_size],
        
        R.get(edge_embed)(V.B1)[param_size],
        R.get(edge_embed)(V.B2)[param_size],
        R.get(edge_embed)(V.B3)[param_size],
        R.get(edge_embed)(V.B4)[param_size],

        R.special.alldiff(...))]

    # Aggregating into a common predicate
    template += [R.get(f"{layer_name}_circular")(V.X) <= R.get(subgraph)(V.X)[param_size] for subgraph in
                 [f"{layer_name}_brick", f"{layer_name}_heterocycle"]]
    template += [R.get(f"{layer_name}_pattern")(V.X) <= R.get(f"{layer_name}_circular")(V.X)[param_size]]

    return template