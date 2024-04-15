from neuralogic.core import R, V


def get_oxy(layer_name: str, param_size: tuple, carbon: str, oxygen: str, hydrogen: str):
    template = []
    
    # Defining an alcoholic group (R-O-H)
    #TODO: this won't work for datasets with implicit hydrogens
    template += [R.get(f"{layer_name}_alcoholic")(V.C) <= (
                 R.get(f"{layer_name}_saturated")(V.C),
                 R.get(f"{layer_name}_hydroxyl")(V.O),
                 R.hidden.get(f"{layer_name}_single_bonded")(V.C, V.O, V.B1),
                 R.get(f"{layer_name}_bond_message")(V.C, V.O, V.B1))]

    # Defining a ketone (R1-C(=O)-R2)
    template += [R.get(f"{layer_name}_ketone")(V.C) <= (
                 R.get(f"{layer_name}_carbonyl_group")(V.C, V.O, V.R1, V.R2),
                 R.get(carbon)(V.R1), 
                 R.get(carbon)(V.R2))]
    
    # Defining an aldehyde (R-C(=O)-H)
    #TODO: this won't work for datasets with implicit hydrogens
    template += [R.get(f"{layer_name}_aldehyde")(V.C) <= (
                 R.get(f"{layer_name}_carbonyl_group")(V.C, V.O, V.R, V.H),
                 R.get(carbon)(V.R),
                 R.get(hydrogen)(V.H))]

    # Defining acyl halide group (R-C(=O)-X)
    template += [R.get(f"{layer_name}_acyl_halide")(V.C) <= (
                 R.get(f"{layer_name}_carbonyl_group")(V.C, V.O, V.R, V.X),
                 R.get(carbon)(V.R),
                 R.get(f"{layer_name}_halogen")(V.X))]

    # Defining carboxylic acid (R-C(=O)-OH)
    template += [R.get(f"{layer_name}_carboxylic_acid")(V.C) <= (
                 R.get(f"{layer_name}_carbonyl_group")(V.C, V.O, V.R, V.O1),
                 R.get(carbon)(V.R), 
                 R.get(f"{layer_name}_hydroxyl")(V.O1))]

    # Defining carboxylic acid anhydride (R1-C(=O)-O-C(=O)-R2)
    # TODO: should this be propagated on C or on R?
    template += [R.get(f"{layer_name}_carboxylic_acid_anhydride")(V.C1, V.C2) <= (
                 R.get(f"{layer_name}_carbonyl_group")(V.C1, V.X1, V.O12, V.R1),
                 R.get(oxygen)(V.O12),
                 R.get(f"{layer_name}_carbonyl_group")(V.C2, V.X2, V.O12, V.R2),
                 R.special.alldiff(V.C1, V.C2))]

    # Defining an ester group (R1-C(=O)-O-R2)
    # TODO: will fail for HC(=O)-O-CH
    template += [R.get(f"{layer_name}_ester")(V.X) <= R.get(f"{layer_name}_ester")(V.X, V.Y)]
    template += [R.get(f"{layer_name}_ester")(V.R1, V.R2) <= (
                 R.get(f"{layer_name}_carbonyl_group")(V.C, V.X, V.R1, V.O),
                 R.get(carbon)(V.R1), 
                 R.get(oxygen)(V.O),
                 R.get(carbon)(V.R2), 
                 R.hidden.get(f"{layer_name}_single_bonded")(V.O, V.R2, V.B),
                 R.get(f"{layer_name}_bond_message")(V.O, V.R2, V.B))]

    # Defining carbonate ester group (R1-O-C(=O)-O-R2)
    template += [R.get(f"{layer_name}_carbonate_ester")(V.X) <= R.get(f"{layer_name}_carbonate_ester")(V.X, V.Y)]
    template += [R.get(f"{layer_name}_carbonate_ester")(V.R1, V.R2) <= (
                 R.get(f"{layer_name}_carbonyl_group")(V.C, V.X, V.O1, V.O2),
                 R.get(oxygen)(V.O1), 
                 R.get(oxygen)(V.O2),
                 R.get(carbon)(V.R1), 
                 R.hidden.get(f"{layer_name}_single_bonded")(V.R1, V.O1, V.B1),
                 R.get(f"{layer_name}_bond_message")(V.O1, V.R1, V.B1),
                 R.get(carbon)(V.R2), 
                 R.hidden.get(f"{layer_name}_single_bonded")(V.R2, V.O2, V.B2),
                 R.get(f"{layer_name}_bond_message")(V.O2, V.R2, V.B2))]

    # Defining the ether group (R1-O-R2)
    template += [R.get(f"{layer_name}_ether")(V.X) <= R.get(f"{layer_name}_ether")(V.X, V.Y)]
    template += [R.get(f"{layer_name}_ether")(V.C, V.R) <= (
                 R.get(carbon)(V.C), 
                 R.get(oxygen)(V.O), 
                 R.get(carbon)(V.R),
                 R.hidden.get(f"{layer_name}_single_bonded")(V.C, V.O, V.B1),
                 R.hidden.get(f"{layer_name}_single_bonded")(V.O, V.R, V.B2),
                 R.get(f"{layer_name}_bond_message")(V.C, V.O, V.B1),
                 R.get(f"{layer_name}_bond_message")(V.R, V.O, V.B2),
                 R.special.alldiff(...))]
    
    # Aggregating oxygen patterns
    template += [R.get(f"{layer_name}_oxy_groups")(V.X) <= R.get(f"{layer_name}_alcoholic")(V.X)[param_size]]
    template += [R.get(f"{layer_name}_oxy_groups")(V.X) <= R.get(f"{layer_name}_ketone")(V.X)[param_size]]
    template += [R.get(f"{layer_name}_oxy_groups")(V.X) <= R.get(f"{layer_name}_aldehyde")(V.X)[param_size]]
    template += [R.get(f"{layer_name}_oxy_groups")(V.X) <= R.get(f"{layer_name}_acyl_halide")(V.X)[param_size]]
    template += [R.get(f"{layer_name}_oxy_groups")(V.X) <= R.get(f"{layer_name}_carboxylic_acid")(V.X)[param_size]]
    template += [R.get(f"{layer_name}_oxy_groups")(V.X) <= R.get(f"{layer_name}_carboxylic_acid_anhydride")(V.X, V.Y)[param_size]]
    template += [R.get(f"{layer_name}_oxy_groups")(V.X) <= R.get(f"{layer_name}_ester")(V.X)[param_size]]
    template += [R.get(f"{layer_name}_oxy_groups")(V.X) <= R.get(f"{layer_name}_carbonate_ester")(V.X)[param_size]]
    template += [R.get(f"{layer_name}_oxy_groups")(V.X) <= R.get(f"{layer_name}_ether")(V.X)[param_size]]

    return template