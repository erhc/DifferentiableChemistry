from neuralogic.core import R, V

def get_sulfuric(layer_name: str, param_size: tuple, carbon: str, hydrogen: str, sulfur: str, nitrogen: str):
    template = []

    # Defining thiocyanate group (R-S-C≡N)
    template += [R.get(f"{layer_name}_thiocyanate")(V.R) <= (
                 R.get(carbon)(V.C),
                 R.get(sulfur)(V.S), 
                 R.get(nitrogen)(V.N), 
                 R.get(carbon)(V.R),
                 R.hidden.get(f"{layer_name}_triple_bonded")(V.C, V.N, V.B1),
                 R.hidden.get(f"{layer_name}_single_bonded")(V.C, V.S, V.B2),
                 R.hidden.get(f"{layer_name}_single_bonded")(V.S, V.R, V.B3),
                 R.get(f"{layer_name}_bond_message")(V.C, V.N, V.B1),
                 R.get(f"{layer_name}_bond_message")(V.C, V.S, V.B2),
                 R.get(f"{layer_name}_bond_message")(V.S, V.R, V.B3),
                 R.special.alldiff(...))]
    
    # Defining isothiocyanate group (R-N=C=S)
    template += [R.get(f"{layer_name}_isothiocyanate")(V.R) <= (
                 R.get(carbon)(V.C),
                 R.get(sulfur)(V.S), 
                 R.get(nitrogen)(V.N), 
                 R.get(carbon)(V.R),
                 R.hidden.get(f"{layer_name}_double_bonded")(V.C, V.S, V.B1),
                 R.hidden.get(f"{layer_name}_double_bonded")(V.C, V.N, V.B2),
                 R.hidden.get(f"{layer_name}_single_bonded")(V.N, V.R, V.B3),
                 R.get(f"{layer_name}_bond_message")(V.C, V.S, V.B1),
                 R.get(f"{layer_name}_bond_message")(V.C, V.N, V.B2),
                 R.get(f"{layer_name}_bond_message")(V.N, V.R, V.B3),
                 R.special.alldiff(...))]

    # Defining sulfide group (R1-S-R2)
    template += [R.get(f"{layer_name}_sulfide")(V.R1, V.R2) <= (
                 R.get(carbon)(V.R1),
                 R.get(sulfur)(V.S), 
                 R.get(carbon)(V.R2),
                 R.hidden.get(f"{layer_name}_single_bonded")(V.R1, V.S, V.B1),
                 R.hidden.get(f"{layer_name}_single_bonded")(V.S, V.R2, V.B2),
                 R.get(f"{layer_name}_bond_message")(V.R1, V.S, V.B1),
                 R.get(f"{layer_name}_bond_message")(V.S, V.R2, V.B2),
                 R.special.alldiff(...))]
    
    # Defining disulfide group (R1-S-S-R2)
    template += [R.get(f"{layer_name}_disulfide")(V.C1, V.C2) <= (
                 R.get(carbon)(V.C1), 
                 R.get(sulfur)(V.S1),
                 R.get(sulfur)(V.S2), 
                 R.get(carbon)(V.C2),
                 R.hidden.get(f"{layer_name}_single_bonded")(V.C1, V.S1, V.B1),
                 R.hidden.get(f"{layer_name}_single_bonded")(V.S1, V.S2, V.B12),
                 R.hidden.get(f"{layer_name}_single_bonded")(V.S2, V.C2, V.B2),
                 R.get(f"{layer_name}_bond_message")(V.C1, V.S1, V.B1),
                 R.get(f"{layer_name}_bond_message")(V.S2, V.C2, V.B2),
                 R.get(f"{layer_name}_bond_message")(V.S1, V.S2, V.B12),
                 R.special.alldiff(...))]

    # Defining thiol group (R-S-H)
    # TODO: this won't work for datasets with implicit hydrogens
    template += [R.get(f"{layer_name}_thiol")(V.C) <= (
                 R.get(carbon)(V.C), 
                 R.get(sulfur)(V.S), 
                 R.get(hydrogen)(V.H),
                 R.hidden.get(f"{layer_name}_single_bonded")(V.C, V.S, V.B1),
                 R.get(f"{layer_name}_single_bonded")(V.S, V.H, V.B2),
                 R.get(f"{layer_name}_bond_message")(V.C, V.S, V.B1),
                 R.get(f"{layer_name}_bond_message")(V.S, V.H, V.B2),
                 R.special.alldiff(...))]

    # Aggregating sulfuric groups
    template += [R.get(f"{layer_name}_sulfuric_groups")(V.X) <= R.get(f"{layer_name}_isothiocyanate")(V.X)[param_size]]
    template += [R.get(f"{layer_name}_sulfuric_groups")(V.X) <= R.get(f"{layer_name}_thiocyanate")(V.X)[param_size]]
    template += [R.get(f"{layer_name}_sulfuric_groups")(V.X) <= R.get(f"{layer_name}_thiol")(V.X)[param_size]]
    template += [R.get(f"{layer_name}_sulfuric_groups")(V.X) <= R.get(f"{layer_name}_sulfide")(V.X, V.Y)[param_size]]
    template += [R.get(f"{layer_name}_sulfuric_groups")(V.X) <= R.get(f"{layer_name}_disulfide")(V.X, V.Y)[param_size]]

    return template