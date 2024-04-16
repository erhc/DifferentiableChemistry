from neuralogic.core import R, V

def get_hydrocarbons(layer_name: str, param_size: tuple, carbon: str):
    template = []

    # Defining the benzene ring
    template += [R.get(f"{layer_name}_benzene_ring")(V.A) <= R.get(f"{layer_name}_benzene_ring")(V.A, V.B)]
    template += [R.get(f"{layer_name}_benzene_ring")(V.A, V.B) <= R.get(f"{layer_name}_benzene_ring")(V.A, V.B, V.C, V.D, V.E, V.F)]
    template += [R.get(f"{layer_name}_benzene_ring")(V.A, V.B, V.C, V.D, V.E, V.F) <= (
                 R.get(f"{layer_name}_aromatic_bonded")(V.A, V.B, V.B1),
                 R.get(f"{layer_name}_aromatic_bonded")(V.B, V.C, V.B2),
                 R.get(f"{layer_name}_aromatic_bonded")(V.C, V.D, V.B3),
                 R.get(f"{layer_name}_aromatic_bonded")(V.D, V.E, V.B4),
                 R.get(f"{layer_name}_aromatic_bonded")(V.E, V.F, V.B5),
                 R.get(f"{layer_name}_aromatic_bonded")(V.F, V.A, V.B6),
                 R.get(carbon)(V.A), R.get(carbon)(V.B), R.get(carbon)(V.C),
                 R.get(carbon)(V.D), R.get(carbon)(V.E), R.get(carbon)(V.F),
                 R.get(f"{layer_name}_bond_message")(V.A, V.B, V.B1),
                 R.get(f"{layer_name}_bond_message")(V.B, V.C, V.B2),
                 R.get(f"{layer_name}_bond_message")(V.C, V.D, V.B3),
                 R.get(f"{layer_name}_bond_message")(V.D, V.E, V.B4),
                 R.get(f"{layer_name}_bond_message")(V.E, V.F, V.B5),
                 R.get(f"{layer_name}_bond_message")(V.F, V.A, V.B6),
                 R.special.alldiff(...))]

    # Defining an alkene (R-C=C-R), alkyne group (R-C≡C-R)
    template += [R.get(f"{layer_name}_alkene_bond")(V.C1, V.C2) <= (
                 R.get(carbon)(V.C1), 
                 R.get(carbon)(V.C2), 
                 R.hidden.get(f"{layer_name}_double_bonded")(V.C1, V.C2, V.B),
                 R.get(f"{layer_name}_bond_message")(V.C1, V.C2, V.B))]
    
    template += [R.get(f"{layer_name}_alkyne_bond")(V.C1, V.C2) <= (
                 R.get(carbon)(V.C1), 
                 R.get(carbon)(V.C2), 
                 R.hidden.get(f"{layer_name}_triple_bonded")(V.C1, V.C2, V.B),
                 R.get(f"{layer_name}_bond_message")(V.C1, V.C2, V.B))]

    # Aggregating hydrocarbon groups
    template += [R.get(f"{layer_name}_hydrocarbon_groups")(V.C1) <= R.get(f"{layer_name}_alkene_bond")(V.C1, V.C2)[param_size]]
    template += [R.get(f"{layer_name}_hydrocarbon_groups")(V.C1) <= R.get(f"{layer_name}_alkyne_bond")(V.C1, V.C2)[param_size]]
    template += [R.get(f"{layer_name}_hydrocarbon_groups")(V.A) <= R.get(f"{layer_name}_benzene_ring")(V.A)[param_size]]

    return template