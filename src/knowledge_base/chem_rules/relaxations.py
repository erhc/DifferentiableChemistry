from neuralogic.core import R, V

def get_relaxations(layer_name: str, param_size: tuple, connection: str, carbon: str):
    template = []

    # Defining a relaxed aliphatic and aromatic bond messages
    template += [R.get(f"{layer_name}_relaxed_aliphatic_bonded")(V.X, V.Y) <= (R.get(f"{layer_name}_relaxed_aliphatic_bonded")(V.X, V.Y, V.B))]
    template += [R.get(f"{layer_name}_relaxed_aliphatic_bonded")(V.X, V.Y, V.B) <= (
                 R.get(connection)(V.X, V.Y, V.B), 
                 R.get(f"{layer_name}_aliphatic_bond")(V.B),
                 R.get(f"{layer_name}_bond_message")(V.X, V.Y, V.B)[param_size])]

    template += [R.get(f"{layer_name}_relaxed_aromatic_bonded")(V.X, V.Y) <= (R.get(f"{layer_name}_relaxed_aromatic_bonded")(V.X, V.Y, V.B))]
    template += [R.get(f"{layer_name}_relaxed_aromatic_bonded")(V.X, V.Y, V.B) <= (
                 R.get(connection)(V.X, V.Y, V.B), 
                 R.get(f"{layer_name}_aromatic_bond")(V.B),
                 R.get(f"{layer_name}_bond_message")(V.X, V.Y, V.B)[param_size])]

    # Defining a relaxed carbonyl group (key atom type connected to another using an aliphatic bond)
    template += [R.get(f"{layer_name}_relaxed_carbonyl_group")(V.X, V.Y) <= (
                 R.get(f"{layer_name}_key_atom")(V.X)[param_size], 
                 R.get(f"{layer_name}_key_atom")(V.Y)[param_size],
                 R.get(f"{layer_name}_relaxed_aliphatic_bonded")(V.X, V.Y))]

    template += [R.get(f"{layer_name}_relaxed_carbonyl_group")(V.C, V.O, V.R1, V.R2) <= (
                 R.get(f"{layer_name}_relaxed_carbonyl_group")(V.C, V.O),
                 R.get(f"{layer_name}_relaxed_aliphatic_bonded")(V.C, V.R1),
                 R.get(f"{layer_name}_relaxed_aliphatic_bonded")(V.C, V.R2),
                 R.special.alldiff(...))]

    # Defining a relaxed aromatic ring
    template += [R.get(f"{layer_name}_relaxed_benzene_ring")(V.A) <= R.get(f"{layer_name}_relaxed_benzene_ring")(V.A, V.B)]
    template += [R.get(f"{layer_name}_relaxed_benzene_ring")(V.A, V.B) <= (
                 R.get(f"{layer_name}_relaxed_benzene_ring")(V.A, V.B, V.C, V.D, V.E, V.F))]

    template += [R.get(f"{layer_name}_relaxed_benzene_ring")(V.A, V.B, V.C, V.D, V.E, V.F) <= (
                 R.get(f"{layer_name}_relaxed_aromatic_bonded")(V.A, V.B),
                 R.get(f"{layer_name}_relaxed_aromatic_bonded")(V.B, V.C),
                 R.get(f"{layer_name}_relaxed_aromatic_bonded")(V.C, V.D),
                 R.get(f"{layer_name}_relaxed_aromatic_bonded")(V.D, V.E),
                 R.get(f"{layer_name}_relaxed_aromatic_bonded")(V.E, V.F),
                 R.get(f"{layer_name}_relaxed_aromatic_bonded")(V.F, V.A),
                 R.special.alldiff(...))]

    

    # Defining a potential group
    template += [R.get(f"{layer_name}_potential_group")(V.C) <= (
                 R.get(f"{layer_name}_relaxed_aliphatic_bonded")(V.C, V.X)[param_size],
                 R.get(f"{layer_name}_noncarbon")(V.X)[param_size],
                 R.get(carbon)(V.C))]

    # Defining a relaxed carbonyl derivative
    template += [R.get(f"{layer_name}_carbonyl_derivatives")(V.X) <= (
                 R.get(f"{layer_name}_relaxed_carbonyl_group")(V.C, V.O, V.R, V.X),
                 R.get(f"{layer_name}_noncarbon")(V.X)[param_size])]
    template += [R.get(f"{layer_name}_carbonyl_derivatives")(V.C) <= (
                 R.get(f"{layer_name}_relaxed_carbonyl_group")(V.C, V.X, V.R, V.R2),
                 R.get(f"{layer_name}_noncarbon")(V.X)[param_size])]
    template += [R.get(f"{layer_name}_carbonyl_derivatives")(V.C) <= (
                 R.get(f"{layer_name}_relaxed_carbonyl_group")(V.C, V.X, V.R, V.R2),
                 R.get(f"{layer_name}_noncarbon")(V.C)[param_size])]


    # Aggregating relaxations
    template += [R.get(f"{layer_name}_relaxed_functional_group")(V.X) <= (
                 R.get(f"{layer_name}_relaxed_carbonyl_group")(V.X, V.Y, V.R1, V.R2)[param_size])]
    template += [R.get(f"{layer_name}_relaxed_functional_group")(V.X) <= (
                 R.get(f"{layer_name}_relaxed_aliphatic_bonded")(V.X, V.Y)[param_size])]
    template += [R.get(f"{layer_name}_relaxed_functional_group")(V.X) <= (
                 R.get(f"{layer_name}_relaxed_aromatic_bonded")(V.X, V.Y)[param_size])]
    template += [R.get(f"{layer_name}_relaxed_functional_group")(V.X) <= (
                 R.get(f"{layer_name}_relaxed_benzene_ring")(V.X)[param_size])]
    template += [R.get(f"{layer_name}_relaxed_functional_group")(V.X) <= (
                 R.get(f"{layer_name}_potential_group")(V.X)[param_size])]
    template += [R.get(f"{layer_name}_relaxed_functional_group")(V.X) <= (
                 R.get(f"{layer_name}_carbonyl_derivatives")(V.X)[param_size])]

    return template