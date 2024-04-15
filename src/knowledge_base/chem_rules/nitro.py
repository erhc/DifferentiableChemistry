from neuralogic.core import R, V

def get_nitro(layer_name: str, param_size: tuple, carbon: str, oxygen: str, hydrogen: str, nitrogen: str):
    template = []

    # Creating a negation for carbonyl group predicate
    # template += [R.get(f"{layer_name}_n_carbonyl")(V.C) <= (R.get(f"{layer_name}_carbonyl_group")(V.C))]

    # Defining amine group (R1-C-N(-R2)-R3)
    # TODO: this won't work for datasets with implicit hydrogens (primary, secondary amines)
    template += [R.get(f"{layer_name}_amine")(V.N) <= (
                # TODO: now it is relaxed ebcause of some issue when using it (predicate has no input)
                #  ~R.hidden.get(f"{layer_name}_n_carbonyl")(V.C),
                 R.get(f"{layer_name}_amino_group")(V.N, V.C, V.R1, V.R2))]
    
    template += [R.get(f"{layer_name}_amino_group")(V.N, V.R1, V.R2, V.R3) <= (
                 R.get(carbon)(V.R1), 
                 R.get(nitrogen)(V.N),
                 R.hidden.get(f"{layer_name}_single_bonded")(V.N, V.R1, V.B1),
                 R.hidden.get(f"{layer_name}_single_bonded")(V.N, V.R2, V.B2),
                 R.hidden.get(f"{layer_name}_single_bonded")(V.N, V.R3, V.B3),
                 R.get(f"{layer_name}_bond_message")(V.N, V.R1, V.B1),
                 R.get(f"{layer_name}_bond_message")(V.N, V.R2, V.B2),
                 R.get(f"{layer_name}_bond_message")(V.N, V.R3, V.B3),
                 R.special.alldiff(...))]


    # Defining quaternary ammonium ion (R1-N(-R2)(-R3)-R4)
    template += [R.get(f"{layer_name}_quat_ammonion")(V.N) <= (
                 R.get(nitrogen)(V.N), 
                 R.get(carbon)(V.C),
                 R.get(f"{layer_name}_amino_group")(V.N, V.R1, V.R2, V.R3),
                 R.hidden.get(f"{layer_name}_single_bonded")(V.N, V.C, V.B),
                 R.get(f"{layer_name}_bond_message")(V.N, V.C, V.B),
                 R.special.alldiff(...))]

    # Defining amide group (R-C(=O)-N(-R1)-R2)
    template += [R.get(f"{layer_name}_amide")(V.R) <= (R.get(f"{layer_name}_amide")(V.R, V.R1, V.R2))]
    template += [R.get(f"{layer_name}_amide")(V.R1) <= (R.get(f"{layer_name}_amide")(V.R, V.R1, V.R2))]

    template += [R.get(f"{layer_name}_amide")(V.R, V.R1, V.R2) <= (
                 R.get(f"{layer_name}_carbonyl_group")(V.C, V.O, V.R, V.N),
                 R.get(f"{layer_name}_amino_group")(V.N, V.C, V.R1, V.R2),
                 R.special.alldiff(...))]

    # Defining imine group (R1-C(=N-R)-R2)
    template += [R.get(f"{layer_name}_imine")(V.R) <= R.get(f"{layer_name}_imine")(V.R, V.R1, V.R2)]
    template += [R.get(f"{layer_name}_imine")(V.R1) <= R.get(f"{layer_name}_imine")(V.R, V.R1, V.R2)]
    template += [R.get(f"{layer_name}_imine")(V.R, V.R1, V.R2) <= (
                 R.get(carbon)(V.C), 
                 R.get(nitrogen)(V.N),
                 R.hidden.get(f"{layer_name}_double_bonded")(V.C, V.N, V.B),
                 R.hidden.get(f"{layer_name}_single_bonded")(V.C, V.R1, V.B1),
                 R.hidden.get(f"{layer_name}_single_bonded")(V.C, V.R2, V.B2),
                 R.hidden.get(f"{layer_name}_single_bonded")(V.N, V.R, V.B3),
                 R.get(f"{layer_name}_bond_message")(V.N, V.C, V.B),
                 R.get(f"{layer_name}_bond_message")(V.C, V.R1, V.B1),
                 R.get(f"{layer_name}_bond_message")(V.C, V.R2, V.B2),
                 R.get(f"{layer_name}_bond_message")(V.N, V.R, V.B3),
                 R.special.alldiff(...))]

    # Defining imide group (R1-C(=O)-N(-R)-C(=O)-R2)
    template += [R.get(f"{layer_name}_imide")(V.R) <= R.get(f"{layer_name}_imide")(V.R, V.R1, V.R2)]
    template += [R.get(f"{layer_name}_imide")(V.R1) <= R.get(f"{layer_name}_imide")(V.R, V.R1, V.R2)]
    template += [R.get(f"{layer_name}_imide")(V.R, V.R1, V.R2) <= (
                 R.get(carbon)(V.C1), 
                 R.get(nitrogen)(V.N), 
                 R.get(carbon)(V.C2),
                 R.get(f"{layer_name}_carbonyl_group")(V.C1, V.O1, V.R1, V.N),
                 R.get(f"{layer_name}_carbonyl_group")(V.C2, V.O2, V.R2, V.N),
                 R.hidden.get(f"{layer_name}_single_bonded")(V.N, V.R, V.B),
                 R.get(f"{layer_name}_bond_message")(V.N, V.R, V.B),
                 R.special.alldiff(...))]

    # Defining azide group (R-N=N=N)
    template += [R.get(f"{layer_name}_azide")(V.C) <= (
                 R.get(carbon)(V.C), 
                 R.get(nitrogen)(V.N1), 
                 R.get(nitrogen)(V.N2), 
                 R.get(nitrogen)(V.N3),
                 R.hidden.get(f"{layer_name}_single_bonded")(V.C, V.N1, V.B1),
                 R.hidden.get(f"{layer_name}_double_bonded")(V.N1, V.N2, V.B2),
                 R.hidden.get(f"{layer_name}_double_bonded")(V.N2, V.N3, V.B3),
                 R.get(f"{layer_name}_bond_message")(V.C, V.N, V.B1),
                 R.get(f"{layer_name}_bond_message")(V.N1, V.N2, V.B2),
                 R.get(f"{layer_name}_bond_message")(V.N2, V.N3, V.B3),
                 R.special.alldiff(...))]

    # Defining azo group (R1-N=N-R2)
    template += [R.get(f"{layer_name}_azo")(V.C1, V.C2) <= (
                 R.get(carbon)(V.C1), 
                 R.get(nitrogen)(V.N1), 
                 R.get(nitrogen)(V.N2), 
                 R.get(carbon)(V.C2),
                 R.hidden.get(f"{layer_name}_single_bonded")(V.C1, V.N1, V.B1),
                 R.hidden.get(f"{layer_name}_double_bonded")(V.N1, V.N2, V.B2),
                 R.hidden.get(f"{layer_name}_single_bonded")(V.N2, V.C2, V.B3),
                 R.get(f"{layer_name}_bond_message")(V.C1, V.N, V.B1),
                 R.get(f"{layer_name}_bond_message")(V.N1, V.N2, V.B2),
                 R.get(f"{layer_name}_bond_message")(V.N2, V.C2, V.B3),
                 R.special.alldiff(...))]

    # Defining cyanate group (R-O-C≡N)
    template += [R.get(f"{layer_name}_cyanate")(V.R) <= (
                 R.get(carbon)(V.C), 
                 R.get(nitrogen)(V.N), 
                 R.get(oxygen)(V.O),
                 R.get(carbon)(V.R),
                 R.hidden.get(f"{layer_name}_triple_bonded")(V.C, V.N, V.B1),
                 R.hidden.get(f"{layer_name}_single_bonded")(V.C, V.O, V.B2),
                 R.hidden.get(f"{layer_name}_single_bonded")(V.O, V.R, V.B3),
                 R.get(f"{layer_name}_bond_message")(V.C, V.N, V.B1),
                 R.get(f"{layer_name}_bond_message")(V.C, V.O, V.B2),
                 R.get(f"{layer_name}_bond_message")(V.O, V.R, V.B3),
                 R.special.alldiff(...))]
    
    # Defining isocyanate group (R-N=C=O)
    template += [R.get(f"{layer_name}_isocyanate")(V.R) <= (
                 R.get(carbon)(V.C), 
                 R.get(nitrogen)(V.N),
                 R.get(oxygen)(V.O), 
                 R.get(carbon)(V.R),
                 R.hidden.get(f"{layer_name}_double_bonded")(V.C, V.N, V.B1),
                 R.hidden.get(f"{layer_name}_double_bonded")(V.C, V.O, V.B2),
                 R.hidden.get(f"{layer_name}_single_bonded")(V.N, V.R, V.B3),
                 R.get(f"{layer_name}_bond_message")(V.C, V.N, V.B1),
                 R.get(f"{layer_name}_bond_message")(V.C, V.O, V.B2),
                 R.get(f"{layer_name}_bond_message")(V.N, V.R, V.B3),
                 R.special.alldiff(...))]

    # Defining nitro group (R-N(=O)-O)
    template += [R.get(f"{layer_name}_nitro_group")(V.R) <= (R.get(f"{layer_name}_nitro_group")(V.R, V.N, V.O1, V.O2))]
    template += [R.get(f"{layer_name}_nitro_group")(V.R, V.N, V.O1, V.O2) <= (
                 R.get(nitrogen)(V.N), 
                 R.get(oxygen)(V.O1), 
                 R.get(oxygen)(V.O2),
                 R.hidden.get(f"{layer_name}_single_bonded")(V.R, V.N, V.B1),
                 R.hidden.get(f"{layer_name}_double_bonded")(V.N, V.O1, V.B2),
                 R.hidden.get(f"{layer_name}_single_bonded")(V.N, V.O2, V.B3),
                 R.get(f"{layer_name}_bond_message")(V.R, V.N, V.B1),
                 R.get(f"{layer_name}_bond_message")(V.N, V.O1, V.B2),
                 R.get(f"{layer_name}_bond_message")(V.N, V.O2, V.B3),
                 R.special.alldiff(...))]
    template += [R.get(f"{layer_name}_nitro")(V.C) <= (R.get(carbon)(V.C), R.get(f"{layer_name}_nitro_group")(V.C))]

    # Defining nitrate group (R-O-N(=O)-O)
    template += [R.get(f"{layer_name}_nitrate")(V.R) <= (R.get(f"{layer_name}_nitrate")(V.R, V.O, V.N, V.O1, V.O2))]
    template += [R.get(f"{layer_name}_nitrate")(V.C, V.O, V.N, V.O1, V.O2) <= (
                 R.get(carbon)(V.C), 
                 R.get(oxygen)(V.O), 
                 R.get(f"{layer_name}_nitro_group")(V.O, V.N, V.O1, V.O2),
                 R.hidden.get(f"{layer_name}_single_bonded")(V.C, V.O, V.B),
                 R.get(f"{layer_name}_bond_message")(V.C, V.O, V.B),
                 R.special.alldiff(...))]

    # Defining carbamate group (R-O-C(=O)-N(-R1)-R2)
    template += [R.get(f"{layer_name}_carbamate")(V.R) <= R.get(f"{layer_name}_carbamate")(V.R, V.R1, V.R2)]
    template += [R.get(f"{layer_name}_carbamate")(V.R1) <= R.get(f"{layer_name}_carbamate")(V.R, V.R1, V.R2)]
    template += [R.get(f"{layer_name}_carbamate")(V.R, V.R1, V.R2) <= (
                 R.get(f"{layer_name}_amide")(V.O, V.R1, V.R2),
                 R.get(oxygen)(V.O),
                 R.hidden.get(f"{layer_name}_single_bonded")(V.O, V.R, V.B1),
                 R.get(f"{layer_name}_bond_message")(V.O, V.R, V.B1),
                 R.special.alldiff(...))]

    # Defining azidrine (*(C-C=N-))
    template += [R.get(f"{layer_name}_aziridine")(V.C1) <= (
                 R.get(carbon)(V.C1),
                 R.get(carbon)(V.C1),
                 R.get(nitrogen)(V.N),
                 R.hidden.get(f"{layer_name}_single_bonded")(V.C1, V.C2, V.B1),
                 R.hidden.get(f"{layer_name}_single_bonded")(V.N, V.C1, V.B2),
                 R.hidden.get(f"{layer_name}_double_bonded")(V.N, V.C2, V.B3),
                 R.get(f"{layer_name}_bond_message")(V.C1, V.C2, V.B1),
                 R.get(f"{layer_name}_bond_message")(V.N, V.C1, V.B2),
                 R.get(f"{layer_name}_bond_message")(V.N, V.C2, V.B3),
                 R.special.alldiff(...))]
    
    # Aggregating the nitrogen groups
    template += [R.get(f"{layer_name}_nitrogen_groups")(V.X) <= R.get(f"{layer_name}_amine")(V.X)[param_size]]
    template += [R.get(f"{layer_name}_nitrogen_groups")(V.X) <= R.get(f"{layer_name}_quat_ammonion")(V.X)[param_size]]
    template += [R.get(f"{layer_name}_nitrogen_groups")(V.X) <= R.get(f"{layer_name}_amide")(V.X)[param_size]]
    template += [R.get(f"{layer_name}_nitrogen_groups")(V.X) <= R.get(f"{layer_name}_imine")(V.X)[param_size]]
    template += [R.get(f"{layer_name}_nitrogen_groups")(V.X) <= R.get(f"{layer_name}_azide")(V.X)[param_size]]
    template += [R.get(f"{layer_name}_nitrogen_groups")(V.X) <= R.get(f"{layer_name}_imide")(V.X)[param_size]]
    template += [R.get(f"{layer_name}_nitrogen_groups")(V.X) <= R.get(f"{layer_name}_azo")(V.X, V.Y)[param_size]]
    template += [R.get(f"{layer_name}_nitrogen_groups")(V.X) <= R.get(f"{layer_name}_isocyanate")(V.X)[param_size]]
    template += [R.get(f"{layer_name}_nitrogen_groups")(V.X) <= R.get(f"{layer_name}_cyanate")(V.X)[param_size]]
    template += [R.get(f"{layer_name}_nitrogen_groups")(V.X) <= R.get(f"{layer_name}_nitro")(V.X)[param_size]]
    template += [R.get(f"{layer_name}_nitrogen_groups")(V.X) <= R.get(f"{layer_name}_nitrate")(V.X)[param_size]]
    template += [R.get(f"{layer_name}_nitrogen_groups")(V.X) <= R.get(f"{layer_name}_carbamate")(V.X)[param_size]]
    template += [R.get(f"{layer_name}_nitrogen_groups")(V.X) <= R.get(f"{layer_name}_aziridine")(V.X)[param_size]]

    return template