from chemdiff.knowledge_base.chem_rules.general import get_general
from chemdiff.knowledge_base.chem_rules.hydrocarbons import get_hydrocarbons
from chemdiff.knowledge_base.chem_rules.nitro import get_nitro
from chemdiff.knowledge_base.chem_rules.oxy import get_oxy
from chemdiff.knowledge_base.chem_rules.relaxations import get_relaxations
from chemdiff.knowledge_base.chem_rules.sulfuric import get_sulfuric
from neuralogic.core import R, V


def get_chem_rules(
    layer_name: str,
    node_embed: str,
    edge_embed: str,
    connection: str,
    param_size: int,
    halogens: list,
    output_layer_name: str ="predict",
    single_bond=None,
    double_bond=None,
    triple_bond=None,
    aromatic_bonds=None,
    carbon=None,
    hydrogen=None,
    oxygen=None,
    nitrogen=None,
    sulfur=None,
    path=None,
    hydrocarbons=False,
    nitro=False,
    sulfuric=False,
    oxy=False,
    relaxations=False,
    key_atoms: list = [],
    funnel=False,
):
    template = []
    if funnel:
        param_size = 1
    
    for b in aromatic_bonds:
        template += [
            (R.get(f"{layer_name}_aromatic_bond")(V.B)[param_size,] <= R.get(b)(V.B))
        ]

    for a in halogens:
        template += [
            (R.get(f"{layer_name}_halogen")(V.X)[param_size,] <= R.get(a)(V.X))
        ]
    if relaxations:
        for b in [single_bond, double_bond, triple_bond]:
            template += [
                (
                    R.get(f"{layer_name}_aliphatic_bond")(V.B)[param_size,]
                    <= R.get(b)(V.B)
                )
            ]

        for a in key_atoms:
            template += [
                (R.get(f"{layer_name}_key_atom")(V.A)[param_size,] <= R.get(a)(V.A))
            ]
            template += [
                (R.get(f"{layer_name}_noncarbon")(V.A)[param_size,] <= R.get(a)(V.A))
            ]

        template += [
            (R.get(f"{layer_name}_key_atom")(V.A)[param_size,] <= R.get(carbon)(V.A))
        ]

    if param_size == 1:
        template += [R.get(output_layer_name)[param_size,] <= R.get(f"{layer_name}_chem_rules")]
        param_size = (param_size,)
    else:
        template += [R.get(output_layer_name)[1, param_size] <= R.get(f"{layer_name}_chem_rules")]
        param_size = (param_size, param_size)

    template += get_general(
        layer_name,
        node_embed,
        edge_embed,
        connection,
        param_size,
        single_bond,
        double_bond,
        triple_bond,
        f"{layer_name}_aromatic_bond",
        hydrogen,
        carbon,
        oxygen,
    )

    template += [
        R.get(f"{layer_name}_functional_group")(V.X)[param_size]
        <= R.get(f"{layer_name}_general_groups")(V.X)
    ]

    if path:
        template += [
            R.get(f"{layer_name}_connected_groups")(V.X, V.Y)
            <= (
                R.get(f"{layer_name}_functional_group")(V.X)[param_size],
                R.get(f"{layer_name}_functional_group")(V.Y)[param_size],
                R.get(path)(V.X, V.Y),
            )
        ]
        if relaxations:
            template += [
                R.get(f"{layer_name}_connected_groups")(V.X, V.Y)
                <= (
                    R.get(f"{layer_name}_relaxed_functional_group")(V.X)[param_size],
                    R.get(f"{layer_name}_relaxed_functional_group")(V.Y)[param_size],
                    R.get(path)(V.X, V.Y),
                )
            ]
        template += [
            R.get(f"{layer_name}_chem_rules")(V.X)[param_size]
            <= R.get(f"{layer_name}_connected_groups")(V.X, V.Y)
        ]

    if hydrocarbons:
        template += get_hydrocarbons(layer_name, param_size, carbon)
        template += [
            R.get(f"{layer_name}_functional_group")(V.X)[param_size]
            <= R.get(f"{layer_name}_hydrocarbon_groups")(V.X)
        ]
    if oxy:
        template += get_oxy(layer_name, param_size, carbon, oxygen, hydrogen)
        template += [
            R.get(f"{layer_name}_functional_group")(V.X)[param_size]
            <= R.get(f"{layer_name}_oxy_groups")(V.X)
        ]
    if nitro:
        template += get_nitro(
            layer_name, param_size, carbon, oxygen, hydrogen, nitrogen
        )
        template += [
            R.get(f"{layer_name}_functional_group")(V.X)[param_size]
            <= R.get(f"{layer_name}_nitrogen_groups")(V.X)
        ]
    if sulfuric:
        template += get_sulfuric(
            layer_name, param_size, carbon, hydrogen, sulfur, nitrogen
        )
        template += [
            R.get(f"{layer_name}_functional_group")(V.X)[param_size]
            <= R.get(f"{layer_name}_sulfuric_groups")(V.X)
        ]
    if relaxations:
        template += get_relaxations(layer_name, param_size, connection, carbon)
        template += [
            R.get(f"{layer_name}_chem_rules")(V.X)[param_size]
            <= R.get(f"{layer_name}_relaxed_functional_group")(V.X)
        ]

    template += [
        R.get(f"{layer_name}_chem_rules")(V.X)[param_size]
        <= R.get(f"{layer_name}_functional_group")(V.X)
    ]
    template += [
            R.get(f"{layer_name}_chem_rules")[param_size]
            <= R.get(f"{layer_name}_chem_rules")(V.X)
        ]

    return template
