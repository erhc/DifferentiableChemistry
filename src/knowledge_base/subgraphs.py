from neuralogic.core import R, V
from knowledge_base.subgraph_rules.circular import get_circular
from knowledge_base.subgraph_rules.collective import get_collective
from knowledge_base.subgraph_rules.cycles import get_cycles
from knowledge_base.subgraph_rules.path import get_path
from knowledge_base.subgraph_rules.yshape import get_y_shape
from knowledge_base.subgraph_rules.nbhoods import get_nbhoods

def get_subgraphs(layer_name: str, node_embed: str, edge_embed: str, connection: str, param_size: int,
                  max_cycle_size: int = 10, max_depth: int = 5,
                  single_bond=None, double_bond=None, carbon=None, atom_types=None, aliphatic_bond=None,
                  cycles=False, paths=False, y_shape=False, nbhoods=False, circular=False, collective=False):
    template = []
    
    # Aggregating the patterns
    template += [R.predict[1, param_size] <= R.get(f"{layer_name}_subgraph_pattern")]

    param_size = (param_size, param_size)

    # Adding patterns
    if cycles or circular or collective:
        template += get_cycles(layer_name, node_embed, edge_embed, connection, param_size, max_cycle_size)
    if paths or collective:
        template += get_path(layer_name, node_embed, edge_embed, connection, param_size, max_depth)
    if y_shape:
        template += get_y_shape(layer_name, node_embed, edge_embed, connection, param_size, double_bond)
    if nbhoods:
        for t in atom_types:
            # TODO: unstable param size
            template += [R.get(f"{layer_name}_key_atoms")(V.X)[param_size[0],] <= R.get(t)(V.X)]
        template += get_nbhoods(layer_name, node_embed, edge_embed, connection, param_size, carbon, f"{layer_name}_key_atoms")
    if circular:
        template += get_circular(layer_name, node_embed, edge_embed, connection, param_size, carbon, single_bond, double_bond)
    if collective:
        for t in aliphatic_bond:
            template += [R.get(f"{layer_name}_aliphatic_bond")(V.X)[param_size[0],] <= R.get(t)(V.X)]
        template += get_collective(layer_name, node_embed, edge_embed, connection, param_size, carbon, f"{layer_name}_aliphatic_bond", max_depth)


    template += [R.get(f"{layer_name}_subgraph_pattern")(V.X) <= R.get(f"{layer_name}_pattern")(V.X)[param_size]]

    template += [R.get(f"{layer_name}_subgraph_pattern") <= R.get(f"{layer_name}_subgraph_pattern")(V.X)[param_size]]

    return template