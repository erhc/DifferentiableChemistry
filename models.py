from neuralogic.core import R, V, Transformation, Aggregation, Combination

GNN = "gnn"
RGCN = "rgcn"
KGNN_LOCAL = "kgnn_local"
KGNN_GLOBAL = "kgnn"
EGO_GNN = "ego"
GATED_GNN = "gated_gnn"
DIFFUSION = "diffusion"
CWN = "cw_net"
SUBGRAPH = "sgn"


def get_model(model: str, model_name: str, layers: int, node_embed: str, edge_embed: str,
              connection: str, param_size: int,
              edge_types=None, max_depth=1):
    """
        Parameters
        ----------
        model: type of the model ("gnn", "rgcn", "kgnn_local", "kgnn", "ego", "gated_gnn", "diffusion", "cw_net", "sgn")
        model_name: name for the model
        layers: number of layers
        node_embed: embedding for the nodes
        edge_embed: embedding for the edges
        connection: connection predicate, expects connection(X, Y, E), where X and Y are nodes and E is edge
        param_size: int parameter size of the embeddings
        edge_types: list of strings representing predicates defining edge types
        max_depth: max depth of recursion for gated GNNs, diffusion GCNs, max ring size for CW networks or max order for SGNs
        Returns
        -------
        template: a list of rules defining the model
        """
    if model == GNN:
        return StandardGNN_model(model_name, layers, node_embed, edge_embed, connection, param_size)
    elif model == RGCN:
        return RGCN_model(model_name, layers, node_embed, edge_embed, connection, param_size, edge_types)
    elif model == KGNN_LOCAL:
        return KGNN_model(model_name, layers, node_embed, param_size, edge_embed, connection, local=True)
    elif model == KGNN_GLOBAL:
        return KGNN_model(model_name, layers, node_embed, param_size, local=False)
    elif model == EGO_GNN:
        return EgoGNN_model(model_name, layers, node_embed, edge_embed, connection, param_size)
    elif model == GATED_GNN:
        return GatedGNN_model(model_name, layers, node_embed, edge_embed, connection, param_size, max_depth)
    elif model == DIFFUSION:
        return DiffusionCNN_model(model_name, layers, node_embed, edge_embed, connection, param_size, max_depth)
    elif model == CWN:
        return CWNet_model(model_name, layers, node_embed, edge_embed, connection, param_size, max_depth)
    elif model == SUBGRAPH:
        return SGN_model(model_name, layers, node_embed, edge_embed, connection, param_size, max_depth)
    else:
        raise Exception("Invalid model name")


def StandardGNN_model(model_name: str, layers: int, node_embed: str, edge_embed: str, connection: str, param_size: int):
    def get_gnn(layer_name: str, node_embed: str, param_size: tuple):
        return [(R.get(layer_name)(V.X) <= (R.get(node_embed)(V.X)[param_size],
                                            R.get(node_embed)(V.Y)[param_size],
                                            R.get(connection)(V.X, V.Y, V.B), R.get(edge_embed)(V.B)))]

    template = [(R.get(f"{model_name}_GNN_0")(V.X) <= R.get(node_embed)(V.X))]
    for i in range(layers):
        template += get_gnn(f"{model_name}_GNN_{i + 1}", f"{model_name}_GNN_{i}", (param_size, param_size))

    template += [(R.get(f"{model_name}_GNN")(V.X) <= R.get(f"{model_name}_GNN_{layers}")(V.X))]
    template += [(R.predict[1, param_size] <= R.get(f"{model_name}_GNN")(V.X))]

    return template


def RGCN_model(model_name: str, layers: int, node_embed: str, edge_embed: str, connection: str, param_size: int,
               edge_types):
    def get_rgcn(layer_name: str, node_embed: str, param_size: tuple):
        return [(R.get(layer_name)(V.X) <= (R.get(node_embed)(V.X)[param_size],
                                            R.get(node_embed)(V.Y)[param_size],
                                            R.get(connection)(V.X, V.Y, V.B),
                                            R.get(edge_embed)(V.B),
                                            R.get(f"{model_name}_RGCN_edge_types")(V.B)[param_size]))]

    template = [(R.get(f"{model_name}_RGCN_0")(V.X) <= R.get(node_embed)(V.X))]

    for t in edge_types:
        template.append((R.get(f"{model_name}_RGCN_edge_types")(V.B) <= R.get(t)(V.B)[param_size,]))

    for i in range(layers):
        template += get_rgcn(f"{model_name}_RGCN_{i + 1}", f"{model_name}_RGCN_{i}", (param_size, param_size))

    template += [(R.get(f"{model_name}_RGCN")(V.X) <= R.get(f"{model_name}_RGCN_{layers}")(V.X))]
    template += [(R.predict[1, param_size] <= R.get(f"{model_name}_RGCN")(V.X))]

    return template


def KGNN_model(model_name: str, layers: int, node_embed: str, param_size: int,
               edge_embed: str = None, connection: str = None, local=True):
    def get_k_set(layer_name: str, prev_layer: str, param_size: tuple):
        body = [R.get(node_embed)(V.X)[param_size],
                R.get(prev_layer)(V.Y)[param_size]]
        if local:
            body += [R.get(connection)(V.X, V.Y, V.B),
                     R.get(edge_embed)(V.B)[param_size]]

        return [R.get(layer_name)(V.X) <= body]

    template = [(R.get(f"{model_name}_kGNN_0")(V.X) <= (R.get(node_embed)(V.X)))]

    for i in range(layers):
        template += get_k_set(f"{model_name}_kGNN_{i + 1}", f"{model_name}_kGNN_{i}", (param_size, param_size))

    template += [(R.get(f"{model_name}_kGNN")(V.X) <= R.get(f"{model_name}_kGNN_{layers}")(V.X))]
    template += [(R.predict[1, param_size] <= R.get(f"{model_name}_kGNN")(V.X))]

    return template


def EgoGNN_model(model_name: str, layers: int, node_embed: str,
                 edge_embed: str, connection: str, param_size: int):
    def get_ego(layer_name: str, node_embed: str, param_size: tuple):
        template = []
        template += [R.get(layer_name + "_multigraph")(V.X) <= (
            R.get(connection)(V.X, V.Y, V.B),
            R.get(edge_embed)(V.B)[param_size],
            R.get(node_embed)(V.Y)[param_size])]

        template += [R.get(layer_name)(V.X) <= (
            R.get(connection)(V.X, V.Y, V.B),
            R.get(layer_name + "_multigraph")(V.Y)[param_size])]
        return template

    template = [(R.get(f"{model_name}_ego_0")(V.X) <= (R.get(node_embed)(V.X)))]

    for i in range(layers):
        template += get_ego(f"{model_name}_ego_{i + 1}", f"{model_name}_ego_{i}", (param_size, param_size))

    template += [(R.get(f"{model_name}_ego")(V.X) <= R.get(f"{model_name}_ego_{layers}")(V.X))]
    template += [(R.predict[1, param_size] <= R.get(f"{model_name}_ego")(V.X))]

    return template


def GatedGNN_model(model_name: str, layers: int, node_embed: str,
                   edge_embed: str, connection: str, param_size: int, max_depth=1):
    def get_gated_gnn(layer_name, node_embed, param_size):
        template = []
        template += [(R.get(layer_name + "_h")(V.X, 0) <= (R.get(node_embed)(V.X)[param_size]))]

        template += [(R.get(layer_name + "_a_in")(V.X, V.T) <= (R.get(connection)(V.X, V.Y, V.B),
                                                                R.get(edge_embed)(V.B)[param_size],
                                                                R.get(layer_name + "_h")(V.Y, V.Z)[param_size],
                                                                R.special.next(V.Z, V.T))) | [Aggregation.SUM,
                                                                                              Transformation.IDENTITY]]

        template += [(R.get(layer_name + "_a_out")(V.X, V.T) <= (R.get(connection)(V.X, V.Y, V.B),
                                                                 R.get(edge_embed)(V.B)[param_size],
                                                                 R.get(layer_name + "_h")(V.Y, V.Z)[param_size],
                                                                 R.special.next(V.Z, V.T))) | [Aggregation.SUM,
                                                                                               Transformation.IDENTITY]]

        template += [
            (R.get(layer_name + "_update_gate")(V.X, V.T) <= (R.get(layer_name + "_a_in")(V.X, V.T)[param_size],
                                                              R.get(layer_name + "_a_out")(V.X, V.T)[param_size],
                                                              R.get(layer_name + "_h")(V.X, V.Z)[param_size],
                                                              R.special.next(V.Z, V.T))) | [Transformation.SIGMOID]]

        template += [(R.get(layer_name + "_reset_gate")(V.X, V.T) <= (R.get(layer_name + "_a_in")(V.X, V.T)[param_size],
                                                                      R.get(layer_name + "_a_out")(V.X, V.T)[
                                                                          param_size],
                                                                      R.get(layer_name + "_h")(V.X, V.Z)[param_size],
                                                                      R.special.next(V.Z, V.T))) | [
                         Transformation.SIGMOID]]

        template += [(R.get(layer_name + "_h_tright")(V.X, V.T) <= (R.get(layer_name + "_reset_gate")(V.X, V.T),
                                                                    R.get(layer_name + "_h")(V.X, V.Z),
                                                                    R.special.next(V.Z, V.T))) | [
                         Transformation.IDENTITY, Combination.ELPRODUCT]]

        template += [(R.get(layer_name + "_h_tilde")(V.X, V.T) <= (R.get(layer_name + "_a_in")(V.X, V.T)[param_size],
                                                                   R.get(layer_name + "_a_out")(V.X, V.T)[param_size],
                                                                   R.get(layer_name + "_h_tright")(V.X, V.T)[
                                                                       param_size])) | [Transformation.TANH,
                                                                                        Aggregation.SUM]]

        template += [(R.get(layer_name + "_h_right")(V.X, V.T) <= (R.get(layer_name + "_update_gate")(V.X, V.T),
                                                                   R.get(layer_name + "_h_tilde")(V.X, V.T))) | [
                         Transformation.IDENTITY, Combination.ELPRODUCT]]

        template += [(R.get(layer_name + "_h_left")(V.X, V.T) <= (R.get(layer_name + "_update_gate")(V.X, V.T),
                                                                  R.get(layer_name + "_h")(V.X, V.Z),
                                                                  R.special.next(V.Z, V.T))) | [Transformation.IDENTITY,
                                                                                                Combination.ELPRODUCT]]

        template += [(R.get(layer_name + "_h")(V.X, V.T) <= (R.get(layer_name + "_h_left")(V.X, V.T),
                                                             R.get(layer_name + "_h_right")(V.X, V.T))) | [
                         Aggregation.SUM, Transformation.IDENTITY]]

        template += [(R.get(layer_name)(V.X) <= R.get(layer_name + "_h")(V.X, max_depth))]
        return template

    template = [(R.get(f"{model_name}_gated_0")(V.X) <= (R.get(node_embed)(V.X)))]

    for i in range(max_depth):
        template += [(R._next(i, i + 1))]

    for i in range(layers):
        template += get_gated_gnn(f"{model_name}_gated_{i + 1}", f"{model_name}_gated_{i}", (param_size, param_size))

    template += [(R.get(f"{model_name}_gated")(V.X) <= R.get(f"{model_name}_gated_{layers}")(V.X))]
    template += [(R.predict[1, param_size] <= R.get(f"{model_name}_gated")(V.X))]

    return template


def DiffusionCNN_model(model_name: str, layers: int, node_embed: str,
                       edge_embed: str, connection: str, param_size: int, max_depth: int):
    def get_path(layer_name: str, param_size: tuple):
        template = []
        template += [
            R.get(layer_name)(V.X, V.Y, 0) <= (R.get(edge_embed)(V.B)[param_size], R.get(connection)(V.X, V.Y, V.B))]
        template += [R.get(layer_name)(V.X, V.Y, V.T) <= (R.get(edge_embed)(V.B)[param_size],
                                                          R.get(layer_name)(V.Z, V.Y, V.T1)[param_size],
                                                          R.get(connection)(V.X, V.Z, V.B),
                                                          R.special.next(V.T1, V.T))]

        for i in range(max_depth):
            template += [(R._next(i, i + 1))]

        template += [R.get(layer_name)(V.X, V.Y) <= (R.get(layer_name)(V.X, V.Y, max_depth))]

        return template

    def get_diffusion(layer_name: str, node_embed: str, param_size: tuple):
        template = []
        template += [
            (R.get(layer_name + "_Z")(V.X) <= (
                R.get(f"{model_name}_diff_path")(V.X, V.Y), R.get(node_embed)(V.Y)[param_size])) | [
                Aggregation.SUM]]
        template += [R.get(layer_name + "_Z")(V.X) <= R.get(node_embed)(V.X)[param_size]]
        template += [
            (R.get(layer_name)(V.X) <= (R.get(layer_name + "_Z")(V.X))) | [Transformation.SIGMOID, Aggregation.SUM]]

        return template

    template = [(R.get(f"{model_name}_diff_0")(V.X) <= (R.get(node_embed)(V.X)))]
    template += get_path(f"{model_name}_diff_path", (param_size, param_size))

    for i in range(layers):
        template += get_diffusion(f"{model_name}_diff_{i + 1}", f"{model_name}_diff_{i}", (param_size, param_size))

    template += [(R.get(f"{model_name}_diff")(V.X) <= R.get(f"{model_name}_diff_{layers}")(V.X))]
    template += [(R.predict[1, param_size] <= R.get(f"{model_name}_diff")(V.X))]

    return template


def CWNet_model(model_name: str, layers: int, node_embed: str,
                edge_embed: str, connection: str, param_size: int, max_ring_size: int = 7):
    def bottom_up_features(layer_name: str, prev_layer: str, param_size: tuple):
        template = []
        # atoms aggregate to bonds, bonds to rings
        template += [R.get(layer_name + "_bond_features")(V.B) <= (R.get(connection)(V.X, V.Y, V.B),
                                                                   R.get(f"{prev_layer}_node")(V.X)[param_size],
                                                                   R.get(f"{prev_layer}_node")(V.Y)[param_size])]

        # bonds in same cycle
        def get_bond_cycle(n):
            body = [R.get(connection)(f"X{i}", f"X{(i + 1) % n}", f"B{i}") for i in range(n)]
            body.extend(R.get(f"{prev_layer}_edge")(f"B{i}")[param_size] for i in range(n))
            body.append(R.special.alldiff(...))
            return [R.get(layer_name + "_cycle_features") <= body]

        for i in range(3, max_ring_size):
            template += get_bond_cycle(i)

        template += [R.get(layer_name + "_features")(V.B) <= (R.get(layer_name + "_bond_features")(V.B)[param_size])]
        template += [R.get(layer_name + "_features")(V.X) <= (R.get(layer_name + "_cycle_features")[param_size])]
        return template

    def top_down_features(layer_name: str, prev_layer: str, param_size: tuple):
        template = []

        def get_cycle_messages(n):
            body = [R.get(connection)(f"X{i}", f"X{(i + 1) % n}", f"B{i}") for i in range(n)]
            body.extend(R.get(f"{prev_layer}_edge")(f"B{i}")[param_size] for i in range(n))
            body.append(R.special.alldiff(...))

            return [R.get(layer_name + "_cycle_message")(V.B0) <= body]

        for i in range(3, max_ring_size):
            template += get_cycle_messages(i)

        # atoms sharing a bond share messages, bonds in the same ring
        template += [R.get(layer_name + "_atom_nbhood")(V.X) <= (R.get(connection)(V.X, V.Y, V.B),
                                                                 R.get(f"{prev_layer}_node")(V.Y)[param_size],
                                                                 R.get(layer_name + "_bond_nbhood")(V.B)[param_size])]
        template += [R.get(layer_name + "_bond_nbhood")(V.B) <= (R.get(layer_name + "_cycle_message")(V.B)[param_size])]

        template += [R.get(layer_name + "_nbhood")(V.B) <= (R.get(layer_name + "_bond_nbhood")(V.B)[param_size])]
        template += [R.get(layer_name + "_nbhood")(V.B) <= (R.get(layer_name + "_atom_nbhood")(V.B)[param_size])]
        return template

    def get_cw(layer_name: str, prev_layer: str, param_size: tuple):
        template = []
        template += bottom_up_features(layer_name, prev_layer, param_size)
        template += top_down_features(layer_name, prev_layer, param_size)

        template += [R.get(layer_name)(V.X) <= (R.get(layer_name + "_nbhood")(V.X)[param_size])]
        template += [R.get(layer_name)(V.X) <= (R.get(layer_name + "_features")(V.X)[param_size])]

        template += [R.get(layer_name + "_node")(V.X) <= (R.get(node_embed)(V.X)[param_size])]
        template += [R.get(layer_name + "_node")(V.X) <= (R.get(layer_name)(V.X)[param_size])]
        template += [R.get(layer_name + "_node")(V.X) <= (R.get(prev_layer + "_node")(V.X)[param_size])]

        template += [R.get(layer_name + "_edge")(V.X) <= (R.get(edge_embed)(V.X)[param_size])]
        template += [R.get(layer_name + "_edge")(V.X) <= (R.get(layer_name)(V.X)[param_size])]
        template += [R.get(layer_name + "_edge")(V.X) <= (R.get(prev_layer + "_edge")(V.X)[param_size])]
        return template

    template = [(R.get(f"{model_name}_cw_0_node")(V.X) <= (R.get(node_embed)(V.X)))]
    template += [(R.get(f"{model_name}_cw_0_edge")(V.X) <= (R.get(edge_embed)(V.X)))]

    for i in range(layers):
        template += get_cw(f"{model_name}_cw_{i + 1}", f"{model_name}_cw_{i}", (param_size, param_size))

    template += [(R.get(f"{model_name}_cw")(V.X) <= R.get(f"{model_name}_cw_{layers}")(V.X))]
    template += [(R.predict[1, param_size] <= R.get(f"{model_name}_cw")(V.X))]

    return template


def SGN_model(model_name: str, layers: int, node_embed: str,
              edge_embed: str, connection: str, param_size: int, max_order: int = 3):
    def get_sgn(layer_name: str, node_embed: str, param_size: tuple):
        template = []

        template += [R.get(f"{layer_name}_order_0")(V.B1, V.B2) <= (R.get(connection)(V.X, V.Y, V.B1),
                                                                    R.get(connection)(V.Y, V.Z, V.B2),
                                                                    R.get(edge_embed)(V.B1)[param_size],
                                                                    R.get(edge_embed)(V.B2)[param_size],
                                                                    R.get(node_embed)(V.X)[param_size],
                                                                    R.get(node_embed)(V.Y)[param_size],
                                                                    R.get(node_embed)(V.Z)[param_size])]

        for i in range(max_order):
            template += [R.get(f"{layer_name}_order_{i + 1}")(V.X, V.Z) <= (
            R.get(f"{layer_name}_order_{i}")(V.X, V.Y)[param_size],
            R.get(f"{layer_name}_order_{i}")(V.Y, V.Z)[param_size])]
        template += [R.get(layer_name)(V.X) <= R.get(f"{layer_name}_order_{max_order}")(V.X, V.Z)]

        return template

    template = []

    for i in range(layers):
        template += [
            (R.get(f"{model_name}_sgn_{i}")(V.X) <= (R.get(node_embed)(V.X)))]  # add node embedding or prev layer?
        template += get_sgn(f"{model_name}_sgn_{i + 1}", f"{model_name}_sgn_{i}", (param_size, param_size))
        template += [(R.get(f"{model_name}_sgn_{i + 1}")(V.X) <= (R.get(f"{model_name}_sgn_{i}")(V.X)))]

    template += [(R.get(f"{model_name}_sgn")(V.X) <= R.get(f"{model_name}_sgn_{layers}")(V.X))]
    template += [(R.predict[1, param_size] <= R.get(f"{model_name}_sgn")(V.X))]

    return template
