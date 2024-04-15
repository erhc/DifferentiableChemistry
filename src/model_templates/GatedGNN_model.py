from neuralogic.core import R, V, Transformation, Aggregation, Combination

def GatedGNN_model(model_name: str, layers: int, node_embed: str, edge_embed: str, connection: str, param_size: int, **kwargs):
    max_depth = kwargs.get('max_depth', 1)
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