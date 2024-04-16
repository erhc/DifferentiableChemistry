from model_templates.StandardGNN_model import StandardGNN_model
from model_templates.RGCN_model import RGCN_model
from model_templates.KGNN_model import KGNN_model
from model_templates.EgoGNN_model import EgoGNN_model
from model_templates.GatedGNN_model import GatedGNN_model
from model_templates.DiffusionCNN_model import DiffusionCNN_model
from model_templates.CWNet_model import CWNet_model
from model_templates.SGN_model import SGN_model

model_type_functions = {
    "gnn": StandardGNN_model,
    "rgcn": RGCN_model,
    "kgnn": KGNN_model,
    "ego": EgoGNN_model,
    "gated_gnn": GatedGNN_model,
    "diffusion": DiffusionCNN_model,
    "cw_net": CWNet_model,
    "sgn": SGN_model
}

def get_model(model, model_name, layers, node_embed, edge_embed, connection, param_size, **kwargs):
    """
    Parameters
    ----------
    model: type of the model
    model_name: name for the model
    layers: number of layers
    node_embed: embedding for the nodes
    edge_embed: embedding for the edges
    connection: connection predicate, expects connection(X, Y, E), where X and Y are nodes and E is edge
    param_size: int parameter size of the embeddings
    edge_types: list of strings representing predicates defining edge types

    max_depth: Applicable for k-GNNs, Gated GNNs, Diffusion GNNs, SGNs
    local: Applicable for k-GNNs, default True
    Returns
    -------
    template: a list of rules defining the model
    """
    model_function = model_type_functions.get(model)

    if model_function is None:
        raise Exception(f"Invalid model name: {model}\nPlease use one of the following: {list(model_type_functions.keys())}")

    return model_function(model_name=model_name, layers=layers, node_embed=node_embed, edge_embed=edge_embed, connection=connection, param_size=param_size, **kwargs)