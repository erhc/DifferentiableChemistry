from neuralogic.core import Settings
from neuralogic.nn.loss import MSE
from neuralogic.nn import get_evaluator
from neuralogic.optim import Adam
import numpy as np
import mlflow
import dataset_templates
from models import get_model
from knowledge_base.chemrules import get_chem_rules
from knowledge_base.subgraphs import get_subgraphs
from typing import Union
from sklearn.model_selection import train_test_split
# from IPython.display import clear_output


# def train_test_cycle(template, dataset, lr=0.001, epochs=100, split=0.75):
#     settings = Settings(optimizer=Adam(lr=lr), epochs=epochs, error_function=MSE())
#     evaluator = get_evaluator(template, settings)

#     built_dataset = evaluator.build_dataset(dataset)
#     dataset_len = len(built_dataset.samples)

#     train_size = int(dataset_len * split)

#     idx = random.sample(list(range(dataset_len)), train_size)
#     rest = list(set(range(dataset_len)) - set(idx))
#     train_dataset = np.array(built_dataset.samples)[idx]
#     test_dataset = np.array(built_dataset.samples)[rest]
#     average_losses = []

#     for current_total_loss, number_of_samples in evaluator.train(train_dataset):
#         #clear_output(wait=True)
#         #plt.ylabel("Loss")
#         #plt.xlabel("Epoch")

#         #plt.xlim(0, settings.epochs)

#         train_loss = current_total_loss / number_of_samples
#         #print(train_loss)

#         average_losses.append(train_loss)

#         #plt.plot(average_losses, label="Average loss")

#         #plt.legend()
#         #plt.pause(0.001)
#         #plt.show()

#     loss = []
#     for sample, y_hat in zip(test_dataset, evaluator.test(test_dataset, generator=False)):
#         loss.append(round(y_hat) != sample.java_sample.target.value)

#     test_loss = sum(loss) / len(test_dataset)

#     return train_loss, test_loss, evaluator

def train_test_cycle(template, dataset, lr=0.001, epochs=100, split_ratio=0.75, optimizer=Adam, error_function=MSE):
    """
    Train and test the model based on the provided template and dataset.

    :param template: The template used for the evaluator.
    :param dataset: The dataset to train and test on.
    :param lr: Learning rate for the optimizer.
    :param epochs: Number of training epochs.
    :param split_ratio: The ratio to split the dataset into training and testing.
    :param optimizer: The optimizer class to be used.
    :param error_function: The error function to be used.
    :return: The training loss, testing loss, and the evaluator object.
    """
    settings = Settings(optimizer=optimizer(lr=lr), epochs=epochs, error_function=error_function())
    evaluator = get_evaluator(template, settings)
    built_dataset = evaluator.build_dataset(dataset)
    
    train_dataset, test_dataset = train_test_split(built_dataset.samples, train_size=split_ratio, random_state=42)
    train_losses = train_model(evaluator, train_dataset, settings.epochs)
    test_loss = evaluate_model(evaluator, test_dataset)

    return np.mean(train_losses), test_loss, evaluator

def train_model(evaluator, train_dataset, epochs):
    """
    Train the model on the training dataset.

    :param evaluator: The evaluator object used for training.
    :param train_dataset: The dataset to train on.
    :param epochs: Number of training epochs.
    :return: List of average training losses per epoch.
    """
    average_losses = []
    for epoch in range(epochs):
        current_total_loss, number_of_samples = next(evaluator.train(train_dataset))
        train_loss = current_total_loss / number_of_samples
        average_losses.append(train_loss)
        # Optional: Logging and visualization logic here
    return average_losses

def evaluate_model(evaluator, test_dataset):
    """
    Evaluate the model on the test dataset.

    :param evaluator: The evaluator object used for testing.
    :param test_dataset: The dataset to test on.
    :return: The testing loss.
    """
    loss = [round(y_hat) != sample.java_sample.target.value for sample, y_hat in zip(test_dataset, evaluator.test(test_dataset, generator=False))]
    return sum(loss) / len(test_dataset)

def init_test(dataset_name: str, model_name: str, param_size: int, layers: int, max_depth: int = 1, 
              max_subgraph_depth: int = 5, max_cycle_size: int = 10, subgraphs: Union[tuple, bool, None] = None, 
              chem_rules: Union[tuple, bool, None] = None, test_name: str = "test"):
    """
    Initialize the test setup by configuring the dataset and model along with optional chemical rules and subgraphs.

    :param dataset_name: Name of the dataset to use.
    :param model_name: Name of the model to apply.
    :param param_size: The size of the parameters.
    :param layers: Number of layers in the model.
    :param max_depth: Maximum depth for the model.
    :param max_subgraph_depth: Maximum depth for subgraph processing.
    :param max_cycle_size: Maximum size of cycles in subgraphs.
    :param subgraphs: Tuple containing flags for different subgraph types.
    :param chem_rules: Tuple containing chemical rule configurations.
    :param test_name: Prefix for the template predicates or name of the test. Should not be empty string. - default: "test"
    :return: A tuple containing the template and dataset.
    """
    template, dataset, dataset_info = dataset_templates.get_dataset(dataset_name, param_size)

    template += get_model(model_name, test_name, layers, dataset_info.node_embed, dataset_info.edge_embed, 
                                 dataset_info.connection, param_size, edge_types=dataset_info.bond_types, 
                                 max_depth=max_depth)

    if chem_rules:
        hydrocarbons, oxy, nitro, sulfuric, relaxations = chem_rules
        chem_path = f"{test_name}_sub_path" if subgraphs and subgraphs[1] else None

        template += get_chem_rules(f"{test_name}_chem", dataset_info.node_embed, dataset_info.edge_embed, 
                                   dataset_info.connection, param_size, dataset_info.halogens, 
                                   single_bond=dataset_info.single_bond, double_bond=dataset_info.double_bond, 
                                   triple_bond=dataset_info.triple_bond, aromatic_bonds=dataset_info.aromatic_bonds, 
                                   carbon=dataset_info.carbon, hydrogen=dataset_info.hydrogen, oxygen=dataset_info.oxygen, 
                                   nitrogen=dataset_info.nitrogen, sulfur=dataset_info.sulfur, path=chem_path, 
                                   hydrocarbons=hydrocarbons, nitro=nitro, sulfuric=sulfuric, oxy=oxy, 
                                   relaxations=relaxations, key_atoms=dataset_info.key_atom_type)

    if subgraphs:
        cycles, paths, y_shape, nbhoods, circular, collective = subgraphs
        template += get_subgraphs(f"{test_name}_sub", dataset_info.node_embed, dataset_info.edge_embed, 
                                  dataset_info.connection, param_size, max_cycle_size=max_cycle_size, 
                                  max_depth=max_subgraph_depth, single_bond=dataset_info.single_bond, 
                                  double_bond=dataset_info.double_bond, carbon=dataset_info.carbon, 
                                  atom_types=dataset_info.atom_types, aliphatic_bond=dataset_info.aliphatic_bond, 
                                  cycles=cycles, paths=paths, y_shape=y_shape, nbhoods=nbhoods, circular=circular, 
                                  collective=collective)

    return template, dataset



'''
dataset_name = ["MUTAG"]
model_name = ["gnn", "rgcn", "kgnn_local", "kgnn", "ego", "gated_gnn", "diffusion", "cw_net", "sgn"]
param_size = int
layers = int
lr = float(0, 1)
epochs = int (50, ...)
split = float(0.7, 0.8)
max_depth = int
chemical_rules = [True, True, True, True, True]  # or None

subgraphs = [True, True, True, True, True, True]
'''


def main_rci(dataset_name, model_name, param_size, layers, lr, epochs, split, max_depth, max_cycle_size,
             max_subgraph_depth,
             chemical_rules, subgraphs):
    mlflow.set_tracking_uri("file:/home/hodziemi/chem_scripts/results")
    with mlflow.start_run():
        template, dataset = init_test(dataset_name, model_name, param_size, layers, max_depth=max_depth,
                                      max_subgraph_depth=max_subgraph_depth, max_cycle_size=max_cycle_size,
                                      subgraphs=subgraphs, chem_rules=chemical_rules)
        train_loss, test_loss, eval = train_test_cycle(template, dataset, lr, epochs, split)

        mlflow.log_param("dataset", dataset_name)
        mlflow.log_param("model", model_name)
        mlflow.log_param("parameter_size", param_size)
        mlflow.log_param("num_layers", layers)
        mlflow.log_param("learning_rate", lr)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("train_test_split", split)

        if not chemical_rules:
            chemical_rules = [None, None, None, None, None]

        for i, l in enumerate(["hydrocarbons", "oxy", "nitro", "sulfuric", "relaxations"]):
            mlflow.log_param(l, chemical_rules[i])
        if not subgraphs:
            subgraphs = [None, None, None, None, None, None]

        for i, l in enumerate(["cycles", "paths", "y_shape", "nbhoods", "circular", "collective"]):
            mlflow.log_param(l, subgraphs[i])

        mlflow.log_metric("train_loss", train_loss)
        mlflow.log_metric("test_loss", test_loss)

    return train_loss


def main_opt(trial, dataset_name, model_name, chemical_rules, subgraphs, stage):

    mlflow.set_tracking_uri("http://localhost:2222")
    mlflow.set_experiment(f"chem_test_{stage}_stage - hodziemi")
    with mlflow.start_run():
        max_subgraph_depth = 0
        max_ring_size = 0
        if chemical_rules:
            chemical_rules = [trial.suggest_categorical(i, [True, False]) for i in
                              ["hydrocarbons", "oxy", "nitro", "sulfuric", "relaxations"]]
        if subgraphs:
            max_subgraph_depth = 4
            max_ring_size = 6
            subgraphs = [trial.suggest_categorical(i, [True, False]) for i in
                         ["cycles", "paths", "y_shape", "nbhoods", "circular", "collective"]]

        param_size = 3 if dataset_name == "mutag" else 4 
        layers = 3 
        if model_name == "sgn":
            max_depth = 4 if dataset_name == "mutag" else 3
        elif model_name == "cw_net":
            max_depth = 8 if dataset_name == "mutag" else 7
        elif model_name == "diffusion":
            max_depth = 2 if dataset_name == "mutag" else 3
        else:
            max_depth = 1

        lr = 0.005
        epochs = 200
        split = 0.7
        
        if stage == "prepare":
            param_size = trial.suggest_int("param_size", 2, 10)
            layers = trial.suggest_int("layers", 1, 5)
            if model_name in ("sgn", "diffusion"):
                max_depth = trial.suggest_int("max_depth", 1, 5)
            elif model_name == "cw_net":
                max_depth = trial.suggest_int("max_depth", 1, 10)

        template, dataset = init_test(dataset_name, model_name, param_size, layers, max_depth, max_subgraph_depth,
                                      max_ring_size, subgraphs=subgraphs, chem_rules=chemical_rules)
        if model_name == "gated_gnn":
            print(template)
        train_loss, test_loss, eval = train_test_cycle(template, dataset, lr, epochs, split)
        
        mlflow.log_param("dataset", dataset_name)
        mlflow.log_param("model", model_name)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("parameter_size", param_size)
        mlflow.log_param("num_layers", layers)
        mlflow.log_param("learning_rate", lr)
        # for chemical rules
        if not chemical_rules:
            mlflow.log_param("chem_rules", None)
        else:
            for i, l in enumerate(["hydrocarbons", "oxy", "nitro", "sulfuric", "relaxations"]):
                mlflow.log_param(l, chemical_rules[i])
        if not subgraphs:
            mlflow.log_param("subgraphs", None)
        else:
            mlflow.log_param("subgraph_depth", max_subgraph_depth)
            mlflow.log_param("ring_size", max_ring_size)
            for i, l in enumerate(["cycles", "paths", "y_shape", "nbhoods", "circular", "collective"]):
                mlflow.log_param(l, subgraphs[i])

        mlflow.log_metric("train_loss", train_loss)
        mlflow.log_metric("test_loss", test_loss)

    return test_loss
