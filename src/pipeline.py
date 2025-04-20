from typing import Union
from sklearn.metrics import roc_auc_score

import chemdiff.dataset_templates as dataset_templates
import numpy as np
from chemdiff.knowledge_base.chemrules import get_chem_rules
from chemdiff.knowledge_base.subgraphs import get_subgraphs
from chemdiff.models import get_model
from neuralogic.core import Settings, R, V
from neuralogic.nn import get_evaluator
from neuralogic.nn.loss import MSE
from neuralogic.optim import Adam
from sklearn.model_selection import train_test_split

def train_test_cycle(
    template,
    dataset,
    lr=0.001,
    epochs=100,
    split_ratio=0.75,
    optimizer=Adam,
    error_function=MSE,
    batches=1
):
    """
    Train and test the model based on the provided template and dataset.

    :param template: The template used for the evaluator.
    :param dataset: The dataset to train and test on.
    :param lr: Learning rate for the optimizer.
    :param epochs: Number of training epochs.
    :param split_ratio: The ratio to split the dataset into training and testing.
    :param optimizer: The optimizer class to be used.
    :param error_function: The error function to be used.
    :return: The training loss, testing loss, AUROC validation score and the evaluator object.
    """
    settings = Settings(
        optimizer=optimizer(lr=lr), epochs=epochs, error_function=error_function()
    )
    # print(f"Building dataset in {batches} batches")
    evaluator = get_evaluator(template, settings)
    built_dataset = evaluator.build_dataset(dataset, batch_size=batches)

    train_dataset, test_dataset = train_test_split(
        built_dataset.samples, train_size=split_ratio, random_state=42
    )
    # print("Training model")
    train_losses = train_model(evaluator, train_dataset, settings.epochs)
    test_loss, auroc_score = evaluate_model(evaluator, test_dataset)

    return np.mean(train_losses), test_loss, auroc_score, evaluator


def train_model(evaluator, train_dataset, epochs, early_stopping_rounds=10, early_stopping_threshold=0.001):
    """
    Train the model on the training dataset.

    :param evaluator: The evaluator object used for training.
    :param train_dataset: The dataset to train on.
    :param epochs: Number of training epochs.
    :return: List of average training losses per epoch.
    """
    average_losses = []
    best_loss = float('inf')
    rounds_without_improvement = 0

    for epoch in range(epochs):
        current_total_loss, number_of_samples = next(evaluator.train(train_dataset))
        train_loss = current_total_loss / number_of_samples
        average_losses.append(train_loss)

        if train_loss < best_loss - early_stopping_threshold:
            best_loss = train_loss
            rounds_without_improvement = 0
        else:
            rounds_without_improvement += 1
        # print(f"Epoch {epoch + 1}/{epochs} | Train loss: {train_loss} | Best loss: {best_loss} | Difference: {best_loss - train_loss}")

        if rounds_without_improvement >= early_stopping_rounds:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            break

    return average_losses


def evaluate_model(evaluator, test_dataset):
    """
    Evaluate the model on the test dataset.

    :param evaluator: The evaluator object used for testing.
    :param test_dataset: The dataset to test on.
    :return: The testing loss and AUROC score.
    """
    # loss = [
    #     round(y_hat) != sample.java_sample.target.value
    #     for sample, y_hat in zip(
    #         test_dataset, evaluator.test(test_dataset, generator=False)
    #     )
    # ]
    # return sum(loss) / len(test_dataset)

    predictions = []
    targets = []
    for sample, y_hat in zip(test_dataset, evaluator.test(test_dataset, generator=False)):
        predictions.append(y_hat)
        targets.append(sample.java_sample.target.value)
    
    loss = sum(round(pred) != target for pred, target in zip(predictions, targets)) / len(test_dataset)
    auroc_score = roc_auc_score(targets, predictions)
    
    return loss, auroc_score


def init_test(
    dataset_name: str,
    model_name: str,
    param_size: int,
    layers: int,
    max_depth: int = 1,
    max_subgraph_depth: int = 5,
    max_cycle_size: int = 10,
    subgraphs: Union[tuple, bool, None] = None,
    chem_rules: Union[tuple, bool, None] = None,
    test_name: str = "test",
    architecture: str = "parallel",
    examples=None,
    queries=None,
    funnel=False
):
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
    :param architecture: The architecture to use for the model. - default: "parallel" [""parallel", "featurized", "explainable"]
    :param funnel: create an informational funnel in the knowledge base. - default: False
    :return: A tuple containing the template and dataset.
    """
    template, dataset, dataset_info = dataset_templates.get_dataset(
        dataset_name, param_size, examples, queries
    )
    
    if architecture == "parallel":
        io_layers = {
            "nn_input": dataset_info.node_embed,
            "nn_output": "predict",

            "chem_input": dataset_info.node_embed,
            "chem_output": "predict",

            "subg_input": dataset_info.node_embed,
            "subg_output": "predict",
            }
    elif architecture == "featurized":
        io_layers = {
            "nn_input": "kb_features",
            "nn_output": "predict",

            "chem_input": dataset_info.node_embed,
            "chem_output": "predict",

            "subg_input": dataset_info.node_embed,
            "subg_output": "predict",
            }
        template += [
            (R.get("kb_features")(V.X) <= R.get(dataset_info.node_embed)(V.X)),
            (R.get("kb_features")(V.X) <= R.get(f"{test_name}_sub_subgraph_pattern")(V.X)),
            (R.get("kb_features")(V.X) <= R.get(f"{test_name}_chem_chem_rules")(V.X))
            ]
        
    elif architecture == "explainable":
        io_layers = {
            "nn_input": dataset_info.node_embed,
            "nn_output": "predict",

            "chem_input": "kb_features",
            "chem_output": "predict",

            "subg_input": "kb_features",
            "subg_output": "predict",
            }
        template += [
            (R.get("kb_features")(V.X) <= R.get(dataset_info.node_embed)(V.X)),
            (R.get("kb_features")(V.X) <= R.get(f"{test_name}_{model_name.split('_')[0]}")(V.X))
            ]
    else:
        raise ValueError(f"Invalid architecture: {architecture}. Please use one of the following: ['parallel', 'featurized', 'explainable']")

    local = False
    if model_name == "kgnn_local":
        local = True
        model_name = "kgnn"

    template += get_model(
        model_name,
        test_name,
        layers,
        io_layers["nn_input"],
        dataset_info.edge_embed,
        dataset_info.connection,
        param_size,
        edge_types=dataset_info.bond_types,
        max_depth=max_depth,
        local=local,
        output_layer_name=io_layers["nn_output"],
    )

    if chem_rules:
        try:
            hydrocarbons, oxy, nitro, sulfuric, relaxations = chem_rules
        except:
            hydrocarbons, oxy, nitro, sulfuric, relaxations = (True, ) * 5

        chem_path = f"{test_name}_sub_path" if subgraphs and ((type(subgraphs) in (list, tuple) and subgraphs[1]) or subgraphs) else None

        template += get_chem_rules(
            f"{test_name}_chem",
            io_layers["chem_input"],
            dataset_info.edge_embed,
            dataset_info.connection,
            param_size,
            dataset_info.halogens,
            output_layer_name=io_layers["chem_output"],
            single_bond=dataset_info.single_bond,
            double_bond=dataset_info.double_bond,
            triple_bond=dataset_info.triple_bond,
            aromatic_bonds=dataset_info.aromatic_bonds,
            carbon=dataset_info.carbon,
            hydrogen=dataset_info.hydrogen,
            oxygen=dataset_info.oxygen,
            nitrogen=dataset_info.nitrogen,
            sulfur=dataset_info.sulfur,
            path=chem_path,
            hydrocarbons=hydrocarbons,
            nitro=nitro,
            sulfuric=sulfuric,
            oxy=oxy,
            relaxations=relaxations,
            key_atoms=dataset_info.key_atom_type,
            funnel=funnel,
        )

    if subgraphs:
        try:
            cycles, paths, y_shape, nbhoods, circular, collective = subgraphs
        except:
            cycles, paths, y_shape, nbhoods, circular, collective = (True, ) * 6
        
        template += get_subgraphs(
            f"{test_name}_sub",
            io_layers["subg_input"],
            dataset_info.edge_embed,
            dataset_info.connection,
            param_size,
            max_cycle_size=max_cycle_size,
            max_depth=max_subgraph_depth,
            output_layer_name=io_layers["subg_output"],
            single_bond=dataset_info.single_bond,
            double_bond=dataset_info.double_bond,
            carbon=dataset_info.carbon,
            atom_types=dataset_info.atom_types,
            aliphatic_bond=dataset_info.aliphatic_bond,
            cycles=cycles,
            paths=paths,
            y_shape=y_shape,
            nbhoods=nbhoods,
            circular=circular,
            collective=collective,
            funnel=funnel,
        )

    return template, dataset
