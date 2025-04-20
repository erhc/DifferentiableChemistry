import mlflow
import ray
from ray import tune
from ray.tune.search.optuna import OptunaSearch
try:
    from pipeline import init_test, train_test_cycle
except:
    import os

    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

    from pipeline import init_test, train_test_cycle


def main(config):
    with mlflow.start_run(experiment_id="562876490460634"):
        # max_subgraph_depth = 0
        # max_ring_size = 0
        # if chemical_rules:
        #     chemical_rules = [trial.suggest_categorical(i, [True, False]) for i in
        #                       ["hydrocarbons", "oxy", "nitro", "sulfuric", "relaxations"]]
        # if subgraphs:
        #     max_subgraph_depth = 4
        #     max_ring_size = 6
        #     subgraphs = [trial.suggest_categorical(i, [True, False]) for i in
        #                  ["cycles", "paths", "y_shape", "nbhoods", "circular", "collective"]]

        # param_size = 3 if dataset_name == "mutag" else 4 
        # layers = 3 
        # if model_name == "sgn":
        #     max_depth = 4 if dataset_name == "mutag" else 3
        # elif model_name == "cw_net":
        #     max_depth = 8 if dataset_name == "mutag" else 7
        # elif model_name == "diffusion":
        #     max_depth = 2 if dataset_name == "mutag" else 3
        # else:
        #     max_depth = 1

        dataset_name = config["dataset_name"]
        model_name = config["model_name"]
        max_subgraph_depth = config.get("max_subgraph_depth", 0)
        max_ring_size = config.get("max_ring_size", 0)
        subgraphs = config["subgraphs"]
        chemical_rules = config["chemical_rules"]


        param_size = config['parameter_size']
        max_depth = config['max_depth']
        layers = config['num_layers']
        lr = config['learning_rate']
        # lr = 0.005
        epochs = 200
        split = 0.7

        template, dataset = init_test(dataset_name, model_name, param_size, layers, max_depth, max_subgraph_depth,
                                      max_ring_size, subgraphs=subgraphs, chem_rules=chemical_rules)
        
        train_loss, test_loss, evaluator = train_test_cycle(template, dataset, lr, epochs, split)
        
        mlflow.log_param("dataset", dataset_name)
        mlflow.log_param("model", model_name)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("parameter_size", param_size)
        mlflow.log_param("num_layers", layers)
        mlflow.log_param("learning_rate", lr)

        # for chemical rules
        # if not chemical_rules:
        #     mlflow.log_param("chem_rules", None)
        # else:
        #     for i, l in enumerate(["hydrocarbons", "oxy", "nitro", "sulfuric", "relaxations"]):
        #         mlflow.log_param(l, chemical_rules[i])
        # if not subgraphs:
        #     mlflow.log_param("subgraphs", None)
        # else:
        #     mlflow.log_param("subgraph_depth", max_subgraph_depth)
        #     mlflow.log_param("ring_size", max_ring_size)
        #     for i, l in enumerate(["cycles", "paths", "y_shape", "nbhoods", "circular", "collective"]):
        #         mlflow.log_param(l, subgraphs[i])

        mlflow.log_metric("train_loss", train_loss)
        mlflow.log_metric("test_loss", test_loss)

    return test_loss

def main_tune(hyperparameter_space, num_samples):
    tuner = tune.Tuner(
        main,
        tune_config=tune.TuneConfig(
            search_alg=OptunaSearch(
                metric="mean_loss",
                mode="min"),
            num_samples=num_samples,
        ),
        param_space=hyperparameter_space,
    )
    return tuner.fit()