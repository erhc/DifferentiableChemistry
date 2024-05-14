# Databricks notebook source
from pipeline import init_test, train_test_cycle
import mlflow

def main(trial, dataset_name, model_name, chemical_rules, subgraphs):
    with mlflow.start_run(experiment_id="562876490460634"):
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

        lr = 0.0005 # 0.005 for mutag
        epochs = 100 # 200 for mutag
        split = 0.7
        

        param_size = 6 #trial.suggest_int("param_size", 2, 10)
        layers = 2 #trial.suggest_int("layers", 1, 5)
        if model_name in ("sgn", "diffusion"):
            max_depth = trial.suggest_int("max_depth", 1, 5)
        elif model_name == "cw_net":
            max_depth = trial.suggest_int("max_depth", 1, 10)

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


# COMMAND ----------

from functools import partial
import optuna
from multiprocessing import Pool

models = [
    "gnn",
    "rgcn",
    "kgnn",
    "kgnn_local",
    "ego",
    "diffusion",
    "cw_net",
    "sgn"
    ]

dataset_name = 'ptc_fr'
chem = 'full'

def optimize_main(model_name):
    study = optuna.create_study(study_name=f"{model_name}_{chem}_{dataset_name}", load_if_exists=True)
    c, s = (True, True)
    if chem == 'bare':
        c, s = (False, False)
    main_function = partial(main, dataset_name=dataset_name, model_name=model_name, chemical_rules=c, subgraphs=s)
    study.optimize(main_function, n_trials=200, catch=(KeyError))

pool = Pool(processes=len(models))
pool.map(optimize_main, models)
pool.close()

# COMMAND ----------

