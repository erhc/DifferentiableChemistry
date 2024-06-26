# Databricks notebook source
# MAGIC %sh apt-get -y install graphviz

# COMMAND ----------

from pipeline import init_test, train_test_cycle
import mlflow

test_number, dataset_name, model_name, param_size, layers, lr, epochs, split, max_depth, max_subgraph_depth, max_ring_size = (0, "mutagen", "gnn", 1, 2, 0.005, 200, 0.7, 3, 5, 7)

subgraphs = False
chemical_rules = False

template, dataset = init_test(dataset_name, model_name, param_size, 1, 1, max_subgraph_depth,
                                      max_ring_size, subgraphs=subgraphs, chem_rules=chemical_rules)
train_loss, test_loss, evaluator = train_test_cycle(template, dataset, lr, epochs, split)

# COMMAND ----------

(train_loss, test_loss)
# Base: (0.13667614718541443, 0.21052631578947367)
# KB: (0.03685028514419569, 0.17543859649122806), (0.06960871065204183, 0.10526315789473684)
# KB & implicit h: (0.059649797099622474, 0.15789473684210525), (0.07573930347188278, 0.21052631578947367)
# NO KB, implicit h: (0.16030929646730002, 0.2982456140350877), (0.1545645257555831, 0.2982456140350877)

# COMMAND ----------

#template.draw(filename="all_rules.png")
template.draw()

# COMMAND ----------

# MAGIC %pdb on

# COMMAND ----------

(train_loss, test_loss)

# COMMAND ----------

# MAGIC %sh ls /miniconda3/envs/adoption-hodzice_chem_env_2/bin/python

# COMMAND ----------

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

        lr = 0.005
        epochs = 200
        split = 0.7
        

        param_size = trial.suggest_int("param_size", 2, 10)
        layers = trial.suggest_int("layers", 1, 5)
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

dataset_name = 'mutagen'

def optimize_main(model_name):
    study = optuna.create_study(study_name=model_name, load_if_exists=True)
    main_function = partial(main, dataset_name=dataset_name, model_name=model_name, chemical_rules=True, subgraphs=True)
    study.optimize(main_function, n_trials=200, catch=(KeyError))

pool = Pool(processes=len(models))
pool.map(optimize_main, models)
pool.close()

# COMMAND ----------

from neuralogic.core import Settings
from neuralogic.nn.loss import MSE
from neuralogic.nn import get_evaluator
from neuralogic.optim import Adam

settings = Settings(optimizer=Adam(lr=lr), epochs=epochs, error_function=MSE(), iso_value_compression=False, chain_pruning=False)
evaluator = get_evaluator(template, settings)

built_dataset = evaluator.build_dataset(dataset)

# COMMAND ----------

built_dataset[-1].draw()

# COMMAND ----------

built_dataset[1].draw()

# COMMAND ----------

built_dataset[0].draw()

# COMMAND ----------

