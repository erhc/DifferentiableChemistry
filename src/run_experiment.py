# Databricks notebook source
# MAGIC %pip install neuralogic==0.7.16 optuna==3.6.1 torch==1.13.1 torch_geometric==2.5.3 mlflow
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("model_name", "gnn")
dbutils.widgets.text("dataset_name", "mutagen")
dbutils.widgets.text("chem", "bare")
dbutils.widgets.text("num_trials", "200")
dbutils.widgets.dropdown("architecture", "parallel", ["parallel", "explainable", "featurized"])
dbutils.widgets.text("batches", "10")

# COMMAND ----------

import os
# Change the working directory to the parent directory

# COMMAND ----------

import mlflow
from chemdiff.pipeline import init_test, train_test_cycle


def main(trial, dataset_name, model_name, chemical_rules, subgraphs, architecture="parallel", batches=1):
    with mlflow.start_run(experiment_id="2185254781789756"):
        max_subgraph_depth = 0
        max_cycle_size = 0
        if chemical_rules:
            chemical_rules = [
                trial.suggest_categorical(i, [True, False])
                for i in ["hydrocarbons", "oxy", "nitro", "sulfuric", "relaxations"]
            ]
        if subgraphs:
            max_subgraph_depth = trial.suggest_int("max_subgraph_depth", 1, 8)
            max_cycle_size = trial.suggest_int("max_cycle_size", 3, 10)
            subgraphs = [
                trial.suggest_categorical(i, [True, False])
                for i in [
                    "cycles",
                    "paths",
                    "y_shape",
                    "nbhoods",
                    "circular",
                    "collective",
                ]
            ]

        param_size = trial.suggest_int("param_size", 1, 4)
        funnel = False #trial.suggest_categorical("funnel", [True, False])
        layers = trial.suggest_int("layers", 1, 4)
        if model_name in ["sgn", "diffusion", "cw_net"]:
            max_depth = trial.suggest_int("max_depth", 2, 10)
        else:
            max_depth = 1

        lr = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True) 
        epochs = 500
        split = 0.7

        template, dataset = init_test(
            dataset_name,
            model_name,
            param_size,
            layers,
            max_depth,
            max_subgraph_depth=max_subgraph_depth,
            max_cycle_size=max_cycle_size,
            architecture=architecture,
            subgraphs=subgraphs,
            chem_rules=chemical_rules,
            funnel=funnel
        )

        train_loss, test_loss, auroc_score, evaluator = train_test_cycle(
            template, dataset, lr, epochs, split, batches=batches
        )

        mlflow.log_param("dataset", dataset_name)
        mlflow.log_param("model", model_name)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("parameter_size", param_size)
        mlflow.log_param("num_layers", layers)
        mlflow.log_param("learning_rate", lr)
        mlflow.log_param("architecture", architecture)
        mlflow.log_param("funnel", funnel)
        # for chemical rules
        if not chemical_rules:
            mlflow.log_param("chem_rules", None)
        else:
            for i, l in enumerate(
                ["hydrocarbons", "oxy", "nitro", "sulfuric", "relaxations"]
            ):
                mlflow.log_param(l, chemical_rules[i])
        if not subgraphs:
            mlflow.log_param("subgraphs", None)
        else:
            mlflow.log_param("subgraph_depth", max_subgraph_depth)
            mlflow.log_param("cycle_size", max_cycle_size)
            for i, l in enumerate(
                ["cycles", "paths", "y_shape", "nbhoods", "circular", "collective"]
            ):
                mlflow.log_param(l, subgraphs[i])

        mlflow.log_metric("train_loss", train_loss)
        mlflow.log_metric("test_loss", test_loss)
        mlflow.log_metric("auroc", auroc_score)

    return test_loss

# COMMAND ----------

model_name = dbutils.widgets.get("model_name")
dataset_name = dbutils.widgets.get("dataset_name")
chem = dbutils.widgets.get("chem")
num_trials = int(dbutils.widgets.get("num_trials"))
arch = dbutils.widgets.get("architecture")
batches = int(dbutils.widgets.get("batches"))

# COMMAND ----------

from functools import partial
from multiprocessing import Pool

import optuna

study = optuna.create_study(
    study_name=f"{model_name}_{chem}_{dataset_name}_{arch}"
)
c, s = (True, True)
if chem == 'bare':
    c, s = (False, False)
main_function = partial(
    main,
    dataset_name=dataset_name,
    model_name=model_name,
    chemical_rules=c,
    subgraphs=s,
    architecture=arch,
    batches=batches
)
study.optimize(main_function, n_trials=num_trials, catch=(KeyError))

# COMMAND ----------


