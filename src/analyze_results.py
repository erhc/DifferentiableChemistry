# Databricks notebook source
import mlflow
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# Retrieve all runs from the experiment
runs = mlflow.search_runs(experiment_ids="562876490460634")
runs = runs[(runs['params.dataset'] != 'mutagen') & (runs['params.learning_rate'] == '0.0005')]

# Check for duplicate rows for specified columns
columns_to_check = [col for col in runs.columns if 'param' in col or 'metric' in col]

duplicates = runs.duplicated(subset=columns_to_check, keep='first')
print(f"Number of duplicate rows: {duplicates.sum()}")

# Remove duplicate rows
runs = runs[~duplicates]
runs = runs[(runs['metrics.train_loss'].notnull()) & (runs['metrics.test_loss'].notnull())]


bare_runs = runs[runs['params.circular'].isnull()]
full_runs = runs[runs['params.circular'].notnull()]


models = ['gnn', 'rgcn', 'kgnn', 'kgnn_local', 'ego', 'diffusion', 'cw_net', 'sgn']

for m in models:
    print(f"{m}: ({len(bare_runs[bare_runs['params.model'] == m])}, {len(full_runs[full_runs['params.model'] == m])})")

# COMMAND ----------

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Plotting train loss vs test loss for each model for bare runs as a boxplot
sns.boxplot(data=bare_runs, x='params.model', y='metrics.train_loss', color='blue')
sns.boxplot(data=bare_runs, x='params.model', y='metrics.test_loss', color='orange')
plt.xlabel('Model')
plt.ylabel('Loss')
plt.title('Train Loss vs Test Loss for Bare Runs')
plt.show()

# Plotting train loss vs test loss for each model for full runs as a boxplot
sns.boxplot(data=full_runs, x='params.model', y='metrics.train_loss', color='blue')
sns.boxplot(data=full_runs, x='params.model', y='metrics.test_loss', color='orange')
plt.xlabel('Model')
plt.ylabel('Loss')
plt.title('Train Loss vs Test Loss for Full Runs')
plt.show()

# COMMAND ----------

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Group runs by num_layers and param_size
grouped_runs = bare_runs#[bare_runs['metrics.test_loss'] <= 0.2]

# Plot the performance of test_loss for different num_layers and param_size
sns.lineplot(data=grouped_runs, x='params.parameter_size', y='metrics.train_loss', hue='params.model')
plt.xlabel('Parameter Size')
plt.ylabel('Test Loss')
plt.title(f"Performance on Test Loss for Layers")
plt.show()

# COMMAND ----------

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Group runs by num_layers and param_size
grouped_runs = full_runs[full_runs['metrics.test_loss'] <= 0.2]

# Plot the performance of test_loss for different num_layers and param_size
sns.boxplot(data=grouped_runs, x='params.num_layers', y='metrics.test_loss', order=[str(i) for i in range(1, 6)])#, hue='params.num_layers')
plt.xlabel('Parameter Size')
plt.ylabel('Test Loss')
plt.title(f"Performance on Test Loss for Layers")
plt.show()

# COMMAND ----------

runs.columns

# COMMAND ----------

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Take just the best 50% from each model
top_bare = bare_runs#.groupby('params.model').apply(lambda x: x.nsmallest(50, 'metrics.test_loss'))
top_full = full_runs#.groupby('params.model').apply(lambda x: x.nsmallest(50, 'metrics.test_loss'))

# Join the top 50% datasets
top_bare['Dataset'] = 'Without KB'
top_full['Dataset'] = 'With KB'

merged_data = pd.concat([top_bare, top_full])

# Plot the test loss using boxplot for both datasets
sns.boxplot(x='params.model', y='metrics.test_loss', hue='Dataset', data=merged_data, palette='Set2')
plt.xlabel('Model')
plt.ylabel('Test Loss')
plt.title('Test Loss for Bare Runs vs Full Runs')
plt.legend()
plt.show()

# COMMAND ----------

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Retrieve all runs from the experiment
# runs = mlflow.search_runs(experiment_ids="562876490460634")

# Plot the train and test loss using boxplot for the top 10% of each model
sns.boxplot(x='params.model', y='loss', hue='variable', data=pd.melt(bare_runs, id_vars=['params.model'], value_vars=['metrics.train_loss', 'metrics.test_loss'],var_name='variable', value_name='loss'), palette='Set2')
plt.xlabel('Model')
plt.ylabel('Loss')
plt.title('Train and Test Loss per Model')
plt.show()

# COMMAND ----------

import numpy as np
runs = mlflow.search_runs(experiment_ids="562876490460634")
bare_runs = runs[(runs['metrics.train_loss'].notnull()) & (runs['metrics.test_loss'].notnull()) & (runs['params.circular'].isnull())]
np.unique(bare_runs['params.model'])

# COMMAND ----------

pd.melt(bare_runs, id_vars=['run_id'], value_vars=['metrics.train_loss', 'metrics.test_loss'], var_name='variable', value_name='value')

# COMMAND ----------

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

runs = mlflow.search_runs(experiment_ids="562876490460634")
bare_runs = runs[(runs['metrics.train_loss'].notnull()) & (runs['metrics.test_loss'].notnull()) & (runs['params.circular'].isnull())]
full_runs = runs[(runs['metrics.train_loss'].notnull()) & (runs['metrics.test_loss'].notnull()) & (runs['params.circular'].notnull())]

# Combine bare_runs and full_runs into a single dataframe
combined_runs_df = pd.concat([bare_runs, full_runs])

# Melt the combined dataframe
combined_runs_melted = pd.melt(combined_runs_df, id_vars=['params.model'], value_vars=['metrics.test_loss'], var_name='variable', value_name='value')

# Plot the combined dataframe
sns.boxplot(x='params.model', y='value', hue='param.subgraphs', data=combined_runs_melted, palette='Set2')
plt.xlabel('Model')
plt.ylabel('Test Loss')
plt.title('Test Loss on Combined Runs per Model')
plt.show()

# COMMAND ----------

import mlflow
import pandas as pd

# Retrieve all runs from the experiment
runs = mlflow.search_runs(experiment_ids="562876490460634")

# Filter runs for each model
models = ['gnn', 'rgcn', 'kgnn', 'kgnn_local', 'ego', 'diffusion', 'cw_net', 'sgn']
model_runs = {}
for model in models:
    model_runs[model] = runs[runs['params.model'] == model]

# Find best hyperparameter ranges for each model
best_hyperparameter_ranges = {}
for model, runs in model_runs.items():
    best_runs = runs.nsmallest(5, 'metrics.test_loss')
    best_hyperparameter_ranges[model] = {
        'param_size': (best_runs['params.parameter_size'].min(), best_runs['params.parameter_size'].max()),
        'num_layers': (best_runs['params.num_layers'].min(), best_runs['params.num_layers'].max()),
        'learning_rate': (best_runs['params.learning_rate'].min(), best_runs['params.learning_rate'].max()),
    }

# Convert results to pandas DataFrame
results_df = pd.DataFrame(best_hyperparameter_ranges)

# Save results to a CSV file
results_df.to_csv('hyperparameter_ranges.csv', index=False)

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

        lr = 0.0005 # 0.005 for mutag
        epochs = 10
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

# freeze on param_size 6 and num_layers 2

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
chem = 'bare'

def optimize_main(model_name):
    study = optuna.create_study(study_name=f"{model_name}_{chem}_{dataset_name}", load_if_exists=True)
    c, s = (True, True)
    if chem == 'bare':
        c, s = (False, False)
    main_function = partial(main, dataset_name=dataset_name, model_name=model_name, chemical_rules=c, subgraphs=s)
    study.optimize(main_function, n_trials=100, catch=(KeyError))

pool = Pool(processes=len(models))
pool.map(optimize_main, models)
pool.close()

# COMMAND ----------

