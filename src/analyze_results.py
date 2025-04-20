# Databricks notebook source
# MAGIC %pip install neuralogic==0.7.16 optuna==3.6.1 torch==1.13.1 torch_geometric==2.5.3 mlflow seaborn

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns


# Retrieve all runs from the experiment
runs = mlflow.search_runs(experiment_ids="2185254781789756")
mutagen_runs = runs[
    (runs['params.dataset'] == 'mutagen') #& (runs['params.learning_rate'] == '0.0005')
]

# # Check for duplicate rows for specified columns
# columns_to_check = [col for col in runs.columns if 'param' in col or 'metric' in col]

# duplicates = runs.duplicated(subset=columns_to_check, keep='first')
# print(f"Number of duplicate rows: {duplicates.sum()}")

# # Remove duplicate rows
# runs = runs[~duplicates]
# runs = runs[
#     (runs['metrics.train_loss'].notnull()) & (runs['metrics.test_loss'].notnull())
# ]


# bare_runs = runs[runs['params.circular'].isnull()]
# full_runs = runs[runs['params.circular'].notnull()]


# models = ['gnn', 'rgcn', 'kgnn', 'kgnn_local', 'ego', 'diffusion', 'cw', 'sgn']

# for m in models:
#     print(
#         f"{m}: ({len(bare_runs[bare_runs['params.model'] == m])}, {len(full_runs[full_runs['params.model'] == m])})"
#     )

# COMMAND ----------

print(len(mutagen_runs))
print(len(mutagen_runs[(mutagen_runs['metrics.test_loss'] < 0.3) & (mutagen_runs['metrics.train_loss'] < 0.3)]))

# COMMAND ----------

bad_runs = mutagen_runs[(mutagen_runs['metrics.test_loss'] > 0.3) & (mutagen_runs['metrics.train_loss'] > 0.3)][['metrics.train_loss', 'metrics.test_loss', 'params.model', 'params.architecture']]

# Plotting train loss statistics per model and architecture
sns.boxplot(data=bad_runs, x='params.model', y='metrics.train_loss', hue='params.architecture')
plt.xlabel('Model')
plt.ylabel('Train Loss')
plt.title('Train Loss Statistics per Model and Architecture')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

# Plotting test loss statistics per model and architecture
sns.boxplot(data=bad_runs, x='params.model', y='metrics.test_loss', hue='params.architecture')
plt.xlabel('Model')
plt.ylabel('Test Loss')
plt.title('Test Loss Statistics per Model and Architecture')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

# COMMAND ----------

melted_df

# COMMAND ----------

mutagen_runs = pd.concat([
    mutagen_runs.assign(loss_value=mutagen_runs['metrics.train_loss'], loss_type='Train'),
    mutagen_runs.assign(loss_value=mutagen_runs['metrics.test_loss'], loss_type='Test')
])

# Create the grouped bar chart
g = sns.catplot(data=mutagen_runs, kind="box", x="loss_type", y="loss_value", hue="architecture_type", col="params.model", aspect=1, showfliers=False)
g.set_axis_labels("Models", "Loss")
g.fig.suptitle('Comparison of Train and Test Loss by Model and Architecture', y=1.05)

plt.show()


# COMMAND ----------

mutagen_runs = runs[
    (runs['params.dataset'] == 'mutagen')
]
# mutagen_runs = mutagen_runs[(mutagen_runs['metrics.test_loss'] < 0.3) & (mutagen_runs['metrics.train_loss'] < 0.3)]

# Adding a new column based on the architecture
mutagen_runs['architecture_type'] = mutagen_runs['params.architecture'].apply(lambda x: 'bare' if x == 'parallel' else 'enhanced')

# mutagen_runs = pd.concat([
#     mutagen_runs.assign(loss_value=mutagen_runs['metrics.train_loss'], loss_type='Train'),
#     mutagen_runs.assign(loss_value=mutagen_runs['metrics.test_loss'], loss_type='Test')
# ])
# mutagen_runs['model_archtype'] = mutagen_runs[['params.model', 'loss_type']].apply(lambda x: f"{x['params.model']} {x['loss_type']}", axis=1)

# mutagen_runs = mutagen_runs.sort_values(by='model_archtype')
# # Plotting train loss vs test loss for each model for bare runs as a boxplot without outliers
# plt.figure(figsize=(10, 6))
# sns.boxplot(data=mutagen_runs, x='model_archtype', y='loss_value', hue='architecture_type', showfliers=False)
# plt.xlabel('Model')
# plt.xticks(rotation='vertical')
# plt.ylabel('Loss')
# plt.title('Train Loss per Architecture Type')
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.show()

# Plotting train loss vs test loss for each model for bare runs as a boxplot without outliers
plt.figure(figsize=(10, 6))
sns.boxplot(data=mutagen_runs, x='params.model', y='metrics.train_loss', hue='architecture_type', showfliers=False)
plt.xlabel('Model')
plt.ylabel('Loss')
plt.title('Train Loss per Architecture Type')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

# Plotting test loss for each model for bare runs as a boxplot without outliers
plt.figure(figsize=(10, 6))
sns.boxplot(data=mutagen_runs, x='params.model', y='metrics.test_loss', hue='architecture_type', showfliers=False)
plt.xlabel('Model')
plt.ylabel('Loss')
plt.title('Test Loss per Architecture Type')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()


# Plotting train loss vs test loss for each model for bare runs as a boxplot without outliers
plt.figure(figsize=(10, 6))
sns.boxplot(data=mutagen_runs[mutagen_runs['architecture_type'] == 'enhanced'], x='params.model', y='metrics.train_loss', hue='params.architecture', showfliers=False)
plt.xlabel('Model')
plt.ylabel('Loss')
plt.title('Train Loss per Architecture Type')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

# Plotting test loss for each model for bare runs as a boxplot without outliers
plt.figure(figsize=(10, 6))
sns.boxplot(data=mutagen_runs[mutagen_runs['architecture_type'] == 'enhanced'], x='params.model', y='metrics.test_loss', hue='params.architecture', showfliers=False)
plt.xlabel('Model')
plt.ylabel('Loss')
plt.title('Test Loss per Architecture Type')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

# COMMAND ----------

# Plotting train and test loss dependency on various parameters as lines
parameters = ['params.num_layers', 'params.subgraph_depth', 'params.max_depth', 'params.parameter_size']

for param in parameters:
    plt.figure(figsize=(10, 6))
    sorted_mutagen_runs = mutagen_runs.sort_values(by=param)
    sns.lineplot(data=sorted_mutagen_runs, 
                 x=param, 
                 y='metrics.train_loss', 
                 label="Train Loss",
                #  style='params.architecture',
                 )
    sns.lineplot(data=sorted_mutagen_runs, 
                 x=param, 
                 y='metrics.test_loss', 
                 label="Test Loss",
                #  style='params.architecture',
                 )
    plt.xlabel(param)
    plt.ylabel('Loss')
    plt.title(f'Loss vs {param}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()

# COMMAND ----------

import pandas as pd

# Transforming mutagen_runs DataFrame to include loss_value and loss_type
mutagen_runs_expanded = pd.concat([
    mutagen_runs.assign(loss_value=mutagen_runs['metrics.train_loss'], loss_type='Train'),
    mutagen_runs.assign(loss_value=mutagen_runs['metrics.test_loss'], loss_type='Test')
])

parameters = [
    'params.oxy', 
    'params.nitro', 'params.y_shape',
    'params.hydrocarbons', 'params.collective',
    'params.cycles', 'params.sulfuric',
    'params.circular', 'params.nbhoods', 'params.paths',
    'params.relaxations'
]

for param in parameters:
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=mutagen_runs_expanded, 
                 x='loss_type', 
                 y="loss_value", 
                 hue=param,
                 dodge=True
                 )
    plt.xlabel('loss_type')
    plt.ylabel('Loss')
    plt.title(f'Loss vs {param}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

# COMMAND ----------

non_mutagen_runs = runs[
    (runs['params.dataset']!= 'mutagen') & (runs['params.dataset'])
    # ((runs['params.dataset'] == 'carcinogenous') | (runs['params.dataset'] == 'cyp2d6_substrate'))
]


for dataset in non_mutagen_runs['params.dataset'].unique():

    # Plotting train loss vs test loss for each model for bare runs as a boxplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=non_mutagen_runs[non_mutagen_runs['params.dataset'] == dataset], x='params.architecture', y='metrics.train_loss', showfliers=False)
    plt.xlabel('Model')
    plt.ylabel('Loss')
    plt.title(f'Train Loss per architecture on {dataset}')
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()

    # Plotting train loss vs test loss for each model for bare runs as a boxplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=non_mutagen_runs[non_mutagen_runs['params.dataset'] == dataset], x='params.architecture', y='metrics.test_loss', showfliers=False)
    plt.xlabel('Model')
    plt.ylabel('Loss')
    plt.title(f'Test Loss per architecture on {dataset}')
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()

# COMMAND ----------

# show plot that shows the accuracy per dataset, that it it effective even without added rules on various datasets
non_mutagen_runs = runs[
    (runs['params.dataset'] != 'anti_sarscov2_activity') &
    (runs['params.dataset'])
]

non_mutagen_runs = pd.concat([
    non_mutagen_runs.assign(loss_value=non_mutagen_runs['metrics.train_loss'], loss_type='Train'),
    non_mutagen_runs.assign(loss_value=non_mutagen_runs['metrics.test_loss'], loss_type='Test')
])


# Plotting train loss vs test loss for each model for bare runs as a boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(data=non_mutagen_runs, x='params.dataset', y='loss_value', hue='loss_type', showfliers=False)
plt.xlabel('Model')
plt.xticks(rotation='vertical')
plt.ylabel('Loss')
plt.title(f'Train Loss per dataset')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()



# COMMAND ----------


