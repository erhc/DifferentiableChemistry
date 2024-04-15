# Databricks notebook source
# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the CSV file into a pandas DataFrame
#data = pd.concat([pd.read_csv('mutag_runs.csv').dropna(subset=['model']), pd.read_csv('runs.csv').dropna(subset=['model'])])
data = pd.read_csv('ptcmm_runs.csv').dropna(subset=['model'])
data = data[data['model'] != "gated_gnn"]

#data = data[np.isnan(data['num_layers'])]
#data = data[data['test_loss'] < 0.8]

# COMMAND ----------

np.unique(data['num_layers'], return_counts=True)

# COMMAND ----------

subgraphs = ["cycles", "paths", "y_shape", "nbhoods", "circular", "collective"]
chem_rules = ["hydrocarbons", "oxy", "nitro", "sulfuric", "relaxations"]


# COMMAND ----------

import numpy as np
#sum(data['model'] == "kgnn")
#np.unique(data['y_shape'])
order = ["diffusion", "cw_net", "sgn", "ego", "kgnn", "kgnn_local", "rgcn", "gnn"]
print("\n\n".join([f"{model}:\n{data[data['model'] == model]['test_loss'].describe()}" for model in np.unique(data['model'])]))

# COMMAND ----------

ax = sns.boxplot(data, y="test_loss", x="model", order=order)
ax.set_xlabel("Model")
ax.set_ylabel("Test loss")
ax.set_title("Model performance on PTC MR dataset with added rules")
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

# COMMAND ----------

# make a plot for each model, bare vs rule
data2 = pd.read_csv('ptcmm_runs_bare.csv').dropna(subset=['model'])
data2 = data2[data2['model'] != "gated_gnn"]
#data2 = data2[data2['test_loss'] < 0.8]
data2 = data2.assign(rules=False)

data1 = data.assign(rules=True)
data1 = pd.concat([data1, data2], axis=0)

ax = sns.boxplot(data1, y="test_loss", x="model", order=order, hue="rules")
ax.set_xlabel("Model")
ax.set_ylabel("Test loss")
ax.set_title("Model performance on PTC MM dataset")
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

# COMMAND ----------

# ptc: 
# layers 3
# max_depth sgn 2, cw 7, dif 3
# param_size 4
# cox:
# layers 2
# params 5
# dhfr:
# layers 5
# params 7
# er:
# layers 3
# params 3

data.columns

# COMMAND ----------

sns.lineplot(data=data, x='parameter_size', y='test_loss')#, hue="model")

# COMMAND ----------

sorted_data = data.sort_values('test_loss')[:100]
print("\n\n".join([f"{model}:\n{sorted_data[sorted_data['model'] == model]['test_loss'].describe()}" for model in np.unique(data['model'])]))
individual = {r:sum(sorted_data[r]) for r in chem_rules + subgraphs}
counts = sorted_data[chem_rules + subgraphs].apply(lambda x: tuple(x), axis=1).value_counts()
counts = dict(counts)

# COMMAND ----------

from matplotlib.ticker import FuncFormatter
ax = sns.countplot(sorted_data[:20], x="model", order=order)
ax.set_xlabel("Model")
ax.set_ylabel("Count")
ax.set_title("Model distribution in the top 20 runs on PTC MM dataset")
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
formatter = FuncFormatter(lambda y, _: '{:.0f}'.format(y))
plt.gca().yaxis.set_major_formatter(formatter)

for p in ax.patches:
    height = p.get_height()
    width = p.get_width()
    x = p.get_x()
    y = p.get_y()
    ax.text(x+width/2, y+height/2, str(int(height)), ha="center", fontsize=10, color='white')
    


# COMMAND ----------

chem_data = pd.melt(sorted_data[:20], id_vars=['Start Time', "test_loss", "train_loss", "model"], value_vars=chem_rules + subgraphs, var_name="variables")
chem_data = chem_data[chem_data['value'] == True].drop(columns=['value'])
ax = sns.countplot(chem_data, x="variables", order=chem_rules + subgraphs)
ax.set_xlabel("Rule")
ax.set_ylabel("Count")
ax.set_title("Rule distribution in the top 20 runs on PTC MM dataset")
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
formatter = FuncFormatter(lambda y, _: '{:.0f}'.format(y))
plt.gca().yaxis.set_major_formatter(formatter)

for p in ax.patches:
    height = p.get_height()
    width = p.get_width()
    x = p.get_x()
    y = p.get_y()
    ax.text(x+width/2, y+height/2, str(int(height)), ha="center", fontsize=10, color='white')

# COMMAND ----------

sorted_data[chem_rules + subgraphs].apply(lambda x: tuple(x), axis=1)[:10].value_counts()

# COMMAND ----------

df_stacked = pd.melt(data, id_vars=['model'], value_vars=['train_loss', 'test_loss'],
                     var_name='Loss Type', value_name='Loss')

# Create the boxplot
ax = sns.boxplot(x='model', y='Loss', hue='Loss Type', data=df_stacked, order = order)

# Set the title and labels
plt.title('Train and test loss by model on PTC MM dataset')
plt.xlabel('Model')
plt.ylabel('Loss')
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

# COMMAND ----------

data['Generalization Error'] = abs(data['train_loss'] - data['test_loss'])

# Create the boxplot
sns.boxplot(x='model', y='Generalization Error', data=data)

# Set the title and labels
plt.title('Generalization Error by Model')
plt.xlabel('Model')
plt.ylabel('Generalization Error')


# COMMAND ----------

# best split (0.73, 0.76)
# best epochs ~175
# best lr ~0.005
# best param_size = 3
# best layers ~3

# Extract the x and y columns from the DataFrame
x = data['train_loss']
y = data['Generalization Error']

# Plot x vs y
plt.scatter(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('x vs y')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Dataset definition

# COMMAND ----------

! pip install neuralogic
! pip install torch_geometric
from neuralogic.nn import get_evaluator
from neuralogic.core import R, Template, V, Transformation, Aggregation
from neuralogic.core.settings import Settings
from neuralogic.dataset import Dataset
from neuralogic.optim import SGD
from neuralogic.utils.data import Mutagenesis

# COMMAND ----------

from torch_geometric.datasets import TUDataset
from neuralogic.dataset import TensorDataset, Data, FileDataset

ds = TUDataset(".", name="PTC_FM")

dataset = TensorDataset(data=[Data.from_pyg(data)[0] for data in ds], number_of_classes=2, 
                        one_hot_decode_edge_features=True, one_hot_decode_features=True,
                        feature_name="atom", edge_name="bond")

# COMMAND ----------

dataset.dump_to_file("./ptcfm_queries.txt", "./examples.txt")

# COMMAND ----------

with open("examples.txt") as f:
  lines = f.readlines()

# COMMAND ----------

import re

new_lines = []
for line in lines:
  preds = [p.strip(" ,\n.") for p in line.split("<1>") if p.strip(" ,\n") != ""]
  bonds = [b for b in preds if "bond" in b]
  other = set(preds) - set(bonds)

  numbers = re.findall(r'\d+', line)

  c = max(map(int, numbers)) + 1

  new_line = []
  idx = {}
  for b in bonds:
    match = re.match(r"bond_(\d+)\((\d+), (\d+)\)", b)
    x = match.group(2)
    y = match.group(3)
    t = match.group(1)
    
    if (x, y) not in idx.keys() and (y, x) not in idx.keys():
      bond = f"{c}"
      idx[(x, y)] = bond
      idx[(y, x)] = bond
      c += 1
      new_line.append(f"<1> bond({x}, {y}, {bond}),<1> bond({y}, {x}, {bond}),<1> b_{t}({bond})")


  new_lines.append(", ".join(new_line) + ",<1> " + ",<1> ".join(other) + ".")

with open("ptcfm_examples.txt", "w") as f:
  f.writelines("\n".join(new_lines))

# COMMAND ----------

dataset = FileDataset(examples_file="new_examples.txt", queries_file="queries.txt")

# COMMAND ----------

from neuralogic.core import Template, R, V

template = Template()

atom_types = [f"atom_{i}" for i in range(18)]
key_atoms = ["atom_1", "atom_2", "atom_3", "atom_7"]
bond_types = ["b_1", "b_2", "b_3", "b_0"]

template.add_rules([
    (R.bond_embed(V.B)[3,] <= R.get(bond)(V.B)) for bond in bond_types
    ])

template.add_rules([
    (R.atom_embed(V.A)[3,] <= R.get(atom)(V.A)) for atom in atom_types
    ])

template +=  R.layer_1(V.X) <= (R.atom_embed(V.X)[3, 3], R.atom_embed(V.Y)[3, 3],
                                R.bond(V.X, V.Y, V.B), R.bond_embed(V.B))

template += R.predict[1, 3] <= R.layer_1(V.X)

# COMMAND ----------

from neuralogic.core import Settings
from neuralogic.nn.loss import MSE
from neuralogic.nn import get_evaluator
from neuralogic.optim import Adam
import random
import numpy as np
from IPython.display import clear_output
import matplotlib.pyplot as plt


settings = Settings(optimizer=Adam(lr=0.01), epochs=10, error_function=MSE())
evaluator = get_evaluator(template, settings)

built_dataset = evaluator.build_dataset(dataset)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Training

# COMMAND ----------

from neuralogic.core import Settings
from neuralogic.nn.loss import MSE
from neuralogic.nn import get_evaluator
from neuralogic.optim import Adam
import random
import numpy as np
from IPython.display import clear_output
import matplotlib.pyplot as plt

def train_test_cycle(template, dataset, lr=0.001, epochs=100, split=0.75):
  settings = Settings(optimizer=Adam(lr=lr), epochs=epochs, error_function=MSE())
  evaluator = get_evaluator(template, settings)

  built_dataset = evaluator.build_dataset(dataset)
  dataset_len = len(built_dataset.samples)

  train_size = int(dataset_len*split)

  idx = random.sample(list(range(dataset_len)), train_size)
  rest = list(set(range(dataset_len)) - set(idx))
  train_dataset = np.array(built_dataset.samples)[idx]
  test_dataset = np.array(built_dataset.samples)[rest]
  average_losses = []

  for current_total_loss, number_of_samples in evaluator.train(train_dataset):
      clear_output(wait=True)
      plt.ylabel("Loss")
      plt.xlabel("Epoch")

      plt.xlim(0, settings.epochs)
      
      train_loss = current_total_loss/number_of_samples
      print(train_loss)

      average_losses.append(train_loss)
      
      plt.plot(average_losses, label="Average loss")

      plt.legend()
      plt.pause(0.001)
      plt.show()

  loss = []
  for sample, y_hat in zip(test_dataset, evaluator.test(test_dataset, generator=False)):
      loss.append(round(y_hat) != sample.java_sample.target.value)

  test_loss = sum(loss) / len(test_dataset)

  return train_loss, test_loss, evaluator


# COMMAND ----------

test_losses = []
for i in range(1):
  print("Training and testing model #{}".format(i))
  train_loss, test_loss, eval = train_test_cycle(template, dataset)
  test_losses.append(test_loss)

np.average(test_losses)

# COMMAND ----------

