# Databricks notebook source
# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

! pip install neuralogic
from neuralogic.nn import get_evaluator
from neuralogic.core import R, Template, V
from neuralogic.core.settings import Settings
from neuralogic.dataset import Dataset
from neuralogic.optim import SGD
from neuralogic.utils.data import Mutagenesis

# COMMAND ----------

# MAGIC %md
# MAGIC ## Dataset definition

# COMMAND ----------

_, dataset = Mutagenesis()

template = Template()

template.add_rules([
    (R.atom_embed(V.A)[3,] <= R.get(atom)(V.A)) for atom in ["c", "o", "br", "i", "f", "h", "n", "cl"]
])


template.add_rules([
    (R.bond_embed(V.B)[3,] <= R.get(bond)(V.B)) for bond in ["b_1", "b_2", "b_3", "b_4", "b_5", "b_7"]
])


# COMMAND ----------

# MAGIC %md
# MAGIC ##k-GNNs
# MAGIC

# COMMAND ----------

template += R.k_0(V.X) <= (R.atom_embed(V.X)[3, 3])

def get_k_set(layer_name: str, prev_layer: str, node_embed: str, param_size: tuple, edge_embed: str=None, connection: str=None, local=True):
    body = [R.get(node_embed)(V.X)[param_size],
            R.get(prev_layer)(V.Y)[param_size]]
    if local:
      body += [R.get(connection)(V.X, V.Y, V.B), 
               R.get(edge_embed)(V.B)[param_size]] 

    return R.get(layer_name)(V.X) <= body

# COMMAND ----------

# MAGIC %md
# MAGIC Local

# COMMAND ----------

max_k = 10
for i in range(1, max_k+1):
    template += get_k_set(f"k_{i}", f"k_{i-1}", "atom_embed", (3, 3), "bond_embed", "bond")

# COMMAND ----------

# MAGIC %md
# MAGIC Global

# COMMAND ----------

max_k = 10
for i in range(1, max_k+1):
    template += get_k_set(f"k_{i}", f"k_{i-1}", "atom_embed", (3, 3), local=False)

# COMMAND ----------

template += R.predict[1, 3] <= (R.get(f"k_{max_k}")(V.X))

# COMMAND ----------

# MAGIC %md
# MAGIC ## ego-GNN rules

# COMMAND ----------

def get_ego(layer_name: str, node_embed: str, edge_embed: str, connection: str, param_size: tuple):
  template = []
  template += [R.get(layer_name + "_multigraph")(V.X) <= (R.get(connection)(V.X, V.Y, V.B), R.get(edge_embed)(V.B)[param_size], R.get(node_embed)(V.Y)[param_size])]
  template += [R.get(layer_name + "_ego")(V.X) <= (R.get(connection)(V.X, V.Y, V.B), R.get(layer_name + "_multigraph")(V.Y)[param_size])]
  template += [R.get(layer_name)(V.X) <= R.get(layer_name + "_ego")(V.X)[param_size]]
  return template

'''template += R.multigraph(V.X) <= (R._bond(V.X, V.Y, V.B), R.bond_embed(V.B)[3, 3], R.atom_embed(V.Y)[3, 3])
template += R.ego_graph(V.X)[3, 3] <= (R._bond(V.X, V.Y, V.B), R.multigraph(V.Y)[3, 3])'''

template += get_ego("l1", "atom_embed", "bond_embed", "bond", (3, 3))

# COMMAND ----------

template += R.predict[1, 3] <= (R.l1(V.X))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Training

# COMMAND ----------

print(template)
template.draw()

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