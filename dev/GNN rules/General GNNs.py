# Databricks notebook source
# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

! pip install neuralogic
from neuralogic.nn import get_evaluator
from neuralogic.core import R, Template, V, Transformation, Aggregation
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
    (R.feature(V.A)[3,] <= R.get(atom)(V.A)) for atom in ["c", "o", "br", "i", "f", "h", "n", "cl"]
])

R.layer_1(V.X)[1,] <= (R.edge(V.Y, V.X), R.feature(V.Y)[1,])
R.layer_1(V.X)[1,] <= R.feature(V.X)[1,]
R.predict <= R.layer_1(V.X)[1,]

# COMMAND ----------

for i in range(3):
  template += get_gnn(f"layer_{i+1}", f"layer_{i}", "bond_embed", "bond", (3, 3))
template += R.layer_0(V.X) <= R.atom_embed(V.X)
template += R.predict[1, 3] <= R.layer_3(V.X)

# COMMAND ----------



# COMMAND ----------

# R-GCNs only sum over different types of relations
template = Template()

template.add_rules([
    (R.atom_embed(V.A)[3,] <= R.get(atom)(V.A)) for atom in ["c", "o", "br", "i", "f", "h", "n", "cl"]
])

template.add_rules([
    (R.bond_embed(V.B)[3,] <= R.get(bond)(V.B)) for bond in ["b_1", "b_2", "b_3", "b_4", "b_5", "b_7"]
])

template.add_rules([
    (R.bond_type(V.B) <= R.get(bond)(V.B)[3,]) for bond in ["b_1", "b_2", "b_3", "b_4", "b_5", "b_7"]
])

template +=  R.layer_1(V.X) <= (R.atom_embed(V.X)[3, 3], R.atom_embed(V.Y)[3, 3], R.bond(V.X, V.Y, V.B), R.bond_embed(V.B)[3, 3], R.bond_type(V.B))

template +=  R.layer_2(V.X) <= (R.layer_1(V.X)[3, 3], R.layer_1(V.Y)[3, 3], R.bond(V.X, V.Y, V.B), R.bond_embed(V.B)[3, 3], R.bond_type(V.B))
template +=  R.layer_3(V.X) <= (R.layer_2(V.X)[3, 3], R.layer_2(V.Y)[3, 3], R.bond(V.X, V.Y, V.B), R.bond_embed(V.B)[3, 3], R.bond_type(V.B))


template += R.predict[1, 3] <= R.layer_3(V.X)

def get_rgcn(layer_name: str, node_embed: str, edge_embed: str, connection: str, edge_aggregation: str, param_size: tuple):
  return (R.get(layer_name)(V.X) <= (R.get(node_embed)(V.X)[param_size],
                                     R.get(node_embed)(V.Y)[param_size],
                                     R.get(connection)(V.X, V.Y, V.B),
                                     R.get(edge_embed)(V.B),
                                     R.get(edge_aggregation)(V.B)[param_size]))

# COMMAND ----------

### CONTROL ###
template +=  R.layer_0(V.X) <= (R.atom_embed(V.X))
template +=  R.layer_1(V.X) <= (R.layer_0(V.X)[3, 3], R.layer_0(V.Y)[3, 3], R.bond(V.X, V.Y, V.B), R.bond_embed(V.B)[3, 3])

template +=  R.layer_2(V.X) <= (R.layer_1(V.X)[3, 3], R.layer_1(V.Y)[3, 3], R.bond(V.X, V.Y, V.B), R.bond_embed(V.B)[3, 3])
template +=  R.layer_3(V.X) <= (R.layer_2(V.X)[3, 3], R.layer_2(V.Y)[3, 3], R.bond(V.X, V.Y, V.B), R.bond_embed(V.B)[3, 3])


template += R.predict[1, 3] <= R.layer_3(V.X)

# COMMAND ----------

print(template)
template.draw()

# COMMAND ----------

print(template)
template.draw()

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

