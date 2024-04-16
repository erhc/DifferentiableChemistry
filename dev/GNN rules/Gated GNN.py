# Databricks notebook source
# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

! pip install neuralogic
from neuralogic.nn import get_evaluator
from neuralogic.core import R, Template, V, Transformation, Aggregation, Combination
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
# MAGIC ## Gated GNN rules

# COMMAND ----------

def get_gated_gnn(layer_name, node_embed, edge_embed, connection, param_size, max_depth=1):
  template = []
  template += [(R.get(layer_name + "_h")(V.X, 0) <= (R.get(node_embed)(V.X)[param_size]))]

  template += [(R.get(layer_name + "_a_in")(V.X, V.T) <= (R.get(connection)(V.X, V.Y, V.B), 
                                                          R.get(edge_embed)(V.B)[param_size], 
                                                          R.get(layer_name + "_h")(V.Y, V.Z)[param_size], 
                                                          R.special.next(V.Z, V.T))) | [Aggregation.SUM, Transformation.IDENTITY]]
  
  template += [(R.get(layer_name + "_a_out")(V.X, V.T) <= (R.get(connection)(V.X, V.Y, V.B), 
                                                           R.get(edge_embed)(V.B)[param_size], 
                                                           R.get(layer_name + "_h")(V.Y, V.Z)[param_size], 
                                                           R.special.next(V.Z, V.T))) | [Aggregation.SUM, Transformation.IDENTITY]]


  template += [(R.get(layer_name + "_update_gate")(V.X, V.T) <= (R.get(layer_name + "_a_in")(V.X, V.T)[param_size], 
                                                                 R.get(layer_name + "_a_out")(V.X, V.T)[param_size], 
                                                                 R.get(layer_name + "_h")(V.X, V.Z)[param_size], 
                                                                 R.special.next(V.Z, V.T))) | [Transformation.SIGMOID]]

  template += [(R.get(layer_name + "_reset_gate")(V.X, V.T) <= (R.get(layer_name + "_a_in")(V.X, V.T)[param_size], 
                                                                R.get(layer_name + "_a_out")(V.X, V.T)[param_size], 
                                                                R.get(layer_name + "_h")(V.X, V.Z)[param_size], 
                                                                R.special.next(V.Z, V.T))) | [Transformation.SIGMOID]]


  template += [(R.get(layer_name + "_h_tright")(V.X, V.T) <= (R.get(layer_name + "_reset_gate")(V.X, V.T),
                                                              R.get(layer_name + "_h")(V.X, V.Z),
                                                              R.special.next(V.Z, V.T))) | [Transformation.IDENTITY, Combination.ELPRODUCT]]

  template += [(R.get(layer_name + "_h_tilde")(V.X, V.T) <= (R.get(layer_name + "_a_in")(V.X, V.T)[param_size],
                                                             R.get(layer_name + "_a_out")(V.X, V.T)[param_size],
                                                             R.get(layer_name + "_h_tright")(V.X, V.T)[param_size])) | [Transformation.TANH, Aggregation.SUM]]

  template += [(R.get(layer_name + "_h_right")(V.X, V.T) <= (R.get(layer_name + "_update_gate")(V.X, V.T),
                                                             R.get(layer_name + "_h_tilde")(V.X, V.T))) | [Transformation.IDENTITY, Combination.ELPRODUCT]]

  template += [(R.get(layer_name + "_h_left")(V.X, V.T) <= (R.get(layer_name + "_update_gate")(V.X, V.T),
                                                            R.get(layer_name + "_h")(V.X, V.Z),
                                                            R.special.next(V.Z, V.T))) | [Transformation.IDENTITY, Combination.ELPRODUCT]]

  template += [(R.get(layer_name + "_h")(V.X, V.T) <= (R.get(layer_name + "_h_left")(V.X, V.T),
                                                       R.get(layer_name + "_h_right")(V.X, V.T))) | [Aggregation.SUM, Transformation.IDENTITY]]
  
  for i in range(max_depth):
    template += [(R._next(i, i + 1))]

  template += [(R.get(layer_name)(V.X) <= R.get(layer_name + "_h")(V.X, max_depth))]
  return template

# COMMAND ----------

max_depth = 10
#template += (R._next(i, i + 1) for i in range(max_depth +1))
template += get_gated_gnn("l1", "atom_embed", "bond_embed", "bond", (3, 3), max_depth=max_depth)
template += (R.predict[1, 3] <= R.l1(V.X))

# COMMAND ----------

################## CONTROL #######################

template += R.get("h")(V.X, 0) <= (R.get("atom_embed")(V.X)[3,3])

template += (R.get("activation_in")(V.X, V.T) <= (R.get("bond")(V.X, V.Y, V.B), R.get("bond_embed")(V.B)[3,3], R.get("h")(V.Y, V.Z)[3,3], R.special.next(V.Z, V.T))) | [Aggregation.SUM, Transformation.IDENTITY]  # [D,] = infeatures(V.X, V.Y)
template += (R.get("activation_out")(V.X, V.T) <= (R.get("bond")(V.X, V.Y, V.B), R.get("bond_embed")(V.B)[3,3], R.get("h")(V.Y, V.Z)[3,3], R.special.next(V.Z, V.T))) | [Aggregation.SUM, Transformation.IDENTITY] # [D,] = outfeatures(V.X, V.Y)


template += (R.get("update_gate")(V.X, V.T) <= (R.get("activation_in")(V.X, V.T)[3, 3], R.get("activation_out")(V.X, V.T)[3, 3], R.get("h")(V.X, V.Z)[3, 3], R.special.next(V.Z, V.T))) | [Transformation.SIGMOID]

template += (R.get("reset_gate")(V.X, V.T) <= (R.get("activation_in")(V.X, V.T)[3, 3], R.get("activation_out")(V.X, V.T)[3, 3], R.get("h")(V.X, V.Z)[3, 3], R.special.next(V.Z, V.T))) | [Transformation.SIGMOID]


template += (R.get("h_tright")(V.X, V.T) <= (R.get("reset_gate")(V.X, V.T), R.get("h")(V.X, V.Z), R.special.next(V.Z, V.T))) | [Transformation.IDENTITY, Combination.ELPRODUCT]

template += (R.get("h_tilde")(V.X, V.T) <= (R.get("activation_in")(V.X, V.T)[3, 3], R.get("activation_out")(V.X, V.T)[3, 3], R.get("h_tright")(V.X, V.T)[3, 3])) | [Transformation.TANH, Aggregation.SUM]

template += (R.get("h_right")(V.X, V.T) <= (R.get("update_gate")(V.X, V.T), R.get("h_tilde")(V.X, V.T))) | [Transformation.IDENTITY, Combination.ELPRODUCT]

template += (R.get("h_left")(V.X, V.T) <= (R.get("update_gate")(V.X, V.T), R.get("h")(V.X, V.Z), R.special.next(V.Z, V.T))) | [Transformation.IDENTITY, Combination.ELPRODUCT]

template += (R.get("h")(V.X, V.T) <= (R.get("h_left")(V.X, V.T), R.get("h_right")(V.X, V.T))) | [Aggregation.SUM, Transformation.IDENTITY]

# COMMAND ----------

max_depth = 10
template += (R._next(i, i + 1) for i in range(max_depth))
template += (R.l1(V.X) <= R.h(V.X, max_depth))
template += (R.predict[1, 3] <= R.l1(V.X))


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