# Databricks notebook source
! pip install neuralogic

# COMMAND ----------

from neuralogic.nn import get_evaluator
from neuralogic.core import R, Template, V
from neuralogic.core.settings import Settings
from neuralogic.dataset import Dataset
from neuralogic.optim import SGD

# COMMAND ----------

train_dataset = Dataset()
template = Template()

template.add_rules([
    # Captures triangle
    R.triangle(V.X)[1,] <= (
        R.edge(V.X, V.Y), R.feature(V.Y)[1,], # why 1
        R.edge(V.Y, V.Z), R.feature(V.Z)[1,],
        R.edge(V.Z, V.X), R.feature(V.X)[1,],
    ),

    # Captures general graph
    R.general(V.X)[1,] <= (R.edge(V.Y, V.X), R.feature(V.Y)[1,]),
    R.general(V.X)[1,] <= R.feature(V.X)[1,],

    R.predict <= R.general(V.X)[1,],
    R.predict <= R.triangle(V.X)[1,],
])

train_dataset.add_example(
    [
        R.edge(1, 2), R.edge(2, 3), R.edge(3, 4), R.edge(4, 1),
        R.edge(2, 1), R.edge(3, 2), R.edge(4, 3), R.edge(1, 4),
        R.edge(1, 6), R.edge(3, 6), R.edge(4, 5), R.edge(2, 5),
        R.edge(6, 1), R.edge(6, 3), R.edge(5, 4), R.edge(5, 2),

        R.feature(1), R.feature(2), R.feature(3),
        R.feature(4), R.feature(5), R.feature(6),
    ],
)

train_dataset.add_example(
    [
        R.edge(1, 2), R.edge(2, 3), R.edge(3, 4), R.edge(4, 1),
        R.edge(2, 1), R.edge(3, 2), R.edge(4, 3), R.edge(1, 4),
        R.edge(1, 6), R.edge(4, 6), R.edge(3, 5), R.edge(2, 5),
        R.edge(6, 1), R.edge(6, 4), R.edge(5, 3), R.edge(5, 2),

        R.feature(1), R.feature(2), R.feature(3),
        R.feature(4), R.feature(5), R.feature(6),
    ],
)

train_dataset.add_queries([
    R.predict[1],
    R.predict[0],
])

# COMMAND ----------

# MAGIC %md
# MAGIC # Gated GNN rules

# COMMAND ----------

# g_and = sum over all neighbors, in/outfeatures is learnable vector for R.feature shared among neighbors but not between nodes
template += R.activation_in(V.X) <= (R.edge(V.X, V.Y), R.feature(V.Y)[1,]) # for infeatures(V.X)  # when feature is a vector?
template += R.activation_out(V.X) <= (R.Aedge(V.X, V.Y), R.feature(V.Y)[1,]) # for outfeatures(V.X)

#template += R.activation(V.X) <= (R.activation_in(V.X)[1,], R.activation_out(V.Y)[1,]) # concatenated + bias

# COMMAND ----------

template += R.update_gate(V.X) <= (R.activation(V.X)[1,], R.feature()[1,]) # feature from last layer, g_and = activation + feature, 
# g_or = sigmoid?
template += R.reset_gate(V.X) <= (R.activation(V.X)[1,], R.feature()[1,]) 

# how to multiply gate*features
template += R.h_tilde(V.X) <= (R.activation(V.X)[1,], R.reset_gate(V.X)*R.feature(V.X)[1,]) # tanh activation, * - elementwise
template += R.next_feature(V.X) <= ((1-R.update_gate(V.X))*R.feature(V.X), R.update_gate(V.X)*R.h_tilde(V.X)) # sum of terms


# COMMAND ----------

# MAGIC %md
# MAGIC Subgraph networks

# COMMAND ----------

# create nodes from edges, connect nodes if the original edges share a node, repeat
# first order SGN
template += R.subgraph_line(V.E1, V.E2) <= (R.edge(V.X, V.Z, V.E1), R.edge(V.Y, V.Z, V.E2)) 
template += R.subgraph_line(V.E2, V.E1) <= R.subgraph_line(V.E1, V.E2)
# for directed graphs?

# second order SGN
template += R.subgraph_triangle(V.E1, V.E2) <= (R.subgraph_line(V.E1, V.E), R.subgraph_line(V.E2, V.E))
template += R.subgraph_triangle(V.E2, V.E1) <= R.subgraph_triangle(V.E1, V.E2)
# higher orders not needed probably

# add GNN propagation rules to this?
# how to make it general for motifs

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------



# COMMAND ----------

settings = Settings(optimizer=SGD(), epochs=200)
neuralogic_evaluator = get_evaluator(template, settings)

for _ in neuralogic_evaluator.train(train_dataset):
    pass

graphs = ["a", "b"]

for graph_id, predicted in enumerate(neuralogic_evaluator.test(train_dataset)):
    print(f"Graph {graphs[graph_id]} is predicted to be class: {int(round(predicted))} | {predicted}")