# Databricks notebook source
# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

! pip install neuralogic
from neuralogic.core import Template, R, V
from neuralogic.dataset import Dataset

# COMMAND ----------

template = Template()

# GNN-like graph propagation
template +=  R.layer_1(V.X) <= (R.bond(V.X, V.Y, V.B))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Dataset definition

# COMMAND ----------

train_dataset = Dataset()
train_dataset.add_example(
    [
        R.o(1), R.h(2), R.h(3), # water
        R.bond(1, 2, 10), R.bond(1, 3, 11),
        R.b_1(10), R.b_1(11)
    ],
)
train_dataset.add_example(
    [
        R.o(1), R.h(2), R.h(3), R.h(4), R.h(5), R.c(6), # methanol
        R.bond(1, 2, 10), R.bond(6, 3, 11), R.bond(6, 4, 12), R.bond(6, 5, 13), R.bond(1, 6, 14),
        R.b_1(10), R.b_1(11), R.b_1(12), R.b_1(13), R.b_1(14)
    ],
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Bonding rules

# COMMAND ----------

template += R.bond_message(V.X, V.Y, V.B) <= (R.atom_embed(V.X)[3, 3], R.atom_embed(V.Y)[3, 3], R.bond_embed(V.B)[3,3])

template += R.single_bonded(V.X, V.Y) <= (R.bond(V.X, V.Y, V.B), R.b_1(V.B), R.bond_message(V.X, V.Y, V.B))
template += R.single_bonded(V.X, V.Y, V.B) <= (R.bond(V.X, V.Y, V.B), R.b_1(V.B), R.bond_message(V.X, V.Y, V.B))

template += R.double_bonded(V.X, V.Y) <= (R.bond(V.X, V.Y, V.B), R.b_2(V.B), R.bond_message(V.X, V.Y, V.B))
template += R.double_bonded(V.X, V.Y, V.B) <= (R.bond(V.X, V.Y, V.B), R.b_2(V.B), R.bond_message(V.X, V.Y, V.B))

template += R.triple_bonded(V.X, V.Y) <= (R.bond(V.Y, V.X, V.B), R.b_3(V.B), R.bond_message(V.X, V.Y, V.B))
template += R.triple_bonded(V.X, V.Y, V.B) <= (R.bond(V.Y, V.X, V.B), R.b_3(V.B), R.bond_message(V.X, V.Y, V.B))

template += R.aromatic_bonded(V.X, V.Y) <= (R.bond(V.X, V.Y, V.B), R.b_7(V.B), R.bond_message(V.X, V.Y, V.B)) # maybe also 4, 5 - contained on furan ring both on same oxygen
template += R.aromatic_bonded(V.X, V.Y, V.B) <= (R.bond(V.X, V.Y, V.B), R.b_7(V.B), R.bond_message(V.X, V.Y, V.B))

# COMMAND ----------

# define contains(Molecule, Atom)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Saturation

# COMMAND ----------

from neuralogic.core import Aggregation

template += R.saturated(V.X) <= (R.c(V.X),
                                 R.single_bonded(V.X, V.Y1),
                                 R.single_bonded(V.X, V.Y2),
                                 R.single_bonded(V.X, V.Y3),
                                 R.single_bonded(V.X, V.Y4),
                                 R.special.alldiff(...))
template += (R.saturated <= (R.saturated(V.X))) | [Aggregation.MIN]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Predicition

# COMMAND ----------

from neuralogic.core.constructs.function.reshape import Transformation
template += R.predict[1,] <= R.layer_1(V.X) 
template += R.predict[1,] <= R.saturated

template += R.predict | [Transformation.SIGMOID]

train_dataset.add_queries([
    R.predict[0],
    R.predict[1],
])
print(template)
template.draw()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Training

# COMMAND ----------

from neuralogic.core import Settings
from neuralogic.nn.loss import MSE, CrossEntropy
from neuralogic.nn import get_evaluator
from neuralogic.optim import Adam

settings = Settings(optimizer=Adam(lr=0.1), epochs=100, error_function=CrossEntropy())#, chain_pruning=True, iso_value_compression=True, prune_only_identities=True)
evaluator = get_evaluator(template, settings)

built_dataset = evaluator.build_dataset(train_dataset)

# COMMAND ----------

built_dataset[1].draw()

# COMMAND ----------

from IPython.display import clear_output
import matplotlib.pyplot as plt

average_losses = []

for current_total_loss, number_of_samples in evaluator.train(built_dataset):
    clear_output(wait=True)
    plt.ylabel("Loss")
    plt.xlabel("Epoch")

    plt.xlim(0, settings.epochs)

    average_losses.append(current_total_loss / number_of_samples)
    
    plt.plot(average_losses, label="Average loss")

    plt.legend()
    plt.pause(0.001)
    plt.show()

# COMMAND ----------

for sample, y_hat in zip(built_dataset.samples, evaluator.test(built_dataset)):
    print(f"Target: {sample.java_sample.target}, Predicted: {round(y_hat)} ({y_hat})")

# COMMAND ----------

