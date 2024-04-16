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
        R.bond(1, 2, 10), R.bond(2,1,10),
      R.bond(1, 3, 11), R.bond(3,1,11),
        R.b_1(10), R.b_1(11)
    ],
)
train_dataset.add_example(
    [
        R.o(1), R.h(2), R.h(3), R.h(4), R.h(5), R.c(6), # methanol
        R.bond(1, 2, 10), R.bond(2,1,10),
      R.bond(6, 3, 11), R.bond(3,6,11),
      R.bond(6, 4, 12), R.bond(4,6,12),
      R.bond(6, 5, 13), R.bond(5,6,13),
     R.bond(1, 6, 14), R.bond(6, 1, 14),
        R.b_1(10), R.b_1(11), R.b_1(12), R.b_1(13), R.b_1(14)
    ],
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Bonding rules

# COMMAND ----------

## check bonding rules if working well
template += R.single_bonded(V.X, V.Y) <= (R.bond(V.X, V.Y, V.B), R.b_1(V.B))
template += R.double_bonded(V.X, V.Y) <= (R.bond(V.X, V.Y, V.B), R.b_2(V.B))
template += R.double_bonded(V.X, V.Y) <= (R.bond(V.X, V.Y, V.B), R.b_2(V.B))
template += R.triple_bonded(V.X, V.Y) <= (R.bond(V.Y, V.X, V.B), R.b_3(V.B))
template += R.aromatic_bonded(V.X, V.Y) <= (R.bond(V.X, V.Y, V.B), R.b_7(V.B)) # maybe also 4, 5 - contained on furan ring both on same oxygen

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
                                 R.special.alldiff(...)) # with alldiff it doesn't work
template += (R.saturated <= (R.saturated(V.X))) | [Aggregation.MIN]

# COMMAND ----------

# MAGIC %md
# MAGIC Alcohol

# COMMAND ----------

'''template += R.alcohol(V.Mol) <= (R.contains(V.Mol, V.C), R.alcoholic(V.C))
template += R.alcoholic(V.C) <= (R.saturated(V.C), R.o(V.O), R.h(V.H), 
                                 R.single_bonded(V.C, V.O),
                                 R.single_bonded(V.O, V.H))'''

template += R.alcoholic <= (R.saturated(V.C), R.o(V.O), R.h(V.H), 
                            R.single_bonded(V.C, V.O),
                            R.single_bonded(V.O, V.H))

# COMMAND ----------

# MAGIC %md
# MAGIC Carbonyl

# COMMAND ----------

# carbonyl group contains a carbon double bonded with oxygen
template += R.carbonyl(V.Mol) <= (R.contains(V.Mol, V.C), R.contains(V.Mol, V.O), R.carbonyl_group(V.O, V.C))
template += R.carbonyl_group(V.C, V.O) <= (R.c(V.C), R.o(V.O), R.double_bonded(V.O, V.C))
template += R.carbonyl_group(V.C, V.O, V.R1, V.R2) <= (R.c(V.C), R.o(V.O), R.double_bonded(V.O, V.C), R.single_bonded(V.C, V.R1), R.single_bonded(V.C, V.R2))

# COMMAND ----------

# MAGIC %md
# MAGIC Ketone

# COMMAND ----------

template += R.ketone(V.Mol) <= (R.contains(V.Mol, V.C), 
                                R.carbonyl_group(V.C, _, V.R1, V.R2),
                                R.c(V.R1), R.c(V.R2))

# COMMAND ----------

# MAGIC %md
# MAGIC Aldehyde

# COMMAND ----------

template += R.aldehyde(V.Mol) <= (R.contains(V.Mol, V.C), 
                                  R.carbonyl_group(V.C, _, V.R, V.H),
                                  R.c(V.R), R.h(V.H))

# COMMAND ----------

# MAGIC %md
# MAGIC Acylhalide

# COMMAND ----------

template += R.acyl_fluoride(V.Mol) <= (R.contains(V.Mol, V.C), 
                                       R.carbonyl_group(V.C, _, V.R, V.X),
                                       R.c(V.R), R.f(V.X))

template += R.acyl_chloride(V.Mol) <= (R.contains(V.Mol, V.C), 
                                       R.carbonyl_group(V.C, _, V.R, V.X),
                                       R.c(V.R), R.cl(V.X))

template += R.acyl_bromide(V.Mol) <= (R.contains(V.Mol, V.C), 
                                      R.carbonyl_group(V.C, _, V.R, V.X),
                                      R.c(V.R), R.br(V.X))

template += R.acyl_iodide(V.Mol) <= (R.contains(V.Mol, V.C), 
                                     R.carbonyl_group(V.C, _, V.R, V.X),
                                     R.c(V.R), R.i(V.X))


template += R.acyl_halide(V.Mol) <= R.acyl_fluoride(V.Mol)
template += R.acyl_halide(V.Mol) <= R.acyl_chloride(V.Mol)
template += R.acyl_halide(V.Mol) <= R.acyl_bromide(V.Mol)
template += R.acyl_halide(V.Mol) <= R.acyl_iodide(V.Mol)

# COMMAND ----------

# MAGIC %md
# MAGIC Carboxylic acid

# COMMAND ----------

template += R.carboxylic_acid(V.Mol) <= (R.contains(V.Mol, V.C), R.carbonyl_group(V.C, _, V.R, V.O),
                                R.c(V.R), R.o(V.O),
                                R.h(V.H), R.single_bonded(V.O, V.H))

# COMMAND ----------

# MAGIC %md
# MAGIC Carboxylic acid anhydrides

# COMMAND ----------

template += R.carboxylic_acid_anhydride(V.Mol) <= (R.contains(V.Mol, V.C1), R.contains(V.Mol, V.C1),
                                R.carbonyl_group(V.C1, _, V.O12, _),
                                R.o(V.O12),
                                R.carbonyl_group(V.C2, _, V.O12, _))

# COMMAND ----------

# MAGIC %md
# MAGIC Ester

# COMMAND ----------

template += R.ester(V.Mol) <= (R.contains(V.Mol, V.C), R.carbonyl_group(V.C, _, V.R, V.O),
                                R.c(V.R), R.o(V.O),
                                R.h(V.H), R.single_bonded(V.O, V.H))

template += R.carbonate_ester(V.Mol) <= (R.contains(V.Mol, V.C), R.carbonyl_group(V.C, _, V.O1, V.O2),
                                R.o(V.O1), R.o(V.O2),
                                
                                R.c(V.R1), R.single_bonded(V.R1, V.O1),
                                R.c(V.R2), R.single_bonded(V.O2, V.R2))

# COMMAND ----------

# MAGIC %md
# MAGIC Ether

# COMMAND ----------

template += R.ether(V.Mol) <= (R.contains(V.Mol, V.C), R.c(V.C), 
                                R.o(V.O), R.c(V.R), 
                                R.single_bonded(V.C, V.O),
                                R.single_bonded(V.O, V.R))



# COMMAND ----------

# MAGIC %md
# MAGIC Peroxyde

# COMMAND ----------

# peroxyde
template += R.peroxyde(V.Mol) <= (R.contains(V.Mol, V.C), R.c(V.C), # or saturated?
                                  R.o(V.O1), R.o(V.O2), R.c(V.R), 
                                  R.single_bonded(V.C, V.O1),
                                  R.single_bonded(V.O1, V.O2),
                                  R.single_bonded(V.O2, V.R))

# hydroperoxyde
template += R.hydroperoxyde(V.Mol) <= (R.contains(V.Mol, V.C), R.c(V.C), # or saturated?
                                       R.o(V.O1), R.o(V.O2), R.h(V.H), 
                                       R.single_bonded(V.C, V.O1),
                                       R.single_bonded(V.O1, V.O2),
                                       R.single_bonded(V.O2, V.H))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Predicition

# COMMAND ----------

template += R.predict[1,] <= R.layer_1(V.X)
template += R.predict[1,] <= R.alcoholic

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

settings = Settings(optimizer=Adam(lr=0.1), epochs=100, error_function=CrossEntropy())
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

