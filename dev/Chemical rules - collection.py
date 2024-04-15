# Databricks notebook source
# MAGIC %md
# MAGIC # Molecule rules

# COMMAND ----------

# MAGIC %md
# MAGIC Install PyNeuraLogic from PyPI

# COMMAND ----------

! pip install neuralogic

# COMMAND ----------

from neuralogic.utils.data import Mutagenesis

_, dataset = Mutagenesis()

# COMMAND ----------

from neuralogic.core import Template, R, V

template = Template()


# COMMAND ----------

# MAGIC %md
# MAGIC Defining molecules

# COMMAND ----------

from neuralogic.dataset import Dataset
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
        R.bond(1, 2, 10), R.bond(6, 3, 11), R.bond(6, 4, 12), R.bond(6, 5, 13),
        R.b_1(10), R.b_1(11), R.b_1(12), R.b_1(13), R.saturated(6)
    ],
)

# COMMAND ----------

# MAGIC %md
# MAGIC GNN-like graph propagation rule

# COMMAND ----------

template +=  R.layer_1(V.X) <= (R.bond(V.X, V.Y, V.B))

# COMMAND ----------

# MAGIC %md
# MAGIC # Functional groups

# COMMAND ----------

# MAGIC %md
# MAGIC Bonding rules

# COMMAND ----------

template += R.single_bonded(V.X, V.Y) <= (R.bond(V.X, V.Y, V.B), R.b_1(V.B))
template += R.single_bonded(V.X, V.Y) <= (R.bond(V.Y, V.X, V.B), R.b_1(V.B))

template += R.double_bonded(V.X, V.Y) <= (R.bond(V.X, V.Y, V.B), R.b_2(V.B))
template += R.double_bonded(V.X, V.Y) <= (R.bond(V.Y, V.X, V.B), R.b_2(V.B))

template += R.triple_bonded(V.X, V.Y) <= (R.bond(V.X, V.Y, V.B), R.b_3(V.B))
template += R.triple_bonded(V.X, V.Y) <= (R.bond(V.Y, V.X, V.B), R.b_3(V.B))


# aromatic, ionic, ...
template += R.aromatic_bonded(V.X, V.Y) <= (R.bond(V.X, V.Y, V.B), R.b_4(V.B))
template += R.aromatic_bonded(V.X, V.Y) <= (R.bond(V.Y, V.X, V.B), R.b_4(V.B)) # or which nuber




# COMMAND ----------

# MAGIC %md
# MAGIC Saturation rule

# COMMAND ----------

from neuralogic.core import Aggregation

template += R.saturated(V.X) <= (R.c(V.X),
                                 R.single_bonded(V.X, V.Y1),
                                 R.single_bonded(V.X, V.Y2),
                                 R.single_bonded(V.X, V.Y3),
                                 R.single_bonded(V.X, V.Y4),
                                 R.special.allldiff(V.Y1, V.Y2, V.Y3, V.Y4))
template += (R.saturated <= (R.saturated(V.C), R.c(V.C))) | [Aggregation.MIN]

# COMMAND ----------

# MAGIC %md
# MAGIC Alkane rule

# COMMAND ----------

# molecule is an alkane if all carbon atoms are saturated & it only contains C & H
template += R.not_hydrocarbyl(V.Mol) <= (R.contains(V.Mol, V.X), ~R.c(V.X), ~R.h(V.X)) # check negation

template += R.not_alkane(V.Mol) <= (R.contains(V.Mol, V.C), ~R.saturated(V.C))
template += R.not_alkane(V.Mol) <= (~R.not_hydrocarbyl(V.Mol))
template += R.alkane(V.Mol) <= (~R.not_alkane(V.Mol))

# COMMAND ----------

# MAGIC %md
# MAGIC Alkene rule

# COMMAND ----------

template += R.alkene(V.Mol) <= (~R.not_hydrocarbyl(V.Mol), 
                                R.contains(V.Mol, V.C1), R.contains(V.Mol, V.C2),
                                R.c(V.C1), R.c(V.C2),
                                R.double_bonded(V.C1, V.C2) # if there exists a double bond between carbons
                                )


# COMMAND ----------

# MAGIC %md
# MAGIC Alkyne rule

# COMMAND ----------

template += R.alkyne(V.Mol) <= (~R.not_hydrocarbyl(V.Mol), 
                                R.contains(V.Mol, V.C1), R.contains(V.Mol, V.C2),
                                R.c(V.C1), R.c(V.C2),
                                R.triple_bonded(V.C1, V.C2) # if there exists a triple bond between carbons
                                )

# COMMAND ----------

# MAGIC %md
# MAGIC Haloalkane rule

# COMMAND ----------

# all carbons saturated & there exists a F, Cl, Br or I atom
template += R.haloalkane(V.Mol) <= R.fluoroalkane(V.Mol)
template += R.haloalkane(V.Mol) <= R.chloroalkane(V.Mol)
template += R.haloalkane(V.Mol) <= R.bromoalkane(V.Mol)
template += R.haloalkane(V.Mol) <= R.iodoalkane(V.Mol)

# COMMAND ----------

template += R.fluoroalkane(V.Mol) <= (R.saturated(V.Mol), 
                                      R.contains(V.Mol, V.F), R.f(V.F), 
                                      R.contains(V.Mol, V.C), R.single_bonded(V.C, V.F))

template += R.chloroalkane(V.Mol) <= (R.saturated(V.Mol), 
                                      R.contains(V.Mol, V.Cl), R.cl(V.Cl), 
                                      R.contains(V.Mol, V.C), R.single_bonded(V.C, V.Cl))

template += R.bromoalkane(V.Mol) <= (R.saturated(V.Mol), 
                                     R.contains(V.Mol, V.Br), R.br(V.Br), 
                                     R.contains(V.Mol, V.C), R.single_bonded(V.C, V.Br))

template += R.iodoalkane(V.Mol) <= (R.saturated(V.Mol), 
                                    R.contains(V.Mol, V.I), R.i(V.I), 
                                    R.contains(V.Mol, V.C), R.single_bonded(V.C, V.I))

# COMMAND ----------

# MAGIC %md
# MAGIC Phenyl rule (benzene ring)

# COMMAND ----------

template += R.benzene_ring(V.A, V.B, V.C, V.D, V.E, V.F) <= (R.aromatic_bonded(V.A, V.B), 
                                                             R.aromatic_bonded(V.B, V.C),
                                                             R.aromatic_bonded(V.C, V.D),
                                                             R.aromatic_bonded(V.D, V.E),
                                                             R.aromatic_bonded(V.E, V.F),
                                                             R.aromatic_bonded(V.F, V.A))

# COMMAND ----------

# MAGIC %md
# MAGIC Alcohol rule

# COMMAND ----------

'''template += R.alcohol(V.Mol) <= (R.contains(V.Mol, V.C), R.alcoholic(V.C))
template += R.alcoholic(V.C) <= (R.saturated(V.C), R.o(V.O), R.h(V.H), 
                                 R.single_bonded(V.C, V.O),
                                 R.single_bonded(V.O, V.H))'''

template += R.alcoholic <= (R.saturated, R.o(V.O), R.h(V.H), 
                                 R.single_bonded(V.C, V.O),
                                 R.single_bonded(V.O, V.H))

# COMMAND ----------

# MAGIC %md
# MAGIC Carbonyl groups rules

# COMMAND ----------

# carbonyl group contains a carbon double bonded with oxygen
template += R.carbonyl(V.Mol) <= (R.contains(V.Mol, V.C), R.contains(V.Mol, V.O), R.carbonyl_group(V.O, V.C))
template += R.carbonyl_group(V.C, V.O) <= (R.c(V.C), R.o(V.O), R.double_bonded(V.O, V.C))
template += R.carbonyl_group(V.C, V.O, V.R1, V.R2) <= (R.c(V.C), R.o(V.O), R.double_bonded(V.O, V.C), R.single_bonded(V.C, V.R1), R.single_bonded(V.C, V.R2))

# COMMAND ----------

# ketone
template += R.ketone(V.Mol) <= (R.contains(V.Mol, V.C), 
                                R.carbonyl_group(V.C, _, V.R1, V.R2),
                                R.c(V.R1), R.c(V.R2))

# aldehyde
template += R.aldehyde(V.Mol) <= (R.contains(V.Mol, V.C), 
                                  R.carbonyl_group(V.C, _, V.R, V.H),
                                  R.c(V.R), R.h(V.H))

# acylhalide
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
# MAGIC Carboxylic acids

# COMMAND ----------

# carboxylic acid
template += R.carboxylic_acid(V.Mol) <= (R.contains(V.Mol, V.C), R.carbonyl_group(V.C, _, V.R, V.O),
                                R.c(V.R), R.o(V.O),
                                R.h(V.H), R.single_bonded(V.O, V.H))

# COMMAND ----------

# MAGIC %md
# MAGIC Esters

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
# MAGIC Peroxydes

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
# MAGIC Ethers

# COMMAND ----------

template += R.ether(V.Mol) <= (R.contains(V.Mol, V.C), R.c(V.C), 
                                R.o(V.O), R.c(V.R), 
                                R.single_bonded(V.C, V.O),
                                R.single_bonded(V.O, V.R))



# COMMAND ----------

# dicarbonyls, acetals, hemiacetals/ketals, orthoesters, orthocarbonate esters, carboxylic acit anhydrides
# nitrogen, sulphur, phosphorus, boron, metals, polymers, heterocycles!, biomolecules

# COMMAND ----------

# MAGIC %md
# MAGIC # Characteristic reactions

# COMMAND ----------

# MAGIC %md
# MAGIC No available datasets for this purpose

# COMMAND ----------

#template += R.produces(Reaction, [V.Reactants], [V.Products]) <= (R.reacts(R.Reaction, React) for React in V.Reactants,
#                                                                    R.Reaction([V.Reactants], [V.Products]))

# COMMAND ----------

# alkanes
# combustion reaction
template += R.reacts(combustion, V.Mol) <= (R.alkane(V.Mol))
template += R.reacts(combustion, V.Mol, V.State) <= (R.alkane(V.Mol), R.favorable_state(R.combustion, V.State))

# make some template for this?? How to add quantities of reactants/products
#template += R.combustion([V.Alkane[n], V.Oxygen[2*n]], [V.Water, V.CX]) <= () # if enough oxygen produce water and CO2, otherwise produce CO or C if even less

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC # Electronegativity rules

# COMMAND ----------

# MAGIC %md
# MAGIC Lacking 3d structure to be able to determine the vectors

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC # Prediction

# COMMAND ----------

template += R.predict[1,] <= R.layer_1(V.X)
template += R.predict[1,] <= R.alcoholic #(V.X)

# COMMAND ----------


train_dataset.add_queries([
    R.predict[0],
    R.predict[1],
])
print(template)

# COMMAND ----------

template.draw()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Training
# MAGIC
# MAGIC When we have our dataset and template ready, it's time to build ("ground") the template over the dataset and start training.
# MAGIC We can do the training manually and write our own custom training loop, but we can also use predefined helpers - *evaluators*,
# MAGIC that handle model and dataset building, training, and more. Evaluators can be customized via `Settings`.
# MAGIC
# MAGIC <sup>Note that building the dataset (=grounding the logic rules and translating into neural networks) may take a while, depending on the complexity of your template.
# MAGIC But this is only done once before the training itself, which takes up most of the time anyway.
# MAGIC <sup>

# COMMAND ----------

from neuralogic.core import Settings
from neuralogic.nn.loss import MSE, CrossEntropy
from neuralogic.nn import get_evaluator
from neuralogic.optim import Adam

settings = Settings(optimizer=Adam(lr=0.001), epochs=100, error_function=MSE())
evaluator = get_evaluator(template, settings)

built_dataset = evaluator.build_dataset(train_dataset)

#evaluator.test(train_dataset)

# COMMAND ----------

built_dataset[1].draw()

# COMMAND ----------

evaluator.test(train_dataset, generator=False)

# COMMAND ----------

# MAGIC %md
# MAGIC Finally, we iterate through the iterator encapsulated in the `train` method of the evaluator, which yields a total loss of the epoch and the number of samples of the current epoch.
# MAGIC We then get access to the results from the training loop that we can further visualize, inspect, log, etc.

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

# MAGIC %md
# MAGIC We can then check the trained model predictions (for the same sample set here) by utilizing the `test` method.

# COMMAND ----------

for sample, y_hat in zip(built_dataset.samples, evaluator.test(built_dataset)):
    print(f"Target: {sample.java_sample.target}, Predicted: {round(y_hat)} ({y_hat})")

# COMMAND ----------

