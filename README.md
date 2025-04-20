# DifferentiableChemistry
This is the repository for the code of my master thesis "Deep learning for computational chemistry with differentiable background knowledge" and upcoming research paper

## Project Structure
The code is structured in a couple of scripts each representing some part of the implementation. 
All of the code is stored in `src` folder, and it might be converted to a package.
Model implementations are stored in `model_template` folders with `models.py` as the entrypoint.
Chemical and subgraph rules are stored in `knowledge_base` folder. 
Dataset files are stored in `datasets` folder, while they are loaded and parsed in `datasets.py`.
The training pipeline as well as other details are in `pipeline.py`.

To run all experiments from scratch, you can do it with `run_experiments.py`. If you want to see the code for running one set of parameters, see `main_notebook.py`, and the code which was used to analyze the results is in `analyze_results.py`. However, both of these might be messy and they are not intended for wider usage.


*Have in mind that the code is just a working draft, it will be updated to address possible issues, as well as to document it properly.*
