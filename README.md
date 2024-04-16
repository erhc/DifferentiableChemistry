# DifferentiableChemistry
This is the repository for the code of my master thesis "Deep learning for computational chemistry with differentiable background knowledge" and upcoming research paper

## Project Structure
The code is structured in a couple of scripts each representing some part of the implementation. 
All of the code is stored in `src` folder, and it might be converted to a package.
Model implementations are stored in `model_template` folders with `models.py` as the entrypoint.
Chemical and subgraph rules are stored in `knowledge_base` folder. 
Dataset files are stored in `datasets` folder, while they are loaded and parsed in `datasets.py`.
The training pipeline as well as other details are in `pipeline.py`.

The `main_notebook.py` is there for my testing and probably very messy, please ignore it. Same goes for `/dev` folder, it contains some notebooks with outdated code.


*Have in mind that the code is just a working draft, it will be updated to address possible issues, as well as to document it properly.*
