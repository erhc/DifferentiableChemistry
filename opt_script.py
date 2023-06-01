from functools import partial
import pipeline
import optuna
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_name')
    parser.add_argument('model_name')
    parser.add_argument('-c', '--chem',
                        action='store_true')
    parser.add_argument('-s', '--subgraph',
                        action='store_true')
    parser.add_argument('--stage')
    args = parser.parse_args()
    study = optuna.create_study()

    main_function = partial(pipeline.main_opt, dataset_name=args.dataset_name, model_name=args.model_name,
                            chemical_rules=args.chem, subgraphs=args.subgraph, stage=args.stage)

    study.optimize(main_function, n_trials=100)

