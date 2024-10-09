from chemdiff.file_datasets.COXTemplate import COXTemplate
from chemdiff.file_datasets.CustomDataset import CustomDataset
from chemdiff.file_datasets.DHFRTemplate import DHFRTemplate
from chemdiff.file_datasets.ERTemplate import ERTemplate
from chemdiff.file_datasets.MutagenesisTemplate import MutagenesisTemplate
from chemdiff.file_datasets.PTCFMTemplate import PTCFMTemplate
from chemdiff.file_datasets.PTCFRTemplate import PTCFRTemplate
from chemdiff.file_datasets.PTCMMTemplate import PTCMMTemplate
from chemdiff.file_datasets.PTCTemplate import PTCTemplate
import os


dataset_templates = {
    "mutagen": MutagenesisTemplate,
    "ptc": PTCTemplate,
    "ptc_fr": PTCFRTemplate,
    "ptc_mm": PTCMMTemplate,
    "ptc_fm": PTCFMTemplate,
    "cox": COXTemplate,
    "dhfr": DHFRTemplate,
    "er": ERTemplate,
    "custom": CustomDataset,
}

dataset_len = {
    'anti_sarscov2_activity': 1484,
    'blood_brain_barrier': 2030,
    'carcinogenous': 280,
    'cox': 303,
    'cyp2c9_substrate': 669,
    'cyp2d6_substrate': 667,
    'cyp3a4_substrate': 670,
    'cyp_p450_3a4_inhibition': 12328,
    'dhfr': 393,
    'er': 446,
    'human_intestinal_absorption': 578,
    'mutagen': 183,
    # 'mutagenic': 7278, # broken
    'oral_bioavailability': 640,
    'p_glycoprotein_inhibition': 1218,
    'pampa_permeability': 2034,
    'ptc': 344,
    'ptc_fm': 349,
    'ptc_fr': 351,
    'ptc_mm': 336,
    'skin_reaction': 404,
    'splice_ai': 7962
    } 



custom_datasets = [
    'anti_sarscov2_activity',
    'blood_brain_barrier',
    'carcinogenous',
    'cyp2c9_substrate',
    'cyp2d6_substrate',
    'cyp3a4_substrate',
    'cyp_p450_3a4_inhibition',
    'human_intestinal_absorption',
    # 'mutagenic', #broken
    'oral_bioavailability',
    'p_glycoprotein_inhibition',
    'pampa_permeability',
    'skin_reaction',
    ]
available_datasets = [
    "mutagen",
    "ptc",
    "ptc_fr",
    "ptc_mm",
    "ptc_fm",
    # "cox",
    "dhfr",
    "er",
    ] + custom_datasets

def get_available_datasets():
    return available_datasets

def get_dataset_len(name):
    return dataset_len.get(name, 0)

def get_dataset(dataset, param_size, examples=None, queries=None):
    """
    Get the dataset template based on the dataset name and parameter size.

    :param dataset: Name of the dataset. Run `get_available_datasets()` to get the list of available datasets.
    :param param_size: Size of the parameter for the dataset template.
    :return: An instance of the corresponding dataset template class.
    :raises ValueError: If the dataset name is invalid.
    """
    # Retrieve the template class from the dictionary
    template_class = None
    if dataset in custom_datasets:
        template_class = CustomDataset
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        examples = os.path.join(current_dir, f"datasets/{dataset}_examples.txt")
        queries = os.path.join(current_dir, f"datasets/{dataset}_queries.txt")
        dataset = "custom"
    else:
        template_class = dataset_templates.get(dataset)

    if template_class is None:
        raise ValueError(
            f"Invalid dataset name: {dataset}\nPlease use one of the following: {list(dataset_templates.keys())}"
        )

    if dataset == "custom":
        return CustomDataset(examples, queries, param_size)
    # Instantiate and return the template class
    return template_class(param_size)
