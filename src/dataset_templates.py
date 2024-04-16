from file_datasets.MutagenesisTemplate import MutagenesisTemplate
from file_datasets.PTCTemplate import PTCTemplate
from file_datasets.PTCFRTemplate import PTCFRTemplate
from file_datasets.PTCFMTemplate import PTCFMTemplate
from file_datasets.PTCMMTemplate import PTCMMTemplate
from file_datasets.COXTemplate import COXTemplate
from file_datasets.DHFRTemplate import DHFRTemplate
from file_datasets.ERTemplate import ERTemplate

dataset_templates = {
    "mutagen": MutagenesisTemplate,
    "ptc": PTCTemplate,
    "ptc_fr": PTCFRTemplate,
    "ptc_mm": PTCMMTemplate,
    "ptc_fm": PTCFMTemplate,
    "cox": COXTemplate,
    "dhfr": DHFRTemplate,
    "er": ERTemplate
}

def get_dataset(dataset, param_size):
    """
    Get the dataset template based on the dataset name and parameter size.

    :param dataset: Name of the dataset.
    :param param_size: Size of the parameter for the dataset template.
    :return: An instance of the corresponding dataset template class.
    :raises ValueError: If the dataset name is invalid.
    """
    # Retrieve the template class from the dictionary
    template_class = dataset_templates.get(dataset)

    if template_class is None:
        raise ValueError(f"Invalid dataset name: {dataset}\nPlease use one of the following: {list(dataset_templates.keys())}")

    # Instantiate and return the template class
    return template_class(param_size)
