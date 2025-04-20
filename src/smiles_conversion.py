from neuralogic.dataset import Data
from torch_geometric.utils import from_smiles, from_networkx

from pysmiles import read_smiles
import networkx
from neuralogic.dataset import TensorDataset, FileDataset

from rdkit import Chem
import networkx as nx
from rdkit.Chem import AddHs, MolFromSmiles, GetPeriodicTable
import os

# Uranium is the heaviest naturally occurring element. 
# Beyond that, elements are typically synthesized in laboratories and have short half-lives
# We will use the first 92 elements in the periodic table as possible atom types
# Alternative: use all 118 elements
MAX_ATOM_TYPES = 92
MAX_EDGE_TYPES = len(Chem.rdchem.BondType.values)

def smiles_to_pyg(smiles: str, explicit_hydrogens=True):
    """
    Converts a SMILES string to a PyTorch Geometric (PyG) graph.

    Args:
        smiles (str): The SMILES representation of the molecule.
        explicit_hydrogens (bool): Add explicit hydrogens, default True

    Returns:
        torch_geometric.data.Data: A PyG graph representing the molecule, with atom and bond attributes.

    """
    mol = MolFromSmiles(smiles)

    if explicit_hydrogens:
        mol = AddHs(mol)

    # Convert mol to a NetworkX graph
    graph = nx.Graph()
    for atom in mol.GetAtoms():
        x = [0,] * MAX_ATOM_TYPES
        x[atom.GetAtomicNum()] = 1 # Atomic numbers start from 0 (0 for Unknown)
        graph.add_node(atom.GetIdx(), x=x, 
                    #    atom_symbol=atom.GetSymbol()
                       )
    for bond in mol.GetBonds():
        edge_attr = [0, ] * MAX_EDGE_TYPES
        for k, i in Chem.rdchem.BondType.values.items():
            if i == bond.GetBondType():
                edge_attr[k] = 1
        graph.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), edge_attr=edge_attr, 
                    #    bond_type=bond.GetBondType()
                       )
    
    # Add atom and bond IDs to the graph
    for node in graph.nodes:
        graph.nodes[node]['atom_id'] = node
    for edge in graph.edges:
        graph.edges[edge]['bond_id'] = mol.GetBondBetweenAtoms(edge[0], edge[1]).GetIdx()
    
    return from_networkx(graph)



def add_hydrogens(smiles: str):
    """Add explicit hydrogens to SMILES"""
    mol = Chem.MolFromSmiles(smiles)
    mol_with_hydrogens = AddHs(mol)
    smiles_with_hydrogens = Chem.MolToSmiles(mol_with_hydrogens)
    return smiles_with_hydrogens


def smiles_to_neuralogic(smiles: str):
    """"Convert SMILES to neuralogic tensor (through pygeometric)"""
    return Data.from_pyg(smiles_to_pyg(smiles))[0]

def smiles_to_networkx(smiles: str):
    """Convert SMILES to networkx network"""
    #smiles_with_hydrogens = add_hydrogens(smiles)
    mol = read_smiles(smiles, explicit_hydrogen=True)
    return mol

def get_atom_mapping(graph: networkx.classes.graph.Graph):
    """Get atom_id<->element mapping for networkx graph"""
    return dict(graph.nodes(data='element'))

def networkx_to_neuralogic(mol):
    """Convert networkx graph to neuralogic tensor (though pygeometric)"""
    pyg_graph = from_networkx(mol)
    return Data.from_pyg(pyg_graph)[0]

def update_predicate(pred, bond_mappings):
    # change bond from `bond_x(a1, a2)` to `bond(a1, a2, B), b_x(B)`
    if pred.startswith('bond'):
        first = int(pred.split('(')[1].split(',')[0])
        second = int(pred.split(',')[1].split(')')[0])
        B = bond_mappings[(first, second)]
        order = int(pred.split('(')[0].split('_')[1])
        new_pred = f'<1> bond({pred.split("(")[1].split(")")[0]}, {B}), <1> b_{order}({B})'
    # change atom from atom_0(0) to the mapping, for example to C(0)
    elif pred.startswith('atom'):

        atomic_number = int(pred.split('(')[0].split('_')[1])
        pt = GetPeriodicTable()
        e_id = pt.GetElementSymbol(atomic_number).lower()
        atom_id = pred.split('(')[1].split(')')[0]
        new_pred = f'<1> {e_id}({atom_id})'
    else:
        new_pred = pred
    return new_pred

def create_queries_file(labels, file_name):
    """"Manually create the *_queries.txt file using list of labels"""
    with open(file_name, 'w') as f:
        for label in labels:
            f.write(f'{label} predict.\n')

def get_dataset_and_mappings(smiles_list, labels=None, file_prefix='', output_location='datasets'):
    """Create the neuralogic dataset from list of smiles and also dump it as text files"""
    assert len(smiles_list) == len(labels) if labels is not None else True

    # convert the SMILES list to list of neuralogic tensors
    pyg_graphs = [smiles_to_pyg(smile) for smile in smiles_list]
    graphs = [Data.from_pyg(g)[0] for g in pyg_graphs]

    # Add label if available - this behaved bit weird in generation of queries.txt -> "[1.0] predict." instead of "1 predict." so generating this file manually instead
    
    if labels is not None:
        for graph, label in zip(graphs, labels):
            if type(label) == int:
                graph.y = label
            elif type(label) == list:
                graph.y = label[0]
    
            

    # Get the atom-element mappings - possible to get from networkx so convert from SMILES to networkx
    # networkx_graphs = [smiles_to_networkx(smile) for smile in smiles_list]
    # element_mappings = [get_atom_mapping(graph) for graph in networkx_graphs]
    # Bond mapping: [{(in_node, out_node): bond_id, ...} for G in graphs]
    bond_mappings = [{(int(g.edge_index[0][i]), int(g.edge_index[1][i])): int(id) for i, id in enumerate(g.bond_id)} for g in pyg_graphs]
    
    # create the dataset
    dataset = TensorDataset(graphs, one_hot_encode_labels=False, one_hot_decode_edge_features=True, one_hot_decode_features=True)
    dataset.edge_name = 'bond'
    dataset.feature_name = 'atom'

    # dump the dataset to text file
    queries_fp = f'{output_location}/{file_prefix}_queries.txt'
    examples_fp = f'{output_location}/{file_prefix}_examples_bad.txt' # this file does not have the desired structure, it is used to create the correct formating but can be deleted
    with open(queries_fp, 'w') as q_file, open(examples_fp, 'w') as e_file:
        dataset.dump(q_file, e_file)

    # update the dataset to have the desired format
    examples_updated_fp = f'{output_location}/{file_prefix}_examples.txt'
    with open(examples_fp, 'r') as in_handle, open(examples_updated_fp, 'w') as out_handle:
        for line, mapping in zip(in_handle.readlines(), bond_mappings):
            # get the predicates
            predicates = [p.strip(" ,\n.") for p in line.split("<1>") if p.strip(" ,\n") != ""]
            # update the predicates
            new_predicates = []
            for predicate in predicates:
                new_predicates.append(update_predicate(predicate, mapping))
            #predicates = [update_predicate(predicate, B, mapping) for predicate in predicates]
            out_handle.write(",".join(new_predicates)+'.\n')

    # Delete the bad examples file
    os.remove(examples_fp)

    # load the dataset from the text file
    dataset = FileDataset(examples_file=examples_updated_fp, queries_file=queries_fp)

    return dataset