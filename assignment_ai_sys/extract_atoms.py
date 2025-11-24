#!/usr/bin/env python
"""Extract all unique atom symbols from the dataset"""
import pickle
from rdkit import Chem
import pandas as pd
from data_preprocessing import CustomData

# Load preprocessed data
with open('drug_data.pkl', 'rb') as f:
    drug_data = pickle.load(f)

# Load SMILES
df = pd.read_csv('drugbank.csv', sep='\t')
drug_smiles = dict(zip(df['ID1'], df['X1']))
drug_smiles.update(dict(zip(df['ID2'], df['X2'])))

# Extract ALL unique atoms
atom_set = set()
for drug_id in list(drug_data.keys()):
    if drug_id in drug_smiles:
        smiles = drug_smiles[drug_id]
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            for atom in mol.GetAtoms():
                atom_set.add(atom.GetSymbol())

atom_list = sorted(list(atom_set))
print(f'Total unique atoms: {len(atom_list)}')
print(f'Atoms: {atom_list}')
