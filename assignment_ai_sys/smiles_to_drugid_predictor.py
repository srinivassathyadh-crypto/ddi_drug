#!/usr/bin/env python
"""
SMILES-based Predictor using Preprocessed Graphs
Maps SMILES to Drug IDs, then uses exact preprocessed graphs from training

Usage:
    python smiles_to_drugid_predictor.py --smiles1 "SMILES_1" --smiles2 "SMILES_2"
"""
import argparse
import pickle
import torch
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from data_preprocessing import CustomData
from model import MPNN_DDI
from torch_geometric.data import Batch
import warnings
warnings.filterwarnings('ignore')

class SMILEStoDrugIDPredictor:
    def __init__(self, model_path='epoch20_valacc0.9548_testacc0.9539.pt.pt',
                 drug_data_path='drug_data.pkl'):
        """Initialize predictor with preprocessed graphs"""
        print("Using device: cpu")
        print("Loading preprocessed drug data...")

        # Load preprocessed graphs
        with open(drug_data_path, 'rb') as f:
            self.drug_data = pickle.load(f)

        print(f"Loaded {len(self.drug_data)} drugs")

        # Load SMILES to Drug ID mapping
        print("Loading SMILES to Drug ID mapping...")
        df = pd.read_csv('drugbank.csv', sep='\t')

        # Create bidirectional mapping
        self.smiles_to_id = {}
        for _, row in df.iterrows():
            id1 = row['ID1'].strip('\"')
            id2 = row['ID2'].strip('\"')
            smiles1 = row['X1'].strip('\"')
            smiles2 = row['X2'].strip('\"')

            self.smiles_to_id[smiles1] = id1
            self.smiles_to_id[smiles2] = id2

        print(f"Mapped {len(self.smiles_to_id)} SMILES strings")

        # Load interaction descriptions
        self.interaction_map = df[['Y', 'Map']].drop_duplicates().set_index('Y')['Map'].to_dict()

        # Get dimensions
        sample_drug = list(self.drug_data.values())[0]
        in_dim = sample_drug.x.shape[-1]
        edge_dim = sample_drug.edge_attr.shape[-1]

        print(f"Node feature dimension: {in_dim}")
        print(f"Edge feature dimension: {edge_dim}")

        # Initialize and load model
        print("Initializing model architecture...")
        self.model = MPNN_DDI(
            in_dim=in_dim,
            edge_dim=edge_dim,
            hidden_dim=64,
            n_iter=3,
            kge_dim=64,
            rel_total=86
        )

        print(f"Loading model weights from {model_path}...")
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        self.model.load_state_dict(checkpoint)
        self.model.eval()
        print("Model loaded successfully!\n")

    def smiles_to_drug_id(self, smiles):
        """Convert SMILES to Drug ID"""
        if smiles in self.smiles_to_id:
            return self.smiles_to_id[smiles], None
        else:
            return None, f"SMILES not found in database (need one of {len(self.smiles_to_id)} available drugs)"

    def get_properties(self, smiles):
        """Extract molecular properties from SMILES"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return {}

            return {
                'molecular_weight': Descriptors.MolWt(mol),
                'logp': Descriptors.MolLogP(mol),
                'h_donors': Descriptors.NumHDonors(mol),
                'h_acceptors': Descriptors.NumHAcceptors(mol),
                'num_atoms': mol.GetNumAtoms(),
                'num_bonds': mol.GetNumBonds()
            }
        except:
            return {}

    def predict_all_relations(self, smiles1, smiles2, top_k=10):
        """
        Predict all interaction types using preprocessed graphs

        Args:
            smiles1: SMILES string for drug 1
            smiles2: SMILES string for drug 2
            top_k: Number of top predictions to display

        Returns:
            Dictionary with predictions and properties
        """
        print(f"Predicting interaction between:")
        print(f"  Drug 1: {smiles1[:60]}...")
        print(f"  Drug 2: {smiles2[:60]}...")
        print("="*80)

        # Convert SMILES to Drug IDs
        drug1_id, error1 = self.smiles_to_drug_id(smiles1)
        if error1:
            return {'error': f"Drug 1: {error1}"}

        drug2_id, error2 = self.smiles_to_drug_id(smiles2)
        if error2:
            return {'error': f"Drug 2: {error2}"}

        print(f"\nMapped to Drug IDs:")
        print(f"  Drug 1: {drug1_id}")
        print(f"  Drug 2: {drug2_id}\n")

        # Get preprocessed graphs
        if drug1_id not in self.drug_data:
            return {'error': f"Drug {drug1_id} not in preprocessed data"}
        if drug2_id not in self.drug_data:
            return {'error': f"Drug {drug2_id} not in preprocessed data"}

        graph1 = self.drug_data[drug1_id]
        graph2 = self.drug_data[drug2_id]

        # Get properties
        props1 = self.get_properties(smiles1)
        props2 = self.get_properties(smiles2)

        # Predict for all 86 interaction types
        with torch.no_grad():
            all_scores = []
            for rel_idx in range(86):
                # Create fresh batches for each relation
                h_data = Batch.from_data_list([graph1], follow_batch=['edge_index'])
                t_data = Batch.from_data_list([graph2], follow_batch=['edge_index'])
                rels = torch.LongTensor([rel_idx])

                score = self.model((h_data, t_data, rels))
                all_scores.append(score.item())

        # Convert to probabilities using softmax
        scores_tensor = torch.tensor(all_scores)
        probabilities = torch.softmax(scores_tensor, dim=0).numpy()

        # Create predictions list
        predictions = []
        for idx in range(86):
            rel_type = idx + 1  # 1-indexed
            prob = probabilities[idx]

            predictions.append({
                'relation_type': rel_type,
                'relation_name': self.interaction_map.get(rel_type, f'Interaction Type {rel_type}'),
                'probability': prob,
                'prediction': 'Yes' if prob >= 0.5 else 'No',
                'confidence': self._get_confidence_level(prob)
            })

        # Sort by probability
        predictions.sort(key=lambda x: x['probability'], reverse=True)

        return {
            'drug1': {'smiles': smiles1, 'drug_id': drug1_id, 'properties': props1},
            'drug2': {'smiles': smiles2, 'drug_id': drug2_id, 'properties': props2},
            'predictions': predictions,
            'top_k': top_k
        }

    def _get_confidence_level(self, prob):
        """Get confidence level based on probability"""
        if prob > 0.9:
            return 'Very High ⭐⭐⭐'
        elif prob > 0.7:
            return 'High ⭐⭐'
        elif prob > 0.5:
            return 'Moderate ⭐'
        else:
            return 'Low'

    def print_results(self, result):
        """Print formatted results"""
        if 'error' in result:
            print(f"\n❌ Error: {result['error']}\n")
            return

        # Drug information
        print("\n" + "="*80)
        print("DRUG INFORMATION")
        print("="*80)

        props1 = result['drug1']['properties']
        print(f"\nDrug 1: {result['drug1']['drug_id']}")
        print(f"SMILES: {result['drug1']['smiles'][:60]}...")
        print(f"  • Molecular Weight: {props1.get('molecular_weight', 0):.2f}")
        print(f"  • LogP: {props1.get('logp', 0):.2f}")
        print(f"  • H-Bond Donors: {props1.get('h_donors', 0)}")
        print(f"  • H-Bond Acceptors: {props1.get('h_acceptors', 0)}")
        print(f"  • Atoms: {props1.get('num_atoms', 0)}, Bonds: {props1.get('num_bonds', 0)}")

        props2 = result['drug2']['properties']
        print(f"\nDrug 2: {result['drug2']['drug_id']}")
        print(f"SMILES: {result['drug2']['smiles'][:60]}...")
        print(f"  • Molecular Weight: {props2.get('molecular_weight', 0):.2f}")
        print(f"  • LogP: {props2.get('logp', 0):.2f}")
        print(f"  • H-Bond Donors: {props2.get('h_donors', 0)}")
        print(f"  • H-Bond Acceptors: {props2.get('h_acceptors', 0)}")
        print(f"  • Atoms: {props2.get('num_atoms', 0)}, Bonds: {props2.get('num_bonds', 0)}")

        # Top predictions
        print("\n" + "="*80)
        print(f"Top {result['top_k']} Most Likely Interactions:")
        print("="*80)

        top_predictions = result['predictions'][:result['top_k']]

        for i, pred in enumerate(top_predictions, 1):
            print(f"\n{i}. Type {pred['relation_type']}: {pred['probability']*100:.2f}% ({pred['confidence']})")
            print(f"   Prediction: {pred['prediction']}")
            print(f"   {pred['relation_name']}")

        # Risk assessment
        print("\n" + "="*80)
        print("RISK ASSESSMENT")
        print("="*80)

        very_high = [p for p in result['predictions'] if p['probability'] > 0.9]
        high = [p for p in result['predictions'] if 0.7 <= p['probability'] <= 0.9]
        moderate = [p for p in result['predictions'] if 0.5 <= p['probability'] < 0.7]

        if very_high:
            print("\n🔴 VERY HIGH RISK Interactions:")
            for pred in very_high:
                print(f"   • Type {pred['relation_type']}: {pred['probability']*100:.2f}%")
                print(f"     {pred['relation_name']}")

        if high:
            print("\n🟠 HIGH RISK Interactions:")
            for pred in high:
                print(f"   • Type {pred['relation_type']}: {pred['probability']*100:.2f}%")
                print(f"     {pred['relation_name']}")

        if moderate:
            print("\n🟡 MODERATE RISK Interactions:")
            for pred in moderate:
                print(f"   • Type {pred['relation_type']}: {pred['probability']*100:.2f}%")
                print(f"     {pred['relation_name']}")

        if not very_high and not high and not moderate:
            print("\n🟢 LOW RISK: No high-confidence interactions detected")

        print("\n" + "="*80)
        print("Model: DGNN (95.48% validation accuracy, 95.39% test accuracy)")
        print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Predict drug interactions from SMILES (using preprocessed graphs)')
    parser.add_argument('--smiles1', type=str, required=True, help='SMILES string for drug 1')
    parser.add_argument('--smiles2', type=str, required=True, help='SMILES string for drug 2')
    parser.add_argument('--top_k', type=int, default=10, help='Number of top predictions to show')
    parser.add_argument('--show_all', action='store_true', help='Show all 86 interaction types')
    parser.add_argument('--output', type=str, help='Output file for results (optional)')

    args = parser.parse_args()

    # Initialize predictor
    predictor = SMILEStoDrugIDPredictor()

    # Make prediction
    top_k = 86 if args.show_all else args.top_k
    result = predictor.predict_all_relations(args.smiles1, args.smiles2, top_k)

    # Print results
    predictor.print_results(result)

    # Save to file if requested
    if args.output:
        import sys
        original_stdout = sys.stdout
        with open(args.output, 'w') as f:
            sys.stdout = f
            predictor.print_results(result)
            sys.stdout = original_stdout
        print(f"✅ Results saved to: {args.output}")


if __name__ == '__main__':
    main()
