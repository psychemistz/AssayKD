#!/usr/bin/env python3
"""
Simplified Data Processing Module

Consolidates all data preprocessing functionality:
- SMILES to molecular graphs
- Enhanced molecular descriptors
- Data validation and normalization
"""

import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.preprocessing import StandardScaler
import pickle
import os
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def get_atom_features(atom):
    """Extract atom features for graph neural network"""
    return [
        atom.GetAtomicNum(),
        atom.GetDegree(),
        atom.GetImplicitValence(),
        atom.GetFormalCharge(),
        int(atom.GetIsAromatic()),
        int(atom.IsInRing()),
        atom.GetTotalNumHs(),
        int(atom.GetHybridization()),
        int(atom.GetChiralTag())
    ]

def get_bond_features(bond):
    """Extract bond features for graph neural network"""
    return [
        int(bond.GetBondType()),
        int(bond.GetIsConjugated()),
        int(bond.IsInRing())
    ]

def smiles_to_graph(smiles):
    """Convert SMILES string to PyTorch Geometric graph"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    mol = Chem.AddHs(mol)
    
    # Extract atom features
    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append(get_atom_features(atom))
    
    if len(atom_features) == 0:
        return None
    
    # Extract bond features and connectivity
    edge_indices = []
    edge_features = []
    
    for bond in mol.GetBonds():
        start_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        
        edge_indices.extend([[start_idx, end_idx], [end_idx, start_idx]])
        bond_feat = get_bond_features(bond)
        edge_features.extend([bond_feat, bond_feat])
    
    # Convert to tensors
    x = torch.tensor(atom_features, dtype=torch.float)
    
    if len(edge_indices) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 3), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.float)
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

def compute_molecular_descriptors(mol):
    """Compute comprehensive molecular descriptors"""
    descriptors = [
        ('MW', Descriptors.ExactMolWt),
        ('LogP', Descriptors.MolLogP), 
        ('TPSA', Descriptors.TPSA),
        ('HBD', Descriptors.NumHDonors),
        ('HBA', Descriptors.NumHAcceptors),
        ('RotBonds', Descriptors.NumRotatableBonds),
        ('AromaticRings', Descriptors.NumAromaticRings),
        ('HeavyAtoms', Descriptors.HeavyAtomCount),
        ('FractionCSP3', Descriptors.FractionCSP3),
        ('RingCount', Descriptors.RingCount),
        ('BalabanJ', Descriptors.BalabanJ),
        ('BertzCT', Descriptors.BertzCT),
        ('Chi0', Descriptors.Chi0),
        ('Chi1', Descriptors.Chi1),
        ('Kappa1', Descriptors.Kappa1),
        ('Kappa2', Descriptors.Kappa2),
        ('MaxEStateIndex', Descriptors.MaxEStateIndex),
        ('MinEStateIndex', Descriptors.MinEStateIndex),
        ('LabuteASA', Descriptors.LabuteASA),
        ('PEOE_VSA1', Descriptors.PEOE_VSA1),
        ('PEOE_VSA2', Descriptors.PEOE_VSA2),
        ('SMR_VSA1', Descriptors.SMR_VSA1),
        ('SMR_VSA2', Descriptors.SMR_VSA2),
        ('SlogP_VSA1', Descriptors.SlogP_VSA1),
        ('SlogP_VSA2', Descriptors.SlogP_VSA2)
    ]
    
    features = []
    feature_names = []
    
    for desc_name, desc_func in descriptors:
        try:
            value = desc_func(mol)
            if value is None or np.isnan(value) or np.isinf(value):
                value = 0.0
            features.append(float(value))
            feature_names.append(desc_name)
        except:
            continue
    
    return features, feature_names

def clean_data(df):
    """Clean and validate input dataframe"""
    print("🧹 Cleaning data...")
    
    # Standardize column names
    column_mapping = {
        'HCT_01(%)': 'normal_01',
        'HCT_10(%)': 'normal_10', 
        'MCC_01(%)': 'mcc26_01',
        'MCC_10(%)': 'mcc26_10',
        'MKL_01(%)': 'mkl1_01',
        'MKL_10(%)': 'mkl1_10',
        'Smile': 'SMILES',
        'smile': 'SMILES',
        'smiles': 'SMILES'
    }
    
    df = df.rename(columns=column_mapping)
    
    if 'SMILES' not in df.columns:
        raise ValueError("No SMILES column found!")
    
    # Clean SMILES
    df = df.dropna(subset=['SMILES'])
    df = df[df['SMILES'].str.strip() != '']
    df = df[~df['SMILES'].isin(['nan', 'NaN', 'NULL', 'null', 'None', 'none'])]
    
    print(f"✓ Cleaned data: {len(df)} valid compounds")
    return df.reset_index(drop=True)

def create_activity_labels(df):
    """Create binary activity labels from viability percentages"""
    print("🎯 Creating activity labels...")
    
    safety_threshold = 80   # >80% viability = safe for normal cells
    toxicity_threshold = 50 # <50% viability = toxic to cancer cells
    
    # Normal cell safety
    if 'normal_10' in df.columns:
        df['normal_safe'] = (df['normal_10'] > safety_threshold).astype(int)
        safe_count = df['normal_safe'].sum()
        print(f"  • normal_safe: {safe_count}/{len(df)} ({safe_count/len(df)*100:.1f}%)")
    
    # MCC26 cancer cell activity  
    if 'mcc26_10' in df.columns:
        df['mcc26_active'] = (df['mcc26_10'] < toxicity_threshold).astype(int)
        active_count = df['mcc26_active'].sum()
        print(f"  • mcc26_active: {active_count}/{len(df)} ({active_count/len(df)*100:.1f}%)")
    
    # MKL1 cancer cell activity
    if 'mkl1_10' in df.columns:
        df['mkl1_active'] = (df['mkl1_10'] < toxicity_threshold).astype(int)
        active_count = df['mkl1_active'].sum()
        print(
            f"  • mkl1_active: {active_count}/{len(df)} "
            f"({active_count/len(df)*100:.1f}%)"
        )
    
    return df

def process_molecules(df):
    """Convert SMILES to graphs and compute descriptors"""
    print(f"🧬 Processing {len(df)} molecules...")
    
    valid_data = []
    graph_data = []
    descriptor_data = []
    feature_names = None
    
    success_count = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Converting molecules"):
        smiles = str(row['SMILES']).strip()
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            
            graph = smiles_to_graph(smiles)
            if graph is None:
                continue
            
            desc_features, desc_names = compute_molecular_descriptors(mol)
            if len(desc_features) == 0:
                continue
            
            if feature_names is None:
                feature_names = desc_names
            
            if len(desc_features) != len(feature_names):
                continue
            
            canonical_smiles = Chem.MolToSmiles(mol)
            row_dict = row.to_dict()
            row_dict['SMILES_canonical'] = canonical_smiles
            
            valid_data.append(row_dict)
            graph_data.append(graph)
            descriptor_data.append(desc_features)
            success_count += 1
            
        except Exception:
            continue
    
    print(f"✓ Successfully processed {success_count}/{len(df)} molecules")
    
    if len(valid_data) == 0:
        raise ValueError("No valid molecules could be processed!")
    
    return valid_data, graph_data, descriptor_data, feature_names

def normalize_descriptors(descriptor_data, feature_names):
    """Normalize molecular descriptors"""
    print("📊 Normalizing descriptors...")
    
    descriptors_array = np.array(descriptor_data)
    scaler = StandardScaler()
    descriptors_normalized = scaler.fit_transform(descriptors_array)
    
    print(f"✓ Normalized {descriptors_array.shape[1]} features for {descriptors_array.shape[0]} molecules")
    return descriptors_normalized, scaler

def save_processed_data(df_clean, graph_data, descriptors_normalized, feature_names, scaler):
    """Save all processed data"""
    print("💾 Saving processed data...")
    
    # Create directories
    for directory in ['data/graphs', 'models']:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    # Add normalized descriptors to dataframe
    for i, name in enumerate(feature_names):
        df_clean[name] = descriptors_normalized[:, i]
    
    # Save files
    df_clean.to_csv('data/processed_data.csv', index=False)
    torch.save(graph_data, 'data/graphs/molecular_graphs.pt')
    
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    with open('models/feature_names.txt', 'w') as f:
        for name in feature_names:
            f.write(f"{name}\n")
    
    print(f"✓ Data saved to data/processed_data.csv and data/graphs/molecular_graphs.pt")
    return 'data/processed_data.csv', 'data/graphs/molecular_graphs.pt'

def process_pipeline(input_file='data/raw_data.csv'):
    """Main processing pipeline"""
    print("🧬 MOLECULAR DATA PROCESSING PIPELINE")
    print("="*60)
    
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    # Load and clean data
    df_original = pd.read_csv(input_file)
    df_clean = clean_data(df_original)
    df_clean = create_activity_labels(df_clean)
    
    # Process molecules
    valid_data, graph_data, descriptor_data, feature_names = process_molecules(df_clean)
    df_final = pd.DataFrame(valid_data)
    
    # Normalize descriptors
    descriptors_normalized, scaler = normalize_descriptors(descriptor_data, feature_names)
    
    # Save everything
    output_file, graph_file = save_processed_data(
        df_final, graph_data, descriptors_normalized, feature_names, scaler
    )
    
    print(f"\n✅ Processing completed!")
    print(f"📁 Files created:")
    print(f"  • {output_file}")
    print(f"  • {graph_file}")
    print(f"  • models/scaler.pkl")
    print(f"  • models/feature_names.txt")
    
    return df_final, graph_data

# In the load_processed_data() function, update file paths:
def load_processed_data():
    """Load already processed data"""
    try:
        df = pd.read_csv('data/graph_processed_data.csv')  # Your existing file
        graph_data = torch.load('data/graphs/molecular_graphs.pt', map_location='cpu')
        
        with open('models/enhanced_feature_names.txt', 'r') as f:  # Your existing file
            feature_names = [line.strip() for line in f]
        
        return df, graph_data, feature_names
    except FileNotFoundError:
        print("❌ Processed data not found. Run process_pipeline() first.")
        return None, None, None

if __name__ == "__main__":
    df, graph_data = process_pipeline()
