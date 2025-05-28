#!/usr/bin/env python3
"""
Simplified Prediction Interface - FIXED VERSION

Provides easy-to-use prediction capabilities for both basic KD and FedKD models:
- Single molecule prediction
- Batch prediction
- Model comparison
"""

import torch
import pandas as pd
import numpy as np
from torch_geometric.data import DataLoader
import pickle
from pathlib import Path
from rdkit import Chem
import warnings
warnings.filterwarnings('ignore')

try:
    from data_processing import smiles_to_graph, compute_molecular_descriptors
    from models_basic import BasicStudentModel
    from models_fedkd import FedKDStudentModel
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure all required files are in the current directory")

def load_trained_model(model_type='basic'):
    """Load a trained model and preprocessing components"""
    print(f"📁 Loading {model_type} model...")
    
    try:
        # For existing model, use the original file
        if model_type == 'basic':
            model_file = 'models/gnn_trained_models.pth'
        else:  # fedkd
            model_file = f'models/{model_type}_trained_model.pth'
        
        if not Path(model_file).exists():
            print(f"❌ Model file not found: {model_file}")
            print(f"Available models:")
            models_dir = Path('models')
            if models_dir.exists():
                for f in models_dir.glob('*.pth'):
                    print(f"  • {f.name}")
            return None, None, None
        
        checkpoint = torch.load(model_file, map_location='cpu')
        
        # Load preprocessing components (your existing files)
        scaler_file = 'models/enhanced_scaler.pkl'
        features_file = 'models/enhanced_feature_names.txt'
        
        with open(scaler_file, 'rb') as f:
            scaler = pickle.load(f)
        
        with open(features_file, 'r') as f:
            feature_names = [line.strip() for line in f]
        
        # Create and load model
        arch_info = checkpoint['model_architecture']
        
        if model_type == 'basic':
            model = BasicStudentModel(
                node_features=arch_info['node_features'],
                edge_features=arch_info['edge_features'],
                hidden_dim=96
            )
        else:  # fedkd
            model = FedKDStudentModel(
                node_features=arch_info['node_features'],
                edge_features=arch_info['edge_features'],
                hidden_dim=96
            )
        
        model.load_state_dict(checkpoint['student_state_dict'])
        model.eval()
        
        print(f"✓ {model_type.title()} model loaded successfully")
        return model, scaler, feature_names
        
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        return None, None, None

def process_smiles_for_prediction(smiles_list, feature_names, scaler):
    """Convert SMILES to graphs and normalized descriptors"""
    print(f"🧬 Processing {len(smiles_list)} molecules...")
    
    valid_graphs = []
    valid_smiles = []
    
    for smiles in smiles_list:
        try:
            # Clean SMILES
            smiles_clean = str(smiles).strip()
            if not smiles_clean:
                continue
            
            # Parse molecule
            mol = Chem.MolFromSmiles(smiles_clean)
            if mol is None:
                continue
            
            # Create graph
            graph = smiles_to_graph(smiles_clean)
            if graph is None:
                continue
            
            # Compute descriptors
            desc_features, desc_names = compute_molecular_descriptors(mol)
            if len(desc_features) == 0:
                continue
            
            # Ensure consistent ordering
            desc_dict = dict(zip(desc_names, desc_features))
            ordered_features = []
            for feat_name in feature_names:
                ordered_features.append(desc_dict.get(feat_name, 0.0))
            
            # Normalize descriptors
            descriptors_normalized = scaler.transform([ordered_features])
            graph.desc_features = torch.tensor(descriptors_normalized[0], dtype=torch.float)
            
            valid_graphs.append(graph)
            valid_smiles.append(Chem.MolToSmiles(mol))
            
        except Exception as e:
            print(f"Warning: Could not process SMILES {smiles}: {e}")
            continue
    
    print(f"✓ Successfully processed {len(valid_graphs)}/{len(smiles_list)} molecules")
    return valid_graphs, valid_smiles

def make_predictions(model, graphs, model_type='basic'):
    """Make predictions with a trained model"""
    
    data_loader = DataLoader(graphs, batch_size=32, shuffle=False)
    
    predictions = {'normal': [], 'mcc26': [], 'mkl1': []}
    
    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            try:
                if model_type == 'basic':
                    preds = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                else:  # fedkd
                    preds = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, 'all', False)
                
                predictions['normal'].extend(preds['normal'].cpu().numpy().flatten())
                predictions['mcc26'].extend(preds['mcc26'].cpu().numpy().flatten())
                predictions['mkl1'].extend(preds['mkl1'].cpu().numpy().flatten())
            except Exception as e:
                print(f"Warning: Batch prediction failed: {e}")
                continue
    
    return predictions

def format_results(smiles_list, predictions, thresholds=None):
    """Format predictions into a results dataframe"""
    
    if thresholds is None:
        thresholds = {'normal': 0.5, 'mcc26': 0.5, 'mkl1': 0.5}
    
    results = []
    
    for i, smiles in enumerate(smiles_list):
        if i >= len(predictions['normal']):
            break
            
        result = {
            'SMILES': smiles,
            'normal_safety_prob': float(predictions['normal'][i]),
            'mcc26_activity_prob': float(predictions['mcc26'][i]),
            'mkl1_activity_prob': float(predictions['mkl1'][i]),
            'normal_safe': predictions['normal'][i] > thresholds['normal'],
            'mcc26_active': predictions['mcc26'][i] > thresholds['mcc26'],
            'mkl1_active': predictions['mkl1'][i] > thresholds['mkl1']
        }
        
        # Calculate selectivity
        result['selectivity_mcc26'] = result['normal_safety_prob'] * result['mcc26_activity_prob']
        result['selectivity_mkl1'] = result['normal_safety_prob'] * result['mkl1_activity_prob']
        result['max_selectivity'] = max(result['selectivity_mcc26'], result['selectivity_mkl1'])
        
        # Selective hit
        result['selective_hit'] = (
            result['normal_safe'] and 
            (result['mcc26_active'] or result['mkl1_active'])
        )
        
        results.append(result)
    
    return pd.DataFrame(results)

def predict_with_model(smiles_list, model_type='basic', thresholds=None):
    """Complete prediction pipeline for a model type"""
    
    # Load model
    model, scaler, feature_names = load_trained_model(model_type)
    if model is None:
        return None
    
    # Process molecules
    graphs, valid_smiles = process_smiles_for_prediction(smiles_list, feature_names, scaler)
    if len(graphs) == 0:
        print("❌ No valid molecules to predict")
        return None
    
    # Make predictions
    predictions = make_predictions(model, graphs, model_type)
    
    # Format results
    results = format_results(valid_smiles, predictions, thresholds)
    
    return results

def compare_predictions(smiles_list, thresholds=None):
    """Compare predictions from both basic KD and FedKD models"""
    print("🔍 Comparing Basic KD vs FedKD predictions...")
    
    # Get predictions from both models
    basic_results = predict_with_model(smiles_list, 'basic', thresholds)
    
    # Try FedKD, but handle if not available
    fedkd_results = predict_with_model(smiles_list, 'fedkd', thresholds)
    
    if basic_results is None:
        print("❌ Could not get basic model predictions")
        return None, None, None
    
    if fedkd_results is None:
        print("⚠️  FedKD model not available, showing only basic model results")
        return basic_results, None, None
    
    # Create comparison dataframe
    comparison = pd.DataFrame()
    comparison['SMILES'] = basic_results['SMILES']
    
    # Add probability predictions
    comparison['basic_normal'] = basic_results['normal_safety_prob']
    comparison['fedkd_normal'] = fedkd_results['normal_safety_prob']
    comparison['basic_mcc26'] = basic_results['mcc26_activity_prob']
    comparison['fedkd_mcc26'] = fedkd_results['mcc26_activity_prob']
    comparison['basic_mkl1'] = basic_results['mkl1_activity_prob']
    comparison['fedkd_mkl1'] = fedkd_results['mkl1_activity_prob']
    
    # Add selectivity scores
    comparison['basic_selectivity'] = basic_results['max_selectivity']
    comparison['fedkd_selectivity'] = fedkd_results['max_selectivity']
    
    # Add hit predictions
    comparison['basic_hit'] = basic_results['selective_hit']
    comparison['fedkd_hit'] = fedkd_results['selective_hit']
    
    # Calculate differences
    comparison['normal_diff'] = comparison['fedkd_normal'] - comparison['basic_normal']
    comparison['mcc26_diff'] = comparison['fedkd_mcc26'] - comparison['basic_mcc26']
    comparison['mkl1_diff'] = comparison['fedkd_mkl1'] - comparison['basic_mkl1']
    comparison['selectivity_diff'] = comparison['fedkd_selectivity'] - comparison['basic_selectivity']
    
    return basic_results, fedkd_results, comparison

def analyze_predictions(results_df, model_name="Model"):
    """Analyze prediction results"""
    print(f"\n📊 {model_name} Prediction Analysis:")
    print(f"  • Total compounds: {len(results_df)}")
    print(f"  • Normal safe: {results_df['normal_safe'].sum()} ({results_df['normal_safe'].mean()*100:.1f}%)")
    print(f"  • MCC26 active: {results_df['mcc26_active'].sum()} ({results_df['mcc26_active'].mean()*100:.1f}%)")
    print(f"  • MKL1 active: {results_df['mkl1_active'].sum()} ({results_df['mkl1_active'].mean()*100:.1f}%)")
    print(f"  • Selective hits: {results_df['selective_hit'].sum()} ({results_df['selective_hit'].mean()*100:.1f}%)")
    print(f"  • Average selectivity: {results_df['max_selectivity'].mean():.3f}")

def test_prediction_interface():
    """Test the prediction interface with example molecules"""
    print("🧪 Testing prediction interface...")
    
    # Test molecules
    test_smiles = [
        'CCO',  # Ethanol
        'CC(=O)OC1=CC=CC=C1C(=O)O',  # Aspirin
        'CN1CCC[C@H]1c2cccnc2',  # Nicotine
        'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O',  # Ibuprofen
        'C1=CC=C(C=C1)C2=CC=CC=N2'  # 2-Phenylpyridine
    ]
    
    compound_names = ['Ethanol', 'Aspirin', 'Nicotine', 'Ibuprofen', '2-Phenylpyridine']
    
    print(f"\nTesting with {len(test_smiles)} molecules...")
    
    # Compare predictions
    basic_results, fedkd_results, comparison = compare_predictions(test_smiles)
    
    if basic_results is not None:
        print("\n📋 Prediction Results:")
        print("="*100)
        print(f"{'Compound':<15} {'Model':<8} {'Normal':<8} {'MCC26':<8} {'MKL1':<8} {'Selectivity':<12} {'Hit':<5}")
        print("-"*100)
        
        for i, name in enumerate(compound_names[:len(basic_results)]):
            # Basic results
            row = basic_results.iloc[i]
            print(f"{name:<15} {'Basic':<8} {row['normal_safety_prob']:<8.3f} "
                  f"{row['mcc26_activity_prob']:<8.3f} {row['mkl1_activity_prob']:<8.3f} "
                  f"{row['max_selectivity']:<12.3f} {'✓' if row['selective_hit'] else '✗':<5}")
            
            # FedKD results (if available)
            if fedkd_results is not None and i < len(fedkd_results):
                fed_row = fedkd_results.iloc[i]
                print(f"{'':<15} {'FedKD':<8} {fed_row['normal_safety_prob']:<8.3f} "
                      f"{fed_row['mcc26_activity_prob']:<8.3f} {fed_row['mkl1_activity_prob']:<8.3f} "
                      f"{fed_row['max_selectivity']:<12.3f} {'✓' if fed_row['selective_hit'] else '✗':<5}")
                
                # Differences
                if comparison is not None and i < len(comparison):
                    comp_row = comparison.iloc[i]
                    print(f"{'':<15} {'Diff':<8} {comp_row['normal_diff']:<8.3f} "
                          f"{comp_row['mcc26_diff']:<8.3f} {comp_row['mkl1_diff']:<8.3f} "
                          f"{comp_row['selectivity_diff']:<12.3f} {'':<5}")
            
            print("-"*100)
        
        # Analysis
        analyze_predictions(basic_results, "Basic Model")
        if fedkd_results is not None:
            analyze_predictions(fedkd_results, "FedKD Model")
        
        return basic_results, fedkd_results, comparison
    
    return None, None, None

def main():
    """Interactive prediction interface"""
    print("🧬 SIMPLIFIED PREDICTION INTERFACE")
    print("="*60)
    
    while True:
        print("\nOptions:")
        print("1. Predict single molecule")
        print("2. Test with example molecules")
        print("3. Compare models (if both available)")
        print("4. Exit")
        
        try:
            choice = input("\nSelect option (1-4): ").strip()
            
            if choice == '1':
                smiles = input("Enter SMILES: ").strip()
                if smiles:
                    print(f"\nPredicting for: {smiles}")
                    
                    basic_results = predict_with_model([smiles], 'basic')
                    
                    if basic_results is not None and len(basic_results) > 0:
                        row = basic_results.iloc[0]
                        print(f"\nBasic Model Results:")
                        print(f"  Normal safety: {row['normal_safety_prob']:.3f} ({'SAFE' if row['normal_safe'] else 'RISKY'})")
                        print(f"  MCC26 activity: {row['mcc26_activity_prob']:.3f} ({'ACTIVE' if row['mcc26_active'] else 'INACTIVE'})")
                        print(f"  MKL1 activity: {row['mkl1_activity_prob']:.3f} ({'ACTIVE' if row['mkl1_active'] else 'INACTIVE'})")
                        print(f"  Selective hit: {'YES ✓' if row['selective_hit'] else 'NO'}")
                        print(f"  Selectivity score: {row['max_selectivity']:.3f}")
                    else:
                        print(f"❌ Could not process SMILES: {smiles}")
            
            elif choice == '2':
                test_prediction_interface()
            
            elif choice == '3':
                smiles_input = input("Enter SMILES (comma-separated): ").strip()
                if smiles_input:
                    smiles_list = [s.strip() for s in smiles_input.split(',')]
                    basic_results, fedkd_results, comparison = compare_predictions(smiles_list)
                    
                    if comparison is not None:
                        print(f"\nComparison completed. Key differences:")
                        print(f"  • Selectivity improvement: {comparison['selectivity_diff'].mean():+.3f}")
                        agreement = (comparison['basic_hit'] == comparison['fedkd_hit']).mean()
                        print(f"  • Hit agreement: {agreement*100:.1f}%")
                    elif basic_results is not None:
                        print(f"\nBasic model predictions completed.")
                        analyze_predictions(basic_results, "Basic Model")
            
            elif choice == '4':
                print("👋 Goodbye!")
                break
            
            else:
                print("Invalid option. Please try again.")
        
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main()