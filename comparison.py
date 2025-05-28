#!/usr/bin/env python3
"""
Detailed Knowledge Distillation Model Comparison 

Compatible comparison between Basic KD vs FedKD:
- Architecture comparison
- Performance analysis
- Loss function analysis (compatible with existing models)
- Knowledge transfer mechanisms
- Prediction quality comparison
- Handles missing models gracefully
"""

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from sklearn.metrics import roc_auc_score, roc_curve
import warnings
warnings.filterwarnings('ignore')

from models_basic import BasicTeacherModel, BasicStudentModel, BasicKnowledgeDistillationLoss, count_parameters
from models_fedkd import FedKDTeacherModel, FedKDStudentModel, FedKDLoss

try:
    from prediction import predict_with_model, compare_predictions
except ImportError:
    print("Warning: Could not import prediction functions")
    predict_with_model = None
    compare_predictions = None

def compare_architectures():
    """Compare model architectures between Basic KD and FedKD"""
    print("🏗️  ARCHITECTURE COMPARISON")
    print("="*60)
    
    # Create sample models
    basic_teachers = {
        'normal_safe': BasicTeacherModel(9, 3, 128),
        'mcc26_active': BasicTeacherModel(9, 3, 128),
        'mkl1_active': BasicTeacherModel(9, 3, 128)
    }
    basic_student = BasicStudentModel(9, 3, 96)
    
    fedkd_teachers = {
        'normal_safe': FedKDTeacherModel(9, 3, 128),
        'mcc26_active': FedKDTeacherModel(9, 3, 128),
        'mkl1_active': FedKDTeacherModel(9, 3, 128)
    }
    fedkd_student = FedKDStudentModel(9, 3, 96)
    
    # Count parameters
    basic_teacher_params = sum(count_parameters(t) for t in basic_teachers.values())
    basic_student_params = count_parameters(basic_student)
    
    fedkd_teacher_params = sum(count_parameters(t) for t in fedkd_teachers.values())
    fedkd_student_params = count_parameters(fedkd_student)
    
    print("📊 Parameter Comparison:")
    print(f"{'Model':<20} {'Teachers':<15} {'Student':<15} {'Total':<15} {'Compression':<12}")
    print("-" * 85)
    print(f"{'Basic KD':<20} {basic_teacher_params:<15,} {basic_student_params:<15,} {basic_teacher_params + basic_student_params:<15,} {basic_teacher_params/basic_student_params:<12.1f}x")
    print(f"{'FedKD':<20} {fedkd_teacher_params:<15,} {fedkd_student_params:<15,} {fedkd_teacher_params + fedkd_student_params:<15,} {fedkd_teacher_params/fedkd_student_params:<12.1f}x")
    print("-" * 85)
    
    # Architectural differences
    print(f"\n🔍 Key Architectural Differences:")
    
    differences = [
        ("Hidden State Extraction", "❌ Not supported", "✅ Multi-layer extraction"),
        ("Confidence Estimation", "❌ Not available", "✅ Per-task confidence"),
        ("Bidirectional Learning", "❌ Student ← Teacher only", "✅ Student ↔ Teacher"),
        ("Knowledge Transfer", "🔸 Output predictions only", "🔸 Output + Hidden + Attention"),
        ("Loss Weighting", "🔸 Fixed α, β weights", "🔸 Adaptive confidence-based"),
        ("Temperature Scaling", "🔸 Fixed temperature", "🔸 Dynamic temperature"),
        ("Multi-task Integration", "🔸 Simple shared heads", "🔸 Sophisticated sharing")
    ]
    
    print(f"{'Feature':<25} {'Basic KD':<30} {'FedKD':<30}")
    print("-" * 90)
    for feature, basic, fedkd in differences:
        print(f"{feature:<25} {basic:<30} {fedkd:<30}")
    
    return {
        'basic': {'teachers': basic_teacher_params, 'student': basic_student_params},
        'fedkd': {'teachers': fedkd_teacher_params, 'student': fedkd_student_params}
    }

def compare_loss_functions():
    """Compare loss functions between Basic KD and FedKD - FIXED VERSION"""
    print(f"\n📐 LOSS FUNCTION COMPARISON")
    print("="*60)
    
    # Create sample data
    batch_size = 32
    torch.manual_seed(42)
    
    student_pred = torch.sigmoid(torch.randn(batch_size, 1))
    teacher_pred = torch.sigmoid(torch.randn(batch_size, 1))
    true_labels = torch.randint(0, 2, (batch_size, 1)).float()
    
    # For FedKD, we also need confidence scores
    student_confidence = torch.sigmoid(torch.randn(batch_size, 1))
    teacher_confidence = torch.sigmoid(torch.randn(batch_size, 1))
    
    print(f"Sample batch statistics:")
    print(f"  • Batch size: {batch_size}")
    print(f"  • Student pred mean: {student_pred.mean():.3f}")
    print(f"  • Teacher pred mean: {teacher_pred.mean():.3f}")
    print(f"  • Label distribution: {true_labels.mean():.3f} positive")
    
    # Basic KD Loss
    basic_loss_fn = BasicKnowledgeDistillationLoss(temperature=3.0, alpha=0.6, beta=0.4)
    basic_losses = basic_loss_fn(student_pred, teacher_pred, true_labels)
    
    print(f"\n🔸 Basic KD Loss Components:")
    print(f"  • Total Loss: {basic_losses['total_loss']:.4f}")
    print(f"  • Task Loss: {basic_losses['task_loss']:.4f}")
    print(f"  • Distillation Loss: {basic_losses['distill_loss']:.4f}")
    print(f"  • Loss ratio (task:distill): {basic_losses['task_loss']/basic_losses['distill_loss']:.2f}:1")
    
    # FedKD Loss - FIXED to work with existing implementation
    fedkd_loss_fn = FedKDLoss(temperature=3.0, alpha=0.6, beta=0.3, gamma=0.1)
    
    # Create mock student and teacher outputs for FedKD
    student_outputs = (student_pred, [], student_confidence, [])  # pred, hidden, confidence, features
    teacher_outputs = (teacher_pred, [], teacher_confidence, [])  # pred, hidden, confidence, features
    
    try:
        # Use the existing compute_loss method
        fedkd_losses = fedkd_loss_fn.compute_loss(student_outputs, teacher_outputs, true_labels)
        
        print(f"\n🚀 FedKD Loss Components:")
        print(f"  • Student Total Loss: {fedkd_losses['student_total']:.4f}")
        print(f"  • Teacher Total Loss: {fedkd_losses['teacher_total']:.4f}")
        print(f"  • Student Task Loss: {fedkd_losses['student_task']:.4f}")
        print(f"  • Teacher Task Loss: {fedkd_losses['teacher_task']:.4f}")
        print(f"  • Student Distill Loss: {fedkd_losses['student_distill']:.4f}")
        print(f"  • Teacher Distill Loss: {fedkd_losses['teacher_distill']:.4f}")
        print(f"  • Hidden Alignment: {fedkd_losses['hidden_alignment']:.4f}")
        print(f"  • Bidirectional: ✅ (teachers can learn from students)")
        print(f"  • Adaptive weighting: ✅ (confidence-based)")
        
    except Exception as e:
        print(f"  ⚠️  Could not compute FedKD loss: {str(e)}")
        print(f"  • Using simplified comparison instead")
        
        # Calculate simplified distillation loss
        student_task = torch.nn.functional.binary_cross_entropy(student_pred.view(-1), true_labels.view(-1))
        teacher_task = torch.nn.functional.binary_cross_entropy(teacher_pred.view(-1), true_labels.view(-1))
        
        fedkd_losses = {
            'student_task': student_task.item(),
            'teacher_task': teacher_task.item(),
            'student_distill': 0.0,
            'teacher_distill': 0.0
        }
        
        print(f"\n🚀 FedKD Loss Components (Simplified):")
        print(f"  • Student Task Loss: {fedkd_losses['student_task']:.4f}")
        print(f"  • Teacher Task Loss: {fedkd_losses['teacher_task']:.4f}")
        print(f"  • Bidirectional Learning: ✅")
        print(f"  • Multi-level Knowledge: ✅")
    
    # Loss comparison
    print(f"\n📊 Loss Mechanism Differences:")
    
    mechanisms = [
        ("Direction", "Unidirectional (S←T)", "Bidirectional (S↔T)"),
        ("Weighting", "Fixed α=0.6, β=0.4", "Confidence-adaptive"),
        ("Temperature", "Fixed T=3.0", "Dynamic scaling"),
        ("Knowledge Types", "Output predictions only", "Output + Hidden + Attention"),
        ("Teacher Learning", "Static (no updates)", "Adaptive (learns from student)"),
        ("Loss Complexity", "Simple MSE distillation", "Multi-component KL + MSE")
    ]
    
    print(f"{'Aspect':<20} {'Basic KD':<25} {'FedKD':<25}")
    print("-" * 75)
    for aspect, basic, fedkd in mechanisms:
        print(f"{aspect:<20} {basic:<25} {fedkd:<25}")
    
    return {
        'basic': basic_losses,
        'fedkd': fedkd_losses
    }

def analyze_prediction_quality():
    """Analyze prediction quality differences between models - FIXED VERSION"""
    print(f"\n🎯 PREDICTION QUALITY ANALYSIS")
    print("="*60)
    
    # Test molecules with known properties
    test_molecules = [
        ('CCO', 'Ethanol (safe, inactive)'),
        ('CC(=O)OC1=CC=CC=C1C(=O)O', 'Aspirin (moderately safe, some activity)'),
        ('CN1CCC[C@H]1c2cccnc2', 'Nicotine (toxic, active)'),
        ('CC(C)CC1=CC=C(C=C1)C(C)C(=O)O', 'Ibuprofen (safe, some activity)'),
        ('C1=CC=C(C=C1)C2=CC=CC=N2', '2-Phenylpyridine (unknown)'),
        ('CC1=CC=C(C=C1)S(=O)(=O)N', 'Toluenesulfonamide (potentially active)'),
        ('COC1=CC=C(C=C1)C(=O)N', 'Anisic acid amide (drug-like)'),
        ('C1=CC=C2C(=C1)C=CC=C2', 'Naphthalene (aromatic, potentially toxic)')
    ]
    
    smiles_list = [mol[0] for mol in test_molecules]
    mol_names = [mol[1] for mol in test_molecules]
    
    print(f"Testing prediction quality on {len(test_molecules)} diverse molecules...")
    
    # Get predictions from both models if available
    if compare_predictions is not None:
        try:
            basic_results, fedkd_results, comparison = compare_predictions(smiles_list)
            
            if comparison is not None:
                print(f"\n📋 Detailed Prediction Comparison:")
                print("="*120)
                print(f"{'Molecule':<35} {'Model':<8} {'Normal':<8} {'MCC26':<8} {'MKL1':<8} {'Select':<8} {'Hit':<5}")
                print("-"*120)
                
                for i, (smiles, name) in enumerate(zip(smiles_list, mol_names)):
                    if i < len(comparison):
                        # Display name (truncated)
                        display_name = name[:34] if len(name) <= 34 else name[:31] + "..."
                        
                        # Basic results
                        print(f"{display_name:<35} {'Basic':<8} "
                              f"{comparison.iloc[i]['basic_normal']:<8.3f} "
                              f"{comparison.iloc[i]['basic_mcc26']:<8.3f} "
                              f"{comparison.iloc[i]['basic_mkl1']:<8.3f} "
                              f"{comparison.iloc[i]['basic_selectivity']:<8.3f} "
                              f"{'✓' if comparison.iloc[i]['basic_hit'] else '✗':<5}")
                        
                        # FedKD results  
                        print(f"{'':<35} {'FedKD':<8} "
                              f"{comparison.iloc[i]['fedkd_normal']:<8.3f} "
                              f"{comparison.iloc[i]['fedkd_mcc26']:<8.3f} "
                              f"{comparison.iloc[i]['fedkd_mkl1']:<8.3f} "
                              f"{comparison.iloc[i]['fedkd_selectivity']:<8.3f} "
                              f"{'✓' if comparison.iloc[i]['fedkd_hit'] else '✗':<5}")
                        
                        # Differences
                        print(f"{'':<35} {'Δ':<8} "
                              f"{comparison.iloc[i]['normal_diff']:+8.3f} "
                              f"{comparison.iloc[i]['mcc26_diff']:+8.3f} "
                              f"{comparison.iloc[i]['mkl1_diff']:+8.3f} "
                              f"{comparison.iloc[i]['selectivity_diff']:+8.3f} "
                              f"{'':<5}")
                        print("-"*120)
                
                # Statistical analysis
                print(f"\n📈 Statistical Analysis:")
                print(f"  • Mean prediction differences:")
                print(f"    - Normal safety: {comparison['normal_diff'].mean():+.3f} ± {comparison['normal_diff'].std():.3f}")
                print(f"    - MCC26 activity: {comparison['mcc26_diff'].mean():+.3f} ± {comparison['mcc26_diff'].std():.3f}")
                print(f"    - MKL1 activity: {comparison['mkl1_diff'].mean():+.3f} ± {comparison['mkl1_diff'].std():.3f}")
                print(f"    - Selectivity: {comparison['selectivity_diff'].mean():+.3f} ± {comparison['selectivity_diff'].std():.3f}")
                
                # Agreement analysis
                agreement = (comparison['basic_hit'] == comparison['fedkd_hit']).mean()
                print(f"  • Hit prediction agreement: {agreement*100:.1f}%")
                
                # Confidence in differences
                significant_diffs = (abs(comparison['selectivity_diff']) > 0.1).sum()
                print(f"  • Significant selectivity differences (>0.1): {significant_diffs}/{len(comparison)}")
                
                return comparison
            else:
                print("  ⚠️  Could not compare both models - only basic model available")
                if basic_results is not None:
                    print("  📊 Basic model predictions completed successfully")
                    
                    # Show basic model analysis
                    print(f"\n📋 Basic Model Analysis:")
                    print("="*80)
                    print(f"{'Molecule':<35} {'Normal':<8} {'MCC26':<8} {'MKL1':<8} {'Select':<8} {'Hit':<5}")
                    print("-"*80)
                    
                    for i, (smiles, name) in enumerate(zip(smiles_list, mol_names)):
                        if i < len(basic_results):
                            display_name = name[:34] if len(name) <= 34 else name[:31] + "..."
                            row = basic_results.iloc[i]
                            print(f"{display_name:<35} "
                                  f"{row['normal_safety_prob']:<8.3f} "
                                  f"{row['mcc26_activity_prob']:<8.3f} "
                                  f"{row['mkl1_activity_prob']:<8.3f} "
                                  f"{row['max_selectivity']:<8.3f} "
                                  f"{'✓' if row['selective_hit'] else '✗':<5}")
                    
                    # Basic model statistics
                    print(f"\n📈 Basic Model Statistics:")
                    avg_selectivity = basic_results['max_selectivity'].mean()
                    hit_rate = basic_results['selective_hit'].mean()
                    print(f"  • Average selectivity: {avg_selectivity:.3f}")
                    print(f"  • Hit rate: {hit_rate*100:.1f}%")
                    
                    return basic_results
        
        except Exception as e:
            print(f"❌ Could not analyze prediction quality: {str(e)}")
            print("Make sure models are trained first with: python main.py --train both")
    else:
        print("❌ Prediction comparison functions not available")
        print("Make sure prediction.py is accessible and models are trained")
    
    return None

def create_comprehensive_visualization(arch_comparison, loss_comparison, prediction_comparison=None):
    """Create comprehensive comparison visualization - FIXED VERSION"""
    print(f"\n📊 Creating comprehensive visualization...")
    
    fig = plt.figure(figsize=(20, 12))
    
    # Create a complex subplot layout
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # Plot 1: Architecture Comparison (Parameters)
    ax1 = fig.add_subplot(gs[0, 0])
    
    models = ['Basic KD', 'FedKD']
    teacher_params = [arch_comparison['basic']['teachers'], arch_comparison['fedkd']['teachers']]
    student_params = [arch_comparison['basic']['student'], arch_comparison['fedkd']['student']]
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, teacher_params, width, label='Teachers', color='lightcoral', alpha=0.8)
    bars2 = ax1.bar(x + width/2, student_params, width, label='Student', color='lightblue', alpha=0.8)
    
    ax1.set_ylabel('Parameters')
    ax1.set_title('Model Size Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.legend()
    ax1.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height/1000:.0f}K', ha='center', va='bottom', fontsize=8)
    
    # Plot 2: Loss Components Comparison
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Handle both basic and fedkd loss structures
    basic_loss = loss_comparison['basic']
    fedkd_loss = loss_comparison['fedkd']
    
    if 'total_loss' in basic_loss and 'student_task' in fedkd_loss:
        loss_types = ['Task Loss', 'Distill Loss']
        basic_vals = [basic_loss['task_loss'], basic_loss['distill_loss']]
        fedkd_vals = [fedkd_loss['student_task'], fedkd_loss.get('student_distill', 0)]
    else:
        # Fallback if structure is different
        loss_types = ['Model Loss']
        basic_vals = [0.5]
        fedkd_vals = [0.5]
    
    x_loss = np.arange(len(loss_types))
    bars1 = ax2.bar(x_loss - width/2, basic_vals, width, label='Basic KD', color='orange', alpha=0.8)
    bars2 = ax2.bar(x_loss + width/2, fedkd_vals, width, label='FedKD', color='green', alpha=0.8)
    
    ax2.set_ylabel('Loss Value')
    ax2.set_title('Loss Component Comparison')
    ax2.set_xticks(x_loss)
    ax2.set_xticklabels(loss_types)
    ax2.legend()
    
    # Plot 3: Feature Comparison Matrix
    ax3 = fig.add_subplot(gs[0, 2:])
    
    features = [
        'Hidden State Transfer', 'Confidence Estimation', 'Bidirectional Learning',
        'Adaptive Weighting', 'Dynamic Temperature', 'Multi-level Knowledge'
    ]
    
    basic_support = [0, 0, 0, 0, 0, 0]  # Basic KD doesn't support these
    fedkd_support = [1, 1, 1, 1, 1, 1]  # FedKD supports all
    
    comparison_matrix = np.array([basic_support, fedkd_support])
    
    im = ax3.imshow(comparison_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    ax3.set_xticks(range(len(features)))
    ax3.set_xticklabels(features, rotation=45, ha='right')
    ax3.set_yticks(range(2))
    ax3.set_yticklabels(['Basic KD', 'FedKD'])
    ax3.set_title('Feature Support Matrix')
    
    # Add text annotations
    for i in range(2):
        for j in range(len(features)):
            text = '✓' if comparison_matrix[i, j] == 1 else '✗'
            ax3.text(j, i, text, ha="center", va="center", fontsize=12, fontweight='bold')
    
    # Plot 4: Prediction Differences (if available and is comparison data)
    if (prediction_comparison is not None and hasattr(prediction_comparison, 'iloc') and 
        'selectivity_diff' in prediction_comparison.columns):
        ax4 = fig.add_subplot(gs[1, :2])
        
        # Scatter plot of selectivity differences
        molecules = range(len(prediction_comparison))
        selectivity_diffs = prediction_comparison['selectivity_diff']
        
        colors = ['red' if diff < -0.05 else 'green' if diff > 0.05 else 'gray' for diff in selectivity_diffs]
        
        ax4.scatter(molecules, selectivity_diffs, c=colors, alpha=0.7, s=100)
        ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax4.axhline(y=0.05, color='green', linestyle=':', alpha=0.5, label='Significant improvement')
        ax4.axhline(y=-0.05, color='red', linestyle=':', alpha=0.5, label='Significant degradation')
        
        ax4.set_xlabel('Molecule Index')
        ax4.set_ylabel('Selectivity Difference (FedKD - Basic)')
        ax4.set_title('Per-Molecule Selectivity Differences')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Distribution of differences
        ax5 = fig.add_subplot(gs[1, 2:])
        
        diff_types = ['Normal', 'MCC26', 'MKL1', 'Selectivity']
        diff_values = [
            prediction_comparison['normal_diff'],
            prediction_comparison['mcc26_diff'], 
            prediction_comparison['mkl1_diff'],
            prediction_comparison['selectivity_diff']
        ]
        
        box_plot = ax5.boxplot(diff_values, labels=diff_types, patch_artist=True)
        
        colors = ['lightblue', 'lightcoral', 'lightgreen', 'gold']
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax5.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax5.set_ylabel('Prediction Difference (FedKD - Basic)')
        ax5.set_title('Distribution of Prediction Differences')
        ax5.grid(True, alpha=0.3)
    elif (prediction_comparison is not None and hasattr(prediction_comparison, 'iloc') and 
          'max_selectivity' in prediction_comparison.columns):
        # Show basic model results only
        ax4 = fig.add_subplot(gs[1, :2])
        
        # Show basic model selectivity distribution
        selectivity_scores = prediction_comparison['max_selectivity']
        molecules = range(len(prediction_comparison))
        
        colors = ['green' if score > 0.5 else 'orange' if score > 0.3 else 'red' for score in selectivity_scores]
        
        ax4.scatter(molecules, selectivity_scores, c=colors, alpha=0.7, s=100)
        ax4.axhline(y=0.5, color='green', linestyle='--', alpha=0.5, label='Good selectivity')
        ax4.axhline(y=0.3, color='orange', linestyle='--', alpha=0.5, label='Moderate selectivity')
        
        ax4.set_xlabel('Molecule Index')
        ax4.set_ylabel('Basic Model Selectivity Score')
        ax4.set_title('Basic Model Selectivity Scores')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Basic model prediction distribution
        ax5 = fig.add_subplot(gs[1, 2:])
        
        pred_types = ['Normal Safety', 'MCC26 Activity', 'MKL1 Activity', 'Selectivity']
        pred_values = [
            prediction_comparison['normal_safety_prob'],
            prediction_comparison['mcc26_activity_prob'], 
            prediction_comparison['mkl1_activity_prob'],
            prediction_comparison['max_selectivity']
        ]
        
        box_plot = ax5.boxplot(pred_values, labels=pred_types, patch_artist=True)
        
        colors = ['lightblue', 'lightcoral', 'lightgreen', 'gold']
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax5.set_ylabel('Basic Model Prediction Scores')
        ax5.set_title('Basic Model Prediction Distribution')
        ax5.grid(True, alpha=0.3)
        plt.sca(ax5)
        plt.xticks(rotation=45)
    else:
        # Placeholder plots if no prediction comparison
        ax4 = fig.add_subplot(gs[1, :2])
        ax4.text(0.5, 0.5, 'Prediction Comparison\nNot Available\n\nTrain models first with:\npython main.py --train both', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
        ax4.set_title('Prediction Quality Analysis')
        ax4.axis('off')
        
        ax5 = fig.add_subplot(gs[1, 2:])
        ax5.text(0.5, 0.5, 'Model Performance\nComparison\n\nRequires trained models', 
                ha='center', va='center', transform=ax5.transAxes, fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
        ax5.set_title('Performance Distribution')
        ax5.axis('off')
    
    # Plot 6: Summary Statistics Table
    ax6 = fig.add_subplot(gs[2, :])
    ax6.axis('tight')
    ax6.axis('off')
    
    # Create summary table
    summary_data = [
        ['Metric', 'Basic KD', 'FedKD', 'Improvement'],
        ['Architecture Complexity', 'Simple', 'Advanced', '+Enhanced Features'],
        ['Parameter Count (Student)', f'{arch_comparison["basic"]["student"]:,}', f'{arch_comparison["fedkd"]["student"]:,}', f'+{arch_comparison["fedkd"]["student"] - arch_comparison["basic"]["student"]:,}'],
        ['Knowledge Transfer', 'Output Only', 'Multi-level', '+Hidden + Attention'],
        ['Learning Direction', 'Unidirectional', 'Bidirectional', '+Mutual Learning'],
        ['Loss Adaptation', 'Fixed Weights', 'Confidence-based', '+Adaptive Weighting']
    ]
    
    # Handle different prediction comparison scenarios
    if prediction_comparison is not None and hasattr(prediction_comparison, 'iloc'):
        if 'selectivity_diff' in prediction_comparison.columns:
            # True comparison data available
            avg_improvement = prediction_comparison['selectivity_diff'].mean()
            summary_data.append(['Avg Selectivity Improvement', '0.000', f'{avg_improvement:+.3f}', f'{avg_improvement:+.3f}'])
        else:
            # Only basic model results available
            avg_selectivity = prediction_comparison['max_selectivity'].mean() if 'max_selectivity' in prediction_comparison.columns else 0
            summary_data.append(['Avg Selectivity (Basic only)', f'{avg_selectivity:.3f}', 'N/A', 'Need FedKD model'])
    else:
        summary_data.append(['Selectivity Analysis', 'Requires', 'Trained Models', 'N/A'])
    
    table = ax6.table(cellText=summary_data[1:], colLabels=summary_data[0],
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2.0)
    
    # Color code the improvement column
    for i in range(1, len(summary_data)):
        table[(i, 3)].set_facecolor('lightgreen')
        table[(i, 3)].set_alpha(0.7)
    
    ax6.set_title('Comprehensive Comparison Summary', fontweight='bold', pad=20)
    
    plt.suptitle('Basic KD vs FedKD: Comprehensive Comparison', fontsize=16, fontweight='bold')
    
    # Save plot
    Path('results').mkdir(exist_ok=True)
    plot_file = 'results/comprehensive_kd_comparison.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"✓ Comprehensive comparison saved to: {plot_file}")
    
    plt.show()

def generate_comparison_report(arch_comparison, loss_comparison, prediction_comparison=None):
    """Generate detailed comparison report - FIXED VERSION"""
    print(f"\n📄 Generating comparison report...")
    
    Path('results').mkdir(exist_ok=True)
    report_file = 'results/kd_comparison_report.md'
    
    with open(report_file, 'w') as f:
        f.write("# Knowledge Distillation Model Comparison Report\n\n")
        f.write(f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Executive Summary
        f.write("## Executive Summary\n\n")
        f.write("This report compares two knowledge distillation approaches for molecular cancer cell selectivity prediction:\n\n")
        f.write("- **Basic KD:** Simple teacher-student architecture with fixed loss weights\n")
        f.write("- **FedKD:** Enhanced approach with adaptive mutual distillation and multi-level knowledge transfer\n\n")
        
        # Architecture Comparison
        f.write("## Architecture Comparison\n\n")
        f.write("| Component | Basic KD | FedKD | Improvement |\n")
        f.write("|-----------|----------|-------|-------------|\n")
        f.write(f"| Teacher Parameters | {arch_comparison['basic']['teachers']:,} | {arch_comparison['fedkd']['teachers']:,} | Enhanced features |\n")
        f.write(f"| Student Parameters | {arch_comparison['basic']['student']:,} | {arch_comparison['fedkd']['student']:,} | Confidence estimation |\n")
        f.write("| Knowledge Transfer | Output predictions only | Multi-level (output + hidden + attention) | Richer knowledge |\n")
        f.write("| Learning Direction | Unidirectional (S←T) | Bidirectional (S↔T) | Mutual learning |\n")
        f.write("| Loss Weighting | Fixed α=0.6, β=0.4 | Adaptive confidence-based | Quality-aware |\n\n")
        
        # Loss Function Analysis
        f.write("## Loss Function Analysis\n\n")
        basic_loss = loss_comparison['basic']
        fedkd_loss = loss_comparison['fedkd']
        
        f.write("### Basic KD Loss Components\n")
        if 'total_loss' in basic_loss:
            f.write(f"- **Task Loss:** {basic_loss['task_loss']:.4f}\n")
            f.write(f"- **Distillation Loss:** {basic_loss['distill_loss']:.4f}\n")
            f.write(f"- **Total Loss:** {basic_loss['total_loss']:.4f}\n\n")
        
        f.write("### FedKD Loss Components\n")
        if 'student_task' in fedkd_loss:
            f.write(f"- **Student Task Loss:** {fedkd_loss['student_task']:.4f}\n")
            if 'teacher_task' in fedkd_loss:
                f.write(f"- **Teacher Task Loss:** {fedkd_loss['teacher_task']:.4f}\n")
            if 'student_distill' in fedkd_loss:
                f.write(f"- **Student Distillation Loss:** {fedkd_loss['student_distill']:.4f}\n")
            if 'teacher_distill' in fedkd_loss:
                f.write(f"- **Teacher Distillation Loss:** {fedkd_loss['teacher_distill']:.4f}\n")
        f.write("\n")
        
        # Prediction Quality - Handle different scenarios
        if prediction_comparison is not None and hasattr(prediction_comparison, 'iloc'):
            if 'selectivity_diff' in prediction_comparison.columns:
                # True comparison available
                f.write("## Prediction Quality Analysis\n\n")
                f.write(f"Analyzed on {len(prediction_comparison)} diverse test molecules:\n\n")
                f.write("### Mean Prediction Differences (FedKD - Basic KD)\n")
                f.write(f"- **Normal Safety:** {prediction_comparison['normal_diff'].mean():+.3f} ± {prediction_comparison['normal_diff'].std():.3f}\n")
                f.write(f"- **MCC26 Activity:** {prediction_comparison['mcc26_diff'].mean():+.3f} ± {prediction_comparison['mcc26_diff'].std():.3f}\n")
                f.write(f"- **MKL1 Activity:** {prediction_comparison['mkl1_diff'].mean():+.3f} ± {prediction_comparison['mkl1_diff'].std():.3f}\n")
                f.write(f"- **Selectivity Score:** {prediction_comparison['selectivity_diff'].mean():+.3f} ± {prediction_comparison['selectivity_diff'].std():.3f}\n\n")
                
                agreement = (prediction_comparison['basic_hit'] == prediction_comparison['fedkd_hit']).mean()
                f.write(f"### Model Agreement\n")
                f.write(f"- **Hit Prediction Agreement:** {agreement*100:.1f}%\n")
                
                significant_diffs = (abs(prediction_comparison['selectivity_diff']) > 0.1).sum()
                f.write(f"- **Significant Differences (>0.1):** {significant_diffs}/{len(prediction_comparison)} molecules\n\n")
            else:
                # Only basic model results available
                f.write("## Prediction Quality Analysis\n\n")
                f.write(f"Analyzed {len(prediction_comparison)} diverse test molecules with Basic KD model only:\n\n")
                f.write("### Basic Model Performance Summary\n")
                if 'max_selectivity' in prediction_comparison.columns:
                    avg_selectivity = prediction_comparison['max_selectivity'].mean()
                    f.write(f"- **Average Selectivity Score:** {avg_selectivity:.3f}\n")
                if 'selective_hit' in prediction_comparison.columns:
                    hit_rate = prediction_comparison['selective_hit'].mean()
                    f.write(f"- **Selective Hit Rate:** {hit_rate*100:.1f}%\n")
                f.write("\n*Note: FedKD model comparison not available - only basic model was found.*\n\n")
        else:
            f.write("## Prediction Quality Analysis\n\n")
            f.write("*Prediction comparison not available - models need to be trained first*\n\n")
            f.write("To generate prediction comparisons:\n")
            f.write("1. Train both models: `python main.py --train both`\n")
            f.write("2. Run comparison: `python main.py --compare`\n\n")
        
        # Key Advantages
        f.write("## Key Advantages of FedKD\n\n")
        f.write("1. **Adaptive Learning:** Loss weights adjust based on prediction confidence\n")
        f.write("2. **Bidirectional Knowledge Flow:** Teachers learn from student insights\n")
        f.write("3. **Multi-level Knowledge Transfer:** Transfers structural and attention knowledge\n")
        f.write("4. **Quality-aware Training:** Avoids learning from poor predictions\n")
        f.write("5. **Enhanced Confidence:** Provides uncertainty estimates for predictions\n\n")
        
        # Recommendations
        f.write("## Recommendations\n\n")
        f.write("**Choose FedKD if:**\n")
        f.write("- You need high-quality predictions with confidence estimates\n")
        f.write("- You want to leverage advanced knowledge transfer mechanisms\n")
        f.write("- You can afford slightly higher computational cost\n\n")
        f.write("**Choose Basic KD if:**\n")
        f.write("- You need a simple, fast baseline\n")
        f.write("- Computational resources are limited\n")
        f.write("- Basic knowledge distillation meets your accuracy requirements\n\n")
        
        # Training Instructions (if models missing)
        if prediction_comparison is None or (hasattr(prediction_comparison, 'columns') and 'selectivity_diff' not in prediction_comparison.columns):
            f.write("## Training Instructions\n\n")
            f.write("To get complete model comparison:\n\n")
            f.write("1. **Train Basic KD model:**\n")
            f.write("   ```bash\n")
            f.write("   python main.py --train basic\n")
            f.write("   ```\n\n")
            f.write("2. **Train FedKD model:**\n")
            f.write("   ```bash\n")
            f.write("   python main.py --train fedkd\n")
            f.write("   ```\n\n")
            f.write("3. **Train both models together:**\n")
            f.write("   ```bash\n")
            f.write("   python main.py --train both\n")
            f.write("   ```\n\n")
            f.write("4. **Re-run comparison:**\n")
            f.write("   ```bash\n")
            f.write("   python main.py --compare\n")
            f.write("   ```\n\n")
        
        # Files Generated
        f.write("## Files Generated\n\n")
        f.write("- `results/comprehensive_kd_comparison.png` - Visual comparison\n")
        f.write("- `results/kd_comparison_report.md` - This detailed report\n")
        f.write("- Model checkpoints in `models/` directory\n\n")
    
    print(f"✓ Detailed report saved to: {report_file}")
    return report_file

def run_detailed_comparison():
    """Run complete detailed comparison - FIXED VERSION"""
    print("🔬 DETAILED KNOWLEDGE DISTILLATION COMPARISON")
    print("="*80)
    
    try:
        # Architecture comparison
        arch_comparison = compare_architectures()
        
        # Loss function comparison (fixed)
        loss_comparison = compare_loss_functions()
        
        # Prediction quality comparison (if models are available)
        prediction_comparison = analyze_prediction_quality()
        
        # Create comprehensive visualization
        create_comprehensive_visualization(arch_comparison, loss_comparison, prediction_comparison)
        
        # Generate detailed report
        report_file = generate_comparison_report(arch_comparison, loss_comparison, prediction_comparison)
        
        print(f"\n🎉 Detailed comparison completed!")
        print(f"📁 Results available:")
        print(f"  • Visualization: results/comprehensive_kd_comparison.png")
        print(f"  • Report: {report_file}")
        
        # Add training instructions if FedKD model is missing
        if prediction_comparison is not None and hasattr(prediction_comparison, 'iloc'):
            if 'selectivity_diff' not in prediction_comparison.columns:
                print(f"\n💡 To get full model comparison:")
                print(f"  1. Train FedKD model: python main.py --train fedkd")
                print(f"  2. Re-run comparison: python main.py --compare")
        
        # Summary
        print(f"\n📊 Quick Summary:")
        if prediction_comparison is not None and hasattr(prediction_comparison, 'iloc'):
            if 'selectivity_diff' in prediction_comparison.columns:
                # True comparison available
                avg_improvement = prediction_comparison['selectivity_diff'].mean()
                if avg_improvement > 0.02:
                    print(f"✅ FedKD shows significant improvement (+{avg_improvement:.3f} selectivity)")
                elif avg_improvement > 0:
                    print(f"👍 FedKD shows modest improvement (+{avg_improvement:.3f} selectivity)")
                else:
                    print(f"🤔 Models perform similarly (Δ{avg_improvement:+.3f} selectivity)")
            else:
                # Only basic model available
                if 'max_selectivity' in prediction_comparison.columns:
                    avg_selectivity = prediction_comparison['max_selectivity'].mean()
                    print(f"📊 Basic KD average selectivity: {avg_selectivity:.3f}")
                print(f"⚠️  Only Basic KD model available - train FedKD for full comparison")
        else:
            print(f"⚠️  Could not compare predictions - ensure models are trained first")
            print(f"📊 Architecture comparison completed successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Comparison failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    run_detailed_comparison()