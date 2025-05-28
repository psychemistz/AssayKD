#!/usr/bin/env python3
"""
Enhanced Main Pipeline Orchestrator

Comprehensive command-line interface for molecular prediction pipeline:
- Data processing with validation
- Enhanced model training (Basic KD vs Enhanced FedKD)
- Hyperparameter optimization integration
- Model evaluation and detailed comparison
- Interactive prediction interface
- Performance monitoring and analysis
- Configuration management for enhanced features
"""

import argparse
import sys
import os
import time
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Import TrainingConfig here for use in run_training
from training import TrainingConfig # ADDED IMPORT

# Setup logging
def setup_logging(log_level: str = 'INFO', log_file: Optional[str] = None):
    """Setup comprehensive logging"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Create logs directory
    Path('logs').mkdir(exist_ok=True)
    
    # Configure logging
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(f'logs/{log_file}'))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=handlers
    )
    
    return logging.getLogger(__name__)

def check_dependencies(verbose: bool = True) -> bool:
    """Check if required packages are installed with version info"""
    if verbose:
        print("🔍 Checking dependencies...")
    
    required_packages = [
        ('torch', 'PyTorch', '1.12+'),
        ('torch_geometric', 'PyTorch Geometric', '2.0+'),
        ('rdkit', 'RDKit', '2022.03+'),
        ('sklearn', 'Scikit-learn', '1.0+'),
        ('pandas', 'Pandas', '1.3+'),
        ('numpy', 'NumPy', '1.20+'),
        ('matplotlib', 'Matplotlib', '3.5+'),
        ('tqdm', 'TQDM', 'any'),
        ('optuna', 'Optuna (for hyperopt)', '2.0+')
    ]
    
    missing_packages = []
    installed_versions = {}
    
    for package, name, version_req in required_packages:
        try:
            if package == 'optuna':
                # Optuna is optional for hyperparameter optimization
                try:
                    module = __import__(package)
                    version = getattr(module, '__version__', 'unknown')
                    installed_versions[package] = version
                    if verbose:
                        print(f"  ✓ {name} ({version}) - Optional")
                except ImportError:
                    if verbose:
                        print(f"  ⚠️  {name} - MISSING (optional for hyperopt)")
                    continue
            else:
                module = __import__(package)
                version = getattr(module, '__version__', 'unknown')
                installed_versions[package] = version
                
                if verbose:
                    print(f"  ✓ {name} ({version})")
                    
        except ImportError:
            if verbose:
                print(f"  ✗ {name} - MISSING (required: {version_req})")
            missing_packages.append((package, name, version_req))
    
    if missing_packages:
        print(f"\n❌ Missing required packages: {len(missing_packages)}")
        print("Install with:")
        print("  pip install torch torch-geometric rdkit scikit-learn pandas numpy matplotlib tqdm")
        print("\nFor hyperparameter optimization, also install:")
        print("  pip install optuna optuna-integration")
        print("\nOr use conda:")
        print("  conda install pytorch pytorch-geometric rdkit scikit-learn pandas numpy matplotlib tqdm -c pytorch -c conda-forge")
        print("  pip install optuna optuna-integration")
        return False
    
    if verbose:
        print("✅ All required dependencies satisfied!")
        
        # Save dependency info
        Path('logs').mkdir(exist_ok=True)
        with open('logs/dependencies.json', 'w') as f:
            json.dump(installed_versions, f, indent=2)
    
    return True

def setup_directories(verbose: bool = True) -> None:
    """Create necessary directories with proper structure"""
    if verbose:
        print("📁 Setting up directory structure...")
    
    directories = [
        'data', 'data/graphs', 'data/processed', 'data/splits',
        'models', 'models/checkpoints', 'models/basic', 'models/fedkd',
        'results', 'results/plots', 'results/reports', 'results/comparisons',
        'results/hyperopt', 'results/hyperopt/plots',
        'logs', 'configs'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        if verbose:
            print(f"  ✓ {directory}/")
    
    if verbose:
        print("✅ Directory structure ready!")

def validate_input_file(input_file: str) -> bool:
    """Validate input file exists and has correct format"""
    if not os.path.exists(input_file):
        print(f"❌ Input file not found: {input_file}")
        return False
    
    try:
        import pandas as pd
        df = pd.read_csv(input_file, nrows=5)  # Just check first few rows
        
        required_columns = ['SMILES']
        optional_columns = ['HCT_01(%)', 'HCT_10(%)', 'MCC_01(%)', 'MCC_10(%)', 'MKL_01(%)', 'MKL_10(%)']
        
        if 'SMILES' not in df.columns:
            print(f"❌ Required column 'SMILES' not found in {input_file}")
            print(f"Available columns: {list(df.columns)}")
            return False
        
        activity_cols = [col for col in optional_columns if col in df.columns]
        if len(activity_cols) == 0:
            print(f"⚠️  No activity columns found. Will use SMILES only.")
        else:
            print(f"✓ Found activity columns: {activity_cols}")
        
        print(f"✅ Input file validation passed: {len(df)} samples (showing first 5)")
        
    except Exception as e:
        print(f"❌ Input file validation failed: {str(e)}")
        return False
    
    return True

def run_data_processing(input_file: str = 'data/raw_data.csv', 
                       force_reprocess: bool = False,
                       validation_split: float = 0.1,
                       test_split: float = 0.2) -> bool:
    """Enhanced data processing pipeline with validation"""
    print("\n🔄 Running enhanced data processing...")
    
    # Check if already processed
    processed_files = [
        'data/processed_data.csv',
        'data/graphs/molecular_graphs.pt',
        'models/scaler.pkl'
    ]
    
    if not force_reprocess and all(os.path.exists(f) for f in processed_files):
        print("  ℹ️  Processed data already exists. Use --force-reprocess to regenerate.")
        return True
    
    # Validate input
    if not validate_input_file(input_file):
        return False
    
    try:
        from data_processing import process_pipeline
        
        print(f"  📊 Processing data from: {input_file}")
        print(f"  🎯 Train/Val/Test split: {1-validation_split-test_split:.1f}/{validation_split:.1f}/{test_split:.1f}")
        
        start_time_dp = time.time() # Renamed to avoid conflict
        df, graph_data = process_pipeline(input_file)
        processing_time = time.time() - start_time_dp
        
        if df is not None:
            print(f"  ✅ Processing completed: {len(df)} compounds processed in {processing_time:.1f}s")
            print(f"  📁 Files created:")
            for file_path in processed_files:
                if os.path.exists(file_path):
                    size_mb = os.path.getsize(file_path) / (1024 * 1024)
                    print(f"    • {file_path} ({size_mb:.1f} MB)")
            
            # Save processing metadata
            metadata = {
                'input_file': input_file,
                'compounds_processed': len(df),
                'processing_time_seconds': processing_time,
                'validation_split': validation_split,
                'test_split': test_split,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            with open('logs/processing_metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return True
        else:
            print("  ❌ Processing failed")
            return False
            
    except Exception as e:
        print(f"  ❌ Processing error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def run_training(model_type_to_train: str = 'both', # Renamed model_type to avoid conflict
                config_dict_from_args: Optional[Dict[str, Any]] = None, # Renamed config
                enhanced_features: bool = True,
                monitor_performance: bool = True) -> bool:
    """Enhanced training pipeline with configuration options"""
    print(f"\n🎓 Running enhanced training pipeline ({model_type_to_train})...")
    
    if config_dict_from_args:
        print(f"  ⚙️  Using custom configuration")
        for key, value in config_dict_from_args.items():
            print(f"    • {key}: {value}")
    
    try:
        from training import train_pipeline, main as training_main
        
        start_time_train = time.time() # Renamed
        
        if model_type_to_train == 'both':
            print("  🚀 Training both Basic KD and Enhanced FedKD models...")
            # Pass the config_dict_from_args to training_main
            basic_results, fedkd_results = training_main(config_dict=config_dict_from_args) # CORRECTED
            success = basic_results is not None and fedkd_results is not None
            
            if success and monitor_performance:
                # Enhanced performance monitoring
                print("\n📊 Training Performance Summary:")
                basic_avg, fedkd_avg = 0.0, 0.0 # Initialize
                if basic_results and 'results' in basic_results:
                    basic_avg = calculate_average_performance(basic_results['results'])
                    print(f"  • Basic KD Average AUC: {basic_avg:.3f}")
                
                if fedkd_results and 'results' in fedkd_results:
                    fedkd_avg = calculate_average_performance(fedkd_results['results'])
                    print(f"  • Enhanced FedKD Average AUC: {fedkd_avg:.3f}")
                    if basic_avg > 0: # Avoid division by zero
                        improvement = fedkd_avg - basic_avg
                        print(f"  • FedKD Improvement: {improvement:+.3f} ({improvement/basic_avg*100:+.1f}%)")
                    else:
                        print(f"  • FedKD Improvement: N/A (Basic KD AUC is {basic_avg})")

        else:
            print(f"  🎯 Training {model_type_to_train.upper()} model...")
            # Create TrainingConfig instance from the dictionary
            training_config_instance = TrainingConfig(config_dict=config_dict_from_args) # CORRECTED
            results = train_pipeline(model_type_to_train, config=training_config_instance) # CORRECTED
            success = results is not None
            
            if success and monitor_performance and results:
                avg_performance = calculate_average_performance(results.get('results', {}))
                print(f"  📊 {model_type_to_train.upper()} Average AUC: {avg_performance:.3f}")
        
        training_time = time.time() - start_time_train
        
        if success:
            print(f"\n  ✅ Training completed in {training_time/60:.1f} minutes")
            
            # Save training metadata
            training_metadata = {
                'model_type': model_type_to_train,
                'training_time_seconds': training_time,
                'enhanced_features': enhanced_features,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'config': config_dict_from_args or {} # Store the dict used
            }
            
            with open('logs/training_metadata.json', 'w') as f:
                json.dump(training_metadata, f, indent=2)
        
        return success
            
    except Exception as e:
        print(f"  ❌ Training error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def run_hyperopt(preset: str = 'standard', 
                                    model_type_hp: str = 'both', # Renamed model_type
                                    n_trials: Optional[int] = None,
                                    timeout: Optional[int] = None) -> bool:
    """Run hyperparameter optimization with different presets"""
    print(f"\n🔍 Running hyperparameter optimization ({preset})...")
    
    try:
        # Check if optuna is available
        try:
            import optuna
            from hyperopt_config import get_optimization_config
            from hyperopt import HyperparameterOptimizer
        except ImportError as e:
            print(f"❌ Hyperparameter optimization requires additional packages:")
            print(f"  pip install optuna optuna-integration")
            print(f"Missing: {str(e)}")
            return False
        
        # Get configuration based on preset
        opt_config_obj = get_optimization_config(preset) # Renamed config
        
        # Override with command line arguments
        if model_type_hp != 'both':
            opt_config_obj.model_type = model_type_hp
        if n_trials is not None:
            opt_config_obj.n_trials = n_trials
        if timeout is not None:
            opt_config_obj.timeout = timeout
        
        print(f"  ⚙️  Configuration:")
        print(f"    • Trials: {opt_config_obj.n_trials}")
        print(f"    • Timeout: {opt_config_obj.timeout/3600:.1f} hours")
        print(f"    • Model type: {opt_config_obj.model_type}")
        print(f"    • Primary metric: {opt_config_obj.primary_metric}")
        print(f"    • CV folds: {opt_config_obj.cv_folds}")
        
        # Ensure data is processed
        if not os.path.exists('data/processed_data.csv'):
            print("📊 Processed data not found. Running data processing first...")
            if not run_data_processing():
                return False
        
        # Create optimizer and run optimization
        optimizer = HyperparameterOptimizer(opt_config_obj)
        
        start_time_hp = time.time() # Renamed
        
        if opt_config_obj.model_type == 'both':
            results = optimizer.optimize_both_models()
        else:
            results = {opt_config_obj.model_type: optimizer.optimize_model(opt_config_obj.model_type)}
        
        optimization_time = time.time() - start_time_hp
        
        # Save results and create visualizations
        optimizer.save_results()
        optimizer.create_visualizations()
        
        print(f"\n🏆 Hyperparameter Optimization Results:")
        for mt, result in results.items(): # Renamed model_type
            print(f"\n{mt.title()} Model:")
            print(f"  • Best {opt_config_obj.primary_metric}: {result['best_value']:.4f}")
            print(f"  • Trials completed: {result['n_trials']}")
            print(f"  • Optimization time: {result['optimization_time']/60:.1f} minutes")
            
            print(f"  • Best parameters:")
            for param, value in result['best_params'].items():
                if isinstance(value, float):
                    print(f"    - {param}: {value:.1e}" if abs(value) < 0.01 and value != 0 else f"    - {param}: {value:.4f}")
                else:
                    print(f"    - {param}: {value}")
        
        print(f"\n✅ Hyperparameter optimization completed in {optimization_time/60:.1f} minutes!")
        print(f"📁 Results saved to results/hyperopt/")
        
        # Ask if user wants to train with optimized parameters
        print(f"\n💡 Next steps:")
        print(f"  1. Review optimization results in results/hyperopt/")
        print(f"  2. Train models with optimized parameters:")
        for mt in results.keys(): # Renamed model_type
            print(f"     python main.py --train {mt} --config results/hyperopt/best_{mt}_config.json")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Hyperparameter optimization error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def calculate_average_performance(results_dict: Dict[str, Any]) -> float: # Renamed results
    """Calculate average performance across all tasks"""
    aucs = []
    # results_dict is like: {'Basic Student': {'normal_safe': 0.6, ...}, 'Basic Teacher (...)': ...}
    for model_name, model_task_results in results_dict.items(): # Renamed model_results
        if isinstance(model_task_results, dict): # This should be the case
            # We are interested in the student model's average performance typically
            if "Student" in model_name: # Focus on student models for overall metric
                for task_name, auc_score in model_task_results.items(): # Renamed task, auc
                    if isinstance(auc_score, (int, float)):
                        aucs.append(auc_score)
    
    return sum(aucs) / len(aucs) if aucs else 0.0


def run_prediction_interface(enhanced_mode: bool = True) -> bool:
    """Enhanced prediction interface with better error handling"""
    print(f"\n🔮 Starting {'enhanced' if enhanced_mode else 'standard'} prediction interface...")
    
    # Check if models exist
    basic_model = 'models/gnn_trained_models.pth'
    fedkd_model = 'models/fedkd_trained_model.pth'
    
    available_models = []
    if os.path.exists(basic_model):
        available_models.append('basic')
    if os.path.exists(fedkd_model):
        available_models.append('fedkd')
    
    if not available_models:
        print("  ❌ No trained models found. Please train models first:")
        print("    python main.py --train both")
        return False
    
    print(f"  ✅ Available models: {', '.join(available_models)}")
    
    try:
        from prediction import main as prediction_main
        prediction_main()
        return True
        
    except Exception as e:
        print(f"  ❌ Prediction error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def run_comparison(detailed: bool = True, save_report: bool = True) -> bool:
    """Enhanced model comparison with detailed analysis"""
    print(f"\n📊 Running {'detailed' if detailed else 'basic'} model comparison...")
    
    try:
        from comparison import run_detailed_comparison
        
        start_time_comp = time.time() # Renamed
        result = run_detailed_comparison()
        comparison_time = time.time() - start_time_comp
        
        if result:
            print(f"  ✅ Comparison completed in {comparison_time:.1f}s")
            
            if save_report:
                # Save comparison metadata
                comparison_metadata = {
                    'comparison_time_seconds': comparison_time,
                    'detailed_analysis': detailed,
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'files_generated': [
                        'results/comprehensive_kd_comparison.png', # This file might be generated by training.py now
                        'results/plots/kd_comparison.png', # More likely path from training.py
                        'results/kd_comparison_report.md'
                    ]
                }
                
                with open('logs/comparison_metadata.json', 'w') as f:
                    json.dump(comparison_metadata, f, indent=2)
        
        return result is not None
        
    except Exception as e:
        print(f"  ❌ Comparison error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def run_validation(quick_val: bool = False) -> bool: # Renamed quick
    """Run validation tests on the pipeline"""
    print(f"\n🧪 Running {'quick' if quick_val else 'comprehensive'} validation tests...")
    
    try:
        # Test 1: Check data integrity
        print("  📊 Testing data integrity...")
        if not os.path.exists('data/processed_data.csv'):
            print("    ❌ Processed data not found")
            return False
        
        import pandas as pd
        df = pd.read_csv('data/processed_data.csv')
        
        if len(df) == 0:
            print("    ❌ Empty processed dataset")
            return False
        
        required_cols = ['SMILES']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"    ❌ Missing required columns: {missing_cols}")
            return False
        
        print(f"    ✅ Data integrity check passed ({len(df)} samples)")
        
        # Test 2: Check model files
        print("  🤖 Testing model availability...")
        model_files = [
            'models/gnn_trained_models.pth',
            'models/fedkd_trained_model.pth'
        ]
        
        available_models_val = [] # Renamed
        for model_file in model_files:
            if os.path.exists(model_file):
                available_models_val.append(model_file)
                print(f"    ✅ Found: {model_file}")
            else:
                print(f"    ⚠️  Missing: {model_file}")
        
        if not available_models_val:
            print("    ❌ No trained models found")
            return False
        
        # Test 3: Quick prediction test (if not quick mode)
        if not quick_val:
            print("  🔮 Testing prediction pipeline...")
            try:
                from prediction import predict_with_model
                test_smiles = ['CCO', 'CC(=O)O']  # Simple test molecules
                
                for mt_val in ['basic', 'fedkd']: # Renamed model_type
                    if any(mt_val in mf for mf in available_models_val):
                        results_pred_test = predict_with_model(test_smiles, mt_val) # Renamed results
                        if results_pred_test is not None and len(results_pred_test) > 0:
                            print(f"    ✅ {mt_val.upper()} predictions working")
                        else:
                            print(f"    ❌ {mt_val.upper()} predictions failed")
                            return False
            
            except Exception as e:
                print(f"    ❌ Prediction test failed: {str(e)}")
                return False
        
        # Test 4: Configuration files
        print("  ⚙️  Testing configuration system...")
        configs_dir = Path('configs')
        if configs_dir.exists():
            config_files = list(configs_dir.glob('*.json'))
            print(f"    ✅ Found {len(config_files)} configuration files")
        else:
            print(f"    ⚠️  No configuration directory found")
        
        # Test 5: Results directory structure
        print("  📁 Testing results directory structure...")
        results_subdirs = ['plots', 'reports', 'comparisons', 'hyperopt']
        missing_dirs = []
        for subdir in results_subdirs:
            if not Path(f'results/{subdir}').exists():
                missing_dirs.append(subdir)
        
        if missing_dirs:
            print(f"    ⚠️  Missing results subdirectories: {missing_dirs}")
        else:
            print(f"    ✅ Results directory structure complete")
        
        print("✅ All validation tests passed!")
        return True
        
    except Exception as e:
        print(f"  ❌ Validation error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def load_config(config_file_path: str) -> Optional[Dict[str, Any]]: # Renamed config_file
    """Load configuration from JSON file"""
    try:
        with open(config_file_path, 'r') as f:
            loaded_config = json.load(f) # Renamed config
        print(f"✅ Loaded configuration from {config_file_path}")
        return loaded_config
    except Exception as e:
        print(f"❌ Error loading config file {config_file_path}: {str(e)}")
        return None

def save_config_template(config_template_type: str = 'standard') -> str: # Renamed config_type
    """Save a configuration template file"""
    configs_templates = { # Renamed configs
        'standard': {
            'batch_size': 32,
            'teacher_epochs': 100,
            'student_epochs': 150,
            'teacher_lr': 1e-3,
            'student_lr': 2e-3,
            'weight_decay': 1e-5,
            'patience': 15,
            'use_features': True,
            'early_stopping': True
        },
        'fast': {
            'batch_size': 64,
            'teacher_epochs': 50, # Should be used by TrainingConfig
            'student_epochs': 75, # Should be used by TrainingConfig
            'teacher_lr': 2e-3,
            'student_lr': 3e-3,
            'weight_decay': 1e-4,
            'patience': 10,
            'use_features': True,
            'early_stopping': True
        },
        'thorough': {
            'batch_size': 16,
            'teacher_epochs': 200,
            'student_epochs': 300,
            'teacher_lr': 5e-4,
            'student_lr': 1e-3,
            'weight_decay': 1e-6,
            'patience': 25,
            'use_features': True,
            'early_stopping': True
        }
    }
    
    config_data_to_save = configs_templates.get(config_template_type, configs_templates['standard']) # Renamed config
    
    Path('configs').mkdir(exist_ok=True)
    config_file_to_save = f'configs/{config_template_type}_config.json' # Renamed config_file
    
    with open(config_file_to_save, 'w') as f:
        json.dump(config_data_to_save, f, indent=2)
    
    print(f"✅ Saved {config_template_type} configuration template to {config_file_to_save}")
    return config_file_to_save


def save_pipeline_summary(pipeline_start_time: float, executed_operations: list, overall_success: bool) -> None: # Renamed variables
    """Save comprehensive pipeline execution summary"""
    total_execution_time = time.time() - pipeline_start_time
    
    summary = {
        'pipeline_execution': {
            'start_time': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(pipeline_start_time)),
            'end_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_time_seconds': total_execution_time,
            'total_time_formatted': f"{total_execution_time/60:.1f} minutes",
            'operations_performed': executed_operations,
            'success': overall_success,
            'exit_code': 0 if overall_success else 1
        },
        'system_info': {
            'python_version': sys.version,
            'working_directory': os.getcwd(),
            'command_line': ' '.join(sys.argv)
        }
    }
    
    with open('logs/pipeline_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n📋 Pipeline Summary:")
    print(f"  • Total execution time: {total_execution_time/60:.1f} minutes")
    print(f"  • Operations: {', '.join(executed_operations)}")
    print(f"  • Status: {'✅ SUCCESS' if overall_success else '❌ FAILED'}")
    print(f"  • Summary saved to: logs/pipeline_summary.json")

def show_help_guide():
    """Show comprehensive help guide"""
    guide = """
🧬 MOLECULAR CANCER CELL SELECTIVITY PREDICTION PIPELINE
========================================================

QUICK START:
-----------
1. Process data:     python main.py --process --input data/your_data.csv
2. Train models:     python main.py --train both --enhanced
3. Compare models:   python main.py --compare --detailed
4. Make predictions: python main.py --predict --enhanced

COMMON WORKFLOWS:
----------------
• Full pipeline:     python main.py --full --enhanced --monitor
• Quick test:        python main.py --validate --quick
• Hyperopt:          python main.py --optimize standard --model-type both
• Custom training:   python main.py --train fedkd --config configs/custom.json

HYPERPARAMETER OPTIMIZATION:
----------------------------
• Quick (30 min):   python main.py --optimize quick
• Standard (2h):    python main.py --optimize standard  
• Thorough (8h):    python main.py --optimize thorough
• Production:       python main.py --optimize production

CONFIGURATION:
--------------
• Create templates:  python main.py --create-config standard
• Use custom:        python main.py --train both --config configs/my_config.json

TROUBLESHOOTING:
---------------
• Check deps:       python main.py --validate --skip-checks
• View logs:        Check logs/ directory for detailed information
• Reset:            Delete models/ and results/ directories to start fresh

For more help: python main.py --help
"""
    print(guide)

def main():
    """Enhanced main pipeline orchestrator with comprehensive options"""
    parser = argparse.ArgumentParser(
        description='Enhanced Molecular Cancer Cell Selectivity Prediction Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --process --input data/my_data.csv    # Process custom data
  python main.py --train basic --config configs/basic.json  # Train with config
  python main.py --train fedkd --enhanced              # Train with enhanced features
  python main.py --train both --monitor                # Train both with monitoring
  python main.py --optimize standard --model-type both # Hyperparameter optimization
  python main.py --predict --enhanced                  # Enhanced prediction interface
  python main.py --compare --detailed --save-report    # Detailed comparison
  python main.py --validate --quick                    # Quick validation
  python main.py --full --enhanced --monitor           # Complete enhanced pipeline
  python main.py --help-guide                          # Show comprehensive help guide
  
Advanced:
  python main.py --train fedkd --log-level DEBUG --log-file training.log
  python main.py --full --config configs/production.json --no-interactive
  python main.py --optimize thorough --trials 200 --timeout 28800
        """
    )
    
    # Main actions (mutually exclusive)
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument('--process', action='store_true',
                             help='Process raw molecular data')
    action_group.add_argument('--train', choices=['basic', 'fedkd', 'both'],
                             help='Train models')
    action_group.add_argument('--predict', action='store_true',
                             help='Run prediction interface')
    action_group.add_argument('--compare', action='store_true',
                             help='Run model comparison')
    action_group.add_argument('--optimize', choices=['quick', 'standard', 'thorough', 'selectivity', 'production'],
                             help='Run hyperparameter optimization')
    action_group.add_argument('--validate', action='store_true',
                             help='Run validation tests')
    action_group.add_argument('--full', action='store_true',
                             help='Run complete pipeline')
    action_group.add_argument('--help-guide', action='store_true',
                             help='Show comprehensive help guide')
    action_group.add_argument('--create-config', choices=['standard', 'fast', 'thorough'],
                             help='Create configuration template file')
    
    # Data processing options
    data_group = parser.add_argument_group('Data Processing Options')
    data_group.add_argument('--input', type=str, default='data/raw_data.csv',
                           help='Input CSV file with molecular data')
    data_group.add_argument('--force-reprocess', action='store_true',
                           help='Force data reprocessing')
    data_group.add_argument('--validation-split', type=float, default=0.1,
                           help='Validation split ratio (default: 0.1)')
    data_group.add_argument('--test-split', type=float, default=0.2,
                           help='Test split ratio (default: 0.2)')
    
    # Training options
    train_group = parser.add_argument_group('Training Options')
    train_group.add_argument('--enhanced', action='store_true',
                            help='Use enhanced FedKD features')
    train_group.add_argument('--monitor', action='store_true',
                            help='Enable performance monitoring')
    train_group.add_argument('--config', type=str, # This will be the path to the config JSON
                            help='JSON config file for training parameters')
    
    # Hyperparameter optimization options
    hyperopt_group = parser.add_argument_group('Hyperparameter Optimization Options')
    hyperopt_group.add_argument('--model-type', choices=['basic', 'fedkd', 'both'], default='both',
                               help='Model type for hyperparameter optimization')
    hyperopt_group.add_argument('--trials', type=int,
                               help='Number of optimization trials (overrides preset)')
    hyperopt_group.add_argument('--timeout', type=int,
                               help='Optimization timeout in seconds (overrides preset)')
    
    # Comparison options
    comp_group = parser.add_argument_group('Comparison Options')
    comp_group.add_argument('--detailed', action='store_true',
                           help='Run detailed comparison analysis')
    comp_group.add_argument('--save-report', action='store_true', default=True,
                           help='Save comparison report')

    # Validation options
    val_group = parser.add_argument_group('Validation Options')
    val_group.add_argument('--quick', action='store_true',
                          help='Run quick validation (skip prediction tests)')
    
    # System options
    sys_group = parser.add_argument_group('System Options')
    sys_group.add_argument('--skip-checks', action='store_true',
                          help='Skip dependency checks')
    sys_group.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                          default='INFO', help='Logging level')
    sys_group.add_argument('--log-file', type=str,
                          help='Log file name (saved in logs/)')
    sys_group.add_argument('--no-interactive', action='store_true',
                          help='Disable interactive prompts')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level, args.log_file)
    
    pipeline_start_time = time.time() # Renamed
    executed_operations = [] # Renamed
    
    # Handle help guide
    if args.help_guide:
        show_help_guide()
        return 0
    
    # Handle config creation
    if args.create_config:
        save_config_template(args.create_config)
        return 0
    
    print("🧬 ENHANCED MOLECULAR CANCER CELL SELECTIVITY PREDICTION")
    print("="*80)
    print(f"🕐 Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load configuration if provided
    loaded_config_dict = None # Renamed config
    if args.config: # args.config is the file path string
        loaded_config_dict = load_config(args.config)
        if loaded_config_dict is None:
            return 1 # Exit if config loading failed
    
    try:
        # Pre-flight checks
        if not args.skip_checks:
            if not check_dependencies():
                save_pipeline_summary(pipeline_start_time, executed_operations, False)
                return 1
            executed_operations.append('dependency_check')
        
        setup_directories()
        executed_operations.append('setup_directories')
        
        overall_success = True # Renamed success
        
        if args.process:
            # Data processing only
            executed_operations.append('data_processing')
            overall_success = run_data_processing(
                args.input, 
                args.force_reprocess,
                args.validation_split,
                args.test_split
            )
            
            if overall_success:
                print(f"\n✅ Data processing completed successfully!")
                print(f"📁 Next step: python main.py --train both --enhanced --monitor")
            else:
                print(f"\n❌ Data processing failed!")
        
        elif args.train:
            # Training with enhanced options
            executed_operations.append(f'training_{args.train}')
            
            # Ensure data is processed first
            if not os.path.exists('data/processed_data.csv'):
                print("📊 Processed data not found. Running data processing first...")
                if not run_data_processing(args.input, args.force_reprocess):
                    overall_success = False
                else:
                    executed_operations.append('auto_data_processing')
            
            if overall_success:
                # Pass the loaded_config_dict (which can be None) to run_training
                overall_success = run_training(
                    args.train, # model_type_to_train
                    config_dict_from_args=loaded_config_dict, # CORRECTED: pass the dict
                    enhanced_features=args.enhanced,
                    monitor_performance=args.monitor
                )
                
                if overall_success:
                    print(f"\n✅ Training completed successfully!")
                    print(f"📁 Next steps:")
                    print(f"  • python main.py --compare --detailed")
                    print(f"  • python main.py --predict --enhanced")
                else:
                    print(f"\n❌ Training failed!")
        
        elif args.predict:
            # Enhanced prediction interface
            executed_operations.append('prediction')
            overall_success = run_prediction_interface(args.enhanced)
            
            if overall_success:
                print(f"\n✅ Prediction interface completed!")
            else:
                print(f"\n❌ Prediction interface failed!")
        
        elif args.compare:
            # Enhanced comparison
            executed_operations.append('comparison')
            overall_success = run_comparison(args.detailed, args.save_report)
            
            if overall_success:
                print(f"\n✅ Comparison completed!")
            else:
                print(f"\n❌ Comparison failed!")
        
        elif args.optimize:
            # Hyperparameter optimization
            executed_operations.append(f'hyperopt_{args.optimize}')
            overall_success = run_hyperopt(
                args.optimize, 
                args.model_type, # This is --model-type for hyperopt, distinct from --train model_type
                args.trials,
                args.timeout
            )
            
            if overall_success:
                print(f"\n✅ Hyperparameter optimization completed!")
                print(f"📁 Next steps:")
                print(f"  • Review results in results/hyperopt/")
                print(f"  • Train with optimized parameters using saved configs")
            else:
                print(f"\n❌ Hyperparameter optimization failed!")
        
        elif args.validate:
            # Validation tests
            executed_operations.append('validation')
            overall_success = run_validation(args.quick)
            
            if overall_success:
                print(f"\n✅ Validation passed!")
            else:
                print(f"\n❌ Validation failed!")
        
        elif args.full:
            # Complete enhanced pipeline
            executed_operations.append('full_pipeline')
            print("🚀 Running complete enhanced pipeline...")
            
            # Step 1: Data processing
            print("\n" + "="*50)
            print("STEP 1: DATA PROCESSING")
            print("="*50)
            if not run_data_processing(args.input, args.force_reprocess):
                overall_success = False
            else:
                executed_operations.append('full_data_processing')
            
            # Step 2: Training both models
            if overall_success:
                print("\n" + "="*50)
                print("STEP 2: MODEL TRAINING")
                print("="*50)
                # Pass loaded_config_dict to run_training for the 'full' pipeline as well
                if not run_training('both', config_dict_from_args=loaded_config_dict, enhanced_features=args.enhanced, monitor_performance=args.monitor):
                    overall_success = False
                else:
                    executed_operations.append('full_training')
            
            # Step 3: Detailed comparison
            if overall_success:
                print("\n" + "="*50)
                print("STEP 3: MODEL COMPARISON")
                print("="*50)
                if not run_comparison(args.detailed, args.save_report):
                    overall_success = False
                else:
                    executed_operations.append('full_comparison')
            
            # Step 4: Validation
            if overall_success:
                print("\n" + "="*50)
                print("STEP 4: VALIDATION")
                print("="*50)
                if not run_validation(args.quick):
                    overall_success = False
                else:
                    executed_operations.append('full_validation')
            
            if overall_success:
                print(f"\n🎉 Complete enhanced pipeline finished successfully!")
                print(f"📁 Results available in:")
                print(f"  • results/ - Plots and reports")
                print(f"  • models/ - Trained model checkpoints")
                print(f"  • logs/ - Execution logs and metadata")
                print(f"🔮 Run 'python main.py --predict --enhanced' for interactive predictions")
                print(f"🔍 Consider 'python main.py --optimize standard' for hyperparameter tuning")
            else:
                print(f"\n❌ Pipeline failed at one of the steps!")
        
        # Save comprehensive summary
        save_pipeline_summary(pipeline_start_time, executed_operations, overall_success)
        
        return 0 if overall_success else 1
        
    except KeyboardInterrupt:
        print(f"\n\n👋 Pipeline interrupted by user")
        save_pipeline_summary(pipeline_start_time, executed_operations, False)
        return 1
    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        save_pipeline_summary(pipeline_start_time, executed_operations, False)
        return 1

if __name__ == "__main__":
    exit(main())