#!/usr/bin/env python3
"""
Training Pipeline 

Comprehensive training system with:
- Data loading and splitting with validation
- Teacher training with performance monitoring
- Student training with knowledge distillation
- Advanced model evaluation and comparison
- Comprehensive logging and model checkpointing
- Support for FedKD features
- Fixed CUDA tensor handling throughout
"""

import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import json
import time
import logging
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Any, Optional, Tuple, List
import warnings
warnings.filterwarnings('ignore')

from data_processing import load_processed_data
from models_basic import BasicTeacherModel, BasicStudentModel, BasicKDTrainer
from models_fedkd import (
    FedKDTeacherModel,
    FedKDStudentModel,
    FedKDTrainer,
    FedKDLoss
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TrainingConfig:
    """Configuration class for training - FIXED with all hyperopt parameters"""

    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        # Learning rates
        self.teacher_lr = 1e-3
        self.student_lr = 2e-3

        # Architecture parameters (ADDED for hyperopt)
        self.teacher_hidden_dim = 128
        self.student_hidden_dim = 96
        self.num_layers = 4
        self.dropout = 0.2

        # Training parameters
        self.batch_size = 32
        self.teacher_epochs = 100
        self.student_epochs = 150
        self.weight_decay = 1e-5
        self.patience = 15
        self.grad_clip = 1.0

        # Knowledge distillation parameters (ADDED for hyperopt)
        self.temperature = 3.0
        self.alpha = 0.6  # Task loss weight
        self.beta = 0.4   # Distillation loss weight
        self.gamma = 0.1  # Hidden state loss weight (for FedKD)
        self.attention_weight = 0.05  # Attention transfer weight (for FedKD)

        # FedKD specific parameters (ADDED for hyperopt)
        self.quality_threshold = 0.1
        self.temp_decay = 0.95

        # Training options
        self.use_features = True
        self.monitor_quality = True
        self.adaptive_lr = True
        self.save_checkpoints = True
        self.early_stopping = True
        self.label_smoothing = 0.0  # Optional label smoothing

        # Data splits
        self.test_size = 0.2
        self.val_size = 0.1
        self.stratify = True

        # Device settings
        self.device = 'auto'  # 'auto', 'cpu', 'cuda'

        # Update with provided config
        if config_dict:
            for key, value in config_dict.items():
                if hasattr(self, key):
                    setattr(self, key, value)
                else:
                    logger.warning(f"Unknown config parameter: {key} = {value}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

    def update(self, updates: Dict[str, Any]) -> None:
        """Update configuration with new values"""
        for key, value in updates.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                logger.warning(f"Unknown config parameter: {key} = {value}")

    def get_model_config(self) -> Dict[str, Any]:
        """Get model architecture configuration"""
        return {
            'teacher_hidden_dim': self.teacher_hidden_dim,
            'student_hidden_dim': self.student_hidden_dim,
            'num_layers': self.num_layers,
            'dropout': self.dropout
        }

    def get_training_config(self) -> Dict[str, Any]:
        """Get training-specific configuration"""
        return {
            'teacher_lr': self.teacher_lr,
            'student_lr': self.student_lr,
            'batch_size': self.batch_size,
            'teacher_epochs': self.teacher_epochs,
            'student_epochs': self.student_epochs,
            'weight_decay': self.weight_decay,
            'patience': self.patience,
            'grad_clip': self.grad_clip
        }

    def get_distillation_config(self) -> Dict[str, Any]:
        """Get knowledge distillation configuration"""
        return {
            'temperature': self.temperature,
            'alpha': self.alpha,
            'beta': self.beta,
            'gamma': self.gamma,
            'attention_weight': self.attention_weight,
            'quality_threshold': self.quality_threshold,
            'temp_decay': self.temp_decay
        }

    def __str__(self) -> str:
        """String representation of configuration"""
        config_str = "TrainingConfig:\n"
        for key, value in self.to_dict().items():
            config_str += f"  {key}: {value}\n"
        return config_str

def setup_device(config: TrainingConfig) -> torch.device:
    """Setup and return the appropriate device"""
    if config.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
        else:
            device = torch.device('cpu')
            logger.info("Using CPU")
    else:
        device = torch.device(config.device)
        logger.info(f"Using specified device: {device}")

    return device

def add_labels_to_graphs(graph_data: List, df: pd.DataFrame) -> List:
    """Add activity labels to graph objects with validation"""
    logger.info("🏷️  Adding labels to graphs...")

    if len(graph_data) != len(df):
        raise ValueError(f"Graph data length ({len(graph_data)}) doesn't match DataFrame length ({len(df)})")

    label_columns = ['normal_safe', 'mcc26_active', 'mkl1_active']
    available_labels = [col for col in label_columns if col in df.columns]

    if not available_labels:
        logger.warning("No activity labels found in DataFrame!")
        return graph_data

    logger.info(f"Available labels: {available_labels}")

    for i, graph in enumerate(graph_data):
        # Add labels as tensors with consistent shape [1]
        graph.y_normal = torch.tensor([float(df.iloc[i].get('normal_safe', 0))], dtype=torch.float)
        graph.y_mcc26 = torch.tensor([float(df.iloc[i].get('mcc26_active', 0))], dtype=torch.float)
        graph.y_mkl1 = torch.tensor([float(df.iloc[i].get('mkl1_active', 0))], dtype=torch.float)

        # Add molecule identifier for tracking
        graph.mol_id = i
        if 'SMILES' in df.columns:
            graph.smiles = df.iloc[i]['SMILES']

    return graph_data

def create_data_splits(graph_data: List,
                              df: pd.DataFrame,
                              config: TrainingConfig) -> Tuple:
    """Create train/validation/test splits with comprehensive validation"""
    logger.info("📊 Creating data splits...")

    indices = list(range(len(df)))

    # Stratification
    stratify_col = None
    if config.stratify and 'normal_safe' in df.columns:
        stratify_col = df['normal_safe'].values
        logger.info("Using stratified splitting based on normal_safe")

    # Split: (train+val) vs test
    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=config.test_size,
        random_state=42,
        stratify=stratify_col
    )

    # Split: train vs val
    if stratify_col is not None:
        stratify_val = df.iloc[train_val_idx]['normal_safe'].values
    else:
        stratify_val = None

    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=config.val_size/(1-config.test_size),
        random_state=42,
        stratify=stratify_val
    )

    # Log split statistics
    logger.info(f"Data split statistics:")
    logger.info(f"  • Train: {len(train_idx)} ({len(train_idx)/len(df)*100:.1f}%)")
    logger.info(f"  • Validation: {len(val_idx)} ({len(val_idx)/len(df)*100:.1f}%)")
    logger.info(f"  • Test: {len(test_idx)} ({len(test_idx)/len(df)*100:.1f}%)")

    # Create data loaders with options
    train_data = [graph_data[i] for i in train_idx]
    val_data = [graph_data[i] for i in val_idx]
    test_data = [graph_data[i] for i in test_idx]

    # Data loaders with better memory management
    train_loader = DataLoader(
        train_data,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,  # Keep 0 for compatibility
        pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_data,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )
    test_loader = DataLoader(
        test_data,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    # Save split information
    split_info = {
        'train_indices': train_idx,
        'val_indices': val_idx,
        'test_indices': test_idx,
        'train_size': len(train_idx),
        'val_size': len(val_idx),
        'test_size': len(test_idx),
        'stratified': config.stratify,
        'random_seed': 42
    }

    Path('data/splits').mkdir(parents=True, exist_ok=True)
    with open('data/splits/split_info.json', 'w') as f:
        json.dump(split_info, f, indent=2)

    return train_loader, val_loader, test_loader, train_idx, val_idx, test_idx

def safe_tensor_to_numpy(tensor_or_value):
    """Safely convert tensor to numpy, handling CUDA tensors"""
    if isinstance(tensor_or_value, torch.Tensor):
        return tensor_or_value.detach().cpu().numpy()
    elif isinstance(tensor_or_value, (list, tuple)):
        return [safe_tensor_to_numpy(item) for item in tensor_or_value]
    else:
        return tensor_or_value

def safe_tensor_item(tensor_or_value):
    """Safely extract scalar value from tensor, handling CUDA tensors"""
    if isinstance(tensor_or_value, torch.Tensor):
        return tensor_or_value.detach().cpu().item()
    else:
        return float(tensor_or_value)

def train_teacher_model(train_loader: DataLoader,
                                val_loader: DataLoader,
                                cell_line: str,
                                model_type: str = 'basic',
                                config: TrainingConfig = None,
                                device: torch.device = None) -> Tuple:
    """Teacher training with comprehensive monitoring and fixed CUDA handling"""

    if config is None:
        config = TrainingConfig()
    if device is None:
        device = torch.device('cpu')

    logger.info(f"🎓 Training {model_type.title()} Teacher for {cell_line}")

    # Get model dimensions from sample batch
    sample_batch = next(iter(train_loader))
    node_features = sample_batch.x.size(1)
    edge_features = sample_batch.edge_attr.size(1)

    # Create teacher model
    if model_type == 'basic':
        model = BasicTeacherModel(node_features, edge_features, hidden_dim=128, dropout=0.3)
    else:  # fedkd
        model = FedKDTeacherModel(node_features, edge_features, hidden_dim=128, dropout=0.3)

    model.to(device)

    # Optimizer with adaptive learning
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.teacher_lr,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )

    # Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        patience=config.patience,
        factor=0.5,
        verbose=True,
        min_lr=1e-6
    )

    # Training tracking with metrics
    best_val_auc = 0
    best_model_state = None
    train_losses = []
    val_aucs = []
    val_precisions = []
    learning_rates = []

    # Early stopping
    patience_counter = 0

    # Training loop with progress bar
    pbar = tqdm(range(config.teacher_epochs), desc=f"Training {cell_line} teacher")

    for epoch in pbar:
        # Training phase
        model.train()
        epoch_losses = []

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            # Forward pass with error handling
            try:
                if model_type == 'basic':
                    pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                else:  # fedkd
                    pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, return_hidden=False)

                # Get labels for this cell line
                if cell_line == 'normal_safe':
                    labels = batch.y_normal.to(device)
                elif cell_line == 'mcc26_active':
                    labels = batch.y_mcc26.to(device)
                elif cell_line == 'mkl1_active':
                    labels = batch.y_mkl1.to(device)
                else:
                    continue

                # Compute loss with label smoothing for robustness
                pred_flat = pred.view(-1)
                labels_flat = labels.view(-1)

                # Optional label smoothing
                if hasattr(config, 'label_smoothing') and config.label_smoothing > 0:
                    smoothed_labels = labels_flat * (1 - config.label_smoothing) + 0.5 * config.label_smoothing
                    loss = F.binary_cross_entropy(pred_flat, smoothed_labels)
                else:
                    loss = F.binary_cross_entropy(pred_flat, labels_flat)

                # Backward pass with gradient clipping
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip)
                optimizer.step()

                # Safe tensor conversion for loss tracking
                epoch_losses.append(safe_tensor_item(loss))

            except Exception as e:
                logger.error(f"Error in training batch: {str(e)}")
                continue

        if not epoch_losses:
            logger.warning(f"No valid training batches for epoch {epoch}")
            continue

        avg_loss = np.mean(epoch_losses)
        train_losses.append(avg_loss)

        # Validation phase with metrics and safe tensor handling
        model.eval()
        val_preds = []
        val_labels = []
        val_losses = []

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)

                try:
                    if model_type == 'basic':
                        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                    else:
                        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, return_hidden=False)

                    if cell_line == 'normal_safe':
                        labels = batch.y_normal
                    elif cell_line == 'mcc26_active':
                        labels = batch.y_mcc26
                    elif cell_line == 'mkl1_active':
                        labels = batch.y_mkl1
                    else:
                        continue

                    # Validation loss
                    pred_flat = pred.view(-1)
                    labels_flat = labels.view(-1)
                    val_loss = F.binary_cross_entropy(pred_flat, labels_flat)
                    val_losses.append(safe_tensor_item(val_loss))

                    # Safe tensor conversion for predictions and labels
                    val_preds.extend(safe_tensor_to_numpy(pred_flat))
                    val_labels.extend(safe_tensor_to_numpy(labels_flat))

                except Exception as e:
                    logger.error(f"Error in validation batch: {str(e)}")
                    continue

        # Calculate metrics with proper error handling
        if len(val_preds) > 0 and len(set(val_labels)) > 1:
            try:
                val_auc = roc_auc_score(val_labels, val_preds)
                val_ap = average_precision_score(val_labels, val_preds)
                val_aucs.append(val_auc)
                val_precisions.append(val_ap)
            except Exception as e:
                logger.warning(f"Error calculating validation metrics: {str(e)}")
                val_auc = 0.5
                val_ap = 0.5
                val_aucs.append(val_auc)
                val_precisions.append(val_ap)
        else:
            val_auc = 0.5
            val_ap = 0.5
            val_aucs.append(val_auc)
            val_precisions.append(val_ap)

        # Learning rate tracking
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)

        # Scheduler step
        scheduler.step(val_auc)

        # Best model tracking with criteria
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0

            # Save best model checkpoint
            if config.save_checkpoints:
                checkpoint_dir = Path(f'models/checkpoints/{model_type}')
                checkpoint_dir.mkdir(parents=True, exist_ok=True)

                checkpoint = {
                    'model_state_dict': best_model_state,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'val_auc': best_val_auc,
                    'val_ap': val_ap,
                    'config': config.to_dict()
                }

                torch.save(checkpoint, checkpoint_dir / f'best_{cell_line}_teacher.pth')
        else:
            patience_counter += 1

        # Update progress bar with safe formatting
        pbar.set_postfix({
            'Loss': f'{avg_loss:.4f}',
            'Val AUC': f'{val_auc:.3f}',
            'Best': f'{best_val_auc:.3f}',
            'LR': f'{current_lr:.1e}'
        })

        # Early stopping
        if config.early_stopping and patience_counter >= config.patience * 2:
            logger.info(f"Early stopping triggered after {epoch+1} epochs")
            break

        # Log detailed progress every 20 epochs
        if (epoch + 1) % 20 == 0:
            logger.info(
                f"Epoch {epoch+1:3d}: Loss={avg_loss:.4f}, "
                f"Val AUC={val_auc:.3f}, Val AP={val_ap:.3f}, "
                f"Best AUC={best_val_auc:.3f}, LR={current_lr:.1e}"
            )

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    logger.info(f"🏆 Best validation AUC for {cell_line}: {best_val_auc:.3f}")

    # Compile training history
    history = {
        'train_losses': train_losses,
        'val_aucs': val_aucs,
        'val_precisions': val_precisions,
        'learning_rates': learning_rates,
        'best_auc': best_val_auc,
        'epochs_trained': len(train_losses),
        'early_stopped': patience_counter >= config.patience * 2
    }

    return model, history


def train_student_model(train_loader: DataLoader,
                                val_loader: DataLoader,
                                teachers: Dict, # Dict of trained teacher models
                                model_type: str = 'basic',
                                config: TrainingConfig = None, # Expects TrainingConfig instance
                                device: torch.device = None) -> Tuple[Optional[torch.nn.Module], Dict]: # Added Optional and return type hint
    """Student training with comprehensive knowledge distillation monitoring and fixed CUDA handling"""

    if config is None:
        config = TrainingConfig() # Ensure config is always an instance
    if device is None:
        # Default to auto-detection if not specified
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Device auto-selected: {device}")


    logger.info(f"🎓 Training {model_type.title()} Student with Knowledge Distillation")

    # Get model dimensions from a sample batch
    try:
        sample_batch = next(iter(train_loader))
        node_features = sample_batch.x.size(1)
        edge_features = sample_batch.edge_attr.size(1)
    except StopIteration:
        logger.error("Training data loader is empty. Cannot proceed with student training.")
        return None, {} # Return None for student and empty history

    student_model_params = {
        'node_features': node_features,
        'edge_features': edge_features,
        'hidden_dim': config.student_hidden_dim,
        'num_layers': max(1, config.num_layers - 1), # Ensure at least 1 layer
        'dropout': config.dropout
    }

    if model_type == 'fedkd':
        # FedKDStudentModel might require teacher's output dimension for hidden state matching
        student_model_params['teacher_target_hidden_dim'] = config.teacher_hidden_dim

    # --- Initialize Student Model, Trainer, and Optimizers ---
    student: Optional[torch.nn.Module] = None # Initialize student to None
    trainer = None
    optimizers_for_trainer = None # This will hold the optimizer(s)

    if model_type == 'basic':
        student = BasicStudentModel(**student_model_params)
        try:
            # BasicKDTrainer now accepts temperature, alpha, and beta
            trainer = BasicKDTrainer(teachers, student, device,
                                     temperature=config.temperature,
                                     alpha=config.alpha,
                                     beta=config.beta) # ADDED beta
        except TypeError as e:
            logger.error(f"Error initializing BasicKDTrainer. Check its __init__ signature. Error: {e}")
            # This log might still be useful if there's a mismatch in future.
            return None, {} # Critical error, stop

        student_optimizer = torch.optim.AdamW(
            student.parameters(), lr=config.student_lr, weight_decay=config.weight_decay
        )
        optimizers_for_trainer = student_optimizer 

    elif model_type == 'fedkd':
        student = FedKDStudentModel(**student_model_params)
        
        fedkd_loss_fn = FedKDLoss( # Instantiate the loss function for FedKD
            temperature=config.temperature,
            alpha=config.alpha,
            beta=config.beta,
            gamma=config.gamma,
            attention_weight=config.attention_weight,
            quality_threshold=config.quality_threshold,
            temp_decay=config.temp_decay
        )
        try:
            # This call should now work if FedKDTrainer.__init__ accepts loss_fn
            trainer = FedKDTrainer(teachers, student, device, loss_fn=fedkd_loss_fn)
        except TypeError as e:
            logger.error(f"Error initializing FedKDTrainer. Check its __init__ signature. Error: {e}")
            return None, {}


        student_optimizer_fedkd = torch.optim.AdamW(
            student.parameters(), lr=config.student_lr, weight_decay=config.weight_decay
        )
        # Teachers might be fine-tuned in FedKD, so they need optimizers too
        teacher_optimizers_dict_fedkd = {}
        for name, teacher_model_instance in teachers.items():
            if teacher_model_instance is not None: # Ensure teacher exists
                teacher_optimizers_dict_fedkd[name] = torch.optim.AdamW(
                    teacher_model_instance.parameters(),
                    lr=config.teacher_lr * 0.1, # Example: smaller LR for teacher fine-tuning
                    weight_decay=config.weight_decay
                )
        # FedKDTrainer expects a dictionary of optimizers: one for student, and a sub-dict for teachers
        optimizers_for_trainer = {
            'student': student_optimizer_fedkd,
            'teachers': teacher_optimizers_dict_fedkd
        }
    else:
        logger.error(f"Unknown model_type: {model_type} in train_student_model.")
        return None, {}

    if student is None or trainer is None or optimizers_for_trainer is None:
        logger.error(f"Failed to initialize student, trainer, or optimizers for model_type '{model_type}'.")
        return None, {}

    student.to(device)
    # Ensure all teacher models are on the correct device as well
    for teacher_name, teacher_model in teachers.items():
        if teacher_model is not None:
            teacher_model.to(device)

    # --- Training Tracking Variables ---
    train_losses_log = []
    teacher_losses_log = []
    val_metrics_log = {key: [] for key in teachers.keys()} # Store history of val metrics per task
    quality_metrics_log = {} # For FedKD quality statistics per epoch
    best_avg_val_auc = 0.0
    best_student_model_state = None
    epochs_without_improvement = 0


    # --- Training Loop ---
    pbar = tqdm(range(config.student_epochs), desc=f"Training {model_type} student")
    for epoch in pbar:
        epoch_student_loss = 0.0
        epoch_teacher_loss_fedkd = 0.0 # Specific to FedKD's teacher updates
        epoch_loss_details = {}

        try:
            student.train() # Set student to train mode
            for teacher_model in teachers.values(): # Set teachers to eval for KD, unless fine-tuned in FedKD
                if teacher_model is not None:
                    if model_type == 'basic' or (model_type == 'fedkd' and not config.get_distillation_config().get('fine_tune_teachers', False)): # Assuming a config for teacher fine-tuning
                        teacher_model.eval()
                    else: # FedKD with teacher fine-tuning
                         teacher_model.train()


            if model_type == 'basic':
                # BasicKDTrainer.train_epoch expects train_loader and the student_optimizer
                epoch_student_loss, epoch_loss_details = trainer.train_epoch(train_loader, optimizers_for_trainer)
                # epoch_teacher_loss_fedkd remains 0 for basic KD
            elif model_type == 'fedkd':
                # FedKDTrainer.train_epoch expects train_loader, the dict of optimizers, and current epoch
                epoch_student_loss, epoch_teacher_loss_fedkd, epoch_loss_details = trainer.train_epoch(
                    train_loader,
                    optimizers_for_trainer, # The dict: {'student': ..., 'teachers': ...}
                    epoch # Pass current epoch for FedKD adaptive strategies
                )
                # Log FedKD quality statistics if available
                if hasattr(trainer.loss_fn, 'get_quality_statistics'):
                    quality_stats = trainer.loss_fn.get_quality_statistics()
                    if quality_stats:
                        serializable_stats = {k: safe_tensor_item(v) if isinstance(v, torch.Tensor) else v for k, v in quality_stats.items()}
                        quality_metrics_log[epoch] = serializable_stats
            
            train_losses_log.append(safe_tensor_item(epoch_student_loss))
            teacher_losses_log.append(safe_tensor_item(epoch_teacher_loss_fedkd))

        except Exception as e_train_epoch:
            logger.error(f"Error during training epoch {epoch} for {model_type} student: {str(e_train_epoch)}")
            import traceback
            traceback.print_exc() # Print full traceback for debugging
            if epoch < 5: # If errors occur very early, it might be a setup issue
                logger.error("Critical error in early training epochs. Aborting student training.")
                return None, {
                    'train_losses': train_losses_log, 'teacher_losses': teacher_losses_log, 
                    'val_metrics': val_metrics_log, 'quality_metrics': quality_metrics_log,
                    'best_auc': best_avg_val_auc, 'epochs_trained': epoch
                } # Return partial history
            continue # Try to continue to the next epoch

        # --- Validation Phase ---
        student.eval() # Set student to evaluation mode for validation
        current_epoch_val_aucs = []
        current_epoch_val_aps = []
        
        # Prepare to log metrics for each task for this epoch
        current_epoch_task_metrics = {key: {'auc': 0.5, 'ap': 0.0} for key in teachers.keys()}


        with torch.no_grad():
            for cell_line_task in teachers.keys(): # Iterate through each task/cell line
                batch_preds_for_task = []
                batch_labels_for_task = []

                for batch_data in val_loader:
                    batch_data = batch_data.to(device)
                    try:
                        # Student model's forward pass
                        # BasicStudentModel might return dict or single tensor based on 'cell_line' arg
                        # FedKDStudentModel returns a dict of all head outputs
                        if model_type == 'basic':
                            # BasicStudent needs to be called per task if its forward expects a task name
                            # Or it returns all heads and we select. Assuming it returns all heads.
                            student_outputs = student(batch_data.x, batch_data.edge_index, batch_data.edge_attr, batch_data.batch, cell_line='all')
                        else: # fedkd
                            student_outputs = student(batch_data.x, batch_data.edge_index, batch_data.edge_attr, batch_data.batch, cell_line='all', return_hidden=False)

                        # Get the specific prediction for the current cell_line_task
                        pred_for_current_task = None
                        task_key_map = {'normal_safe': 'normal', 'mcc26_active': 'mcc26', 'mkl1_active': 'mkl1'}
                        student_output_key = task_key_map.get(cell_line_task)

                        if isinstance(student_outputs, dict) and student_output_key in student_outputs:
                            pred_for_current_task = student_outputs[student_output_key]
                        elif isinstance(student_outputs, torch.Tensor) and model_type == 'basic' and cell_line_task == 'normal_safe': # Heuristic for single-output basic
                             pred_for_current_task = student_outputs
                        
                        if pred_for_current_task is None:
                            # logger.warning(f"No prediction output for {cell_line_task} from student in validation batch.")
                            continue

                        # Get corresponding labels
                        labels_for_current_task = None
                        if cell_line_task == 'normal_safe': labels_for_current_task = batch_data.y_normal
                        elif cell_line_task == 'mcc26_active': labels_for_current_task = batch_data.y_mcc26
                        elif cell_line_task == 'mkl1_active': labels_for_current_task = batch_data.y_mkl1
                        
                        if labels_for_current_task is not None:
                            batch_preds_for_task.extend(safe_tensor_to_numpy(pred_for_current_task.view(-1)))
                            batch_labels_for_task.extend(safe_tensor_to_numpy(labels_for_current_task.view(-1)))

                    except Exception as e_val_batch:
                        logger.error(f"Error in validation batch for {cell_line_task}: {str(e_val_batch)}")
                        continue # to next batch

                # Calculate metrics for the current cell_line_task after iterating all batches
                task_val_auc = 0.5
                task_val_ap = 0.0 # Default for single class or error
                if len(batch_preds_for_task) > 0 and len(batch_labels_for_task) > 0:
                    np_labels = np.array(batch_labels_for_task)
                    np_preds = np.array(batch_preds_for_task)
                    if len(set(np_labels)) > 1: # Requires at least two classes for AUC/AP
                        try:
                            task_val_auc = roc_auc_score(np_labels, np_preds)
                            task_val_ap = average_precision_score(np_labels, np_preds)
                        except ValueError as e_metrics: # e.g. "Only one class present in y_true"
                            logger.warning(f"Could not calculate AUC/AP for {cell_line_task} (epoch {epoch}): {e_metrics}. Defaulting scores.")
                            # task_val_auc remains 0.5, task_val_ap remains 0.0 or could be based on class counts
                            if sum(np_labels) == 0: task_val_ap = 0.0
                            elif sum(np_labels) == len(np_labels): task_val_ap = 1.0

                    elif len(set(np_labels)) == 1: # Only one class present
                         # logger.debug(f"Only one class present for {cell_line_task} in validation (epoch {epoch}). AUC defaults to 0.5.")
                         if sum(np_labels) == 0 : task_val_ap = 0.0 # All negative
                         elif sum(np_labels) == len(np_labels): task_val_ap = 1.0 # All positive

                current_epoch_val_aucs.append(task_val_auc)
                current_epoch_val_aps.append(task_val_ap)
                val_metrics_log[cell_line_task].append({'auc': task_val_auc, 'ap': task_val_ap}) # Log per epoch
                current_epoch_task_metrics[cell_line_task] = {'auc': task_val_auc, 'ap': task_val_ap}


        # --- Post-Validation Logic (Best Model, Early Stopping, Logging) ---
        avg_epoch_val_auc = np.mean(current_epoch_val_aucs) if current_epoch_val_aucs else 0.0

        if avg_epoch_val_auc > best_avg_val_auc:
            best_avg_val_auc = avg_epoch_val_auc
            if student is not None: # Ensure student model exists
                 best_student_model_state = copy.deepcopy(student.state_dict())
            epochs_without_improvement = 0
            if config.save_checkpoints and student is not None:
                checkpoint_dir = Path(f'models/checkpoints/{model_type}')
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                torch.save({
                    'student_state_dict': best_student_model_state, 'epoch': epoch,
                    'avg_val_auc': best_avg_val_auc, 'val_metrics_at_best': current_epoch_task_metrics, # Save metrics of this best epoch
                    'config': config.to_dict()
                }, checkpoint_dir / 'best_student.pth')
        else:
            epochs_without_improvement += 1

        # Update progress bar
        pbar_postfix = {
            'S_Loss': f'{safe_tensor_item(epoch_student_loss):.4f}',
            'Avg_AUC': f'{avg_epoch_val_auc:.3f}',
            'Best': f'{best_avg_val_auc:.3f}'
        }
        if model_type == 'fedkd':
            pbar_postfix['T_Loss'] = f'{safe_tensor_item(epoch_teacher_loss_fedkd):.4f}'
            if epoch in quality_metrics_log and 'quality_gap_avg' in quality_metrics_log[epoch]: # Check specific key
                pbar_postfix['Q_Gap'] = f"{quality_metrics_log[epoch]['quality_gap_avg']:.3f}"
        pbar.set_postfix(pbar_postfix)

        # Detailed epoch logging
        if (epoch + 1) % (config.student_epochs // 5 or 5) == 0 or epoch == config.student_epochs -1 : # Log ~5 times or on last epoch
            log_msg_epoch = f"Epoch {epoch+1:3d}: Student Loss={safe_tensor_item(epoch_student_loss):.4f}, Avg Val AUC={avg_epoch_val_auc:.3f}"
            if model_type == 'fedkd': log_msg_epoch += f", Teacher Loss={safe_tensor_item(epoch_teacher_loss_fedkd):.4f}"
            logger.info(log_msg_epoch)
            for task_name_log, metrics_log in current_epoch_task_metrics.items():
                logger.info(f"  {task_name_log}: AUC={metrics_log['auc']:.3f}, AP={metrics_log['ap']:.3f}")


        # Early stopping check
        if config.early_stopping and epochs_without_improvement >= config.patience:
            logger.info(f"Early stopping triggered at epoch {epoch+1} due to no improvement in Avg Val AUC for {config.patience} epochs.")
            break
            
    # --- End of Training Loop ---

    # Load the best student model state if training was successful
    if best_student_model_state is not None and student is not None:
        student.load_state_dict(best_student_model_state)
        logger.info(f"🏆 Loaded best student model with Avg Val AUC: {best_avg_val_auc:.3f}")
    elif student is None:
        logger.error("Student model was not successfully initialized or trained.")
        return None, {} # Return None if student is None
    else:
        logger.warning(f"No best model state was saved (Best Avg Val AUC remained {best_avg_val_auc:.3f}). Using model from last epoch.")


    # Compile final history dictionary
    final_history = {
        'train_losses': train_losses_log,
        'teacher_losses': teacher_losses_log, # For FedKD, this tracks teacher update losses
        'val_metrics_per_epoch': val_metrics_log, # Detailed epoch-wise validation metrics
        'quality_metrics_per_epoch': quality_metrics_log, # For FedKD
        'best_avg_val_auc': best_avg_val_auc,
        'epochs_trained': epoch + 1 # Total epochs run (completed or early stopped)
    }

    return student, final_history


def evaluate_models(models: Dict,
                           test_loader: DataLoader,
                           device: torch.device = None,
                           save_detailed_results: bool = True) -> Dict:
    """Model evaluation with comprehensive metrics, safe tensor handling,
       and proper handling of single-task teachers."""

    if device is None:
        device = torch.device('cpu')

    logger.info("📊 Model evaluation on test set...")

    results = {}  # For primary metric, e.g., AUC
    detailed_results = {} # For all metrics

    all_evaluation_tasks = ['normal_safe', 'mcc26_active', 'mkl1_active']

    for model_name, model_instance in models.items():
        if model_instance is None:
            logger.warning(f"Model instance for {model_name} is None. Skipping evaluation.")
            continue

        logger.info(f"Evaluating {model_name}...")
        model_instance.eval()
        model_instance.to(device)
        
        # Initialize results for this model if not already present
        if model_name not in results:
            results[model_name] = {}
        if model_name not in detailed_results:
            detailed_results[model_name] = {}

        is_teacher_model = 'Teacher' in model_name
        
        for current_eval_task in all_evaluation_tasks:
            
            teacher_specific_task_name = None
            if is_teacher_model:
                # Try to parse the teacher's specific task from its name, e.g., "Basic Teacher (normal_safe)"
                try:
                    # Regex to find content within the last parentheses
                    import re
                    match = re.search(r'\(([^)]+)\)[^(]*$', model_name)
                    if match:
                        teacher_specific_task_name = match.group(1)
                except Exception as e_parse:
                    logger.warning(f"Could not parse task from teacher model name {model_name}: {e_parse}")

                if teacher_specific_task_name and current_eval_task != teacher_specific_task_name:
                    # This teacher is specialized for a different task.
                    # logger.debug(f"  Skipping evaluation of specialized teacher {model_name} on off-target task {current_eval_task}.")
                    results[model_name][current_eval_task] = "N/A" # Mark as Not Applicable
                    detailed_results[model_name][current_eval_task] = {
                        "status": "Not Applicable (off-target task for specialized teacher)",
                        'auc': None, 'ap': None, 'f1': None, 'accuracy': None, 'precision': None, 'recall': None,
                        'n_samples': 0, 'n_positive': 0
                    }
                    continue # Move to the next evaluation task for this teacher

            # If we're here: it's a student model OR a teacher on its designated task.
            batch_true_labels_list = []
            batch_pred_probs_list = []

            with torch.no_grad():
                for batch_idx, batch_data in enumerate(test_loader):
                    batch_data = batch_data.to(device)
                    
                    # 1. Get true labels for the current_eval_task from the batch
                    labels_for_task = None
                    if current_eval_task == 'normal_safe':
                        labels_for_task = batch_data.y_normal
                    elif current_eval_task == 'mcc26_active':
                        labels_for_task = batch_data.y_mcc26
                    elif current_eval_task == 'mkl1_active':
                        labels_for_task = batch_data.y_mkl1
                    
                    if labels_for_task is None:
                        logger.warning(f"Labels for task '{current_eval_task}' not found in batch {batch_idx}. Skipping batch for this task.")
                        continue

                    # 2. Get model predictions for the current_eval_task
                    predictions_for_task_probs = None
                    try:
                        if is_teacher_model: # And it's the correct task (due to the check above)
                            # Teacher model outputs directly for its specific task
                            pred_output_tensor = model_instance(batch_data.x, batch_data.edge_index, batch_data.edge_attr, batch_data.batch)
                            if hasattr(pred_output_tensor, 'view'):
                                predictions_for_task_probs = pred_output_tensor.view(-1)
                        
                        elif 'Student' in model_name:
                            # Student model outputs a dictionary of predictions for all tasks
                            student_all_task_outputs = model_instance(batch_data.x, batch_data.edge_index, batch_data.edge_attr, batch_data.batch)
                            
                            student_output_key_map = {'normal_safe': 'normal', 'mcc26_active': 'mcc26', 'mkl1_active': 'mkl1'}
                            student_head_key = student_output_key_map.get(current_eval_task)

                            if isinstance(student_all_task_outputs, dict) and student_head_key in student_all_task_outputs:
                                predictions_for_task_probs = student_all_task_outputs[student_head_key].view(-1)
                            else:
                                logger.warning(f"Student model {model_name} did not provide output for head '{student_head_key}' (task: {current_eval_task}) in batch {batch_idx}.")
                        
                        # If predictions were obtained for this batch
                        if predictions_for_task_probs is not None:
                            batch_pred_probs_list.extend(safe_tensor_to_numpy(predictions_for_task_probs))
                            batch_true_labels_list.extend(safe_tensor_to_numpy(labels_for_task.view(-1)))
                        # else: Prediction for this task was None from the model in this batch

                    except Exception as e_batch_eval:
                        logger.error(f"Error during batch {batch_idx} evaluation for {model_name} on {current_eval_task}: {str(e_batch_eval)}")
                        # import traceback; traceback.print_exc() # Uncomment for detailed debugging
                        continue # to next batch
            
            # After processing all batches for the current model and current_eval_task:
            if len(batch_pred_probs_list) > 0 and len(batch_true_labels_list) > 0:
                # Convert collected lists to numpy arrays for scikit-learn
                np_true_labels = np.array(batch_true_labels_list)
                np_pred_probs = np.array(batch_pred_probs_list)
                np_pred_binary = (np_pred_probs > 0.5).astype(int)

                # Check for sufficient variation for metrics
                if (len(set(np_true_labels)) > 1 or \
                   (len(set(np_true_labels)) == 1 and sum(np_true_labels) == 0 and len(np_pred_probs) > 0) or \
                   (len(set(np_true_labels)) == 1 and sum(np_true_labels) == len(np_true_labels) and len(np_pred_probs) > 0)):
                    try:
                        from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score, f1_score
                        
                        auc = roc_auc_score(np_true_labels, np_pred_probs) if len(set(np_true_labels)) > 1 else 0.5
                        ap = average_precision_score(np_true_labels, np_pred_probs) if len(set(np_true_labels)) > 1 else \
                             (0.0 if sum(np_true_labels) == 0 else (1.0 if sum(np_true_labels) == len(np_true_labels) else 0.5))
                        
                        accuracy = accuracy_score(np_true_labels, np_pred_binary)
                        precision = precision_score(np_true_labels, np_pred_binary, zero_division=0)
                        recall = recall_score(np_true_labels, np_pred_binary, zero_division=0)
                        f1 = f1_score(np_true_labels, np_pred_binary, zero_division=0)

                        results[model_name][current_eval_task] = auc # Store primary metric
                        detailed_results[model_name][current_eval_task] = {
                            'auc': auc, 'ap': ap, 'accuracy': accuracy, 
                            'precision': precision, 'recall': recall, 'f1': f1,
                            'n_samples': len(np_true_labels), 
                            'n_positive': int(sum(np_true_labels)),
                            'status': 'Evaluated'
                        }
                        logger.info(f"  {model_name} - {current_eval_task}: AUC={auc:.3f}, AP={ap:.3f}, F1={f1:.3f}")

                    except Exception as e_metrics_calc:
                        logger.error(f"Error calculating metrics for {model_name} - {current_eval_task}: {str(e_metrics_calc)}")
                        results[model_name][current_eval_task] = 0.0 # Fallback
                        detailed_results[model_name][current_eval_task] = {
                            'auc': 0.0, 'ap': 0.0, 'f1': 0.0, 'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0,
                            'n_samples': len(np_true_labels), 'n_positive': int(sum(np_true_labels)),
                            'status': f'Error in metrics calculation: {e_metrics_calc}'
                        }
                else: # Insufficient label variation after collecting all batches
                    logger.warning(f"  {model_name} - {current_eval_task}: Insufficient label variation for full metric calculation. Collected {len(np_true_labels)} labels ({int(sum(np_true_labels))} positive).")
                    results[model_name][current_eval_task] = 0.5 # AUC for no info
                    detailed_results[model_name][current_eval_task] = {
                        'auc': 0.5, 
                        'ap': (0.0 if sum(np_true_labels) == 0 else (1.0 if sum(np_true_labels) == len(np_true_labels) else 0.5)) if len(np_pred_probs)>0 else 0.0,
                        'f1': 0.0, 'accuracy': float(sum(np_true_labels == np_pred_binary))/len(np_true_labels) if len(np_true_labels)>0 else 0.0,
                        'precision':0.0, 'recall':0.0,
                        'n_samples': len(np_true_labels), 'n_positive': int(sum(np_true_labels)),
                        'status': 'Insufficient label variation'
                    }
            else: # No predictions or labels were collected for this model on this task across all batches
                # This case should be rare if test_loader is not empty and models/labels are correctly defined.
                # Avoid re-logging if it was already marked N/A for an off-target teacher.
                if not (is_teacher_model and current_eval_task != teacher_specific_task_name):
                    logger.warning(f"  {model_name} - {current_eval_task}: No predictions/labels collected after all batches. Evaluation skipped.")
                    results[model_name][current_eval_task] = 0.0
                    detailed_results[model_name][current_eval_task] = {
                        'auc': 0.0, 'ap': 0.0, 'f1': 0.0, 'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0,
                        'n_samples': 0, 'n_positive': 0,
                        'status': 'No predictions or labels collected'
                    }
    
    # Save detailed results
    if save_detailed_results:
        Path('results').mkdir(parents=True, exist_ok=True)
        # Ensure all numerical values in detailed_results are Python native types for JSON
        serializable_detailed_results = {}
        for m_name, tasks in detailed_results.items():
            serializable_detailed_results[m_name] = {}
            for t_name, metrics in tasks.items():
                serializable_detailed_results[m_name][t_name] = {}
                for metric_key, metric_val in metrics.items():
                    if isinstance(metric_val, (np.generic, np.ndarray)):
                        serializable_detailed_results[m_name][t_name][metric_key] = metric_val.item() if metric_val.size == 1 else metric_val.tolist()
                    elif isinstance(metric_val, (float, int, str)) or metric_val is None:
                        serializable_detailed_results[m_name][t_name][metric_key] = metric_val
                    else:
                        serializable_detailed_results[m_name][t_name][metric_key] = str(metric_val) # Fallback to string

        with open('results/detailed_evaluation_results.json', 'w') as f:
            json.dump(serializable_detailed_results, f, indent=2)
        logger.info("Detailed evaluation results saved to results/detailed_evaluation_results.json")

    return results

def train_random_forest_baseline(df: pd.DataFrame,
                                        feature_names: List[str],
                                        train_idx: List[int],
                                        test_idx: List[int],
                                        config: TrainingConfig = None) -> Dict:
    """Random Forest baseline with comprehensive evaluation"""

    logger.info(f"🌲 Training Random Forest baseline (FAIR: {len(train_idx)} train samples, same as KD models)...")

    if config is None:
        config = TrainingConfig()

    X = df[feature_names].values
    train_indices = train_idx  # FAIR: Use same 70% as KD models

    rf_results = {}
    rf_detailed = {}

    for cell_line in ['normal_safe', 'mcc26_active', 'mkl1_active']:
        if cell_line not in df.columns:
            logger.warning(f"Cell line {cell_line} not found in data")
            continue

        y = df[cell_line].values
        X_train, X_test = X[train_indices], X[test_idx]
        y_train, y_test = y[train_indices], y[test_idx]

        # Random Forest with hyperparameter tuning
        rf = RandomForestClassifier(
            n_estimators=300,  # Increased for better performance
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'  # Handle class imbalance
        )

        try:
            rf.fit(X_train, y_train)

            # Comprehensive evaluation
            y_pred_proba = rf.predict_proba(X_test)[:, 1]
            y_pred = rf.predict(X_test)

            # Calculate metrics
            from sklearn.metrics import (
                accuracy_score, precision_score, recall_score,
                f1_score, average_precision_score
            )
            
            auc = roc_auc_score(y_test, y_pred_proba) if len(set(y_test)) > 1 else 0.5
            ap = average_precision_score(y_test, y_pred_proba) if len(set(y_test)) > 1 else (0.0 if sum(y_test)==0 else 1.0 if sum(y_test)==len(y_test) else 0.5)

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)

            rf_results[cell_line] = auc
            rf_detailed[cell_line] = {
                'auc': auc,
                'ap': ap,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'feature_importance': rf.feature_importances_.tolist()
            }

            logger.info(f"  RF {cell_line}: AUC={auc:.3f}, AP={ap:.3f}, F1={f1:.3f}")

        except Exception as e:
            logger.error(f"Error training RF for {cell_line}: {str(e)}")
            rf_results[cell_line] = 0.5
            rf_detailed[cell_line] = {
                'auc': 0.5, 'ap': 0.5, 'accuracy': 0.5, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0,
                'feature_importance': [0.0] * len(feature_names)
            }

    # Save detailed RF results
    Path('results').mkdir(exist_ok=True)
    with open('results/rf_detailed_results.json', 'w') as f:
        json.dump(rf_detailed, f, indent=2)

    return rf_results

def create_comparison_plots(basic_results: Dict,
                                   fedkd_results: Dict,
                                   rf_results: Dict,
                                   save_individual_plots: bool = True) -> None:
    """Create comparison visualizations"""

    logger.info("📊 Creating comparison visualizations...")

    Path('results/plots').mkdir(parents=True, exist_ok=True)

    # Set style for better plots
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")

    # Main comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Knowledge Distillation Model Comparison', fontsize=16, fontweight='bold')

    cell_lines = ['normal_safe', 'mcc26_active', 'mkl1_active']
    cell_names = ['Normal Safety', 'MCC26 Activity', 'MKL1 Activity']

    # Plot 1: Performance comparison
    ax1 = axes[0, 0]
    x = np.arange(len(cell_lines))
    width = 0.25

    basic_aucs = [basic_results.get('Basic Student', {}).get(cl, 0) for cl in cell_lines]
    fedkd_aucs = [fedkd_results.get('FedKD Student', {}).get(cl, 0) for cl in cell_lines]
    rf_aucs_plot = [rf_results.get(cl, 0) for cl in cell_lines] # Renamed to avoid conflict

    bars1 = ax1.bar(x - width, basic_aucs, width, label='Basic KD', alpha=0.8)
    bars2 = ax1.bar(x, fedkd_aucs, width, label='FedKD', alpha=0.8)
    bars3 = ax1.bar(x + width, rf_aucs_plot, width, label='Random Forest', alpha=0.8)

    # Add value labels
    for bars_group in [bars1, bars2, bars3]: # Renamed to avoid conflict
        for bar in bars_group:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax1.set_xlabel('Cell Lines', fontweight='bold')
    ax1.set_ylabel('AUC Score', fontweight='bold')
    ax1.set_title('Model Performance Comparison', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(cell_names, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.4, 1.0)

    # Plot 2: Improvement analysis
    ax2 = axes[0, 1]

    basic_improvements = [basic_aucs[i] - rf_aucs_plot[i] for i in range(len(cell_lines))]
    fedkd_improvements = [fedkd_aucs[i] - rf_aucs_plot[i] for i in range(len(cell_lines))]

    bars1_imp = ax2.bar(x - width/2, basic_improvements, width, label='Basic KD vs RF', alpha=0.8) # Renamed
    bars2_imp = ax2.bar(x + width/2, fedkd_improvements, width, label='FedKD vs RF', alpha=0.8) # Renamed

    # Add value labels
    for bars_group in [bars1_imp, bars2_imp]: # Renamed
        for bar in bars_group:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2.,
                    height + 0.005 if height > 0 else height - 0.005,
                    f'{height:+.3f}', ha='center',
                    va='bottom' if height > 0 else 'top', fontsize=9, fontweight='bold')

    ax2.set_xlabel('Cell Lines', fontweight='bold')
    ax2.set_ylabel('AUC Improvement over RF', fontweight='bold')
    ax2.set_title('Improvement Analysis', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(cell_names, rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)

    # Plot 3: Model comparison heatmap
    ax3 = axes[1, 0]

    comparison_data = np.array([basic_aucs, fedkd_aucs, rf_aucs_plot])
    im = ax3.imshow(comparison_data, cmap='RdYlGn', aspect='auto', vmin=0.4, vmax=1.0)

    ax3.set_xticks(range(len(cell_names)))
    ax3.set_xticklabels(cell_names, rotation=45)
    ax3.set_yticks(range(3))
    ax3.set_yticklabels(['Basic KD', 'FedKD', 'Random Forest'])
    ax3.set_title('Performance Heatmap', fontweight='bold')

    # Add text annotations
    for i in range(3):
        for j_idx in range(len(cell_names)): # Renamed j
            ax3.text(j_idx, i, f'{comparison_data[i, j_idx]:.3f}',
                           ha="center", va="center", fontweight='bold')

    cbar = plt.colorbar(im, ax=ax3)
    cbar.set_label('AUC Score', fontweight='bold')

    # Plot 4: Statistical summary
    ax4 = axes[1, 1]

    model_names_plot = ['Basic KD', 'FedKD', 'Random Forest'] # Renamed
    means = [np.mean(basic_aucs), np.mean(fedkd_aucs), np.mean(rf_aucs_plot)]
    stds = [np.std(basic_aucs), np.std(fedkd_aucs), np.std(rf_aucs_plot)]

    bars_stat = ax4.bar(model_names_plot, means, yerr=stds, capsize=5, alpha=0.8) # Renamed

    # Add value labels
    for i, (bar, mean_val, std_val) in enumerate(zip(bars_stat, means, stds)): # Renamed mean, std
        ax4.text(bar.get_x() + bar.get_width()/2., mean_val + std_val + 0.01,
                f'{mean_val:.3f}±{std_val:.3f}', ha='center', va='bottom', fontweight='bold')

    ax4.set_ylabel('Average AUC Score', fontweight='bold')
    ax4.set_title('Statistical Summary', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.set_xticklabels(model_names_plot, rotation=45)

    plt.tight_layout()

    # Save main plot
    main_plot_file = 'results/plots/kd_comparison.png'
    plt.savefig(main_plot_file, dpi=300, bbox_inches='tight')
    logger.info(f"Main comparison plot saved to: {main_plot_file}")

    if save_individual_plots:
        # plt.show() # Comment out for non-interactive environments
        pass
    else:
        plt.close()

    # Additional individual plots if requested
    if save_individual_plots:
        # Individual performance plots for each cell line
        for i, (cell_line_key, cell_name_val) in enumerate(zip(cell_lines, cell_names)): # Renamed
            fig_ind, ax_ind = plt.subplots(figsize=(8, 6)) # Renamed fig, ax

            model_names_ind = ['Basic KD', 'FedKD', 'Random Forest'] # Renamed
            values_ind = [basic_aucs[i], fedkd_aucs[i], rf_aucs_plot[i]] # Renamed

            bars_ind = ax_ind.bar(model_names_ind, values_ind, alpha=0.8) # Renamed

            for bar, value_item in zip(bars_ind, values_ind): # Renamed value
                ax_ind.text(bar.get_x() + bar.get_width()/2., value_item + 0.01,
                       f'{value_item:.3f}', ha='center', va='bottom', fontweight='bold')

            ax_ind.set_ylabel('AUC Score', fontweight='bold')
            ax_ind.set_title(f'{cell_name_val} Performance Comparison', fontweight='bold')
            ax_ind.set_ylim(0, 1.0)
            ax_ind.grid(True, alpha=0.3)

            individual_plot_file = f'results/plots/{cell_line_key}_comparison.png'
            plt.savefig(individual_plot_file, dpi=300, bbox_inches='tight')
            logger.info(f"Individual plot saved to: {individual_plot_file}")
            plt.close(fig_ind) # Close specific figure


def save_models(models: Dict,
                        model_type: str,
                        config: TrainingConfig, # config is an instance of TrainingConfig
                        results: Dict) -> None: # results is pipeline_results
    """Save models with comprehensive metadata - FIXED PYTORCH GEOMETRIC PICKLING"""

    logger.info(f"💾 Saving {model_type} models...")

    # Create model directories
    model_dir = Path(f'models/{model_type}')
    model_dir.mkdir(parents=True, exist_ok=True)

    # Save models with FIXED pickling for PyTorch Geometric
    if 'teachers' in models and 'student' in models:
        try:
            # Create comprehensive checkpoint - FIXED for PyTorch Geometric
            # `results` parameter is `pipeline_results`
            # `models` parameter is `pipeline_results['models']`
            # `config` parameter is the `TrainingConfig` instance
            checkpoint = {
                'model_type': model_type,
                'config': config.to_dict(),
                # Store only serializable parts of 'results' (which is pipeline_results)
                'evaluation_metrics': results.get('results', {}),
                'rf_evaluation_metrics': results.get('rf_results', {}),
                'training_histories': results.get('histories', {}),
                'data_split_info': results.get('data_splits', {}),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'model_architecture': {
                    # Get these from the config object for consistency
                    'node_features': 9,  # Assuming this is fixed based on data_processing.py
                    'edge_features': 3,  # Assuming this is fixed based on data_processing.py
                    'teacher_hidden_dim': config.teacher_hidden_dim,
                    'student_hidden_dim': config.student_hidden_dim,
                    'num_layers': config.num_layers, # Add other relevant arch params from config
                    'dropout': config.dropout
                }
            }

            # Save student model - only state dict to avoid pickling issues
            if models['student'] is not None: # `models` here is `pipeline_results['models']`
                checkpoint['student_state_dict'] = models['student'].state_dict()

            # Save teacher models - only state dicts
            teacher_states = {}
            for name, teacher in models['teachers'].items(): # `models` here is `pipeline_results['models']`
                if teacher is not None:
                    teacher_states[name] = teacher.state_dict()
            checkpoint['teacher_state_dicts'] = teacher_states

            # Save to appropriate file
            if model_type == 'basic':
                save_path = 'models/gnn_trained_models.pth'
            else:
                save_path = f'models/{model_type}_trained_model.pth'

            # Use torch.save with protocol 4 for better compatibility
            torch.save(checkpoint, save_path, pickle_protocol=4)
            logger.info(f"Model checkpoint saved to: {save_path}")

            # Save model summary separately as JSON (more reliable)
            summary_path = model_dir / 'model_summary.json'
            # Ensure `models['student']` and `models['teachers'].values()` are not None before accessing parameters
            student_params = 0
            if models.get('student'):
                student_params = sum(p.numel() for p in models['student'].parameters() if p.requires_grad)
            
            total_teacher_params = 0
            if models.get('teachers'):
                for teacher in models['teachers'].values():
                    if teacher:
                        total_teacher_params += sum(p.numel() for p in teacher.parameters() if p.requires_grad)

            summary = {
                'model_type': model_type,
                'num_teachers': len(models.get('teachers', {})),
                'student_parameters': student_params,
                'total_teacher_parameters': total_teacher_params,
                'config': config.to_dict(),
                # Use the 'evaluation_metrics' from the checkpoint for consistency
                'performance_summary': checkpoint.get('evaluation_metrics', {}),
                'timestamp': checkpoint['timestamp']
            }


            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)

            logger.info(f"Model summary saved to: {summary_path}")

        except Exception as e:
            logger.error(f"Error saving models with full checkpoint: {str(e)}") # This error will now be less likely
            logger.info("Attempting simplified save (state dicts only)...") # This fallback remains

            try:
                # Fallback: Save only state dictionaries
                simplified_checkpoint = {
                    'model_type': model_type,
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'model_architecture': {
                        'node_features': 9,
                        'edge_features': 3,
                        'teacher_hidden_dim': config.teacher_hidden_dim,
                        'student_hidden_dim': config.student_hidden_dim,
                        'num_layers': config.num_layers,
                        'dropout': config.dropout
                    },
                    'config_used': config.to_dict() # Save the config used
                }

                # Save student state dict
                if models.get('student') is not None:
                    simplified_checkpoint['student_state_dict'] = models['student'].state_dict()

                # Save teacher state dicts
                teacher_states = {}
                if models.get('teachers') is not None:
                    for name, teacher in models['teachers'].items():
                        if teacher is not None:
                            teacher_states[name] = teacher.state_dict()
                simplified_checkpoint['teacher_state_dicts'] = teacher_states
                
                # Also save evaluation results if available from the `results` (pipeline_results) argument
                if results and 'results' in results:
                    simplified_checkpoint['evaluation_metrics'] = results.get('results', {})
                if results and 'rf_results' in results:
                    simplified_checkpoint['rf_evaluation_metrics'] = results.get('rf_results', {})


                # Save with simplified checkpoint
                if model_type == 'basic':
                    save_path = 'models/gnn_trained_models.pth'
                else:
                    save_path = f'models/{model_type}_trained_model.pth'

                torch.save(simplified_checkpoint, save_path, pickle_protocol=4)
                logger.info(f"Simplified model checkpoint saved to: {save_path}")

                # Save config and results separately as JSON
                config_path = model_dir / 'training_config.json'
                with open(config_path, 'w') as f:
                    json.dump(config.to_dict(), f, indent=2)

                results_path = model_dir / 'training_results.json'
                # Make results JSON serializable
                serializable_eval_results = {} # Renamed to avoid conflict with `results` argument
                if results and 'results' in results: # `results` is `pipeline_results`
                    # Access the actual evaluation metrics dict: results['results']
                    for key, value in results['results'].items():
                        if isinstance(value, dict):
                            serializable_eval_results[key] = {k: float(v) if isinstance(v, (int, float, np.float32, np.float64)) else str(v)
                                                       for k, v in value.items()}
                        else:
                            serializable_eval_results[key] = float(value) if isinstance(value, (int, float, np.float32, np.float64)) else str(value)
                
                if results and 'rf_results' in results:
                     serializable_eval_results['rf_results'] = {
                         k: float(v) if isinstance(v, (int, float, np.float32, np.float64)) else str(v)
                         for k, v in results['rf_results'].items()
                     }


                with open(results_path, 'w') as f:
                    json.dump(serializable_eval_results, f, indent=2)

                logger.info(f"Training config saved to: {config_path}")
                logger.info(f"Training results saved to: {results_path}")

            except Exception as e2:
                logger.error(f"Failed to save models even with simplified approach: {str(e2)}")
                # Not raising e2 to allow pipeline to finish, but error is logged.

def train_pipeline(model_type: str = 'basic',
                          config: Optional[TrainingConfig] = None) -> Optional[Dict]:
    """Main training pipeline with comprehensive monitoring and fixed CUDA handling"""

    if config is None:
        config = TrainingConfig()

    logger.info(f"🧬 {model_type.upper()} KNOWLEDGE DISTILLATION PIPELINE")
    logger.info("="*80)

    # Setup device
    device = setup_device(config)

    # Load data
    logger.info("📊 Loading processed data...")
    df, graph_data, feature_names = load_processed_data()
    if df is None:
        logger.error("Failed to load processed data")
        return None

    logger.info(f"Loaded {len(df)} compounds with {len(feature_names)} features")

    # Add labels and create splits
    graph_data = add_labels_to_graphs(graph_data, df)
    train_loader, val_loader, test_loader, train_idx, val_idx, test_idx = create_data_splits(
        graph_data, df, config
    )

    # Phase 1: Teacher training
    logger.info(f"\n🎓 Phase 1: Teacher Training")
    teachers = {}
    teacher_histories = {}

    for cell_line in ['normal_safe', 'mcc26_active', 'mkl1_active']:
        if cell_line not in df.columns:
            logger.warning(f"Cell line {cell_line} not found in data, skipping...")
            continue

        teacher, history = train_teacher_model(
            train_loader, val_loader, cell_line, model_type, config, device
        )
        teachers[cell_line] = teacher
        teacher_histories[f'teacher_{cell_line}'] = history

    if not teachers:
        logger.error("No teachers were successfully trained")
        return None

    # Phase 2: Student training
    logger.info(f"\n🎓 Phase 2: Student Training")
    student, student_history = train_student_model(
        train_loader, val_loader, teachers, model_type, config, device
    )
    if student is None: # Check if student training failed
        logger.error(f"{model_type} student training failed.")
        return None


    # Phase 3: Evaluation
    logger.info(f"\n📊 Phase 3: Evaluation")
    models_to_evaluate = {f'{model_type.title()} Student': student}

    # Add teachers for evaluation
    for cell_line_key, teacher_model in teachers.items(): # Renamed
        models_to_evaluate[f'{model_type.title()} Teacher ({cell_line_key})'] = teacher_model

    results = evaluate_models(models_to_evaluate, test_loader, device)

    # Random Forest baseline
    rf_results = train_random_forest_baseline(df, feature_names, train_idx, test_idx, config)

    # Compile comprehensive results
    pipeline_results = {
        'results': results,
        'rf_results': rf_results,
        'models': {'teachers': teachers, 'student': student},
        'histories': {**teacher_histories, 'student': student_history},
        'config': config.to_dict(),
        'data_splits': {
            'train_size': len(train_idx),
            'val_size': len(val_idx),
            'test_size': len(test_idx)
        }
    }

    # Save models
    save_models(pipeline_results['models'], model_type, config, pipeline_results)

    logger.info(f"✅ {model_type} pipeline completed successfully!")

    return pipeline_results

def main(config_dict: Optional[Dict[str, Any]] = None) -> Tuple[Optional[Dict], Optional[Dict]]:
    """Main function for training and comparing both models with safe tensor handling"""

    logger.info("🚀 COMPREHENSIVE KD COMPARISON PIPELINE")
    logger.info("="*80)

    # Create configurations
    basic_config = TrainingConfig(config_dict)
    # Basic KD might not use additional features by design, or this can be a config choice
    # basic_config.use_features = False # Example: if Basic GNN doesn't use other features

    fedkd_config = TrainingConfig(config_dict)
    # FedKD might use all features by design
    # fedkd_config.use_features = True

    # Train Basic KD
    logger.info("\n" + "="*50)
    logger.info("TRAINING BASIC KD")
    logger.info("="*50)
    basic_pipeline = train_pipeline('basic', basic_config)

    # Train FedKD
    logger.info("\n" + "="*50)
    logger.info("TRAINING FEDKD")
    logger.info("="*50)
    fedkd_pipeline = train_pipeline('fedkd', fedkd_config)

    if basic_pipeline and fedkd_pipeline:
        # Create comparison visualization
        create_comparison_plots(
            basic_pipeline['results'],
            fedkd_pipeline['results'],
            basic_pipeline['rf_results'] # Assuming RF results are consistent or re-run if needed
        )

        # Comprehensive final summary
        logger.info(f"\n" + "="*80)
        logger.info("🏆 FINAL COMPARISON SUMMARY")
        logger.info("="*80)

        cell_lines = ['normal_safe', 'mcc26_active', 'mkl1_active']

        # Calculate comprehensive statistics
        basic_aucs = [basic_pipeline['results'].get('Basic Student', {}).get(cl, 0) for cl in cell_lines]
        fedkd_aucs = [fedkd_pipeline['results'].get('FedKD Student', {}).get(cl, 0) for cl in cell_lines]
        rf_aucs = [basic_pipeline['rf_results'].get(cl, 0) for cl in cell_lines]

        basic_avg = np.mean(basic_aucs) if basic_aucs else 0
        fedkd_avg = np.mean(fedkd_aucs) if fedkd_aucs else 0
        rf_avg = np.mean(rf_aucs) if rf_aucs else 0
        
        basic_std = np.std(basic_aucs) if basic_aucs else 0
        fedkd_std = np.std(fedkd_aucs) if fedkd_aucs else 0
        rf_std = np.std(rf_aucs) if rf_aucs else 0


        logger.info(f"Average AUC Scores (mean ± std):")
        logger.info(f"  • Basic KD: {basic_avg:.3f} ± {basic_std:.3f}")
        logger.info(f"  • FedKD: {fedkd_avg:.3f} ± {fedkd_std:.3f}")
        logger.info(f"  • Random Forest: {rf_avg:.3f} ± {rf_std:.3f}")

        logger.info(f"\nPerformance Improvements:")
        fedkd_vs_basic = fedkd_avg - basic_avg
        fedkd_vs_rf = fedkd_avg - rf_avg
        basic_vs_rf = basic_avg - rf_avg
        
        # Avoid division by zero if basic_avg or rf_avg is 0
        fedkd_vs_basic_perc = (fedkd_vs_basic / basic_avg * 100) if basic_avg != 0 else float('inf')
        fedkd_vs_rf_perc = (fedkd_vs_rf / rf_avg * 100) if rf_avg != 0 else float('inf')
        basic_vs_rf_perc = (basic_vs_rf / rf_avg * 100) if rf_avg != 0 else float('inf')


        logger.info(f"  • FedKD vs Basic KD: {fedkd_vs_basic:+.3f} ({fedkd_vs_basic_perc:+.1f}%)")
        logger.info(f"  • FedKD vs RF: {fedkd_vs_rf:+.3f} ({fedkd_vs_rf_perc:+.1f}%)")
        logger.info(f"  • Basic KD vs RF: {basic_vs_rf:+.3f} ({basic_vs_rf_perc:+.1f}%)")


        # Statistical significance testing
        p_value_ttest = None # Initialize
        try:
            from scipy import stats
            if len(basic_aucs) > 1 and len(fedkd_aucs) > 1 and len(basic_aucs) == len(fedkd_aucs):
                # Filter out any non-numeric if they somehow got in (e.g. None for failed tasks)
                valid_basic_aucs = [auc for auc in basic_aucs if isinstance(auc, (int, float))]
                valid_fedkd_aucs = [auc for auc in fedkd_aucs if isinstance(auc, (int, float))]
                if len(valid_basic_aucs) == len(valid_fedkd_aucs) and len(valid_basic_aucs) > 1:
                    t_stat, p_value_ttest = stats.ttest_rel(valid_fedkd_aucs, valid_basic_aucs)
                    logger.info(f"\nStatistical Analysis:")
                    logger.info(f"  • Paired t-test p-value: {p_value_ttest:.4f}")
                    if p_value_ttest < 0.05:
                        logger.info(f"  • ✅ FedKD significantly outperforms Basic KD (p < 0.05)")
                    else:
                        logger.info(f"  • 🤔 No significant difference between models (p ≥ 0.05)")
                else:
                    logger.warning("Not enough valid paired data points for t-test.")
            else:
                logger.warning("Paired t-test requires at least 2 paired samples for each group.")
        except ImportError:
            logger.warning("SciPy not available for statistical testing")


        # Model complexity analysis
        if 'models' in basic_pipeline and basic_pipeline['models'].get('student') and \
           'models' in fedkd_pipeline and fedkd_pipeline['models'].get('student'):
            basic_student_params = sum(p.numel() for p in basic_pipeline['models']['student'].parameters() if p.requires_grad)
            fedkd_student_params = sum(p.numel() for p in fedkd_pipeline['models']['student'].parameters() if p.requires_grad)

            logger.info(f"\nModel Complexity:")
            logger.info(f"  • Basic KD Student: {basic_student_params:,} parameters")
            logger.info(f"  • FedKD Student: {fedkd_student_params:,} parameters")
            param_increase = fedkd_student_params - basic_student_params
            param_increase_perc = (param_increase / basic_student_params * 100) if basic_student_params != 0 else float('inf')
            logger.info(f"  • Parameter increase: {param_increase:,} (+{param_increase_perc:.1f}%)")


        # Quality improvement analysis for FedKD
        if 'histories' in fedkd_pipeline and 'student' in fedkd_pipeline['histories']:
            student_history = fedkd_pipeline['histories']['student']
            if 'quality_metrics' in student_history and student_history['quality_metrics']:
                try:
                    # Get the last available quality metrics
                    last_epoch_key = sorted(student_history['quality_metrics'].keys())[-1]
                    final_quality = student_history['quality_metrics'][last_epoch_key]
                    
                    logger.info(f"\nFedKD Quality Metrics (Epoch {last_epoch_key}):")
                    if 'student_quality_avg' in final_quality:
                        logger.info(f"  • Student Quality: {final_quality['student_quality_avg']:.3f}")
                    if 'teacher_quality_avg' in final_quality:
                        logger.info(f"  • Teacher Quality: {final_quality['teacher_quality_avg']:.3f}")
                    if 'current_temperature' in final_quality:
                        logger.info(f"  • Temperature: {final_quality['current_temperature']:.3f}")
                except (IndexError, KeyError, TypeError) as e:
                    logger.warning(f"Could not retrieve final FedKD quality metrics: {e}")
            else:
                 logger.info(f"\nFedKD completed (quality metrics not available or empty in history).")


        # Recommendation
        logger.info(f"\n🎯 Recommendation:")
        if fedkd_vs_basic > 0.02:  # 2% improvement threshold
            logger.info(f"✅ Use FedKD - shows significant improvement over Basic KD")
        elif fedkd_vs_basic > 0:
            logger.info(f"👍 FedKD shows modest improvement - consider based on computational budget")
        else:
            logger.info(f"🤔 Basic KD performs similarly - consider simpler approach")

        # Save comprehensive results
        comprehensive_results = {
            # Storing full pipeline results can be very large due to models.
            # Consider storing paths or summaries instead.
            # 'basic_pipeline_summary': {k: v for k, v in basic_pipeline.items() if k != 'models'},
            # 'fedkd_pipeline_summary': {k: v for k, v in fedkd_pipeline.items() if k != 'models'},
            'comparison_summary': {
                'basic_avg_auc': basic_avg,
                'fedkd_avg_auc': fedkd_avg,
                'rf_avg_auc': rf_avg,
                'fedkd_improvement_over_basic': fedkd_vs_basic,
                'statistical_significance_p_value': p_value_ttest if p_value_ttest is not None else "N/A"
            },
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }

        Path('results').mkdir(exist_ok=True)
        with open('results/comprehensive_comparison_results.json', 'w') as f:
            json.dump(comprehensive_results, f, indent=2)

        logger.info(f"\n📁 Results saved to:")
        logger.info(f"  • results/comprehensive_comparison_results.json")
        logger.info(f"  • results/plots/kd_comparison.png")
        logger.info(f"  • models/gnn_trained_models.pth (Basic KD)")
        logger.info(f"  • models/fedkd_trained_model.pth (FedKD)")

        return basic_pipeline, fedkd_pipeline

    logger.error("One or both training pipelines failed.")
    return basic_pipeline, fedkd_pipeline


if __name__ == "__main__":
    basic_results, fedkd_results = main()