#!/usr/bin/env python3
"""
Basic Knowledge Distillation Models

Simple but effective teacher-student architecture:
- Graph neural network backbone
- Specialized teacher models per cell line  
- Multi-task student model
- Basic knowledge distillation loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GlobalAttention
from torch_geometric.nn import global_mean_pool, global_max_pool

class MolecularGNN(nn.Module):
    """Core Graph Neural Network for molecular representation"""
    
    def __init__(self, node_features=9, edge_features=3, hidden_dim=128, num_layers=4, dropout=0.2):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Initial embeddings
        self.node_embedding = nn.Linear(node_features, hidden_dim)
        self.edge_embedding = nn.Linear(edge_features, hidden_dim)
        
        # Graph attention layers
        self.gat_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        for i in range(num_layers):
            self.gat_layers.append(
                GATConv(hidden_dim, hidden_dim // 8, heads=8, dropout=dropout, 
                       edge_dim=hidden_dim, concat=True)
            )
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            self.dropouts.append(nn.Dropout(dropout))
        
        # Attention pooling
        gate_nn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.global_att_pool = GlobalAttention(gate_nn)
        
        # Final representation
        self.graph_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, x, edge_index, edge_attr, batch):
        # Initial embeddings
        x = self.node_embedding(x)
        edge_attr = self.edge_embedding(edge_attr)
        
        residual = x
        
        # Graph attention layers with residual connections
        for i, (gat, bn, dropout) in enumerate(zip(self.gat_layers, self.batch_norms, self.dropouts)):
            x_new = gat(x, edge_index, edge_attr=edge_attr)
            x_new = bn(x_new)
            x_new = F.relu(x_new)
            x_new = dropout(x_new)
            
            # Residual connection every 2 layers
            if i > 0 and i % 2 == 0:
                x_new = x_new + residual
                residual = x_new
            
            x = x_new
        
        # Global pooling (3 methods)
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x_att = self.global_att_pool(x, batch)
        
        # Combine and process
        graph_repr = torch.cat([x_mean, x_max, x_att], dim=1)
        return self.graph_mlp(graph_repr)

class BasicTeacherModel(nn.Module):
    """Specialized teacher model for one cell line"""
    
    def __init__(self, node_features=9, edge_features=3, hidden_dim=128, num_layers=4, dropout=0.3):
        super().__init__()
        
        self.gnn = MolecularGNN(node_features, edge_features, hidden_dim, num_layers, dropout)
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, edge_index, edge_attr, batch):
        graph_repr = self.gnn(x, edge_index, edge_attr, batch)
        return self.classifier(graph_repr)

class BasicStudentModel(nn.Module):
    """Multi-task student model with shared backbone"""
    
    def __init__(self, node_features=9, edge_features=3, hidden_dim=96, num_layers=3, dropout=0.2):
        super().__init__()
        
        # Smaller GNN backbone
        self.gnn = MolecularGNN(node_features, edge_features, hidden_dim, num_layers, dropout)
        
        # Separate heads for each cell line
        head_dim = hidden_dim // 2
        
        self.head_normal = nn.Sequential(
            nn.Linear(hidden_dim, head_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_dim, head_dim // 2),
            nn.ReLU(),
            nn.Linear(head_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.head_mcc26 = nn.Sequential(
            nn.Linear(hidden_dim, head_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_dim, head_dim // 2),
            nn.ReLU(),
            nn.Linear(head_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.head_mkl1 = nn.Sequential(
            nn.Linear(hidden_dim, head_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_dim, head_dim // 2),
            nn.ReLU(),
            nn.Linear(head_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, edge_index, edge_attr, batch, cell_line='all'):
        graph_repr = self.gnn(x, edge_index, edge_attr, batch)
        
        if cell_line == 'normal':
            return self.head_normal(graph_repr)
        elif cell_line == 'mcc26':
            return self.head_mcc26(graph_repr)
        elif cell_line == 'mkl1':
            return self.head_mkl1(graph_repr)
        else:
            return {
                'normal': self.head_normal(graph_repr),
                'mcc26': self.head_mcc26(graph_repr),
                'mkl1': self.head_mkl1(graph_repr)
            }

class BasicKnowledgeDistillationLoss(nn.Module):
    """Basic knowledge distillation loss function"""
    
    def __init__(self, temperature: float = 3.0, alpha: float = 0.6, beta: float = 0.4): # Added type hints
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha  # Weight for task loss
        self.beta = beta    # Weight for distillation loss
        if not (0 <= alpha <= 1 and 0 <= beta <= 1 and 0 <= alpha + beta <= 1):
             # Or ensure alpha + beta sums to 1, depending on desired behavior
             # For this structure, alpha and beta are independent coefficients for their respective losses.
             # If they are meant to be weights of a convex combination (sum to 1), the logic might differ.
             # The current usage implies: total_loss = self.alpha * task_loss + self.beta * distill_loss
             # So, alpha and beta don't strictly need to sum to 1.
             pass # print(f"Warning: Loss weights alpha({alpha}), beta({beta}) might need adjustment.")
    
    def forward(self, student_pred, teacher_pred, true_labels):
        """Compute combined loss"""
        
        # Flatten tensors for consistency
        student_pred = student_pred.view(-1)
        teacher_pred = teacher_pred.view(-1)
        true_labels = true_labels.view(-1)
        
        # Task loss: student vs true labels
        task_loss = F.binary_cross_entropy(student_pred, true_labels)
        
        # Distillation loss: student vs teacher (with temperature scaling)
        student_logits = torch.logit(student_pred.clamp(1e-7, 1-1e-7)) / self.temperature
        teacher_logits = torch.logit(teacher_pred.clamp(1e-7, 1-1e-7)) / self.temperature
        
        # Use MSE for simplicity (works well for binary classification)
        distill_loss = F.mse_loss(
            torch.sigmoid(student_logits), 
            torch.sigmoid(teacher_logits)
        ) * (self.temperature ** 2)
        
        # Combined loss
        total_loss = self.alpha * task_loss + self.beta * distill_loss
        
        return {
            'total_loss': total_loss,
            'task_loss': task_loss,
            'distill_loss': distill_loss
        }

class BasicKDTrainer:
    """Basic knowledge distillation trainer"""
    
    def __init__(self, teachers, student, device='cpu', 
                 temperature: float = 3.0, alpha: float = 0.6, beta: float = 0.4): # MODIFIED: Added KD params
        super().__init__() # Added super().__init__() if it might inherit in the future, though not strictly needed here.
        self.teachers = teachers
        self.student = student
        self.device = device
        
        # Instantiate loss_fn with passed parameters
        self.loss_fn = BasicKnowledgeDistillationLoss(
            temperature=temperature, 
            alpha=alpha, 
            beta=beta
        )
        
        # Move models to device
        for teacher_model_instance in teachers.values(): # Renamed for clarity
            if teacher_model_instance is not None:
                teacher_model_instance.to(device)
        if student is not None:
            student.to(device)
    
    def train_step(self, batch, optimizer): # optimizer is the student_optimizer
        """Single training step for Basic KD"""
        batch = batch.to(self.device)
        optimizer.zero_grad()
        
        # Get student predictions for all tasks
        # BasicStudentModel's forward returns a dict: {'normal': ..., 'mcc26': ..., 'mkl1': ...}
        student_preds_dict = self.student(batch.x, batch.edge_index, batch.edge_attr, batch.batch, cell_line='all')
        
        total_combined_loss = 0.0 # Sum of (alpha*task_loss + beta*distill_loss) for each task
        
        # Store loss components for logging/details
        # These will be lists, and we'll average them later
        all_task_losses = []
        all_distill_losses = []
        
        num_tasks_processed = 0

        # Train on each cell line task
        for cell_line_task_name, teacher_model in self.teachers.items():
            if teacher_model is None:
                continue # Skip if a teacher for this task is missing

            # Get true labels for the current cell_line_task_name
            labels_for_task = None
            student_pred_for_task = None
            
            student_output_key_map = {'normal_safe': 'normal', 'mcc26_active': 'mcc26', 'mkl1_active': 'mkl1'}
            student_head_key = student_output_key_map.get(cell_line_task_name)

            if cell_line_task_name == 'normal_safe':
                labels_for_task = batch.y_normal.to(self.device)
            elif cell_line_task_name == 'mcc26_active':
                labels_for_task = batch.y_mcc26.to(self.device)
            elif cell_line_task_name == 'mkl1_active':
                labels_for_task = batch.y_mkl1.to(self.device)
            else:
                # Should not happen if teachers.keys() are from the defined set
                continue 
            
            if student_head_key and isinstance(student_preds_dict, dict) and student_head_key in student_preds_dict:
                student_pred_for_task = student_preds_dict[student_head_key]
            else:
                # This case needs careful handling: if BasicStudentModel doesn't return a dict,
                # or the key is missing. For now, assume it returns the dict.
                # If BasicStudentModel returns a single tensor for 'normal' when cell_line='normal'
                # then the student forward call needs to be per-task, which is less efficient.
                # The current student(..., cell_line='all') suggests it returns a dict.
                # print(f"Warning: Student prediction for {student_head_key} (task {cell_line_task_name}) not found in student_preds_dict.")
                continue


            # Get teacher prediction (teacher model is already in eval mode from train_student_model)
            with torch.no_grad():
                # BasicTeacherModel's forward returns a single tensor output
                teacher_pred_for_task = teacher_model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            
            # Compute loss using self.loss_fn
            # loss_fn returns a dict: {'total_loss': ..., 'task_loss': ..., 'distill_loss': ...}
            loss_components = self.loss_fn(student_pred_for_task, teacher_pred_for_task, labels_for_task)
            
            total_combined_loss += loss_components['total_loss']
            all_task_losses.append(loss_components['task_loss'].item())
            all_distill_losses.append(loss_components['distill_loss'].item())
            num_tasks_processed += 1
        
        if num_tasks_processed > 0:
            avg_batch_total_loss = total_combined_loss / num_tasks_processed
            avg_batch_total_loss.backward() # Backpropagate the average loss
            
            # Gradient clipping (applied to student's parameters)
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=1.0) # Max norm from config if available
            
            optimizer.step() # Update student's parameters
            
            # Prepare details for logging
            loss_details_for_batch = {
                'student_total_loss_avg': avg_batch_total_loss.item(),
                'student_task_loss_avg': sum(all_task_losses) / num_tasks_processed if all_task_losses else 0.0,
                'student_distill_loss_avg': sum(all_distill_losses) / num_tasks_processed if all_distill_losses else 0.0
            }
            return avg_batch_total_loss.item(), loss_details_for_batch
        else:
            return 0.0, {} # No tasks processed in this batch

    def train_epoch(self, data_loader, optimizer): # optimizer is the student_optimizer
        """Full training epoch for Basic KD"""
        self.student.train() # Student is in training mode
        for teacher in self.teachers.values(): # Teachers are set to eval mode before student training loop
            if teacher is not None:
                teacher.eval() 
        
        epoch_total_loss_sum = 0.0
        epoch_task_loss_sum = 0.0
        epoch_distill_loss_sum = 0.0
        num_batches_processed = 0
        
        for batch in data_loader: # tqdm can be added here if desired
            batch_avg_total_loss, batch_loss_details = self.train_step(batch, optimizer)
            
            if batch_loss_details: # If the step was productive
                epoch_total_loss_sum += batch_avg_total_loss # This is already an item (float)
                epoch_task_loss_sum += batch_loss_details.get('student_task_loss_avg', 0.0)
                epoch_distill_loss_sum += batch_loss_details.get('student_distill_loss_avg', 0.0)
                num_batches_processed += 1
        
        if num_batches_processed > 0:
            avg_epoch_total_loss = epoch_total_loss_sum / num_batches_processed
            avg_epoch_task_loss = epoch_task_loss_sum / num_batches_processed
            avg_epoch_distill_loss = epoch_distill_loss_sum / num_batches_processed
            
            epoch_details_aggregated = {
                'student_total_loss': avg_epoch_total_loss, # This is the main loss for student's S_Loss
                'student_task_loss': avg_epoch_task_loss,
                'student_distill_loss': avg_epoch_distill_loss
            }
            # The first return value should be the primary loss for progress bar (S_Loss)
            return avg_epoch_total_loss, epoch_details_aggregated 
        else:
            return 0.0, {} # No batches processed in epoch

# Utility functions
def count_parameters(model):
    """Count trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_model_info(teachers, student):
    """Get information about model architectures"""
    info = {
        'student_params': count_parameters(student),
        'teacher_params': {},
        'total_teacher_params': 0
    }
    
    for name, teacher in teachers.items():
        params = count_parameters(teacher)
        info['teacher_params'][name] = params
        info['total_teacher_params'] += params
    
    info['compression_ratio'] = info['total_teacher_params'] / info['student_params']
    
    return info

if __name__ == "__main__":
    # Test the models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create models
    teachers = {
        'normal_safe': BasicTeacherModel(node_features=9, edge_features=3),
        'mcc26_active': BasicTeacherModel(node_features=9, edge_features=3),
        'mkl1_active': BasicTeacherModel(node_features=9, edge_features=3)
    }
    
    student = BasicStudentModel(node_features=9, edge_features=3)
    
    # Get model info
    info = get_model_info(teachers, student)
    
    print("📊 Basic KD Model Architecture:")
    print(f"  • Student parameters: {info['student_params']:,}")
    print(f"  • Total teacher parameters: {info['total_teacher_params']:,}")
    print(f"  • Compression ratio: {info['compression_ratio']:.1f}x")
    print("✅ Basic KD models created successfully!")
