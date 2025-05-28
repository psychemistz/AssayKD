#!/usr/bin/env python3
"""
Fast FedKD Training Configuration
"""

from training import TrainingConfig

def get_fast_fedkd_config():
    """Get a fast-training FedKD configuration"""
    config = TrainingConfig()
    
    # Speed optimizations
    config.batch_size = 64          # Larger batches
    config.teacher_epochs = 40      # Fewer epochs
    config.student_epochs = 80      # Fewer epochs
    
    # Better learning rates
    config.teacher_lr = 2e-3        # Higher
    config.student_lr = 3e-3        # Higher
    config.weight_decay = 5e-4      # Less regularization
    
    # Smaller models for speed
    config.teacher_hidden_dim = 96
    config.student_hidden_dim = 64
    config.num_layers = 3
    config.dropout = 0.1
    
    # Simplified distillation
    config.temperature = 3.0
    config.alpha = 0.8              # More task focus
    config.beta = 0.2               # Less distillation
    
    return config

if __name__ == "__main__":
    config = get_fast_fedkd_config()
    print("Fast FedKD Config created!")
    print(f"Batch size: {config.batch_size}")
    print(f"Student epochs: {config.student_epochs}")
    print(f"Student LR: {config.student_lr}")
