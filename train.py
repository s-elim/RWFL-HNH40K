import torch
import torch.optim as optim
import os

# Import your custom loss
from losses import RiskWeightedFocalLoss

def evaluate_model(model, val_loader, device):
    """
    Rigorous evaluation loop to calculate metrics for the BMVC paper.
    Tracks Accuracy and False Positive Rate (FPR).
    """
    model.eval()
    
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0

    with torch.no_grad():
        for images, labels, _ in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass (Inference mode uses early exit if batch_size=1, 
            # otherwise processes all for rigorous batch metric calculation)
            logits = model(images)
            probabilities = torch.sigmoid(logits.squeeze(-1))
            
            # Threshold at 0.5 for binary classification
            preds = (probabilities >= 0.5).float()
            
            # Calculate confusion matrix components
            for i in range(len(labels)):
                if preds[i] == 1 and labels[i] == 1:
                    true_positives += 1
                elif preds[i] == 0 and labels[i] == 0:
                    true_negatives += 1
                elif preds[i] == 1 and labels[i] == 0:
                    false_positives += 1
                elif preds[i] == 0 and labels[i] == 1:
                    false_negatives += 1

    total = true_positives + true_negatives + false_positives + false_negatives
    accuracy = (true_positives + true_negatives) / total if total > 0 else 0.0
    fpr = false_positives / (false_positives + true_negatives) if (false_positives + true_negatives) > 0 else 0.0
    
    print(f"Val Acc: {accuracy:.4f} | Val FPR: {fpr:.4f} | TP: {true_positives} TN: {true_negatives} FP: {false_positives} FN: {false_negatives}")
    
    return accuracy, fpr

def train_cascaded_model(model, train_loader, val_loader, num_epochs, device, checkpoint_path):
    """
    Dual-objective training loop that balances the fast stem and the deep scrutinizer.
    Includes epoch-by-epoch evaluation and dynamic checkpointing.
    """
    criterion = RiskWeightedFocalLoss(alpha=0.25, gamma=2.0).to(device)
    
    # Optimizer with weight decay for regularization
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Stem loss weighting factor
    early_loss_weight = 0.4 
    best_accuracy = 0.0

    print(f"Starting training for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        model.train()
        epoch_final_loss = 0.0
        epoch_early_loss = 0.0
        
        for batch_idx, (images, labels, risk_weights) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            risk_weights = risk_weights.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass yields both sets of logits during training
            final_logits, early_logits = model(images)
            
            # Squeeze to match label shape [B]
            final_logits = final_logits.squeeze(-1)
            early_logits = early_logits.squeeze(-1)
            
            # Calculate dual objectives
            loss_final = criterion(final_logits, labels, risk_weights)
            loss_early = criterion(early_logits, labels, risk_weights)
            
            # Total loss balances the deep scrutinizer and the fast stem
            total_loss = loss_final + (early_loss_weight * loss_early)
            
            total_loss.backward()
            
            # Gradient clipping to stabilize transformer training
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_final_loss += loss_final.item()
            epoch_early_loss += loss_early.item()
            
        scheduler.step()
        
        print(f"\n--- Epoch [{epoch+1}/{num_epochs}] ---")
        print(f"Train Final Loss: {epoch_final_loss/len(train_loader):.4f} | Train Stem Loss: {epoch_early_loss/len(train_loader):.4f}")
        
        # Rigorous Validation at the end of every epoch
        val_acc, val_fpr = evaluate_model(model, val_loader, device)
        
        # Save best weights based on highest validation accuracy
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            torch.save(model.state_dict(), checkpoint_path)
            print(f"*** New Best Model Saved! (Acc: {best_accuracy:.4f}) ***")