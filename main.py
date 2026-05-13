import os
import argparse
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

# Import your local modules
from dataset import HNH40KBinaryDataset
from model import CascadedRiskAwareFilter
from train import train_cascaded_model, evaluate_model

def main():
    parser = argparse.ArgumentParser(description="HNH40K Cascaded Risk-Aware Filter")
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'evaluate'], help="Mode: 'train' or 'evaluate'")
    parser.add_argument('--data_dir', type=str, default='/data/selim_sarowar/HNH40K/HNH40K_Dataset', help="Path to your dataset folder")
    parser.add_argument('--epochs', type=int, default=50, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size")
    parser.add_argument('--checkpoint', type=str, default='./weights/best_cascaded_model.pth', help="Path to save/load weights")
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Define Image Transforms (Strict standardization to 192x160)
    transform = transforms.Compose([
        transforms.Resize((192, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 2. Load the Full Dataset
    print(f"Loading dataset from {args.data_dir}...")
    try:
        full_dataset = HNH40KBinaryDataset(root_dir=args.data_dir, transform=transform)
    except RuntimeError as e:
        print(f"Dataset Error: {e}")
        return
    
    # 3. Dynamically Split into 80% Train, 20% Val
    dataset_size = len(full_dataset)
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    print(f"Dataset split: {train_size} training samples, {val_size} validation samples.")

    # 4. Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Ensure the weights directory exists
    os.makedirs(os.path.dirname(args.checkpoint), exist_ok=True)

    # 5. Execute Mode
    if args.mode == 'train':
        print("Initializing Training Pipeline...")
        # Initialize the model here and pass it to the train function
        model = CascadedRiskAwareFilter(exit_threshold=0.1).to(device)
        train_cascaded_model(model, train_loader, val_loader, args.epochs, device, args.checkpoint)
        print(f"Training complete. Best weights are saved at {args.checkpoint}")

    elif args.mode == 'evaluate':
        if not os.path.exists(args.checkpoint):
            raise FileNotFoundError(f"Cannot find checkpoint at {args.checkpoint}. Please train the model first.")
            
        print("Initializing Evaluation Pipeline...")
        model = CascadedRiskAwareFilter(exit_threshold=0.1).to(device)
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        
        print("\n--- Final Stress Test Evaluation ---")
        evaluate_model(model, val_loader, device)

if __name__ == "__main__":
    main()