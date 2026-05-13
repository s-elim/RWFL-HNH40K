import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import HNH40KBinaryDataset
from model import CascadedRiskAwareFilter

def generate_final_report(model_path, data_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    transform = transforms.Compose([
        transforms.Resize((192, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load only the validation portion for the final plot
    full_dataset = HNH40KBinaryDataset(root_dir=data_dir, transform=transform)
    _, val_dataset = torch.utils.data.random_split(
        full_dataset, [int(0.8 * len(full_dataset)), len(full_dataset) - int(0.8 * len(full_dataset))],
        generator=torch.Generator().manual_seed(42)
    )
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = CascadedRiskAwareFilter().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for images, labels, _ in val_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits.squeeze(-1))
            preds = (probs >= 0.5).float()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Generate Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Non-Hand', 'Hand'], yticklabels=['Non-Hand', 'Hand'])
    plt.title('HNH40K Confusion Matrix (Final)')
    plt.ylabel('Ground Truth')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('hnh40k_confusion_matrix.pdf')

    print(f"Final Validation Accuracy: {sum(cm.diagonal())/len(all_labels):.4f}")
    print("Metrics and plots generated successfully.")

if __name__ == "__main__":
    generate_final_report('./weights/best_cascaded_model.pth', '/data/selim_sarowar/HNH40K/HNH40K_Dataset')