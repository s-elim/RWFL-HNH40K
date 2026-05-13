import os
import glob
import torch
from torch.utils.data import Dataset
from PIL import Image

class HNH40KBinaryDataset(Dataset):
    """
    Dataset loader for the standard binary HNH40K folder structure.
    Target structure expects 'hand' and 'non-hand' subdirectories.
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        
        # Explicit folder paths based on your setup
        self.hand_dir = os.path.join(root_dir, 'hand')
        self.non_hand_dir = os.path.join(root_dir, 'non-hand')
        
        self._build_dataset()

    def _build_dataset(self):
        valid_extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
        
        # 1. Load Hand Images (Label: 1.0, Risk Weight: 1.0)
        if os.path.exists(self.hand_dir):
            for ext in valid_extensions:
                for img_path in glob.glob(os.path.join(self.hand_dir, '**', ext), recursive=True):
                    self.samples.append((img_path, 1.0, 1.0))
        else:
            print(f"Warning: 'hand' directory not found at {self.hand_dir}")

        # 2. Load Non-Hand Images (Label: 0.0, Risk Weight: 1.0)
        # Note: Since they are grouped, we default the risk weight to 1.0 for all.
        if os.path.exists(self.non_hand_dir):
            for ext in valid_extensions:
                for img_path in glob.glob(os.path.join(self.non_hand_dir, '**', ext), recursive=True):
                    self.samples.append((img_path, 0.0, 1.0))
        else:
            print(f"Warning: 'non-hand' directory not found at {self.non_hand_dir}")

        if len(self.samples) == 0:
            raise RuntimeError(f"Found 0 images in {self.root_dir}. Check paths and extensions.")
            
        print(f"Successfully loaded {len(self.samples)} images from {self.root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label_val, weight_val = self.samples[idx]
        
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            # If an image is corrupt, load a dummy black image to prevent training crash
            print(f"Error loading {img_path}: {e}")
            img = Image.new('RGB', (192, 160), (0, 0, 0))

        if self.transform:
            img = self.transform(img)
            
        label = torch.tensor(label_val, dtype=torch.float32)
        weight = torch.tensor(weight_val, dtype=torch.float32)
        
        return img, label, weight