from dataset import HNH40KBinaryDataset
data = HNH40KBinaryDataset(root_dir='/data/selim_sarowar/HNH40K/HNH40K_Dataset')
hands = sum(1 for _, label, _ in data.samples if label == 1.0)
non_hands = sum(1 for _, label, _ in data.samples if label == 0.0)
print(f"Hands: {hands} | Non-Hands: {non_hands}")