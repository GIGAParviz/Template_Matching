import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import os
from typing import Dict, List

class LoFTRDataset(Dataset):
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self.image_pairs = self.load_pairs()

    def load_pairs(self) -> List[str]:
        pairs = []
        for file in os.listdir(os.path.join(self.root_dir, "matches")):
            if file.endswith(".npy"):
                pairs.append(file.replace(".npy", ""))
        return pairs


    def __len__(self) -> int:
        return len(self.image_pairs)


    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        pair_name = self.image_pairs[idx]
        match_data = np.load(os.path.join(self.root_dir, "matches", pair_name + ".npy"), allow_pickle=True).item()

        ref_img_name = r"C:\Users\ASUS\OneDrive\Desktop\QATM\dataset_root\Google_17_sample.jpg"
        query_img_name = pair_name + ".jpg"

        ref_img = cv2.imread(os.path.join(self.root_dir, "images", ref_img_name), cv2.IMREAD_GRAYSCALE)
        query_img = cv2.imread(os.path.join(self.root_dir, "images", query_img_name), cv2.IMREAD_GRAYSCALE)
        
        
        
        ref_img = cv2.resize(ref_img, (256, 256)) 
        query_img = cv2.resize(query_img, (256, 256))
        
        ref_tensor = torch.tensor(ref_img, dtype=torch.float32).unsqueeze(0) / 255.0
        query_tensor = torch.tensor(query_img, dtype=torch.float32).unsqueeze(0) / 255.0

        return {
            "image0": ref_tensor,
            "image1": query_tensor,
            "keypoints0": torch.tensor(match_data["keypoints0"], dtype=torch.float32),
            "keypoints1": torch.tensor(match_data["keypoints1"], dtype=torch.float32),
        }



# root_dir = r"C:\Users\ASUS\OneDrive\Desktop\QATM\dataset_root"

# data = LoFTRDataset(root_dir)


# from torch.utils.data import DataLoader

# train_loader = DataLoader(data, batch_size=8, shuffle=True)

# for batch in train_loader:
#     print(batch)
#     break

