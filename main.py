from engine import LoFTR
from data_builder import LoFTRDataset
from train import train_loftr
from torch.utils.data import DataLoader 

root_dir = r"C:\Users\ASUS\OneDrive\Desktop\QATM\dataset_root"
dataset = LoFTRDataset(root_dir)
train_loader = DataLoader(dataset, batch_size=2, shuffle=True)

import torch
dataset = torch.utils.data.Subset(dataset, range(50))

model = LoFTR()
# model = train_loftr(model, train_loader)

loftr_model = train_loftr(model, train_loader)

