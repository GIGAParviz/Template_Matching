import torch.nn as nn
import torch.optim as optim
import torch
from tqdm import tqdm


def train_loftr(model, train_loader, num_epochs=10, lr=0.001) -> nn.Module:

    model.train()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    downsample_factor = 4  
    
    for epoch in tqdm(range(num_epochs)):
        for batch in train_loader:
            images0 = batch['image0']  
            images1 = batch['image1']
            keypoints0 = batch['keypoints0']  
            keypoints1 = batch['keypoints1']  
            
            optimizer.zero_grad()
            outputs = model({'image0': images0, 'image1': images1})
            scores = outputs['scores']  
            
            b, h, w, h2, w2 = scores.shape

            target = torch.zeros_like(scores)

            for b_idx in range(b):
                pts0 = keypoints0[b_idx]  
                pts1 = keypoints1[b_idx]   
                

                pts0_fs = (pts0 / downsample_factor).long()
                pts1_fs = (pts1 / downsample_factor).long()

                for i in range(pts0_fs.shape[0]):
                    y0, x0 = pts0_fs[i]
                    y1, x1 = pts1_fs[i]

                    if 0 <= y0 < h and 0 <= x0 < w and 0 <= y1 < h2 and 0 <= x1 < w2:
                        target[b_idx, y0, x0, y1, x1] = 1.0

            loss = criterion(scores, target)
            loss.backward()
            optimizer.step()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    print("Training complete.")
    return model